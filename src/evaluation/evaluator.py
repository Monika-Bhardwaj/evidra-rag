from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Settings, get_settings
from src.evaluation.metrics import (
    citation_correctness,
    completeness,
    faithfulness,
    numeric_answer_correctness,
    page_correctness,
    retrieval_metrics,
)
from src.pipeline import RagPipeline
from src.schemas import RetrievedChunk
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.query_understanding import QueryClassifier
from src.retrieval.routing import QueryRouter


class Evaluator:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.pipeline = RagPipeline(self.settings)
        self.classifier = QueryClassifier()
        self.expander = QueryExpander()
        self.router = QueryRouter()
        self.results: List[Dict[str, Any]] = []

    def load_questions(self) -> List[Dict[str, Any]]:
        path = self.settings.eval_questions_path_resolved
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["questions"]

    def load_gold_map(self) -> Dict[str, Any]:
        path = self.settings.processed_dir_path / "gold_evidence.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def run(self) -> Dict[str, Any]:
        questions = self.load_questions()
        all_chunks = self.pipeline.vector_store.chunks()
        gold_map = self.load_gold_map()

        for question in questions:
            qid = question["id"]
            question_text = question["question"]
            required = question.get("required_evidence", [])
            key_values = question.get("key_values", [])
            pages = question.get("page", [])
            relevant_ids = {h["chunk_id"] for h in gold_map.get(str(qid), [])}
            if not relevant_ids:
                relevant_ids = None

            intent = self.classifier.classify(question_text)
            expanded = self.expander.expand(question_text, intent)
            route = self.router.route(intent, question_text)

            raw_retrieved, _, _ = self.pipeline.hybrid.retrieve(
                query=question_text,
                expanded_queries=expanded,
                dense_top_k=self.settings.dense_top_k,
                bm25_top_k=self.settings.bm25_top_k,
                top_k=20,
                alpha=self.settings.hybrid_alpha,
                method=self.settings.hybrid_method,
                min_similarity=0.0,
                chunk_types=route.chunk_types or None,
                type_boost=route.type_boost,
            )

            start = time.perf_counter()
            response = self.pipeline.answer(question_text, include_debug=True)
            latency_ms = (time.perf_counter() - start) * 1000.0

            final_evidence = self._evidence_from(response)
            generation_info = response.generation
            row = {
                "id": qid,
                "question": question_text,
                "category": intent.category,
                "pass": bool(
                    response.evidence_sufficient
                    and page_correctness(final_evidence, pages)
                    and all(
                        v
                        for v in numeric_answer_correctness(response.answer, key_values).values()
                    )
                ) if key_values else bool(response.evidence_sufficient),
                "retrieval_raw": retrieval_metrics(raw_retrieved, all_chunks, required, k=6, relevant_ids=relevant_ids),
                "retrieval_final": retrieval_metrics(final_evidence, all_chunks, required, k=6, relevant_ids=relevant_ids),
                "generation": {
                    "numeric_correctness": numeric_answer_correctness(response.answer, key_values),
                    "numeric_recall": (
                        round(
                            sum(
                                1
                                for v in numeric_answer_correctness(
                                    response.answer, key_values
                                ).values()
                                if v
                            )
                            / max(1, len(key_values)),
                            4,
                        )
                        if key_values
                        else 1.0
                    ),
                    "citation_correctness": citation_correctness(response.answer, final_evidence),
                    "faithful": faithfulness(response.answer, final_evidence),
                    "completeness": completeness(response.answer, required),
                    "page_correct": page_correctness(final_evidence, pages),
                    "evidence_sufficient": response.evidence_sufficient,
                },
                "system": {
                    "provider": generation_info.provider if generation_info else None,
                    "model": generation_info.model if generation_info else None,
                    "latency_ms": round(latency_ms, 1),
                    "prompt_tokens": generation_info.prompt_tokens if generation_info else 0,
                    "completion_tokens": generation_info.completion_tokens if generation_info else 0,
                    "est_cost_usd": generation_info.estimated_cost_usd if generation_info else 0.0,
                },
                "evidence": [
                    {
                        "chunk_id": c.chunk.chunk_id,
                        "page": c.chunk.page,
                        "section": c.chunk.section,
                        "chunk_type": c.chunk.chunk_type,
                        "hybrid_score": round(c.hybrid_score, 4),
                    }
                    for c in final_evidence
                ],
                "answer": response.answer,
            }
            self.results.append(row)

        summary = self._summary()
        self._write_results()
        self._render_report()
        self._render_plots()
        return {"summary": summary, "results": self.results}

    def _evidence_from(self, response) -> List[RetrievedChunk]:
        evidence: List[RetrievedChunk] = []
        for source in response.sources:
            chunk = self.pipeline._chunks_by_id.get(source["chunk_id"])
            if chunk is not None:
                evidence.append(
                    RetrievedChunk(
                        chunk=chunk,
                        hybrid_score=1.0,
                        rerank_score=1.0,
                    )
                )
        return evidence

    def _summary(self) -> Dict[str, Any]:
        def avg(key: str, sub: str = "retrieval_raw") -> float:
            values = [
                r[sub][key]
                for r in self.results
                if key in r.get(sub, {}) and isinstance(r[sub][key], (int, float))
            ]
            return round(sum(values) / max(1, len(values)), 4) if values else 0.0

        total_cost = round(sum(r["system"]["est_cost_usd"] for r in self.results), 6)
        total_latency = round(sum(r["system"]["latency_ms"] for r in self.results), 1)
        total_tokens = sum(r["system"]["prompt_tokens"] + r["system"]["completion_tokens"] for r in self.results)
        passes = sum(1 for r in self.results if r["pass"])
        return {
            "n_questions": len(self.results),
            "passed": passes,
            "pass_rate": round(passes / max(1, len(self.results)), 4),
            "recall_at_6": avg("recall_at_k"),
            "precision_at_6": avg("precision_at_k"),
            "mrr": avg("mrr"),
            "ndcg_at_6": avg("ndcg_at_k"),
            "avg_numeric_recall": round(
                sum(r["generation"].get("numeric_recall", 0.0) for r in self.results) / max(1, len(self.results)), 4
            ),
            "avg_completeness": round(
                sum(r["generation"].get("completeness", 0.0) for r in self.results) / max(1, len(self.results)), 4
            ),
            "citation_accuracy": round(
                sum(1 for r in self.results if r["generation"]["citation_correctness"].get("has_citation"))
                / max(1, len(self.results)),
                4,
            ),
            "faithfulness_rate": round(
                sum(1 for r in self.results if r["generation"].get("faithful"))
                / max(1, len(self.results)),
                4,
            ),
            "page_accuracy": round(
                sum(1 for r in self.results if r["generation"].get("page_correct"))
                / max(1, len(self.results)),
                4,
            ),
            "total_latency_ms": total_latency,
            "avg_latency_ms": round(total_latency / max(1, len(self.results)), 1),
            "total_tokens": total_tokens,
            "total_est_cost_usd": total_cost,
            "provider": self.pipeline.llm.provider_name,
            "llm_model": self.pipeline.llm.model,
        }

    def _write_results(self) -> None:
        out = self.settings.processed_dir_path / "evaluation_results.json"
        out.write_text(
            json.dumps({"summary": self._summary(), "results": self.results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _render_report(self) -> None:
        s = self._summary()
        path = self.settings.eval_report_path_resolved
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Evaluation Report — Agent-as-a-Judge RAG")
        lines.append("")
        lines.append(f"- Questions evaluated: **{s['n_questions']}**")
        lines.append(f"- Passed (all key values + page + sufficient evidence): **{s['passed']} / {s['n_questions']}** ({s['pass_rate']:.2%})")
        lines.append(f"- Retrieval Recall@6: **{s['recall_at_6']:.4f}**")
        lines.append(f"- Retrieval Precision@6: **{s['precision_at_6']:.4f}**")
        lines.append(f"- MRR: **{s['mrr']:.4f}**")
        lines.append(f"- nDCG@6: **{s['ndcg_at_6']:.4f}**")
        lines.append(f"- Numeric recall (key values found in answer): **{s['avg_numeric_recall']:.4f}**")
        lines.append(f"- Completeness (required phrases in answer): **{s['avg_completeness']:.4f}**")
        lines.append(f"- Citation accuracy: **{s['citation_accuracy']:.4f}**")
        lines.append(f"- Faithfulness (claims supported by evidence): **{s['faithfulness_rate']:.4f}**")
        lines.append(f"- Page accuracy: **{s['page_accuracy']:.4f}**")
        lines.append(f"- Average latency: **{s['avg_latency_ms']} ms**")
        lines.append(f"- Total tokens: **{s['total_tokens']}**")
        lines.append(f"- Estimated LLM cost: **${s['total_est_cost_usd']:.6f}**")
        lines.append(f"- Generation provider: **{s['provider']}** ({s['llm_model']})")
        lines.append("")
        lines.append("| # | Question | Category | R@6 | nDCG@6 | Num | Cit | Faith | Page | Pass |")
        lines.append("|---|----------|----------|-----|--------|-----|-----|-------|------|------|")
        for r in self.results:
            gen = r["generation"]
            lines.append(
                "| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {} | {} |".format(
                    r["id"],
                    r["question"][:70],
                    r["category"],
                    r["retrieval_raw"]["recall_at_k"],
                    r["retrieval_raw"]["ndcg_at_k"],
                    gen["numeric_recall"],
                    "Y" if gen["citation_correctness"]["has_citation"] else "N",
                    "Y" if gen["faithful"] else "N",
                    "Y" if gen["page_correct"] else "N",
                    "YES" if r["pass"] else "no",
                )
            )
        lines.append("")
        lines.append("## Per-question evidence (top chunks used)")
        for r in self.results:
            lines.append(f"### Q{r['id']} — {r['question']}")
            lines.append(f"**Answer:** {r['answer']}")
            lines.append("")
            evidence_lines = []
            for e in r["evidence"]:
                evidence_lines.append(
                    f"- `{e['chunk_id']}` · page {e['page']} · section '{e['section']}' · type {e['chunk_type']}"
                )
            lines.append("Evidence:" if evidence_lines else "Evidence: _(none — insufficient)_")
            lines.extend(evidence_lines)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _render_plots(self) -> None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "eval_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = [str(r["id"]) for r in self.results]
        recall = [r["retrieval_raw"]["recall_at_k"] for r in self.results]
        ndcg = [r["retrieval_raw"]["ndcg_at_k"] for r in self.results]

        fig, ax = plt.subplots(figsize=(12, 4))
        x = range(len(ids))
        ax.bar([i - 0.2 for i in x], recall, width=0.4, label="Recall@6")
        ax.bar([i + 0.2 for i in x], ndcg, width=0.4, label="nDCG@6")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"Q{i}" for i in ids], rotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Retrieval quality per golden question")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "retrieval_quality.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        numeric = [r["generation"]["numeric_recall"] for r in self.results]
        complete = [r["generation"]["completeness"] for r in self.results]
        ax.bar([i - 0.2 for i in x], numeric, width=0.4, label="Numeric recall")
        ax.bar([i + 0.2 for i in x], complete, width=0.4, label="Completeness")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"Q{i}" for i in ids], rotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Answer quality per golden question")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "answer_quality.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(list(x), [r["system"]["latency_ms"] for r in self.results], color="#1f77b4")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels([f"Q{i}" for i in ids], rotation=45)
        axes[0].set_title("Latency (ms)")
        axes[1].bar(list(x), [r["system"]["est_cost_usd"] for r in self.results], color="#d62728")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels([f"Q{i}" for i in ids], rotation=45)
        axes[1].set_title("Estimated LLM cost (USD)")
        fig.tight_layout()
        fig.savefig(out_dir / "system_metrics.png", dpi=130)
        plt.close(fig)


def main() -> None:
    evaluator = Evaluator(get_settings())
    result = evaluator.run()
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()