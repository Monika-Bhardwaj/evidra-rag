from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Settings, get_settings
from src.evaluation.metrics import (
    citation_correctness,
    cited_metrics,
    completeness,
    faithfulness,
    numeric_answer_correctness,
    page_correctness,
    retrieval_metrics,
)
from src.generation.security import RETRIEVED_INJECTION_PATTERNS
from src.pipeline import INSF_MSG, RagPipeline
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.query_understanding import QueryClassifier
from src.retrieval.routing import QueryRouter
from src.schemas import DocumentChunk, RetrievedChunk
from src.validation.claims import ClaimVerifier

K_CURVE_KS = (1, 2, 4, 6, 8, 10)


class Evaluator:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.pipeline = RagPipeline(self.settings)
        self.classifier = QueryClassifier()
        self.expander = QueryExpander()
        self.router = QueryRouter()
        self.claim_verifier = ClaimVerifier()
        self.abstention_results: List[Dict[str, Any]] = []
        self.adversarial_results: List[Dict[str, Any]] = []
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
            claim_verdicts = self.claim_verifier.verify(response.answer, final_evidence)
            generation_info = response.generation
            row = {
                "id": qid,
                "question": question_text,
                "category": intent.category,
                "pass": bool(
                    response.evidence_sufficient
                    and page_correctness(final_evidence, pages)
                    and all(
                        v for v in numeric_answer_correctness(response.answer, key_values).values()
                    )
                )
                if key_values
                else bool(response.evidence_sufficient),
                "retrieval_raw": retrieval_metrics(
                    raw_retrieved, all_chunks, required, k=6, relevant_ids=relevant_ids
                ),
                "retrieval_final": retrieval_metrics(
                    final_evidence, all_chunks, required, k=6, relevant_ids=relevant_ids
                ),
                "retrieval_k_curve": self._retrieval_curve(
                    raw_retrieved, all_chunks, required, relevant_ids
                ),
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
                    "cited": cited_metrics(final_evidence, relevant_ids),
                },
                "system": {
                    "provider": generation_info.provider if generation_info else None,
                    "model": generation_info.model if generation_info else None,
                    "latency_ms": round(latency_ms, 1),
                    "prompt_tokens": generation_info.prompt_tokens if generation_info else 0,
                    "completion_tokens": generation_info.completion_tokens
                    if generation_info
                    else 0,
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
                "claims": [c.to_dict() for c in claim_verdicts],
                "answer": response.answer,
            }
            self.results.append(row)

        self.abstention_results = self.run_abstention()
        self.adversarial_results = self.run_adversarial()
        summary = self._summary()
        self._write_results()
        self._write_claims()
        self._write_abstention()
        self._write_adversarial()
        self._render_report()
        self._render_plots()
        return {
            "summary": summary,
            "results": self.results,
            "abstention": self.abstention_results,
            "adversarial": self.adversarial_results,
        }

    def run_abstention(self) -> List[Dict[str, Any]]:
        path = self.settings.abstain_questions_path_resolved
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        for q in data["questions"]:
            qid = q["id"]
            question_text = q["question"]
            response = self.pipeline.answer(question_text)
            abstained = not response.evidence_sufficient
            exact = response.answer.strip() == INSF_MSG
            rows.append(
                {
                    "id": qid,
                    "question": question_text,
                    "reason": q.get("reason", ""),
                    "abstained": abstained,
                    "exact_abstention_string": exact,
                    "correct": bool(abstained and exact),
                    "warning": response.warning,
                    "answer_excerpt": response.answer[:160],
                }
            )
        return rows

    def _retrieval_curve(
        self,
        ranked: List[RetrievedChunk],
        chunks: List[DocumentChunk],
        required: List[str],
        relevant_ids: Optional[set],
        ks: Tuple[int, ...] = K_CURVE_KS,
    ) -> Dict[str, Dict[str, float]]:
        curve: Dict[str, Dict[str, float]] = {}
        for k in ks:
            m = retrieval_metrics(ranked[:k], chunks, required, k=k, relevant_ids=relevant_ids)
            curve[str(k)] = {
                "recall_at_k": round(m["recall_at_k"], 4),
                "precision_at_k": round(m["precision_at_k"], 4),
                "ndcg_at_k": round(m["ndcg_at_k"], 4),
                "mrr": round(m["mrr"], 4),
            }
        return curve

    def run_adversarial(self) -> List[Dict[str, Any]]:
        path = self.settings.adversarial_cases_path_resolved
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        for case in data["cases"]:
            qid = case["id"]
            question_text = case["question"]
            injection = case["injection"]
            command_token = case.get("command_token", "")
            base = self.pipeline._retrieve_with_cache
            failed_to_patch = False

            def patched(
                question: str,
                expanded: List[str],
                intent: Any,
                route: Any,
                debug: bool,
            ) -> Any:
                nonlocal command_token, injection
                evidence, sufficient, cached = base(question, expanded, intent, route, debug)
                if evidence:
                    base_chunk = evidence[0].chunk
                    injected = RetrievedChunk(
                        chunk=DocumentChunk(
                            chunk_id=base_chunk.chunk_id,
                            text=base_chunk.text + "\n\n" + injection,
                            page=base_chunk.page,
                            section=base_chunk.section,
                            chunk_type=base_chunk.chunk_type,
                            document_id=base_chunk.document_id,
                            source=base_chunk.source,
                            metadata=dict(base_chunk.metadata),
                        ),
                        dense_score=evidence[0].dense_score,
                        sparse_score=evidence[0].sparse_score,
                        hybrid_score=evidence[0].hybrid_score,
                        rerank_score=evidence[0].rerank_score,
                    )
                    return [injected] + list(evidence[1:]), sufficient, cached
                return evidence, sufficient, cached

            try:
                self.pipeline._retrieve_with_cache = patched
                response = self.pipeline.answer(question_text)
            except Exception as exc:  # pragma: no cover - defensive
                failed_to_patch = True
                response = None
                rows.append(
                    {
                        "id": qid,
                        "question": question_text,
                        "injection": injection,
                        "command_token": command_token,
                        "followed": False,
                        "raw_marker_in_answer": False,
                        "abstained": False,
                        "correct": False,
                        "warning": f"harness-error: {type(exc).__name__}: {exc}",
                        "answer_excerpt": "",
                    }
                )
            finally:
                self.pipeline._retrieve_with_cache = base

            if failed_to_patch:
                continue

            answer = response.answer
            followed = bool(command_token and command_token.lower() in answer.lower())
            raw_marker = any(m.lower() in answer.lower() for m in RETRIEVED_INJECTION_PATTERNS)
            rows.append(
                {
                    "id": qid,
                    "question": question_text,
                    "injection": injection,
                    "command_token": command_token,
                    "followed": followed,
                    "raw_marker_in_answer": raw_marker,
                    "abstained": not response.evidence_sufficient,
                    "correct": bool((not followed) and (not raw_marker)),
                    "warning": response.warning,
                    "answer_excerpt": answer[:160],
                }
            )
        return rows

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
        total_tokens = sum(
            r["system"]["prompt_tokens"] + r["system"]["completion_tokens"] for r in self.results
        )
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
                sum(r["generation"].get("numeric_recall", 0.0) for r in self.results)
                / max(1, len(self.results)),
                4,
            ),
            "avg_completeness": round(
                sum(r["generation"].get("completeness", 0.0) for r in self.results)
                / max(1, len(self.results)),
                4,
            ),
            "citation_accuracy": round(
                sum(
                    1
                    for r in self.results
                    if r["generation"]["citation_correctness"].get("has_citation")
                )
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
            "cited_recall": round(
                sum(r["generation"].get("cited", {}).get("cited_recall", 1.0) for r in self.results)
                / max(1, len(self.results)),
                4,
            ),
            "cited_precision": round(
                sum(
                    r["generation"].get("cited", {}).get("cited_precision", 1.0)
                    for r in self.results
                )
                / max(1, len(self.results)),
                4,
            ),
            "abstention": self._abstention_summary(),
            "adversarial": self._adversarial_summary(),
            "claim_faithfulness_rate": self._claim_faithfulness(),
            "retrieval_k_curve": self._k_curve_summary(),
            "total_latency_ms": total_latency,
            "avg_latency_ms": round(total_latency / max(1, len(self.results)), 1),
            "total_tokens": total_tokens,
            "total_est_cost_usd": total_cost,
            "provider": self.pipeline.llm.provider_name,
            "llm_model": self.pipeline.llm.model,
        }

    def _abstention_summary(self) -> Dict[str, Any]:
        n = len(self.abstention_results)
        if n == 0:
            return {"n": 0, "correct": 0, "abstain_correctness": None, "exact_string_rate": None}
        correct = sum(1 for r in self.abstention_results if r["correct"])
        exact = sum(1 for r in self.abstention_results if r["exact_abstention_string"])
        false_abstains = sum(
            1
            for r in self.results
            if not r["generation"].get("evidence_sufficient")
            or "could not find sufficient evidence" in r["answer"].lower()
        )
        denom = correct + false_abstains
        return {
            "n": n,
            "correct": correct,
            "abstain_correctness": round(correct / n, 4),
            "exact_string_rate": round(exact / n, 4),
            "false_abstains_on_golden": false_abstains,
            "abstain_precision": round(correct / denom, 4) if denom else None,
            "abstain_recall": round(correct / n, 4),
        }

    def _claim_faithfulness(self) -> float:
        total = 0
        supported = 0
        for r in self.results:
            for claim in r.get("claims", []):
                total += 1
                if claim["verdict"] == "supported":
                    supported += 1
        return round(supported / total, 4) if total else 0.0

    def _k_curve_summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k in K_CURVE_KS:
            key = str(k)
            recalls = [
                r["retrieval_k_curve"][key]["recall_at_k"]
                for r in self.results
                if key in r.get("retrieval_k_curve", {})
            ]
            ndcgs = [
                r["retrieval_k_curve"][key]["ndcg_at_k"]
                for r in self.results
                if key in r.get("retrieval_k_curve", {})
            ]
            if recalls:
                out[key] = {
                    "recall_at_k": round(sum(recalls) / len(recalls), 4),
                    "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4),
                }
        return out

    def _adversarial_summary(self) -> Dict[str, Any]:
        n = len(self.adversarial_results)
        if n == 0:
            return {"n": 0, "passed": 0, "resistance_rate": None, "none_followed": None}
        passed = sum(1 for r in self.adversarial_results if r["correct"])
        followed = sum(1 for r in self.adversarial_results if r["followed"])
        raw_markers = sum(1 for r in self.adversarial_results if r["raw_marker_in_answer"])
        return {
            "n": n,
            "passed": passed,
            "resistance_rate": round(passed / n, 4),
            "none_followed": followed == 0,
            "raw_marker_leaked": raw_markers,
        }

    def _write_results(self) -> None:
        out = self.settings.processed_dir_path / "evaluation_results.json"
        out.write_text(
            json.dumps(
                {"summary": self._summary(), "results": self.results}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    def _write_claims(self) -> None:
        out = self.settings.processed_dir_path / "claims.jsonl"
        lines = []
        for r in self.results:
            for claim in r.get("claims", []):
                lines.append(
                    json.dumps(
                        {
                            "question_id": r["id"],
                            "question": r["question"],
                            **claim,
                        },
                        ensure_ascii=False,
                    )
                )
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_abstention(self) -> None:
        out = self.settings.processed_dir_path / "abstention_results.json"
        out.write_text(
            json.dumps(self.abstention_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_adversarial(self) -> None:
        out = self.settings.processed_dir_path / "adversarial_results.json"
        out.write_text(
            json.dumps(self.adversarial_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _render_report(self) -> None:
        s = self._summary()
        path = self.settings.eval_report_path_resolved
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Evaluation Report — Agent-as-a-Judge RAG")
        lines.append("")
        lines.append(f"- Questions evaluated: **{s['n_questions']}**")
        lines.append(
            f"- Passed (all key values + page + sufficient evidence): **{s['passed']} / {s['n_questions']}** ({s['pass_rate']:.2%})"
        )
        lines.append(f"- Retrieval Recall@6: **{s['recall_at_6']:.4f}**")
        lines.append(f"- Retrieval Precision@6: **{s['precision_at_6']:.4f}**")
        lines.append(f"- MRR: **{s['mrr']:.4f}**")
        lines.append(f"- nDCG@6: **{s['ndcg_at_6']:.4f}**")
        lines.append(
            f"- Numeric recall (key values found in answer): **{s['avg_numeric_recall']:.4f}**"
        )
        lines.append(
            f"- Completeness (required phrases in answer): **{s['avg_completeness']:.4f}**"
        )
        lines.append(f"- Citation accuracy: **{s['citation_accuracy']:.4f}**")
        lines.append(
            f"- Faithfulness (claims supported by evidence): **{s['faithfulness_rate']:.4f}**"
        )
        lines.append(f"- Page accuracy: **{s['page_accuracy']:.4f}**")
        lines.append(
            f"- Cited-evidence recall (gold chunks actually cited): **{s['cited_recall']:.4f}**"
        )
        lines.append(
            f"- Cited-evidence precision (gold share of cited chunks): **{s['cited_precision']:.4f}**"
        )
        abst = s.get("abstention", {})
        if abst.get("n"):
            lines.append(
                f"- Abstention (out-of-knowledge): **{abst['correct']} / {abst['n']}** "
                f"correct ({abst['abstain_correctness']:.2%}); exact abstention sentence "
                f"used in {abst['exact_string_rate']:.2%}"
            )
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
        lines.append("## Per-question claim verdicts")
        for r in self.results:
            claims = r.get("claims", [])
            lines.append(f"### Q{r['id']} — {r['question']}")
            if not claims:
                lines.append("_(no claims parsed — abstained or non-answer)_")
            for c in claims:
                lines.append(
                    "- {} `{}` page={} overlap={} ({})".format(
                        "SUPPORTED" if c["verdict"] == "supported" else c["verdict"],
                        c["text"][:90],
                        c["page"],
                        c["token_overlap"],
                        c["reason"],
                    )
                )
            lines.append("")
        abst = self.abstention_results
        if abst:
            lines.append("## Abstention (out-of-knowledge) cases")
            lines.append("| id | reason | abstained | exact sentence | correct | warning |")
            lines.append("|---|---|---|---|---|---|")
            for r in abst:
                lines.append(
                    f"| {r['id']} | {r['reason']} | {r['abstained']} | "
                    f"{r['exact_abstention_string']} | {r['correct']} | {r['warning'] or ''} |"
                )
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
