from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    citation_correctness,
    cited_metrics,
    completeness,
    faithfulness,
    ndcg_at_k,
    numeric_answer_correctness,
    page_correctness,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.metrics import (  # noqa: E402
    mrr as mrr_metric,
)
from src.logging_utils import get_logger  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.embeddings import build_embedder  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.query_expansion import QueryExpander  # noqa: E402
from src.retrieval.query_understanding import QueryClassifier  # noqa: E402
from src.retrieval.routing import QueryRouter, RouteSpec  # noqa: E402
from src.schemas import RetrievedChunk  # noqa: E402

logger = get_logger(__name__)

RETRIEVAL_VARIANTS: List[tuple[str, float, str]] = [
    ("dense", 1.0, "weighted"),
    ("bm25", 0.0, "weighted"),
    ("weighted-0.3", 0.3, "weighted"),
    ("weighted-0.5", 0.5, "weighted"),
    ("weighted-0.7", 0.7, "weighted"),
    ("rrf", 0.5, "rrf"),
]


@dataclass
class SweepConfig:
    name: str
    alpha: float
    method: str
    routing: bool
    expansion: bool
    k: int


def build_sweep_configs(k: int) -> List[SweepConfig]:
    configs: List[SweepConfig] = []
    for name, alpha, method in RETRIEVAL_VARIANTS:
        for routing in (False, True):
            for expansion in (False, True):
                configs.append(
                    SweepConfig(
                        name=name,
                        alpha=alpha,
                        method=method,
                        routing=routing,
                        expansion=expansion,
                        k=k,
                    )
                )
    return configs


def load_index(settings: Settings) -> tuple[Any, BM25Retriever, Any, List[Any]]:
    from src.retrieval.vector_store import FAISSVectorStore, InMemoryVectorStore

    index_dir = settings.index_dir_path
    store = None
    try:
        store = FAISSVectorStore.load(index_dir)
    except Exception as exc:
        logger.warning("FAISS load failed (%s); trying in-memory fallback.", exc)
    if store is None or store.size() == 0:
        store = InMemoryVectorStore.load(index_dir)
    if store.size() == 0:
        raise RuntimeError("No index found. Run `python scripts/build_index.py` first.")
    chunks = store.chunks()
    embedder = build_embedder(settings)
    bm25 = BM25Retriever(chunks)
    return store, bm25, embedder, chunks


def load_questions(settings: Settings) -> List[Dict[str, Any]]:
    data = json.loads(settings.eval_questions_path_resolved.read_text(encoding="utf-8"))
    return data["questions"]


def load_gold_map(settings: Settings) -> Dict[str, Any]:
    path = settings.processed_dir_path / "gold_evidence.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sweep_row(
    q: Dict[str, Any],
    config: SweepConfig,
    gold_map: Dict[str, Any],
    chunks_by_id: Dict[str, Any],
    hybrid: HybridRetriever,
    classifier: QueryClassifier,
    expander: QueryExpander,
    router: QueryRouter,
    settings: Settings,
) -> Dict[str, float]:
    qid = q["id"]
    qtext = q["question"]
    intent = classifier.classify(qtext)
    expanded = expander.expand(qtext, intent) if config.expansion else [qtext]
    route = router.route(intent, qtext) if config.routing else RouteSpec()

    top, sufficient, _ = hybrid.retrieve(
        query=qtext,
        expanded_queries=expanded,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        top_k=config.k,
        alpha=config.alpha,
        method=config.method,
        min_similarity=settings.min_similarity,
        chunk_types=route.chunk_types or None,
        type_boost=route.type_boost,
    )
    ranked_ids = [c.chunk.chunk_id for c in top]
    gold_ids = {h["chunk_id"] for h in gold_map.get(str(qid), [])}
    gold_ids = {cid for cid in gold_ids if cid in chunks_by_id}
    gains = {cid: 1.0 for cid in gold_ids}
    return {
        "hit": float(bool(set(ranked_ids[: config.k]) & gold_ids)),
        "sufficient": float(sufficient),
        "recall": recall_at_k(ranked_ids, gold_ids, config.k),
        "precision": precision_at_k(ranked_ids, gold_ids, config.k),
        "mrr": mrr_metric(ranked_ids, gold_ids),
        "ndcg": ndcg_at_k(ranked_ids, gains, config.k),
    }


def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    n = len(rows)

    def mean(key: str) -> float:
        return round(sum(r[key] for r in rows) / n, 4)

    return {
        "hit_rate": mean("hit"),
        "sufficient_rate": mean("sufficient"),
        "recall_at_k": mean("recall"),
        "precision_at_k": mean("precision"),
        "mrr": mean("mrr"),
        "ndcg_at_k": mean("ndcg"),
    }


def run_retrieval_sweep(
    configs: List[SweepConfig],
    questions: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    chunks_by_id: Dict[str, Any],
    hybrid: HybridRetriever,
    settings: Settings,
) -> List[Dict[str, Any]]:
    classifier = QueryClassifier()
    expander = QueryExpander()
    router = QueryRouter()
    sweep: List[Dict[str, Any]] = []
    for config in configs:
        rows = [
            sweep_row(
                q,
                config,
                gold_map,
                chunks_by_id,
                hybrid,
                classifier,
                expander,
                router,
                settings,
            )
            for q in questions
        ]
        agg = aggregate_rows(rows)
        sweep.append(
            {
                "name": config.name,
                "method": config.method,
                "alpha": config.alpha,
                "routing": config.routing,
                "expansion": config.expansion,
                "top_k": config.k,
                **agg,
            }
        )
        print(
            "[retrieval] {:<14} method={:<8} alpha={:<4} routing={} expansion={} k={} "
            "hit={:.3f} recall@{k}={:.4f} prec@{k}={:.4f} mrr={:.4f} ndcg@k={:.4f}".format(
                config.name,
                config.method,
                config.alpha,
                int(config.routing),
                int(config.expansion),
                config.k,
                agg["hit_rate"],
                agg["recall_at_k"],
                agg["precision_at_k"],
                agg["mrr"],
                agg["ndcg_at_k"],
                k=config.k,
            )
        )
    return sweep


def best_retrieval_config(sweep: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        sweep,
        key=lambda r: (r["hit_rate"], r["ndcg_at_k"], r["recall_at_k"], r["mrr"]),
    )


def run_pipeline_pass(
    override: Dict[str, Any],
    label: str,
    questions: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    settings: Settings,
) -> Dict[str, Any]:
    from src.pipeline import RagPipeline

    variant_settings = settings.model_copy(update=override)
    pipeline = RagPipeline(variant_settings)
    chunks_by_id = pipeline._chunks_by_id
    results: List[Dict[str, Any]] = []

    for q in questions:
        qid = q["id"]
        qtext = q["question"]
        required = q.get("required_evidence", [])
        key_values = q.get("key_values", [])
        pages = q.get("page", [])
        gold_ids = {h["chunk_id"] for h in gold_map.get(str(qid), [])} or None
        relevant_ids = {cid for cid in gold_ids if cid in chunks_by_id} if gold_ids else None

        intent = pipeline.classifier.classify(qtext)
        expanded = pipeline.expander.expand(qtext, intent)
        route = pipeline.router.route(intent, qtext)
        raw, _, _ = pipeline.hybrid.retrieve(
            query=qtext,
            expanded_queries=expanded,
            dense_top_k=variant_settings.dense_top_k,
            bm25_top_k=variant_settings.bm25_top_k,
            top_k=20,
            alpha=variant_settings.hybrid_alpha,
            method=variant_settings.hybrid_method,
            min_similarity=0.0,
            chunk_types=route.chunk_types or None,
            type_boost=route.type_boost,
        )

        start = time.perf_counter()
        response = pipeline.answer(qtext, include_debug=True)
        latency_ms = (time.perf_counter() - start) * 1000.0

        final_evidence: List[RetrievedChunk] = []
        for src in response.sources:
            chunk = chunks_by_id.get(src["chunk_id"])
            if chunk is not None:
                final_evidence.append(
                    RetrievedChunk(chunk=chunk, hybrid_score=1.0, rerank_score=1.0)
                )

        from src.evaluation.metrics import retrieval_metrics

        raw_metrics = retrieval_metrics(
            raw, pipeline.vector_store.chunks(), required, k=6, relevant_ids=relevant_ids
        )
        final_metrics = retrieval_metrics(
            final_evidence, pipeline.vector_store.chunks(), required, k=6, relevant_ids=relevant_ids
        )
        numeric = numeric_answer_correctness(response.answer, key_values)
        passed = (
            bool(
                response.evidence_sufficient
                and page_correctness(final_evidence, pages)
                and all(numeric.values())
            )
            if key_values
            else bool(response.evidence_sufficient)
        )
        results.append(
            {
                "id": qid,
                "pass": passed,
                "retrieval_raw": raw_metrics,
                "retrieval_final": final_metrics,
                "numeric_recall": (
                    round(sum(1 for v in numeric.values() if v) / max(1, len(numeric)), 4)
                    if key_values
                    else 1.0
                ),
                "completeness": completeness(response.answer, required),
                "citation_correct": citation_correctness(response.answer, final_evidence).get(
                    "has_citation"
                ),
                "faithful": faithfulness(response.answer, final_evidence),
                "page_correct": page_correctness(final_evidence, pages),
                "cited": cited_metrics(final_evidence, relevant_ids),
                "latency_ms": round(latency_ms, 1),
                "reranker_used": pipeline.reranker.is_used(),
                "answer": response.answer,
            }
        )

    n = len(results)

    def mean(key: str, sub: Optional[str] = None) -> float:
        values = []
        for r in results:
            val = r[key] if sub is None else r.get(sub, {}).get(key)
            if isinstance(val, (int, float)):
                values.append(val)
        return round(sum(values) / max(1, len(values)), 4)

    summary = {
        "label": label,
        "config": override,
        "n_questions": n,
        "passed": sum(1 for r in results if r["pass"]),
        "pass_rate": round(sum(1 for r in results if r["pass"]) / max(1, n), 4),
        "recall_at_6": mean("recall_at_k", "retrieval_raw"),
        "precision_at_6": mean("precision_at_k", "retrieval_raw"),
        "mrr": mean("mrr", "retrieval_raw"),
        "ndcg_at_6": mean("ndcg_at_k", "retrieval_raw"),
        "avg_numeric_recall": mean("numeric_recall"),
        "avg_completeness": mean("completeness"),
        "citation_accuracy": mean("citation_correct"),
        "faithfulness_rate": mean("faithful"),
        "page_accuracy": mean("page_correct"),
        "cited_recall": mean("cited_recall", "cited"),
        "cited_precision": mean("cited_precision", "cited"),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / max(1, n), 1),
    }
    print(
        "[pipeline] {:<24} pass={}/{:<2} citedR={:.4f} citedP={:.4f} R@6={:.4f} nDCG={:.4f} "
        "page={:.3f} completeness={:.3f} reranker={}".format(
            label,
            summary["passed"],
            n,
            summary["cited_recall"],
            summary["cited_precision"],
            summary["recall_at_6"],
            summary["ndcg_at_6"],
            summary["page_accuracy"],
            summary["avg_completeness"],
            "on" if pipeline.reranker.is_used() else "off",
        )
    )
    return {"summary": summary, "results": results}


def decide(passes: List[Dict[str, Any]]) -> Dict[str, Any]:
    def sort_key(r: Dict[str, Any]) -> Any:
        s = r["summary"]
        return (
            s["pass_rate"],
            s["cited_recall"],
            s["cited_precision"],
            s["ndcg_at_6"],
        )

    ranked = sorted(passes, key=sort_key, reverse=True)
    return ranked[0]


def main() -> None:
    settings = get_settings()
    questions = load_questions(settings)
    gold_map = load_gold_map(settings)
    store, bm25, embedder, chunks = load_index(settings)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    hybrid = HybridRetriever(store, bm25, embedder)

    out_dir = (settings.root_dir / "docs" / "eval_results" / "ablation") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    header = {
        "embedding_model": settings.embedding_model,
        "dense_top_k": settings.dense_top_k,
        "bm25_top_k": settings.bm25_top_k,
        "index_chunks": len(chunks),
        "min_similarity": settings.min_similarity,
        "questions": len(questions),
        "reranker_model": settings.reranker_model,
        "generated_utc": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(
        json.dumps(header, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("== Retrieval sweep (k = 6) ==")
    sweep = run_retrieval_sweep(
        build_sweep_configs(k=6),
        questions,
        gold_map,
        chunks_by_id,
        hybrid,
        settings,
    )
    winner = best_retrieval_config(sweep)

    print("== k sweep on best config {} ==".format(winner["name"]))
    k_sweep = run_retrieval_sweep(
        [
            SweepConfig(
                name=winner["name"],
                alpha=winner["alpha"],
                method=winner["method"],
                routing=winner["routing"],
                expansion=winner["expansion"],
                k=k,
            )
            for k in (4, 8)
        ],
        questions,
        gold_map,
        chunks_by_id,
        hybrid,
        settings,
    )

    selector = best_retrieval_config(sweep)
    default_cfg = {
        "name": "weighted-0.5",
        "method": "weighted",
        "alpha": 0.5,
        "routing": True,
        "expansion": True,
    }
    print("== Full-pipeline passes ==")
    passes = []
    run_names_to_override = [
        (f"default-off ({default_cfg['name']}, rerank off)", {"use_reranker": False}),
        (
            f"best-off ({selector['name']}, rerank off)",
            {
                "use_reranker": False,
                "hybrid_alpha": selector["alpha"],
                "hybrid_method": selector["method"],
                "use_query_routing": selector["routing"],
                "use_query_expansion": selector["expansion"],
            },
        ),
        (
            "default-score (rerank=score)",
            {"use_reranker": True, "reranker_kind": "score"},
        ),
        (
            "default-cross (rerank=cross)",
            {"use_reranker": True, "reranker_kind": "cross"},
        ),
    ]
    for label, override in run_names_to_override:
        p = run_pipeline_pass(override, label, questions, gold_map, settings)
        passes.append(p)

    recommended = decide(passes)

    with (out_dir / "retrieval_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
        writer.writerows(k_sweep)

    with (out_dir / "pipeline_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "passed",
            "n_questions",
            "pass_rate",
            "cited_recall",
            "cited_precision",
            "recall_at_6",
            "precision_at_6",
            "mrr",
            "ndcg_at_6",
            "avg_numeric_recall",
            "avg_completeness",
            "citation_accuracy",
            "faithfulness_rate",
            "page_accuracy",
            "avg_latency_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in passes:
            s = p["summary"]
            writer.writerow({k: s.get(k) for k in fieldnames})

    (out_dir / "pipeline_results.json").write_text(
        json.dumps([p["summary"] for p in passes], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = render_report(header, sweep, k_sweep, winner, selector, passes, recommended)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print("\nAblation report: %s" % (out_dir / "REPORT.md"))


def render_report(
    header: Dict[str, Any],
    sweep: List[Dict[str, Any]],
    k_sweep: List[Dict[str, Any]],
    winner: Dict[str, Any],
    selector: Dict[str, Any],
    passes: List[Dict[str, Any]],
    recommended: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Ablation report — retrieval / evidence layer (Phase 3)")
    lines.append("")
    lines.append("## Fixed settings")
    for key, value in header.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Retrieval sweep (top_k = 6)")
    lines.append("Binary gold relevance (gold chunk ids from `gold_evidence.json`).")
    lines.append("")
    lines.append(
        "| variant | method | alpha | routing | expansion | hit | suff | R@6 | P@6 | MRR | nDCG@6 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(sweep, key=lambda x: (-x["hit_rate"], -x["ndcg_at_k"])):
        lines.append(
            f"| {r['name']} | {r['method']} | {r['alpha']} | {r['routing']} | {r['expansion']} "
            f"| {r['hit_rate']:.3f} | {r['sufficient_rate']:.3f} | {r['recall_at_k']:.4f} "
            f"| {r['precision_at_k']:.4f} | {r['mrr']:.4f} | {r['ndcg_at_k']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Best retrieval config (k=6):** `{selector['name']}` method={selector['method']} "
        f"alpha={selector['alpha']} routing={selector['routing']} expansion={selector['expansion']} "
        f"(hit={selector['hit_rate']:.3f}, nDCG@6={selector['ndcg_at_k']:.4f})."
    )
    lines.append("")
    lines.append("## Window-size sweep (best config)")
    lines.append("| top_k | hit | suff | R@k | P@k | MRR | nDCG@k |")
    lines.append("|---|---|---|---|---|---|---|")
    all_ks = sorted(set(r["top_k"] for r in k_sweep) | {6})
    for k in all_ks:
        row = next((r for r in k_sweep if r["top_k"] == k), None)
        if row is None and k == 6:
            for r in sweep:
                if (
                    r["name"] == selector["name"]
                    and r["routing"] == selector["routing"]
                    and r["expansion"] == selector["expansion"]
                ):
                    row = r
                    break
        if row is None:
            continue
        lines.append(
            f"| {k} | {row['hit_rate']:.3f} | {row['sufficient_rate']:.3f} | {row['recall_at_k']:.4f} "
            f"| {row['precision_at_k']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_k']:.4f} |"
        )
    lines.append("")
    lines.append("## Full-pipeline passes")
    lines.append(
        "| variant | pass | citedR | citedP | R@6 | P@6 | MRR | nDCG@6 | num | comp | cit | faith | page | ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in passes:
        s = p["summary"]
        lines.append(
            f"| {s['label']} | {s['passed']}/{s['n_questions']} | {s['cited_recall']:.4f} | "
            f"{s['cited_precision']:.4f} | {s['recall_at_6']:.4f} | {s['precision_at_6']:.4f} | "
            f"{s['mrr']:.4f} | {s['ndcg_at_6']:.4f} | {s['avg_numeric_recall']:.4f} | "
            f"{s['avg_completeness']:.4f} | {s['citation_accuracy']:.4f} | {s['faithfulness_rate']:.4f} "
            f"| {s['page_accuracy']:.4f} | {s['avg_latency_ms']} |"
        )
    lines.append("")
    rec = recommended["summary"]
    lines.append("## Decision")
    lines.append(
        f"- Recommended variant: **{rec['label']}** (pass {rec['passed']}/{rec['n_questions']}, "
        f"cited-recall {rec['cited_recall']:.4f}, cited-precision {rec['cited_precision']:.4f}, "
        f"nDCG@6 {rec['ndcg_at_6']:.4f})."
    )
    lines.append(
        "- Frozen default remains `weighted` α=0.5 (routing/expansion on) unless the recommended "
        "variant differs and is adopted in `src/config.py`."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
