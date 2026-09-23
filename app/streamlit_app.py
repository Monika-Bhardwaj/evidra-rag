from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import get_settings
from src.pipeline import IndexNotFoundError, RagPipeline

st.set_page_config(page_title="EVIDRA — Evidence-Centric RAG Assistant", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = get_settings()


@st.cache_resource(show_spinner="Loading RAG pipeline (first load downloads models)...")
def load_pipeline() -> RagPipeline:
    return RagPipeline(SETTINGS)


@st.cache_data(ttl=600)
def load_pages() -> list[dict]:
    path = SETTINGS.processed_dir_path / "pages.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


PAGES_BY_NUMBER = {p["page"]: p.get("text", "") for p in load_pages()}


@st.cache_data(ttl=600)
def load_archives() -> list[dict]:
    root = ROOT / "docs" / "eval_results"
    if not root.exists():
        return []
    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        path = run_dir / "evaluation_results.json"
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        abst = summary.get("abstention", {}) or {}
        adv = summary.get("adversarial", {}) or {}
        rows.append(
            {
                "run": run_dir.name,
                "pass_rate": summary.get("pass_rate"),
                "recall@6": summary.get("recall_at_6"),
                "ndcg@6": summary.get("ndcg_at_6"),
                "abstain": abst.get("abstain_correctness"),
                "adversarial": adv.get("resistance_rate"),
                "provider": summary.get("provider"),
            }
        )
    return rows


def confidence_gauge(value: float, label: str = "Grounding confidence") -> None:
    pct = max(0.0, min(1.0, value))
    band = "high" if pct >= 0.7 else "medium" if pct >= 0.4 else "low"
    color = {"high": ":green[high]", "medium": ":orange[medium]", "low": ":red[low]"}[band]
    st.markdown(
        f"**{label}:** {pct:.0%} · ({color})",
        help="Dense-query similarity gate + citation and claim verification.",
    )
    st.progress(pct)


def render_claims(claims: list[dict]) -> None:
    if not claims:
        st.caption("No claims parsed (abstained or non-answer).")
        return
    icon = {"supported": "✅", "low-support": "🟡", "unsupported": "❌"}
    for c in claims:
        verdict = c.get("verdict", "?")
        st.markdown(f"{icon.get(verdict, '❔')} **{verdict}** — {c.get('reason', '')}")
        st.markdown(f"> {c.get('text', '')[:240]}")
        st.caption(
            "page="
            f"{c.get('page')} · token_overlap={c.get('token_overlap')} · "
            f"page_matched={c.get('page_matched')} · numeric_matched={c.get('numeric_matched')}"
        )


def render_evidence(response, jump_to_page: bool = True) -> None:
    debug = response.retrieval_debug
    by_id = {}
    if debug and debug.reranked_hits:
        by_id = {h["chunk_id"]: h for h in debug.reranked_hits}
    for s in response.sources:
        hit = by_id.get(s["chunk_id"])
        title = (
            f"Chunk {s['chunk_id']} · page {s['page']} · section {s['section']}"
            f" · type {s.get('chunk_type', '?')}"
        )
        with st.expander(title):
            if hit:
                st.caption(
                    f"dense={hit['dense_score']:.3f} · sparse={hit['sparse_score']:.3f} · "
                    f"hybrid={hit['hybrid_score']:.3f} · "
                    f"rerank={hit.get('rerank_score')}"
                )
                st.markdown(f"```\n{hit['text'][:1200]}\n```")
            else:
                st.write("(evidence text is not present in this debug payload)")
            if jump_to_page and s["page"] in PAGES_BY_NUMBER:
                if st.button(f"Show page {s['page']} text", key=f"page_{s['chunk_id']}"):
                    st.markdown(PAGES_BY_NUMBER[s["page"]][:3500])
    if not response.sources:
        st.caption("No evidence retrieved — this response was declined (fail-closed).")


def main() -> None:
    try:
        pipeline = load_pipeline()
    except IndexNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.title("EVIDRA — Evidence-Centric RAG Assistant")
    st.caption(
        "Agent-as-a-Judge (arXiv 2410.10934) · claim → evidence → source → validation · "
        "unsupported answers abstain instead of inventing."
    )
    st.caption(
        f"Embedder: {pipeline.embedder.name()} · LLM provider: {pipeline.llm.provider_name} · "
        f"chunks: {len(pipeline._chunks_by_id)}"
    )

    tab_chat, tab_console, tab_eval, tab_arch = st.tabs(
        ["Chat", "RAG Console", "Evaluation", "Architecture"]
    )

    with st.sidebar:
        st.header("System metrics")
        stats = pipeline.stats_summary()
        st.metric("Queries", stats["queries"])
        st.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
        st.metric("Cache hit rate", f"{stats['cache_hit_rate']:.2%}")
        st.metric("Est. LLM cost", f"${stats['total_est_cost_usd']:.6f}")
        st.metric("Total tokens", stats["total_tokens"])
        st.caption("No API keys or secrets are exposed.")

    with tab_chat:
        _chat_tab(pipeline)

    with tab_console:
        _console_tab(pipeline)

    with tab_eval:
        _eval_tab()

    with tab_arch:
        _arch_tab()


def _chat_tab(pipeline: RagPipeline) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the paper"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and grounding the answer..."):
                history = st.session_state.messages[:-1][-6:]
                response = pipeline.answer(prompt, history=history, include_debug=True)
                st.markdown(response.answer)
                if not response.evidence_sufficient:
                    st.warning(
                        "**Abstained (fail-closed):** no sufficient evidence in the provided "
                        "document. The exact abstention sentence was returned."
                    )
                confidence_gauge(response.grounded_confidence)
                with st.expander("Evidence panel", expanded=bool(response.sources)):
                    render_evidence(response, jump_to_page=True)
                with st.expander("Claim → evidence verification", expanded=bool(response.claims)):
                    render_claims(response.claims)
                with st.expander("Retrieval debug"):
                    debug = response.retrieval_debug
                    if debug:
                        st.markdown(f"**Query type:** {debug.query_type}")
                        st.markdown(f"**Expanded queries:** {', '.join(debug.expanded_queries)}")
                        if debug.reranked_hits:
                            st.dataframe(
                                [
                                    {
                                        "chunk": h["chunk_id"],
                                        "type": h["chunk_type"],
                                        "page": h["page"],
                                        "section": h["section"],
                                        "dense": h["dense_score"],
                                        "sparse": h["sparse_score"],
                                        "hybrid": h["hybrid_score"],
                                        "rerank": h["rerank_score"],
                                    }
                                    for h in debug.reranked_hits
                                ],
                                width="stretch",
                            )
                        st.json(debug.to_dict(), expanded=False)
                gen = response.generation
                if gen:
                    st.markdown(
                        f"`model={gen.model}` `tokens={gen.prompt_tokens + gen.completion_tokens}` "
                        f"`latency={gen.latency_ms:.1f}ms` `est_cost=${gen.estimated_cost_usd:.6f}`"
                    )
                if response.warning:
                    st.warning(response.warning)
            st.session_state.messages.append({"role": "assistant", "content": response.answer})


def _console_tab(pipeline: RagPipeline) -> None:
    st.subheader("RAG Console — trace claim → evidence → source")
    question = st.text_input(
        "Query", placeholder="What is OpenHands' average cost?", key="console_q"
    )
    if st.button("Run retrieval & grounding", type="primary") and question:
        response = pipeline.answer(question, include_debug=True)
        st.markdown("#### Answer")
        st.write(response.answer)
        if not response.evidence_sufficient:
            st.warning("Abstained (fail-closed) — no sufficient evidence in the provided document.")
        confidence_gauge(response.grounded_confidence)
        st.markdown("### Claim → evidence mapping")
        render_claims(response.claims)
        st.markdown("### Cited evidence (final top-k)")
        render_evidence(response, jump_to_page=True)
        debug = response.retrieval_debug
        if debug:
            st.markdown("### Retrieved chunks (post-rerank)")
            if debug.reranked_hits:
                st.dataframe(
                    [
                        {
                            "chunk": h["chunk_id"],
                            "type": h["chunk_type"],
                            "page": h["page"],
                            "section": h["section"],
                            "dense": h["dense_score"],
                            "sparse": h["sparse_score"],
                            "hybrid": h["hybrid_score"],
                            "rerank": h["rerank_score"],
                            "snippet": h["text"][:160],
                        }
                        for h in debug.reranked_hits
                    ],
                    width="stretch",
                )
            st.markdown("### Raw retrieval debug")
            st.json(debug.to_dict(), expanded=False)


def _eval_tab() -> None:
    st.subheader("Evaluation dashboard")
    col_run, col_meta = st.columns([1, 3])
    with col_run:
        if st.button("Run evaluation now", type="primary"):
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT)
            with st.spinner("Running `python scripts/evaluate.py`…"):
                proc = subprocess.run(
                    [sys.executable, "scripts/evaluate.py"],
                    cwd=str(ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            if proc.returncode != 0:
                st.error(f"Evaluation failed (rc={proc.returncode}).\n\n{proc.stderr[-2000:]}")
            else:
                st.success("Evaluation finished — archive written to `docs/eval_results/<run>/`.")
                load_archives.clear()
    with col_meta:
        st.caption(
            "Runs `scripts/evaluate.py` in-process with the current index. Each run writes a "
            "versioned archive under `docs/eval_results/<run>/`, leaving the current "
            "`data/processed/evaluation_results.json` for this dashboard."
        )

    results_path = SETTINGS.processed_dir_path / "evaluation_results.json"
    if not results_path.exists():
        st.info("No evaluation results yet. Run `python scripts/evaluate.py` or click above.")
        return
    data = json.loads(results_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    cols = st.columns(4)
    cols[0].metric("Pass rate", f"{summary['pass_rate']:.2%}")
    cols[1].metric("Recall@6", f"{summary['recall_at_6']:.3f}")
    cols[2].metric("nDCG@6", f"{summary['ndcg_at_6']:.3f}")
    cols[3].metric("Faithfulness", f"{summary['faithfulness_rate']:.2%}")
    st.dataframe(
        [
            {
                "id": r["id"],
                "question": r["question"][:60],
                "pass": r["pass"],
                "recall@6": r["retrieval_raw"]["recall_at_k"],
                "ndcg@6": r["retrieval_raw"]["ndcg_at_k"],
                "numeric_recall": r["generation"]["numeric_recall"],
                "faithful": r["generation"]["faithful"],
                "page_correct": r["generation"]["page_correct"],
            }
            for r in data["results"]
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Run archive comparison")
    archives = load_archives()
    if archives:
        st.dataframe(archives, width="stretch", hide_index=True)
        st.caption(
            "Each committed run under `docs/eval_results/` is listed; the latest row is the "
            "current `data/processed/evaluation_results.json` output."
        )
    else:
        st.caption("No committed run archives found under `docs/eval_results/`.")

    eval_root = ROOT / "docs" / "eval_plots"
    for name in ("retrieval_quality", "answer_quality", "system_metrics"):
        img = eval_root / f"{name}.png"
        if img.exists():
            st.image(str(img), caption=name.replace("_", " ").title())


def _arch_tab() -> None:
    docs = ROOT / "docs"
    st.markdown(
        "**EVIDRA dataflow (as built):** user question → query understanding/expansion → "
        "dense (FAISS) + sparse (BM25) retrieval → hybrid fusion → optional rerank → "
        "**sufficiency gate** (fail-closed: raw-question dense cosine must clear "
        "`min_similarity`) → grounded prompt → generation → **citation + claim "
        "validation** (fail-closed) → answer with `[Page X, Section Y]` citations, or the "
        "exact abstention sentence. Retrieved text is neutralized against injected "
        "instructions before it reaches the prompt."
    )
    for name, caption in [
        ("rag_architecture.png", "RAG architecture"),
        ("retrieval_workflow.png", "Retrieval & generation workflow"),
    ]:
        img = docs / name
        if img.exists():
            st.image(str(img), caption=caption, width="stretch")
        else:
            st.warning(f"Diagram missing: {name}. Run `python scripts/generate_diagrams.py`.")


main()
