from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import get_settings
from src.pipeline import IndexNotFoundError, RagPipeline

st.set_page_config(page_title="Agent-as-a-Judge RAG Assistant", layout="wide")

SETTINGS = get_settings()


@st.cache_resource(show_spinner="Loading RAG pipeline (first load downloads models)...")
def load_pipeline() -> RagPipeline:
    return RagPipeline(SETTINGS)


def main() -> None:
    try:
        pipeline = load_pipeline()
    except IndexNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.title("Agent-as-a-Judge Research Paper — RAG Assistant")
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
                with st.expander("Evidence panel", expanded=bool(response.sources)):
                    for s in response.sources:
                        st.markdown(
                            f"**Source** page {s['page']} · section `{s['section']}` · chunk `{s['chunk_id']}`"
                        )
                    if not response.sources:
                        st.markdown("No evidence retrieved — check the debug panel.")
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
                                use_container_width=True,
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
    st.subheader("Retrieval debugger")
    question = st.text_input("Query", placeholder="What is OpenHands' average cost?", key="console_q")
    if st.button("Run retrieval", type="primary") and question:
        response = pipeline.answer(question, include_debug=True)
        st.markdown("#### Answer")
        st.write(response.answer)
        st.markdown("#### Retrieved chunks (post-rerank)")
        debug = response.retrieval_debug
        if debug and debug.reranked_hits:
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
                        "snippet": h["text"][:200],
                    }
                    for h in debug.reranked_hits
                ],
                use_container_width=True,
            )
        st.markdown("#### Raw retrieval debug")
        if debug:
            st.json(debug.to_dict(), expanded=False)


def _eval_tab() -> None:
    st.subheader("Evaluation dashboard")
    results_path = SETTINGS.processed_dir_path / "evaluation_results.json"
    if not results_path.exists():
        st.info("No evaluation results yet. Run `python scripts/evaluate.py`.")
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
        use_container_width=True,
        hide_index=True,
    )
    eval_root = Path(__file__).resolve().parent.parent / "docs" / "eval_plots"
    for name in ("retrieval_quality", "answer_quality", "system_metrics"):
        img = eval_root / f"{name}.png"
        if img.exists():
            st.image(str(img), caption=name.replace("_", " ").title())


def _arch_tab() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    for name, caption in [
        ("rag_architecture.png", "RAG architecture"),
        ("retrieval_workflow.png", "Retrieval & generation workflow"),
    ]:
        img = docs / name
        if img.exists():
            st.image(str(img), caption=caption, use_container_width=True)
        else:
            st.warning(f"Diagram missing: {name}. Run `python scripts/generate_diagrams.py`.")


main()