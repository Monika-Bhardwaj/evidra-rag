from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from src.config import get_settings
from src.pipeline import RagPipeline


def ask(pipeline: RagPipeline, question: str) -> None:
    response = pipeline.answer(question, include_debug=True)
    print("\n" + "=" * 80)
    print("ANSWER:", response.answer)
    print("-" * 80)
    for s in response.sources:
        print(f"[Source] page {s['page']} | section '{s['section']}' | chunk {s['chunk_id']}")
    if response.warning:
        print("WARNING:", response.warning)
    gen = response.generation
    if gen:
        print(
            f"[Metrics] provider={gen.provider} model={gen.model} "
            f"tokens={gen.prompt_tokens + gen.completion_tokens} "
            f"latency={gen.latency_ms:.1f}ms cost=${gen.estimated_cost_usd:.6f}"
        )
    if response.retrieval_debug:
        print("[Debug] query_type:", response.retrieval_debug.query_type)
        for c in response.retrieval_debug.reranked_hits:
            print(
                f"   top: {c['chunk_id']} type={c['chunk_type']} "
                f"dense={c['dense_score']} sparse={c['sparse_score']} "
                f"hybrid={c['hybrid_score']} rerank={c['rerank_score']}"
            )
    print("=" * 80)


def main() -> None:
    pipeline = RagPipeline(get_settings())
    if len(sys.argv) > 1:
        ask(pipeline, " ".join(sys.argv[1:]))
    else:
        print("Ask the paper a question (Ctrl+C or type 'exit' to quit).")
        while True:
            try:
                question = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            ask(pipeline, question)


if __name__ == "__main__":
    main()