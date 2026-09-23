from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _box(ax, x, y, w, h, text, color="#e8f0fe", edge="#1a73e8", size=8):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.2, facecolor=color, edgecolor=edge
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="#444444", style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color, linewidth=1.2
        )
    )


def draw_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    _box(
        ax, 3.4, 13.1, 3.2, 0.6, "PDF SOURCE\nAgent-as-a-Judge.pdf", color="#fff3e0", edge="#e65100"
    )
    _box(
        ax, 3.4, 12.1, 3.2, 0.6, "Document Loader\nPyMuPDF / pypdf", color="#f3e8fd", edge="#6a1b9a"
    )
    _box(
        ax,
        3.4,
        11.1,
        3.2,
        0.6,
        "Text Preprocessor\ncleanup / normalization",
        color="#f3e8fd",
        edge="#6a1b9a",
    )
    _box(
        ax,
        3.4,
        10.1,
        3.2,
        0.6,
        "Semantic Chunker\nsection-aware, tables kept whole",
        color="#e8f5e9",
        edge="#2e7d32",
    )
    _box(
        ax,
        3.4,
        9.1,
        3.2,
        0.6,
        "Metadata Enrichment\npage / section / chunk_type",
        color="#e8f5e9",
        edge="#2e7d32",
    )
    _box(
        ax,
        3.4,
        8.1,
        3.2,
        0.6,
        "Embedding Model\nall-MiniLM-L6-v2 (local)",
        color="#fff3e0",
        edge="#ef6c00",
    )
    _box(
        ax,
        3.4,
        7.1,
        3.2,
        0.6,
        "Vector Database\nFAISS (persisted)",
        color="#f3e8fd",
        edge="#4a148c",
    )
    _box(
        ax,
        0.2,
        6.3,
        2.2,
        0.6,
        "Query Preprocessor\nclassification / expansion",
        color="#e3f2fd",
        edge="#1565c0",
    )
    _box(ax, 2.8, 6.3, 2.1, 0.6, "Dense Retrieval\ncosine (FAISS)", color="#e3f2fd", edge="#1565c0")
    _box(ax, 5.3, 6.3, 2.0, 0.6, "BM25 Retrieval\nlexical", color="#e3f2fd", edge="#1565c0")
    _box(
        ax,
        7.7,
        6.3,
        2.1,
        0.6,
        "Hybrid Fusion\nalpha-weighted / RRF",
        color="#e3f2fd",
        edge="#1565c0",
    )
    _box(
        ax,
        3.4,
        5.3,
        3.2,
        0.6,
        "Reranker (optional)\ncross-encoder",
        color="#e8f5e9",
        edge="#2e7d32",
    )
    _box(
        ax,
        3.4,
        4.3,
        3.2,
        0.6,
        "Top-K Evidence + Context\nconstruction (compact)",
        color="#e8f5e9",
        edge="#2e7d32",
    )
    _box(ax, 3.4, 3.3, 3.2, 0.6, "Grounded LLM Prompt", color="#e3f2fd", edge="#1565c0")
    _box(
        ax,
        3.4,
        2.3,
        3.2,
        0.6,
        "Answer Generator\n(LLM API or offline extractive)",
        color="#e3f2fd",
        edge="#1565c0",
    )
    _box(
        ax,
        3.4,
        1.3,
        3.2,
        0.6,
        "Citation / Grounding\nValidation Layer",
        color="#fce4ec",
        edge="#c62828",
    )
    _box(
        ax,
        3.4,
        0.3,
        3.2,
        0.7,
        "FINAL ANSWER\nanswer + [Page X, Section Y]",
        color="#e8f5e9",
        edge="#1b5e20",
    )

    _arrow(ax, 5.0, 13.1, 5.0, 12.7)
    _arrow(ax, 5.0, 12.1, 5.0, 11.7)
    _arrow(ax, 5.0, 11.1, 5.0, 10.7)
    _arrow(ax, 5.0, 10.1, 5.0, 9.7)
    _arrow(ax, 5.0, 9.1, 5.0, 8.7)
    _arrow(ax, 5.0, 8.1, 5.0, 7.7)
    _arrow(ax, 5.0, 7.1, 5.0, 6.9)
    _arrow(ax, 3.4, 7.4, 3.4, 6.6)
    _arrow(ax, 7.4, 7.4, 7.4, 6.6)
    _arrow(ax, 1.3, 6.3, 2.6, 6.0)
    _arrow(ax, 2.6, 6.3, 4.2, 5.9)
    _arrow(ax, 4.9, 6.3, 6.6, 5.9)
    _arrow(ax, 7.5, 6.3, 8.5, 5.95)
    _arrow(ax, 5.0, 6.3, 5.0, 5.9)
    _arrow(ax, 5.0, 5.3, 5.0, 4.9)
    _arrow(ax, 5.0, 4.3, 5.0, 3.9)
    _arrow(ax, 5.0, 3.3, 5.0, 2.9)
    _arrow(ax, 5.0, 2.3, 5.0, 1.9)
    _arrow(ax, 5.0, 1.3, 5.0, 0.9)

    ax.text(
        8.7,
        1.6,
        "Failure paths:\n"
        "retrieval < threshold\n    -> no-evidence response\n"
        "LLM API failure\n    -> evidence-first response\n"
        "citation validation fails\n    -> low-confidence warning\n"
        "API rate limit -> backoff\n"
        "vector DB unavailable -> in-memory",
        fontsize=7,
        ha="center",
        va="center",
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecea", edgecolor="#b71c1c"),
    )

    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS / "rag_architecture.png", dpi=120)
    plt.close(fig)
    print("Saved docs/rag_architecture.png")


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    steps = [
        ("User Query", "#e3f2fd", "#1565c0"),
        ("Query Understanding\n(classify: numeric / definition / table ...)", "#e3f2fd", "#1565c0"),
        ("Query Expansion\n(optional variants)", "#e3f2fd", "#1565c0"),
        ("Dense Retrieval + Sparse Retrieval\n(FAISS cosine + BM25)", "#e8eaf6", "#283593"),
        ("Hybrid Fusion\n(weighted alpha=0.7 or RRF)", "#e8eaf6", "#283593"),
        ("Reranking\n(optional cross-encoder)", "#f3e5f5", "#6a1b9a"),
        ("Evidence Selection\n(top-k, table-aware for numeric queries)", "#e8f5e9", "#2e7d32"),
        ("Grounded Context Construction", "#e8f5e9", "#2e7d32"),
        ("Generation (LLM, grounded prompt)", "#fff3e0", "#e65100"),
        ("Citation Validation\n(numeric verification, claim support)", "#fce4ec", "#c62828"),
        ("Answer + Citations\n[Page X, Section Y]", "#e8f5e9", "#1b5e20"),
    ]
    y = 11.7
    ys = []
    for text, color, edge in steps:
        _box(ax, 1.6, y, 6.8, 0.82, text, color=color, edge=edge, size=8)
        ys.append(y + 0.41)
        y -= 0.95
    for i in range(len(ys) - 1):
        if i not in (4,):
            _arrow(ax, 5.0, ys[i] - 0.41, 5.0, ys[i + 1] + 0.41)
    _arrow(ax, 5.0, ys[4] - 0.41, 5.0, ys[5] + 0.41)

    ax.text(
        8.3,
        6.5,
        "Low confidence / insufficient evidence\n-> 'I could not find sufficient evidence\n    in the provided document.'",
        fontsize=7,
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecea", edgecolor="#b71c1c"),
    )

    fig.tight_layout()
    fig.savefig(DOCS / "retrieval_workflow.png", dpi=120)
    plt.close(fig)
    print("Saved docs/retrieval_workflow.png")


if __name__ == "__main__":
    draw_architecture()
    draw_workflow()
