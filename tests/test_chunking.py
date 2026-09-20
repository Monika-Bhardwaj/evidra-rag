from __future__ import annotations

from src.ingestion.chunker import SemanticChunker
from src.ingestion.parser import Unit


def test_metadata_preserved() -> None:
    chunker = SemanticChunker(chunk_size_tokens=500, overlap=0.15)
    units = [
        Unit(kind="heading", text="4.4 Cost Analysis", page=11, section=""),
        Unit(
            kind="narrative",
            text="Agent-as-a-Judge cost $30.58 and took 118.43 minutes.",
            page=11,
        ),
    ]
    chunks = chunker.chunk(units)
    assert len(chunks) == 1
    assert chunks[0].page == 11
    assert chunks[0].section == "4.4 Cost Analysis"
    assert chunks[0].document_id == "doc-1"


def test_table_kept_whole() -> None:
    chunker = SemanticChunker(chunk_size_tokens=100, overlap=0.1)
    table_text = "Task | Cost | Time\nOpenHands | $6.38 | 362.41\nMetaGPT | $1.19 | 501.2\nGPT-Pilot | $2.33 | 510.2"
    units = [Unit(kind="table", text=table_text, page=6, section="4.1")]
    chunks = chunker.chunk(units)
    assert chunks
    assert any("OpenHands" in c.text and "MetaGPT" in c.text for c in chunks)
    assert all(c.chunk_type == "table" for c in chunks)


def test_chunk_size_budget() -> None:
    chunker = SemanticChunker(chunk_size_tokens=50, overlap=0.2)
    long_text = "word " * 500
    units = [Unit(kind="narrative", text=long_text, page=1, section="1")]
    chunks = chunker.chunk(units)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 55


def test_result_classification() -> None:
    chunker = SemanticChunker()
    kind = chunker.classify_chunk("4.4 Cost Analysis", "narrative", "cost $30 5% 6% 7%")
    assert kind == "cost-analysis"
    kind = chunker.classify_chunk("4.2", "narrative", "90.44% 60.38% 70.76% 80% 5")
    assert kind == "result"
    kind = chunker.classify_chunk("2.2 DevAI", "definition", "we define the DevAI dataset as 55 tasks")
    assert kind == "definition"


def test_no_cross_page_bleed() -> None:
    chunker = SemanticChunker(chunk_size_tokens=10, overlap=0.1)
    units = [
        Unit(kind="narrative", text="page five text five text five text five text five", page=5, section="2"),
        Unit(kind="narrative", text="page six text six text six text six text six", page=6, section="2"),
    ]
    chunks = chunker.chunk(units)
    assert all(c.page in (5, 6) for c in chunks)