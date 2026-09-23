from __future__ import annotations

import pytest

from src.ingestion.loader import PageDocument, PDFLoader
from src.ingestion.parser import Parser, TextPreprocessor


def test_normalize_whitespace_and_hyphenation() -> None:
    prep = TextPreprocessor()
    cleaned = prep.normalize("Agent-as-a-Judge\nis a   framework\n\n\n\nwith multiple  spaces.")
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_dehyphenation() -> None:
    prep = TextPreprocessor()
    cleaned = prep.normalize("works-\npaces")
    assert "workspaces" in cleaned


def test_heading_detection() -> None:
    parser = Parser(pdf_path=None)
    try:
        assert parser._is_heading("4.4 Cost Analysis", 12.0, True)
        assert parser._is_heading("Appendix K Search Modules", 12.0, True)
        assert parser._is_heading("2.2 DevAI", 10.0, False)
        assert not parser._is_heading(
            "Agent-as-a-Judge is a framework that evaluates agentic systems.", 10.0, False
        )
        assert not parser._is_heading("Table 3 openhands costs", 10.0, True)
    finally:
        parser.close()


def test_repeated_short_lines_detected() -> None:
    span_lines = [
        ("arXiv:2410.10934", 9.9, False),
        ("arXiv:2410.10934", 9.9, False),
        ("arXiv:2410.10934", 9.9, False),
        ("content line here", 9.9, False),
    ]
    repeats = Parser._repeated_short_lines(span_lines)
    assert "arXiv:2410.10934" in repeats
    assert "content line here" not in repeats


def test_parser_produces_sectioned_units(tmp_path, monkeypatch) -> None:
    page = PageDocument(
        page=4,
        text="4.1 Benchmark\nThree open-source frameworks were evaluated on DevAI.\nOpenHands solved requirements.",
        source="Agent-as-a-Judge.pdf",
    )
    parser = Parser(pdf_path=None)
    try:
        units = parser.parse([page])
    finally:
        parser.close()
    kinds = [u.kind for u in units]
    assert "heading" in kinds
    narrative = [u for u in units if u.kind in ("narrative", "definition")]
    assert narrative
    assert "OpenHands solved requirements" in narrative[0].text
    assert narrative[0].section == "4.1 Benchmark"


def test_loader_missing_pdf(tmp_path) -> None:
    loader = PDFLoader(tmp_path / "missing.pdf")
    with pytest.raises(FileNotFoundError):
        loader.load()
