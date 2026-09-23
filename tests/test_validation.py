from __future__ import annotations

from src.schemas import DocumentChunk, RetrievedChunk
from src.validation.claims import ClaimExtractor, ClaimVerifier


def _chunk(page: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"c{page}",
        text=(
            "OpenHands achieved a 9.0% cost reduction and 21.6% latency improvement "
            "for AgentBench requests. The system processed 362.41 seconds total."
        ),
        page=page,
        section="Results",
        chunk_type="narrative",
    )


def _evidence(*pages: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=_chunk(p), hybrid_score=1.0) for p in pages]


def test_claim_extractor_bullets_with_page_markers() -> None:
    answer = (
        "According to the paper:\n"
        "- OpenHands achieved a 9.0% cost reduction [Page 3, Section Results]\n"
        "- The system processed 362.41 seconds total [Page 3, Section Results]"
    )
    claims = ClaimExtractor().extract(answer)
    assert len(claims) == 2
    assert all(c["page"] == 3 for c in claims)
    assert not any("According to the paper" in c["text"] for c in claims)


def test_claim_extractor_skips_meta_and_short_lines() -> None:
    answer = (
        "- I could not find sufficient evidence for this answer in the provided document.\n"
        "- Hi [Page 1, Section X]\n"
        "- Real content sentence cited properly [Page 4, Section Y]"
    )
    claims = ClaimExtractor().extract(answer)
    assert all("could not find sufficient" not in c["text"] for c in claims)
    assert len(claims) == 1


def test_claim_verifier_supports_verbatim_evidence() -> None:
    verifier = ClaimVerifier()
    answer = "- OpenHands achieved a 9.0% cost reduction [Page 3, Section Results]"
    verdicts = verifier.verify(answer, _evidence(3))
    assert len(verdicts) == 1
    claim = verdicts[0]
    assert claim.supported
    assert claim.page_matched
    assert claim.numeric_matched
    assert claim.verdict == "supported"


def test_claim_verifier_rejects_unrelated_claim() -> None:
    verifier = ClaimVerifier()
    answer = "- The Eiffel Tower is located in Paris and is 330 meters tall [Page 7, Section Intro]"
    verdicts = verifier.verify(answer, _evidence(3))
    assert len(verdicts) == 1
    assert not verdicts[0].supported
    assert verdicts[0].verdict == "unsupported"


def test_claim_verifier_rejects_fabricated_number() -> None:
    verifier = ClaimVerifier()
    answer = "- OpenHands achieved a 99.0% cost reduction [Page 3, Section Results]"
    verdicts = verifier.verify(answer, _evidence(3))
    assert not verdicts[0].numeric_matched
    assert not verdicts[0].supported


def test_claim_verifier_rejects_wrong_page() -> None:
    verifier = ClaimVerifier()
    answer = "- OpenHands achieved a 9.0% cost reduction [Page 9, Section Results]"
    verdicts = verifier.verify(answer, _evidence(3))
    assert not verdicts[0].page_matched
    assert not verdicts[0].supported


def test_claim_verifier_page_unknown_still_supported() -> None:
    verifier = ClaimVerifier()
    answer = "- OpenHands achieved a 9.0% cost reduction"
    verdicts = verifier.verify(answer, _evidence(3))
    claim = verdicts[0]
    assert claim.page is None
    assert claim.supported
    assert claim.page_matched  # page=None is treated as not-checkable


def test_claim_extractor_collapses_multiline_bullet() -> None:
    answer = (
        "- DevAI consists of a curated set of 55\n"
        "tasks, each defined by (1) a plain text user query; (2) a set of plain\n"
        "text requirements (for a total of 365 requirements). [Page 4, Section The DevAI Dataset]"
    )
    claims = ClaimExtractor().extract(answer)
    assert len(claims) == 1
    assert claims[0]["page"] == 4
    assert "55" in claims[0]["text"] and "365" in claims[0]["text"]


def test_claim_verifier_low_support_is_not_unsupported() -> None:
    verifier = ClaimVerifier()
    answer = "- (2) Number of Words in User Queries"
    verdicts = verifier.verify(answer, _evidence(3))
    assert len(verdicts) == 1
    assert not verdicts[0].supported
    assert verdicts[0].verdict == "low-support"
    assert verdicts[0].page_matched  # page None, so page_matched True
    assert verdicts[0].verdict != "unsupported"


def test_claim_extractor_drops_debris_lines() -> None:
    answer = "- !\"#$%&'(\n- Real content about OpenHands [Page 3, Section Results]"
    claims = ClaimExtractor().extract(answer)
    assert len(claims) == 1
    assert "OpenHands" in claims[0]["text"]
