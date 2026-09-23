# Future Work

Prioritized by *measured need*, not by fashion. EVIDRA deliberately stops before
over-engineering: no graph RAG, no microservices, no Kubernetes — until an experiment shows a
measurable gain or a concrete deployment requires it.

## High value, low risk — do next

1. **LLM-as-judge grading (optional)**
   The current harness is deterministic by design. A *second*, optional harness could grade
   answer quality with an LLM judge while keeping the deterministic harness as the primary gate.
   Would re-introduce cost + nondeterminism, so it must remain opt-in and separately archived.

2. **Adversarial coverage expansion**
   The 3 injection cases pass. Add cases for: indirect injection across multi-turn history,
   encoded/obfuscated instructions (base64, unicode), and injection inside *tables* (the largest
   single chunk type in this corpus).

3. **Ablation harness as a first-class CI check**
   `docs/eval_results/ablation/` shows the reranker/no-reranker comparison. Promote ablation to
   a script with archive output (like `evaluate.py`) so config changes carry their measurement
   in the same commit.

4. **Retrieval quality on the tail**
   Recall@6 is 0.8148; three golden questions pull it down (Q1/Q16 R@6 0.25, Q7 0.167 — see the
   run report). Investigate: phrase-pair matching for relevance labels (current labels may
   under-credit correct retrieval), and section-anchored retrieval for definition-style facts.

## Medium value — next milestone

5. **Multi-document RAG** (folder of PDFs, DOCX, websites)
   Requires rework of: source identity in chunks/citations, per-document `pages.json`, the
   sufficiency gate (per-source bars), and cross-document fact resolution. Single-document
   limitation is stated in README + failure taxonomy.

6. **Adaptive retrieval**
   Route queries to a specialised retriever ensemble (the paper's own lesson: its best alignment
   came *without* a search module). Would be justified only by a measured recall delta on a
   wider corpus.

7. **Conversational memory with reference resolution**
   Bounded raw history exists; entity/claim tracking across turns would make follow-ups
   ("and the cost?") work on the claim layer rather than raw text.

## Low priority / explicitly deferred

8. **Graph RAG / knowledge graphs** — DevAI's requirement DAGs are a natural fit, but there is no
   measured retrieval problem here that demands it. Revisit only if multi-document + dependency
   questions show the need.
9. **Microservices / Kubernetes / vector DB servers** — actively avoided; the single-process
   pipeline is simpler, faster to reason about, and cheap. Docker compose satisfies the
   deployment requirement.
10. **Multimodal (figures/diagrams)** — the paper's figures carry real signal; requires an
    image index and caption→text alignment. Substantial, visualize as a separate effort.
11. **Cloud vector databases** — only when the local index becomes operationally painful.

## Guardrails for new work

- Every change ships with its measurement (archive) and a regression test.
- Anything that cannot be measured on this corpus with the existing harness needs a new
  deterministic harness first.
- No new dependency without a pinned version and a reason in the commit message.