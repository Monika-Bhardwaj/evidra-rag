# Evaluation Methodology

EVIDRA measures itself with three deterministic harnesses — no LLM judge, no manual grading, no
seed-dependent randomness. This page defines what is measured, how, and what the numbers mean.

## The three harnesses

### 1. Golden set — `src/evaluation/questions.json` (18 questions)

Each question carries `required_evidence` phrases, `expected_values`, and `expected_pages`, all
curated from the paper text. The harness:

1. Runs the full pipeline (`scripts/evaluate.py`) for each question.
2. Builds relevance labels **deterministically**: a chunk is relevant if it contains any required
   phrase. Retrieval metrics are computed from these labels over the retrieved ranking.
3. Grades the produced answer: every expected value must appear (numeric recall), every required
   phrase should appear (completeness), citations must point at pages present in the evidence,
   and the claim layer must mark all claims supported (faithfulness).

Latest run (`docs/eval_results/20260923-153131`, offline-extractive):

| Metric | Value |
|---|---|
| Golden pass (18/18) | **100%** |
| Recall@6 | 0.8148 |
| Precision@6 | 0.2315 |
| MRR | 0.7398 |
| nDCG@6 | 0.6949 |
| Numeric recall (key values in answer) | 1.0000 |
| Completeness (required phrases) | 0.8981 |
| Citation accuracy | 1.0000 |
| Faithfulness (claims supported) | 1.0000 |
| Page accuracy | 1.0000 |

### 2. Abstention set — `src/evaluation/abstain_questions.json` (6 questions)

Out-of-knowledge questions (capital of France, chemical formula of salt, revenue figures, ...).
Correct behaviour is the **exact abstention sentence**, byte-for-byte:

> I could not find sufficient evidence for this answer in the provided document.

Latest run: **6/6 correct**; exact-string rate **1.0**. Before the ABS-6 fix, one question
leaked a plausible-looking answer — the exact-string check exists specifically to catch that
class of regression.

### 3. Adversarial set — `src/evaluation/adversarial_cases.json` (3 cases)

Each case injects a prompt-injection instruction into a retrieved chunk (reveal system prompt,
invent a number, output only a secret string). The harness asserts the injected token never
appears in the answer. **3/3 passing** (`INJ-1`, `INJ-2`, `INJ-3`); INJ-2 additionally expects
abstention if the injection leaks.

## Exact-string abstention

Abstention is not "any refusal": the suite checks the exact sentence. This makes the fail-closed
boundary regression-proof and is what converts "we abstain" from a claim into a measured
invariant.

## What is deliberately NOT measured

- No LLM-as-judge grading — excluded by design to keep evaluation deterministic and CPU-only
  (see `docs/future_work.md` for the option).
- No ±seed variance — thresholds and relevance labels are fixed, so two runs of the same commit
  must be identical. If they are not, that difference is itself a bug to fix.

## Question curation notes

- Q2 was **replaced** (2026-09-23): the original metadata-URL question was structurally
  unanswerable — its gold chunk had dense cosine 0.119 and ranked 86/139. It was swapped for a
  verbatim abstract question. The swap is documented in `questions.json` notes and
  `docs/failure_analysis.md`.
- ABS-6 was added with the exact-string abstention check after the leak post-mortem.

## Runs and reproducibility

`scripts/evaluate.py` archives every run as `docs/eval_results/<timestamp>/`:

```
evaluation_results.json   per-question results + aggregate metrics
claims.jsonl              claim-level verifier output for the run
abstention_results.json   per-question abstention verdicts (exact-string)
evaluation_report.md      rendered report (headline numbers above)
```

The UI's evaluation tab lists archives, re-runs the harness, and compares runs side by side.
Two committed archives document the ABS-6 before/after delta:
`20260923-145944` (pre-fix leak) and `20260923-153131` (fixed). See
`docs/reproducibility.md`.