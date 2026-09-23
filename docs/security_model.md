# Security Model

EVIDRA treats itself as a system answering questions **from one trusted document** while exposed
to untrusted inputs (user prompts and, critically, retrieved text that may have been crafted to
instruct the model). Security is layered, fail-closed, and partially automated-test enforced.

## Threat model

| Threat | In scope? | Defence |
|---|---|---|
| Prompt injection embedded in retrieved chunks | Yes | Neutralization before prompt composition + guard + adversarial harness |
| Direct user jailbreak ("ignore the PDF", "reveal your prompt") | Yes | `JailbreakGuard` + keep-grounded response |
| API abuse (flooding, oversized prompts) | Yes | Rate limit (per-IP token bucket), length cap, optional Basic auth |
| LLM provider outage / hang / degraded upstream | Yes | Timeout, retry/backoff, circuit breaker, evidencing fallback |
| Secrets in the image or repo | Yes | Environment-only keys, dockerignore, `.env` gitignored |
| Multi-tenant isolation, uploaded content, RCE via PDF | Partially | Attacker-controlled PDFs are **not** yet in scope (single trusted corpus); the parser opens files via PyMuPDF/pypdf only |

## Layer 1 — retrieved text is data, not instructions

- Evidence is wrapped in explicit `>>RETRIEVED_EVIDENCE_START<< ... >>RETRIEVED_EVIDENCE_END<<`
  delimiters and the system prompt declares it data.
- `neutralize_retrieved_text` (in `src/generation/security.py`) scrubs instruction-like text from
  evidence **before** prompt composition, so the LLM never even sees the raw injection.
- `JailbreakGuard` matches known patterns (ignore-the-PDF, reveal-system-prompt, invent-a-number,
  do-not-cite, ...); flagged queries still receive a PDF-grounded response with an explicit
  neutralization note.

## Layer 2 — fail-closed generation

- If the LLM fails (error, timeout, rate limit exhausted), the pipeline returns the best
  supporting evidence verbatim with "(Generation failed; no invented content returned.)" — never
  a plausible guess.
- The circuit breaker (3 consecutive failures → 30 s cooldown, half-open probe) prevents a
  wedged provider from hanging every request; providers return `LLMResult(failed=True, error=...)`.
- Unsupported claims anywhere in the answer force the exact abstention sentence.

## Layer 3 — API hardening (`src/api/`)

- `/api/query` length cap (`API_MAX_QUESTION_CHARS`, default 4000).
- Per-IP token-bucket rate limit (`API_RATE_LIMIT_PER_MIN`, default 60; 0 disables).
- HTTP Basic auth when `API_AUTH_USERNAME` **and** `API_AUTH_PASSWORD` are both set
  (`require_auth` dependency).
- `/api/health` = liveness (no pipeline required); `/api/ready` = readiness (503 until the
  index is loaded) — the compose healthcheck uses `/api/ready`, so the UI never starts against a
  half-loaded backend.

## Layer 4 — platform

- Docker runs as a dedicated **non-root user** (`evidra`, uid 10001); `curl` is installed only
  for the healthcheck. Data and index live on named volumes owned by that user.
- No API keys in the image: secrets come from the container environment / `.env` (gitignored).
- `.dockerignore` excludes `.git`, tests, docs, `migrate_opencode_v1_to_v2.py`, and caches.

## Measured security

- The adversarial harness (`src/evaluation/adversarial_cases.json`) injects three instructions
  into retrieved chunks and asserts the injected tokens never leak into answers — **3/3 passing**
  (run `20260923-153131`).
- `tests/test_security.py` exercises injection payloads through the pipeline.
- `docs/rag_failure_taxonomy.md` classifies failure behaviour; the security entries there map to
  the layers above.

## Known boundaries (honest)

- `JailbreakGuard` is rule-based; novel paraphrases are evaluated by the adversarial harness but
  not guaranteed exhaustive.
- The corpus is a single trusted PDF. Persistent multi-tenant uploads would reopen threat model
  questions (document parsing as an attack surface, prompt-injection in *other* documents) — see
  `docs/future_work.md`.