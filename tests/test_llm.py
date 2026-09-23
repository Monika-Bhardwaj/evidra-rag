"""Tests for LLM failure modes: circuit breaker, timeout plumbing, recovery."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.generation import llm as llm_module
from src.generation.llm import (
    CIRCUIT_BREAKER_COOLDOWN_S,
    CIRCUIT_BREAKER_THRESHOLD,
    OpenAIProvider,
    _CircuitState,
    build_llm,
)


def _ok_response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None
    )


def _messages() -> list[dict]:
    return [{"role": "user", "content": "What does the paper report?"}]


def test_circuit_state_opens_after_threshold_and_recovers() -> None:
    state = _CircuitState()
    now = 1000.0
    for i in range(CIRCUIT_BREAKER_THRESHOLD - 1):
        assert not state.record_failure(now + i)
        assert not state.is_open(now + i + 1)
    assert state.record_failure(now + 3)  # opens on the final allowed failure
    assert state.is_open(now + 3 + CIRCUIT_BREAKER_COOLDOWN_S - 0.1)
    assert not state.is_open(now + 3 + CIRCUIT_BREAKER_COOLDOWN_S + 0.1)
    state.record_success()
    assert state.consecutive_failures == 0
    assert not state.is_open(now + 3 + CIRCUIT_BREAKER_COOLDOWN_S + 0.1)


def test_provider_breaker_fails_fast_after_three_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float | None] = []

    def failing(client, messages, model, temperature, max_tokens, timeout=None):
        calls.append(timeout)
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_module, "_chat_completion_with_retry", failing)
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini", timeout_seconds=7.5)

    # Every call reaches the provider while the circuit is closed; each reports
    # the failure verbatim (fail-closed, no fabricated text).
    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        result = provider.chat(_messages())
        assert result.failed is True
        assert not result.text
        assert "provider down" in (result.error or "")
    assert len(calls) == CIRCUIT_BREAKER_THRESHOLD
    assert all(t == 7.5 for t in calls)  # llm_timeout_seconds reaches the request

    # Circuit is now open: fail fast without touching the provider.
    result = provider.chat(_messages())
    assert result.failed is True
    assert "circuit" in (result.error or "").lower()
    assert len(calls) == CIRCUIT_BREAKER_THRESHOLD


def test_breaker_half_open_after_cooldown_and_closes_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "CIRCUIT_BREAKER_COOLDOWN_S", 0.01)
    outcome = {"fail": True}
    calls: list[int] = []

    def flaky(client, messages, model, temperature, max_tokens, timeout=None):
        calls.append(1)
        if outcome["fail"]:
            raise RuntimeError("provider down")
        return _ok_response("recovered")

    monkeypatch.setattr(llm_module, "_chat_completion_with_retry", flaky)
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        provider.chat(_messages())
    assert len(calls) == CIRCUIT_BREAKER_THRESHOLD
    assert provider.chat(_messages()).failed  # open -> short-circuit
    assert len(calls) == CIRCUIT_BREAKER_THRESHOLD

    time.sleep(0.02)  # cooldown elapses -> half-open probe reaches the provider
    outcome["fail"] = False
    result = provider.chat(_messages())
    assert not result.failed
    assert result.text == "recovered"
    assert len(calls) == CIRCUIT_BREAKER_THRESHOLD + 1


def test_success_resets_consecutive_failure_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the reset, 1 failure + success + 2 failures would open the circuit early."""
    calls: list[int] = []
    fail_on_calls = {1, 3, 4, 5}  # fail, recover, then three consecutive failures

    def flaky(client, messages, model, temperature, max_tokens, timeout=None):
        calls.append(len(calls) + 1)
        if len(calls) in fail_on_calls:
            raise RuntimeError("transient")
        return _ok_response()

    monkeypatch.setattr(llm_module, "_chat_completion_with_retry", flaky)
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

    for _ in range(CIRCUIT_BREAKER_THRESHOLD + 2):
        provider.chat(_messages())
    # Calls 1-5 all reached the provider: two successes reset the failure count,
    # so the breaker opened only after three *consecutive* failures (calls 3-5).
    assert len(calls) == CIRCUIT_BREAKER_THRESHOLD + 2
    assert provider.chat(_messages()).failed  # now open
    assert len(calls) == CIRCUIT_BREAKER_THRESHOLD + 2


def test_build_llm_passes_timeout_and_openai_config() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        llm_timeout_seconds=12.5,
        llm_max_tokens=256,
    )
    provider = build_llm(settings)
    assert isinstance(provider, OpenAIProvider)
    assert provider.timeout_seconds == 12.5
    assert provider.max_tokens == 256


def test_build_llm_without_keys_falls_back_to_offline_extractive() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="",
        groq_api_key="",
        gemini_api_key="",
    )
    provider = build_llm(settings)
    assert provider.provider_name == "offline-extractive"
