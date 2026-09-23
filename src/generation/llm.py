from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import tenacity

from src.config import Settings, est_price_for_model
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    failed: bool = False
    error: Optional[str] = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _chat_completion_with_retry(client, messages, model, temperature, max_tokens):
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1.0, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=(
            tenacity.retry_if_exception_type((ConnectionError, TimeoutError))
            | tenacity.retry_if_exception(lambda e: _is_rate_limit(e))
        ),
        reraise=True,
    )
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _call()


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "429" in str(exc) or "apiconnection" in name


class LLMProvider(ABC):
    provider_name = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_rate, self.output_rate = est_price_for_model(model)

    @abstractmethod
    def _chat(self, messages: List[Dict[str, str]]) -> LLMResult: ...

    def chat(self, messages: List[Dict[str, str]]) -> LLMResult:
        return self._chat(messages)

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000 * self.input_rate
            + completion_tokens / 1_000_000 * self.output_rate
        )


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        from openai import OpenAI

        super().__init__(model, temperature, max_tokens)
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def _chat(self, messages: List[Dict[str, str]]) -> LLMResult:
        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        try:
            response = _chat_completion_with_retry(
                self._client, messages, self.model, self.temperature, self.max_tokens
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return LLMResult(
                text="",
                provider=self.provider_name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                failed=True,
                error=str(exc),
            )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        completion_tokens = usage.completion_tokens if usage else estimate_tokens(text)
        return LLMResult(
            text=text,
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._cost(usage.prompt_tokens if usage else prompt_tokens, completion_tokens),
        )


class GroqProvider(OpenAIProvider):
    provider_name = "groq"

    def __init__(
        self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )


class GeminiProvider(OpenAIProvider):
    provider_name = "gemini"

    def __init__(
        self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            temperature=temperature,
            max_tokens=max_tokens,
        )


class LocalExtractiveProvider(LLMProvider):
    provider_name = "offline-extractive"

    def __init__(
        self, model: str = "offline-extractive", temperature: float = 0.0, max_tokens: int = 512
    ) -> None:
        super().__init__(model, temperature, max_tokens)

    def _chat(self, messages: List[Dict[str, str]]) -> LLMResult:
        user_text = messages[-1].get("content", "") if messages else ""
        answer = self._extractive_answer(user_text)
        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        return LLMResult(
            text=answer,
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=estimate_tokens(answer),
            cost_usd=0.0,
        )

    def _extractive_answer(self, user_text: str) -> str:
        blocks = self._parse_evidence(user_text)
        question = self._extract_question(user_text)
        if not blocks:
            return "I could not find sufficient evidence for this answer in the provided document."
        question_tokens = self._meaningful_tokens(question)
        ranked: List[tuple] = []
        for block in blocks:
            sentences = self._split_sentences(block["text"])
            section_tokens = self._meaningful_tokens(block.get("section") or "")
            for sent in sentences:
                score = self._overlap(question_tokens, sent) + 0.5 * len(
                    set(question_tokens) & set(section_tokens)
                )
                if score > 0:
                    ranked.append((score, sent, block))
        ranked.sort(key=lambda t: t[0], reverse=True)
        if not ranked:
            return "I could not find sufficient evidence for this answer in the provided document."
        seen_blocks: Dict[str, Dict] = {}
        chosen: List[tuple] = []
        for _, sent, block in ranked:
            cid = block["chunk_id"]
            if cid in seen_blocks:
                continue
            seen_blocks[cid] = block
            chosen.append((sent, block))
            if len(chosen) >= 2:
                break
        parts = ["According to the paper:"]
        budget_chars = 4200
        used = 0
        for best_sent, block in chosen:
            sentences = self._split_sentences(block["text"])
            select = self._select_sentences(sentences, question_tokens)
            if not select:
                select = [sentences[0]] if sentences else []
            label = f"[Page {block['page']}, Section {block['section']}]"
            for sent in select:
                line = f"- {sent} {label}"
                if used + len(line) > budget_chars and used > 0:
                    break
                parts.append(line)
                used += len(line)
        if len(parts) == 1:
            return "I could not find sufficient evidence for this answer in the provided document."
        return "\n".join(parts)

    def _select_sentences(self, sentences: List[str], question_tokens: List[str]) -> List[str]:
        if not sentences:
            return []
        overlap_flags = []
        for i, sent in enumerate(sentences):
            score = self._overlap(question_tokens, sent)
            numeric = bool(re.search(r"\d", sent))
            overlap_flags.append(score > 0 or numeric)
        keep = [i for i, used in enumerate(overlap_flags) if used]
        for i, used in enumerate(overlap_flags):
            if used:
                if i - 1 >= 0:
                    keep.append(i - 1)
                if i + 1 < len(sentences):
                    keep.append(i + 1)
        keep = sorted(set(keep))
        return [sentences[i] for i in keep]

    def _parse_evidence(self, user_text: str) -> List[Dict[str, str]]:
        pattern = re.compile(
            r"\[Source \d+\]\n"
            r"Page: (\d+)\n"
            r"Section: (.+?)\n"
            r"(?:Chunk: (\S+)\n)?"
            r"Text: ([\s\S]*?)(?=\n\n\[Source |\n\n>>RETRIEVED_EVIDENCE_END|$)"
        )
        blocks = []
        for match in pattern.finditer(user_text):
            blocks.append(
                {
                    "page": match.group(1),
                    "section": match.group(2).strip(),
                    "chunk_id": match.group(3) or "unknown",
                    "text": match.group(4).strip(),
                }
            )
        return blocks

    def _extract_question(self, user_text: str) -> str:
        match = re.search(r"Question: (.+)", user_text)
        return match.group(1).strip() if match else user_text

    @staticmethod
    def _meaningful_tokens(text: str) -> List[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "of",
            "to",
            "and",
            "in",
            "for",
            "how",
            "what",
            "which",
            "was",
            "is",
            "are",
            "were",
            "does",
            "do",
            "did",
            "it",
            "its",
            "that",
            "this",
        }
        tokens = re.findall(r"[a-zA-Z0-9$\-.%]+", text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    @staticmethod
    def _overlap(query_tokens: List[str], sentence: str) -> float:
        sentence_tokens = set(re.findall(r"[a-zA-Z0-9$.\-]+", sentence.lower()))
        if not sentence_tokens:
            return 0.0
        hits = sum(1 for t in query_tokens if t in sentence_tokens or t in sentence.lower())
        return hits / max(1, len(query_tokens) * 0.5)


def build_llm(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider
    if provider == "openai" and settings.openai_api_key:
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    if provider == "groq" and settings.groq_api_key:
        return GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    if provider == "gemini" and settings.gemini_api_key:
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    if provider == "auto":
        if settings.openai_api_key:
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        if settings.gemini_api_key:
            return GeminiProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        if settings.groq_api_key:
            return GroqProvider(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
    logger.info(
        "No LLM API key configured; using offline extractive generation (provider=%s).", provider
    )
    return LocalExtractiveProvider(model="offline-extractive")
