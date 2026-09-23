from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: Literal["auto", "openai", "gemini", "groq", "local"] = "auto"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    chunk_size_tokens: int = 500
    chunk_overlap: float = 0.15

    dense_top_k: int = 48
    bm25_top_k: int = 48
    rerank_top_k: int = 5
    final_top_k: int = 6
    hybrid_alpha: float = 0.5
    hybrid_method: Literal["weighted", "rrf"] = "weighted"
    min_similarity: float = 0.35
    knowledge_boundary_min_sim: float = 0.30
    use_query_expansion: bool = True
    use_query_routing: bool = True
    context_max_tokens: int = 4000

    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_kind: Literal["cross", "score"] = "cross"

    enable_retrieval_cache: bool = True
    cache_dir: str = "data/processed/cache"

    eval_questions_path: str = "src/evaluation/questions.json"
    eval_report_path: str = "docs/evaluation_report.md"
    abstain_questions_path: str = "src/evaluation/abstain_questions.json"
    adversarial_cases_path: str = "src/evaluation/adversarial_cases.json"

    api_max_question_chars: int = 4000
    api_auth_username: str = ""
    api_auth_password: str = ""
    api_rate_limit_per_min: int = 60

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    index_dir: str = "data/processed/index"
    pdf_path: str = "data/raw/Agent-as-a-Judge.pdf"
    document_name: str = "Agent-as-a-Judge.pdf"

    log_level: str = "INFO"

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    def resolve(self, value: str) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = ROOT_DIR / p
        return p

    @property
    def data_dir_path(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def raw_dir_path(self) -> Path:
        return self.resolve(self.raw_dir)

    @property
    def processed_dir_path(self) -> Path:
        return self.resolve(self.processed_dir)

    @property
    def index_dir_path(self) -> Path:
        return self.resolve(self.index_dir)

    @property
    def pdf_path_resolved(self) -> Path:
        p = Path(self.pdf_path)
        if not p.is_absolute():
            p = ROOT_DIR / p
        return p

    @property
    def cache_dir_path(self) -> Path:
        return self.resolve(self.cache_dir)

    @property
    def eval_questions_path_resolved(self) -> Path:
        return self.resolve(self.eval_questions_path)

    @property
    def eval_report_path_resolved(self) -> Path:
        return self.resolve(self.eval_report_path)

    @property
    def abstain_questions_path_resolved(self) -> Path:
        return self.resolve(self.abstain_questions_path)

    @property
    def adversarial_cases_path_resolved(self) -> Path:
        return self.resolve(self.adversarial_cases_path)

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir_path,
            self.processed_dir_path,
            self.index_dir_path,
            self.cache_dir_path,
        ):
            d.mkdir(parents=True, exist_ok=True)


PRICING_PER_1M_TOKENS = [
    {"pattern": "gpt-4o", "input": 2.5, "output": 10.0},
    {"pattern": "gpt-4o-mini", "input": 0.15, "output": 0.6},
    {"pattern": "gpt-4", "input": 30.0, "output": 60.0},
    {"pattern": "gpt-3.5", "input": 0.5, "output": 1.5},
    {"pattern": "gemini", "input": 0.35, "output": 1.05},
    {"pattern": "llama-3.3-70b", "input": 0.59, "output": 0.79},
    {"pattern": "default", "input": 1.0, "output": 2.0},
]


def est_price_for_model(model: str) -> tuple[float, float]:
    for entry in PRICING_PER_1M_TOKENS:
        if entry["pattern"] in model.lower():
            return entry["input"], entry["output"]
    return PRICING_PER_1M_TOKENS[-1]["input"], PRICING_PER_1M_TOKENS[-1]["output"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
