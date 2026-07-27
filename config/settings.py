"""Configuration loader with validation for the RAG pipeline."""

import logging
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))

    # OpenAI / LLM
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-large")
    max_answer_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.0)

    # Vector Store
    vector_store_url: str = Field(default="http://localhost:6333")
    vector_store_collection: str = Field(default="data_aster_docs")
    vector_dimension: int = Field(default=3072)

    # Qdrant Cloud
    qdrant_api_key: Optional[str] = Field(default=None)

    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Retrieval
    top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=5)

    # OCR / Vision
    ocr_engine: str = Field(default="paddle")
    vision_model: str = Field(default="gpt-4o")

    # Evaluation
    judge_model: str = Field(default="gpt-4o")
    evaluation_timeout_seconds: int = Field(default=10800)  # 3 hours

    @field_validator("openai_api_key", mode="after")
    @classmethod
    def check_api_key(cls, v: str) -> str:
        if not v or v.startswith("sk-") is False:
            logger.warning("OPENAI_API_KEY is missing or looks invalid.")
        return v

    @field_validator("data_raw_dir", "data_processed_dir", mode="after")
    @classmethod
    def resolve_paths(cls, v: Path, info) -> Path:
        root = info.data.get("project_root") or Path(".").resolve()
        if not v.is_absolute():
            return (root / v).resolve()
        return v.resolve()

    def ensure_directories(self):
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
