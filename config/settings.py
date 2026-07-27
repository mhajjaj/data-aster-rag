"""Centralized configuration for the RAG pipeline."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default=Path(__file__).resolve().parent.parent.parent)
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

    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Retrieval
    top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=5)

    # OCR / Vision
    ocr_engine: str = Field(default="paddle")  # paddle, easyocr, tesseract
    vision_model: str = Field(default="gpt-4o")  # for image description

    def __post_init__(self):
        self.data_raw_dir = (self.project_root / self.data_raw_dir).resolve()
        self.data_processed_dir = (self.project_root / self.data_processed_dir).resolve()


settings = Settings()
