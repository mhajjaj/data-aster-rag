"""Embedding factory and helpers."""

import logging
from typing import List

import openai
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Unified interface for text embeddings via OpenAI or local sentence-transformers."""

    def __init__(self, model_name: str = None, provider: str = "openai"):
        self.provider = provider
        self.model_name = model_name or settings.embedding_model

        if provider == "openai":
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
        elif provider == "local":
            self.model = SentenceTransformer(self.model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [r.embedding for r in resp.data]
        else:
            return self.model.encode(texts, convert_to_list=True)

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]
