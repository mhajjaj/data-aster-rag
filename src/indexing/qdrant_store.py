"""Qdrant vector store implementation."""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.indexing.vector_store import BaseVectorStore, SearchResult

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-based vector store."""

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.client: Optional[QdrantClient] = None

    def connect(self) -> None:
        logger.info(f"Connecting to Qdrant at {self.url}")
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

    def create_collection(self, collection_name: str, dimension: int) -> None:
        if not self.client:
            self.connect()
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {collection_name} (dim={dimension})")

    def upsert(self, collection_name: str, vectors: List[Dict[str, Any]]) -> None:
        if not self.client:
            self.connect()
        points = [
            PointStruct(
                id=v["id"],
                vector=v["embedding"],
                payload={
                    "text": v["text"],
                    **v.get("metadata", {}),
                },
            )
            for v in vectors
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} points into {collection_name}")

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not self.client:
            self.connect()
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter,
        )
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                text=r.payload.get("text", ""),
                metadata={k: v for k, v in r.payload.items() if k != "text"},
            )
            for r in results
        ]

    def delete_collection(self, collection_name: str) -> None:
        if not self.client:
            self.connect()
        self.client.delete_collection(collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            self.connect()
        return self.client.collection_exists(collection_name)
