"""Vector store interface."""

import abc
from typing import Any, Dict, List, Optional


class SearchResult:
    def __init__(self, id: str, score: float, text: str, metadata: Dict[str, Any]):
        self.id = id
        self.score = score
        self.text = text
        self.metadata = metadata


class BaseVectorStore(abc.ABC):
    """Abstract vector store for indexing and retrieving document chunks."""

    @abc.abstractmethod
    def connect(self) -> None:
        ...

    @abc.abstractmethod
    def create_collection(self, collection_name: str, dimension: int) -> None:
        ...

    @abc.abstractmethod
    def upsert(self, collection_name: str, vectors: List[Dict[str, Any]]) -> None:
        """
        vectors: list of dicts with keys:
            - id: str
            - embedding: List[float]
            - text: str
            - metadata: dict
        """
        ...

    @abc.abstractmethod
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        ...

    @abc.abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        ...

    @abc.abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        ...
