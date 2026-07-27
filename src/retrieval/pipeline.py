"""Retrieval orchestrator: integrates query analysis, jargon resolution, vector search."""

import logging
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.indexing.embeddings import EmbeddingClient
from src.indexing.vector_store import BaseVectorStore
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.jargon_lookup import JargonLookup

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """End-to-end retrieval with query understanding, jargon resolution, and multi-hop search."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_client: EmbeddingClient,
        query_analyzer: Optional[QueryAnalyzer] = None,
        jargon_lookup: Optional[JargonLookup] = None,
    ):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.jargon = jargon_lookup
        self.collection = settings.vector_store_collection

    def retrieve(self, query: str, top_k: int = None, rerank: bool = True) -> List[Dict[str, Any]]:
        top_k = top_k or settings.top_k

        # 1) Analyze query
        analysis = self.query_analyzer.analyze(query)

        # 2) Resolve jargon if needed
        search_queries = analysis.sub_questions[:]
        if analysis.requires_jargon_lookup and self.jargon:
            resolved = [self.jargon.resolve(q) for q in search_queries]
            search_queries.extend(resolved)

        # 3) Retrieve for each sub-question
        all_results: List[Dict[str, Any]] = []
        seen_ids = set()

        for sq in search_queries:
            embedding = self.embedding_client.embed_single(sq)
            results = self.vector_store.search(
                collection_name=self.collection,
                query_embedding=embedding,
                top_k=top_k,
            )
            for r in results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    all_results.append({
                        "id": r.id,
                        "score": r.score,
                        "text": r.text,
                        "metadata": r.metadata,
                        "matched_query": sq,
                    })

        # 4) Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # 5) Optional re-ranking
        if rerank and len(all_results) > settings.rerank_top_k:
            all_results = self._rerank(analysis, all_results)

        final_top_k = settings.rerank_top_k if rerank else top_k
        return all_results[:final_top_k]

    def _rerank(self, analysis, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple cross-encoder style re-ranking (placeholder for real model)."""
        # For now: boost score if key entities are present in text
        for c in candidates:
            text_lower = c["text"].lower()
            for entity in analysis.key_entities:
                if entity.lower() in text_lower:
                    c["score"] += 0.1
            for kw in analysis.keywords:
                if kw.lower() in text_lower:
                    c["score"] += 0.05
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
