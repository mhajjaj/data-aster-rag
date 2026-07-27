"""Text chunking strategies."""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: dict


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict = None,
) -> List[Chunk]:
    """Simple fixed-size chunking with overlap."""
    if not text:
        return []

    metadata = metadata or {}
    chunks: List[Chunk] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                },
            )
        )
        if end >= len(text):
            break
        start += step

    return chunks
