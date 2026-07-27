"""Process parsed documents into embeddable chunks with OCR and image captions."""

import base64
import json
import logging
from typing import Any, Dict, List, Optional

from src.ingestion.base import ParsedDocument
from src.utils.chunking import chunk_text, Chunk
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process parsed documents into structured chunks ready for embedding."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def process(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        """Convert a ParsedDocument into flat chunks for indexing."""
        all_chunks: List[Dict[str, Any]] = []

        # 1) Text chunks from each page
        for chunk in doc.chunks:
            if chunk.text.strip():
                text_chunks = chunk_text(
                    chunk.text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    metadata=chunk.metadata,
                )
                for tc in text_chunks:
                    all_chunks.append({
                        "text": tc.text,
                        "type": "text",
                        "metadata": {
                            **tc.metadata,
                            "source_path": str(doc.source_path),
                            "file_type": doc.file_type,
                        },
                    })

            # 2) Table chunks
            for table in chunk.tables:
                table_text = self._table_to_text(table)
                all_chunks.append({
                    "text": table_text,
                    "type": "table",
                    "metadata": {
                        "source_path": str(doc.source_path),
                        "file_type": doc.file_type,
                        **chunk.metadata,
                    },
                })

            # 3) Image placeholder chunks (hold base64, gen description later)
            for i, img_bytes in enumerate(chunk.images):
                all_chunks.append({
                    "text": f"[Image {i+1} from {chunk.metadata.get('source', doc.source_path)}]",
                    "type": "image",
                    "image_base64": base64.b64encode(img_bytes).decode("utf-8"),
                    "metadata": {
                        "source_path": str(doc.source_path),
                        "file_type": doc.file_type,
                        "image_index": i,
                        **chunk.metadata,
                    },
                })

        return all_chunks

    @staticmethod
    def _table_to_text(table: List[List[str]]) -> str:
        """Convert a table to markdown-like text."""
        if not table:
            return ""
        lines = []
        for i, row in enumerate(table):
            line = " | ".join(str(cell) for cell in row)
            lines.append(line)
            if i == 0:
                lines.append("-" * len(line))
        return "\n".join(lines)

    def describe_images(self, chunks: List[Dict[str, Any]], vision_client=None) -> List[Dict[str, Any]]:
        """Replace image placeholder text with actual descriptions (via GPT-4o Vision)."""
        if vision_client is None:
            try:
                from openai import OpenAI
                vision_client = OpenAI(api_key=settings.openai_api_key)
            except Exception as e:
                logger.warning(f"Cannot init vision client: {e}")
                return chunks

        for chunk in chunks:
            if chunk["type"] == "image" and "image_base64" in chunk:
                try:
                    b64 = chunk["image_base64"]
                    resp = vision_client.chat.completions.create(
                        model=settings.vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image in detail. If it contains charts, graphs, or tables, extract all visible data."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                ],
                            }
                        ],
                        max_tokens=1000,
                    )
                    description = resp.choices[0].message.content
                    chunk["text"] = f"[Image Description]: {description}"
                    chunk["type"] = "image_description"
                    logger.info(f"Described image: {chunk['metadata'].get('source_path')}")
                except Exception as e:
                    logger.warning(f"Failed to describe image: {e}")
                    chunk["text"] = "[Image - could not describe]"
        return chunks
