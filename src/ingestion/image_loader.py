"""Image loader with OCR and optional vision-model captioning."""

import logging
from pathlib import Path
from typing import List

from src.ingestion.base import BaseDocumentLoader, DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)


class ImageLoader(BaseDocumentLoader):
    """Load image files and extract text via OCR + optional description."""

    SUPPORTED_EXTS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"]

    def supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTS

    def load(self, path: Path) -> ParsedDocument:
        logger.info(f"Loading image: {path}")

        with path.open("rb") as f:
            image_bytes = f.read()

        chunk = DocumentChunk(
            text="",
            images=[image_bytes],
            tables=[],
            metadata={
                "source": str(path),
                "file_type": path.suffix.lower().lstrip("."),
                "page_number": 1,
            },
        )

        return ParsedDocument(
            source_path=path,
            file_type="image",
            chunks=[chunk],
            raw_text="",
            metadata={"total_pages": 1},
        )
