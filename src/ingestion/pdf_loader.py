"""PDF loader using PyMuPDF for text, images, and table extraction."""

import io
import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.ingestion.base import BaseDocumentLoader, DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)


class PDFLoader(BaseDocumentLoader):
    """Load PDFs with multi-modal extraction: text, images, tables."""

    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def load(self, path: Path) -> ParsedDocument:
        logger.info(f"Loading PDF: {path}")
        doc = fitz.open(str(path))

        chunks: List[DocumentChunk] = []
        full_text_parts: List[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            images = self._extract_images(page)
            tables = self._extract_tables(page)

            chunk = DocumentChunk(
                text=text,
                images=images,
                tables=tables,
                metadata={
                    "page_number": page_num + 1,
                    "source": str(path),
                    "file_type": "pdf",
                },
            )
            chunks.append(chunk)
            full_text_parts.append(text)

        doc.close()

        return ParsedDocument(
            source_path=path,
            file_type="pdf",
            chunks=chunks,
            raw_text="\n".join(full_text_parts),
            metadata={"total_pages": len(chunks)},
        )

    def _extract_images(self, page: fitz.Page) -> List[bytes]:
        images: List[bytes] = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = page.parent.extract_image(xref)
            if base_image:
                images.append(base_image["image"])
        return images

    def _extract_tables(self, page: fitz.Page) -> List[List[List[str]]]:
        # Placeholder: PyMuPDF table extraction is basic.
        # We will improve this with camelot or pdfplumber later.
        tables: List[List[List[str]]] = []
        tabs = page.find_tables()
        if tabs and tabs.tables:
            for tab in tabs.tables:
                tables.append(tab.extract())
        return tables
