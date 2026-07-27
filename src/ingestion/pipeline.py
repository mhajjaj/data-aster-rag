"""Document ingestion orchestrator."""

import logging
from pathlib import Path
from typing import List, Optional

from src.ingestion.base import BaseDocumentLoader, ParsedDocument
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.image_loader import ImageLoader
from src.ingestion.excel_loader import ExcelLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Auto-detect file types and dispatch to the appropriate loader."""

    def __init__(self, loaders: Optional[List[BaseDocumentLoader]] = None):
        self.loaders = loaders or self._default_loaders()

    def _default_loaders(self) -> List[BaseDocumentLoader]:
        return [
            PDFLoader(),
            ExcelLoader(),
            ImageLoader(),
        ]

    def load(self, path: Path) -> Optional[ParsedDocument]:
        for loader in self.loaders:
            if loader.can_load(path):
                return loader.load(path)
        logger.warning(f"No loader found for {path}")
        return None

    def load_directory(self, directory: Path, pattern: str = "**/*") -> List[ParsedDocument]:
        docs: List[ParsedDocument] = []
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                doc = self.load(file_path)
                if doc:
                    docs.append(doc)
        return docs
