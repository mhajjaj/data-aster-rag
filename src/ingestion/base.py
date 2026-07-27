"""Base document loader interface."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    """A single chunk extracted from a document."""

    text: str = ""
    images: List[bytes] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_multimodal(self) -> bool:
        return bool(self.images or self.tables)


@dataclass
class ParsedDocument:
    """Result of parsing a single source file."""

    source_path: Path
    file_type: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def all_text(self) -> str:
        return "\n".join(c.text for c in self.chunks if c.text)

    def all_images(self) -> List[bytes]:
        images = []
        for c in self.chunks:
            images.extend(c.images)
        return images

    def all_tables(self) -> List[List[List[str]]]:
        tables = []
        for c in self.chunks:
            tables.extend(c.tables)
        return tables


class BaseDocumentLoader(abc.ABC):
    """Abstract base for all document loaders."""

    @abc.abstractmethod
    def load(self, path: Path) -> ParsedDocument:
        """Load and parse a document into structured chunks."""
        ...

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions()

    @abc.abstractmethod
    def supported_extensions(self) -> List[str]:
        ...
