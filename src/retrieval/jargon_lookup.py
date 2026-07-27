"""Jargon / abbreviation lookup from internal company glossary."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class JargonLookup:
    """Load and query internal company glossary to resolve abbreviations."""

    def __init__(self, glossary_path: Optional[Path] = None):
        self.terms: Dict[str, str] = {}
        self.glossary_path = glossary_path
        if glossary_path and glossary_path.exists():
            self.load()

    def load(self) -> None:
        if not self.glossary_path:
            return
        try:
            with open(self.glossary_path, "r", encoding="utf-8") as f:
                self.terms = json.load(f)
            logger.info(f"Loaded {len(self.terms)} glossary terms from {self.glossary_path}")
        except Exception as e:
            logger.error(f"Failed to load glossary: {e}")

    def resolve(self, text: str) -> str:
        """Replace known abbreviations/jargon in text with their definitions."""
        for abbr, definition in self.terms.items():
            text = text.replace(abbr, definition)
        return text

    def lookup(self, term: str) -> Optional[str]:
        return self.terms.get(term.upper(), self.terms.get(term.lower(), None))

    def build_from_docs(self, doc_dir: Path, extractor) -> None:
        """Build glossary from a company knowledge base folder if available."""
        pass  # Future: auto-extract terms from "社内用語集" folder
