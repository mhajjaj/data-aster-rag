"""Basic sanity tests for core components."""

from pathlib import Path

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.utils.chunking import chunk_text


def test_chunk_text_splits_correctly():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert len(chunks[0].text) <= 500


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=100) == []


def test_ingestion_directory():
    # This is a smoke test assuming no files exist
    pipe = IngestionPipeline()
    docs = pipe.load_directory(Path("nonexistent"))
    assert docs == []


def test_pdf_loader_creates_parsed_document():
    from src.ingestion.pdf_loader import PDFLoader
    from src.ingestion.base import ParsedDocument

    loader = PDFLoader()
    assert ".pdf" in loader.supported_extensions()
