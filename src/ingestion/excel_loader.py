"""Excel / CSV loader."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.ingestion.base import BaseDocumentLoader, DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)


class ExcelLoader(BaseDocumentLoader):
    """Load Excel and CSV files as tables."""

    def supported_extensions(self) -> List[str]:
        return [".xlsx", ".xls", ".csv"]

    def load(self, path: Path) -> ParsedDocument:
        logger.info(f"Loading spreadsheet: {path}")

        if path.suffix.lower() == ".csv":
            sheets = {"Sheet1": pd.read_csv(path)}
        else:
            excel = pd.ExcelFile(path)
            sheets = {name: pd.read_excel(path, sheet_name=name) for name in excel.sheet_names}

        chunks: List[DocumentChunk] = []
        all_text_parts: List[str] = []

        for sheet_name, df in sheets.items():
            table = df.astype(str).values.tolist()
            header = df.columns.astype(str).tolist()
            table_with_header = [header] + table

            # Format as markdown-like text for embedding
            text_repr = self._dataframe_to_text(df)

            chunk = DocumentChunk(
                text=text_repr,
                images=[],
                tables=[table_with_header],
                metadata={
                    "sheet_name": sheet_name,
                    "source": str(path),
                    "file_type": path.suffix.lower().lstrip("."),
                },
            )
            chunks.append(chunk)
            all_text_parts.append(text_repr)

        return ParsedDocument(
            source_path=path,
            file_type="spreadsheet",
            chunks=chunks,
            raw_text="\n\n".join(all_text_parts),
            metadata={"sheets": list(sheets.keys())},
        )

    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)
