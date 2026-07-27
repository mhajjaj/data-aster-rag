"""Format answers for SIGNATE submission."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def json_to_submission(results: List[Dict[str, any]], output_csv: Path):
    """
    Convert agent results to the required submission format.
    Expected columns: question_id, answer
    """
    rows = []
    for r in results:
        rows.append({
            "question_id": r.get("question_id", ""),
            "answer": r.get("answer", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Submission written to {output_csv}")
    return output_csv


def truncate_to_tokens(text: str, max_tokens: int = 1000) -> str:
    """
    Rough token truncation using bytes estimate (3 bytes/token for Japanese).
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_tokens * 3:
        return text
    return encoded[: max_tokens * 3].decode("utf-8", errors="ignore")


def validate_submission(submission_csv: Path) -> bool:
    """Validate submission format."""
    df = pd.read_csv(submission_csv)
    required = {"question_id", "answer"}
    if not required.issubset(df.columns):
        logger.error(f"Missing columns: {required - set(df.columns)}")
        return False
    if df["answer"].isnull().any():
        logger.warning("Some answers are empty/null")
    logger.info(f"Validated {len(df)} rows")
    return True
