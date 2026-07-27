"""Query understanding: decompose questions, detect intent, expand terms."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import openai

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    original: str
    rewritten: str
    intent: str  # retrieval, calculation, comparison, aggregation
    key_entities: List[str]
    keywords: List[str]
    sub_questions: List[str]
    requires_images: bool = False
    requires_tables: bool = False
    requires_multiple_cases: bool = False
    requires_jargon_lookup: bool = False


class QueryAnalyzer:
    """Analyze a user question to optimize retrieval."""

    SYSTEM_PROMPT = """
    You are a query analyzer for a Japanese data-consulting RAG system.
    Decompose the user question and output a JSON object with these keys:
    - rewritten: A clearer, more detailed version of the question for retrieval.
    - intent: One of [retrieval, calculation, comparison, aggregation, list]
    - key_entities: List of named entities (company names, project IDs, etc.)
    - keywords: Important keywords to search for
    - sub_questions: Break the question into individual retrievable parts if multi-hop
    - requires_images: true if the answer likely needs image/chart understanding
    - requires_tables: true if the answer likely needs table/spreadsheet data
    - requires_multiple_cases: true if data from multiple projects is needed
    - requires_jargon_lookup: true if internal jargon/abbreviations are used

    Respond ONLY with valid JSON. Do not include markdown or explanations.
    """

    def __init__(self, model: str = None):
        self.model = model or settings.openai_model
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def analyze(self, query: str) -> QueryAnalysis:
        logger.info(f"Analyzing query: {query}")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=1000,
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            return QueryAnalysis(
                original=query,
                rewritten=parsed.get("rewritten", query),
                intent=parsed.get("intent", "retrieval"),
                key_entities=parsed.get("key_entities", []),
                keywords=parsed.get("keywords", []),
                sub_questions=parsed.get("sub_questions", [query]),
                requires_images=parsed.get("requires_images", False),
                requires_tables=parsed.get("requires_tables", False),
                requires_multiple_cases=parsed.get("requires_multiple_cases", False),
                requires_jargon_lookup=parsed.get("requires_jargon_lookup", False),
            )
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return QueryAnalysis(
                original=query,
                rewritten=query,
                intent="retrieval",
                key_entities=[],
                keywords=[],
                sub_questions=[query],
            )
