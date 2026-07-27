"""Agentic RAG Orchestrator: autonomous decision-making pipeline."""

import json
import logging
from typing import Any, Dict, List, Optional

import openai

from config.settings import settings
from src.indexing.embeddings import EmbeddingClient
from src.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Autonomous RAG pipeline.
    Decides how many search rounds are needed, whether to aggregate, or to declare missing info.
    """

    MAX_ROUNDS = 3

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline = None,
        answer_generator: AnswerGenerator = None,
    ):
        self.retrieval = retrieval_pipeline or self._default_retrieval()
        self.generator = answer_generator or AnswerGenerator()
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def _default_retrieval(self) -> RetrievalPipeline:
        vs = QdrantVectorStore(url=settings.vector_store_url)
        emb = EmbeddingClient(provider="openai")
        return RetrievalPipeline(
            vector_store=vs,
            embedding_client=emb,
        )

    def answer(self, question: str) -> Dict[str, Any]:
        """Main entry: agentic multi-round RAG answering a single question."""
        logger.info(f"=" * 60)
        logger.info(f"QUESTION: {question}")

        conversation_history: List[str] = []
        all_retrieved: List[Dict[str, Any]] = []

        for round_idx in range(self.MAX_ROUNDS):
            logger.info(f"--- Round {round_idx + 1} ---")

            # Retrieve
            results = self.retrieval.retrieve(question)
            new_ids = {r["id"] for r in results}
            old_ids = {r["id"] for r in all_retrieved}

            if new_ids.issubset(old_ids) and round_idx > 0:
                logger.info("No new information found, stopping.")
                break

            all_retrieved.extend([r for r in results if r["id"] not in old_ids])
            conversation_history.append(f"Round {round_idx + 1} query: {question}")

            # Try to answer
            candidate = self.generator.generate(question, all_retrieved)

            # Check if we need more info
            action = self._decide_action(question, candidate, all_retrieved, round_idx)

            if action == "done":
                return candidate
            elif action == "refine":
                question = self._refine_query(question, candidate, all_retrieved)
                logger.info(f"Refined query: {question}")
            elif action == "missing":
                return {
                    "answer": "該当するものがない",
                    "citations": [],
                    "confidence": "low",
                    "missing_info": None,
                }
            else:
                # continue with same question
                pass

        # Max rounds reached - return best answer from gathered info
        return self.generator.generate(question, all_retrieved)

    def _decide_action(
        self,
        question: str,
        candidate: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        round_idx: int,
    ) -> str:
        """
        Ask LLM to decide whether the current answer is good enough or needs refinement.
        Returns: done | refine | missing | continue
        """
        if round_idx == self.MAX_ROUNDS - 1:
            return "done"

        prompt = f"""
        質問: {question}

        現在の回答: {candidate.get('answer')}
        信頼度: {candidate.get('confidence')}
        不足情報: {candidate.get('missing_info', 'なし')}

        取得した検索結果数: {len(retrieved)}

        判断してください（JSON出力）:
        {{
            "action": "done" | "refine" | "missing" | "continue",
            "reason": "判断理由",
            "refined_question": "refineの場合の新しい質問文"
        }}

        - done: 回答が十分に正確で信頼性が高い
        - refine: 回答は部分正解だが、追加情報が必要
        - missing: 取得した情報から質問に回答不可能
        - continue: 判断を保留、もう1ラウンド検索を続ける
        """
        try:
            resp = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            decision = json.loads(resp.choices[0].message.content)
            action = decision.get("action", "done")
            logger.info(f"Decision: {action} - {decision.get('reason', '')}")
            return action
        except Exception as e:
            logger.warning(f"Decision step failed: {e}, defaulting to done")
            return "done"

    def _refine_query(
        self,
        original: str,
        candidate: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
    ) -> str:
        """Generate a refined search query based on current findings."""
        missing = candidate.get("missing_info", "")
        prompt = f"""
        元の質問: {original}
        不足している情報: {missing}

        検索結果の要約:
        """
        for i, r in enumerate(retrieved[:5]):
            prompt += f"\n- {r['text'][:200]}..."

        prompt += "\n\n不足情報を補うための、より具体的な検索クエリを1つ日本語で生成してください。検索クエリのみを出力してください。"

        try:
            resp = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return original

    def batch_answer(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Batch process multiple questions.
        questions: list of dicts with "question_id" and "question" keys.
        """
        results = []
        for q in questions:
            answer = self.answer(q["question"])
            results.append({
                "question_id": q.get("question_id", ""),
                "question": q["question"],
                **answer,
            })
        return results
