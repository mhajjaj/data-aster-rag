"""Answer generation with citation, formatting, and grounding."""

import json
import logging
from typing import Any, Dict, List

import openai
from config.settings import settings

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate grounded answers with citations following competition format rules."""

    SYSTEM_PROMPT = """
    あなたはデータアステル社の社内AIアシスタントです。
    提供された検索結果を基に、社内からの質問に正確に回答してください。

    ## 回答ルール
    - 質問の条件（回答形式、単位、小数桁、丸め方、主略称指定、抽出対象の表記）に厳密に従うこと
    - 資料内で定義されているタスクID、アクションID、マイルストーンID、列名、パラメータ名などの識別子は、資料上の表記どおりに回答すること
    - 設問で明示的に指定されている場合を除き、社内用語・略称ではなく通常の表現で記載すること
    - 回答に含める情報の根拠となる資料の出典（ファイル名・ページ番号等）を明記すること
    - 質問で求められている条件に該当する情報や対象が存在しない場合は、「該当するものがない」と回答すること
    - 推測や仮説は含めず、提供された資料に基づく事実のみを回答すること
    - 回答は簡潔に、最大1000トークン以内に収めること

    ## 出力形式
    回答はJSON形式で出力してください：
    {
        "answer": "具体的な回答文",
        "citations": ["出典1", "出典2"],
        "confidence": "high | medium | low",
        "missing_info": "不足している情報があれば記載、なければnull"
    }
    """

    def __init__(self, model: str = None):
        self.model = model or settings.openai_model
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        original_question: str = None,
    ) -> Dict[str, Any]:
        """Generate a grounded answer from retrieved chunks."""
        question = original_question or question

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("metadata", {})
            source = meta.get("source_path", "不明")
            page = meta.get("page_number", "")
            page_info = f" (p.{page})" if page else ""
            context_parts.append(
                f"[検索結果 {i+1}] {source}{page_info}\n{chunk['text']}\n"
            )

        context = "\n".join(context_parts)

        user_prompt = f"質問: {question}\n\n検索結果:\n{context}\n\n上記の検索結果を基に、JSON形式で回答を生成してください。"

        logger.info(f"Generating answer for: {question[:80]}...")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_answer_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            result = json.loads(content)

            # Normalize
            answer = result.get("answer", "")
            if not answer:
                answer = "該当するものがない"

            return {
                "answer": answer,
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", "medium"),
                "missing_info": result.get("missing_info"),
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": "申し訳ございません。回答の生成中にエラーが発生しました。",
                "citations": [],
                "confidence": "low",
                "missing_info": str(e),
            }
