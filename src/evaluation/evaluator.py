"""Evaluation harness for the Data Aster RAG competition.

Evaluates answers on:
1. Submission format compliance (CSV, columns, no empty)
2. Token budget (max 1000 tokens)
3. Instruction adherence (units, IDs, rounding, "該当するものがない")
4. Judge-based scoring (optional LLM-as-a-judge)
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tiktoken

from config.settings import settings
from src.utils.submission import truncate_to_tokens, validate_submission

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
# 1. FORMAT CHECKERS
# ───────────────────────────────────────────────

class FormatChecker:
    """Check if answers conform to competition format rules."""

    # Warning: rough estimator; real token count may differ by model.
    _enc = tiktoken.encoding_for_model("gpt-4")

    @classmethod
    def check_all(cls, df: pd.DataFrame) -> Dict[str, List]:
        """Run all format checks. Returns dict of {check_name: [indices]}"""
        results = {
            "empty_answer": [],
            "over_token_budget": [],
            "trailing_jargon": [],
            "missing_denial_incomplete": [],
        }
        for idx, row in df.iterrows():
            if cls.is_empty(row["answer"]):
                results["empty_answer"].append(idx)
            if cls.is_over_token_budget(row["answer"]):
                results["over_token_budget"].append(idx)
            if cls.has_trailing_jargon(row["answer"]):
                results["trailing_jargon"].append(idx)
        return results

    @classmethod
    def is_empty(cls, answer: str) -> bool:
        return not answer or not str(answer).strip()

    @classmethod
    def is_over_token_budget(cls, answer: str, max_tokens: int = 1000) -> bool:
        # tiktoken-based count
        try:
            tokens = cls._enc.encode(str(answer))
            return len(tokens) > max_tokens
        except Exception:
            # Fallback: ~3 bytes per token for Japanese
            return len(str(answer).encode("utf-8")) > max_tokens * 3

    @classmethod
    def has_trailing_jargon(cls, answer: str) -> bool:
        """Flag answers that still contain unexplained abbreviations.
        This is a heuristic; false positives are expected.
        """
        # Look for isolated uppercase 2-4 letter sequences that may be abbreviations
        patterns = re.findall(r"\b[A-Z]{2,4}\b", str(answer))
        # Exclude common words
        common = {"AI", "IT", "ID", "PK", "FK", "API", "URL", "CSV", "PDF",
                  "USD", "JPY", "GB", "MB", "KB", "TB", "HR", "PR", "QA", "QC"}
        unusual = [p for p in patterns if p not in common]
        return len(unusual) > 2


# ───────────────────────────────────────────────
# 2. JUDGE-BASED SCORER (LLM-as-a-Judge)
# ───────────────────────────────────────────────

class JudgeScorer:
    """Score answers using an LLM judge."""

    JUDGE_PROMPT = """
    あなたは厳格な評価者です。
    以下の質問と正解（もしくは期待される回答形式）、そして参加者の回答を比較し、
    0-100点でスコアリングしてください。

    評価基準:
    - 正確性 (40点): 数値・ID・名称が資料どおりか
    - 完全性 (30点): 質問が要求している全要素が含まれているか
    - 形式遵守 (20点): 単位、小数桁、丸め、指定フォーマットに従っているか
    - 簡潔さ (10点): 不要な推測・余分な説明がないか

    質問: {question}
    期待される形式/正解: {expected}
    参加者の回答: {answer}

    出力はJSON形式:
    {{
        "score": 0-100,
        "accuracy": 0-40,
        "completeness": 0-30,
        "format_compliance": 0-20,
        "conciseness": 0-10,
        "feedback": "具体的な改善点"
    }}
    """

    def __init__(self, model: str = None):
        self.model = model or settings.judge_model
        self._import_openai()

    def _import_openai(self):
        import openai
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def score_single(
        self,
        question: str,
        answer: str,
        expected: str = "",
    ) -> Dict:
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            expected=expected or "（正解データが提供されていません）",
            answer=answer,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            return {
                "score": result.get("score", 0),
                "accuracy": result.get("accuracy", 0),
                "completeness": result.get("completeness", 0),
                "format_compliance": result.get("format_compliance", 0),
                "conciseness": result.get("conciseness", 0),
                "feedback": result.get("feedback", ""),
            }
        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}")
            return {
                "score": 0,
                "accuracy": 0,
                "completeness": 0,
                "format_compliance": 0,
                "conciseness": 0,
                "feedback": f"Error: {e}",
            }

    def score_batch(
        self,
        items: List[Dict[str, str]],
        max_workers: int = 4,
    ) -> List[Dict]:
        """
        items: list of dicts with keys: question, answer, expected (optional)
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self.score_single,
                    it["question"],
                    it["answer"],
                    it.get("expected", ""),
                ): i
                for i, it in enumerate(items)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                results.append((i, fut.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


# ───────────────────────────────────────────────
# 3. MAIN EVALUATOR
# ───────────────────────────────────────────────

class CompetitionEvaluator:
    """End-to-end evaluation runner for competition answers."""

    def __init__(self, use_judge: bool = False, judge_model: str = None):
        self.use_judge = use_judge
        self.judge = JudgeScorer(model=judge_model) if use_judge else None

    def evaluate(
        self,
        submission_csv: Path,
        ground_truth_csv: Optional[Path] = None,
        questions_csv: Optional[Path] = None,
        output_json: Optional[Path] = None,
    ) -> Dict:
        """
        Run full evaluation.

        Args:
            submission_csv: Path to submission CSV (question_id, answer)
            ground_truth_csv: Optional ground truth CSV (question_id, answer)
            questions_csv: Optional questions CSV (question_id, question) for judge context
            output_json: Optional path to write detailed evaluation results
        """
        logger.info("=" * 60)
        logger.info("Starting evaluation...")
        start = time.time()

        # Load submission
        df = pd.read_csv(submission_csv)
        total = len(df)
        logger.info(f"Loaded {total} answers from {submission_csv}")

        # 1) Format checks
        format_issues = FormatChecker.check_all(df)
        format_ok = total - len(format_issues["empty_answer"]) - len(format_issues["over_token_budget"])

        report = {
            "total_questions": total,
            "format_check": {
                "passed": format_ok,
                "failed": total - format_ok,
                "issues": format_issues,
            },
        }

        # 2) Ground truth comparison (exact match + normalized match)
        if ground_truth_csv and ground_truth_csv.exists():
            gt = pd.read_csv(ground_truth_csv)
            merged = df.merge(gt, on="question_id", how="left", suffixes=("", "_gt"))

            exact_matches = 0
            normalized_matches = 0
            for _, row in merged.iterrows():
                pred = str(row["answer"]).strip()
                gold = str(row.get("answer_gt", "")).strip()
                if pred == gold:
                    exact_matches += 1
                if self._normalize(pred) == self._normalize(gold):
                    normalized_matches += 1

            report["ground_truth"] = {
                "exact_match": exact_matches,
                "exact_match_rate": round(exact_matches / total, 4),
                "normalized_match": normalized_matches,
                "normalized_match_rate": round(normalized_matches / total, 4),
            }
        else:
            logger.info("No ground truth provided, skipping exact-match scoring.")

        # 3) Judge-based scoring
        if self.use_judge and questions_csv and questions_csv.exists():
            qdf = pd.read_csv(questions_csv)
            merged_q = df.merge(qdf, on="question_id", how="left")

            gt_map = {}
            if ground_truth_csv and ground_truth_csv.exists():
                gt_map = pd.read_csv(ground_truth_csv).set_index("question_id")["answer"].to_dict()

            items = []
            for _, row in merged_q.iterrows():
                items.append({
                    "question": str(row.get("question", "")),
                    "answer": str(row["answer"]),
                    "expected": gt_map.get(row["question_id"], ""),
                })

            logger.info(f"Running judge scoring on {len(items)} items...")
            judge_results = self.judge.score_batch(items)
            avg_score = sum(r["score"] for r in judge_results) / len(judge_results)

            report["judge"] = {
                "model": self.judge.model,
                "average_score": round(avg_score, 2),
                "scores": judge_results,
            }
        else:
            if self.use_judge:
                logger.info("Judge scoring requested but no questions.csv provided; skipping.")

        elapsed = time.time() - start
        report["elapsed_seconds"] = round(elapsed, 2)

        logger.info(f"Evaluation complete in {elapsed:.1f}s")
        logger.info(f"Format pass: {format_ok}/{total}")
        if "ground_truth" in report:
            logger.info(f"Exact match: {report['ground_truth']['exact_match']}/{total}")
        if "judge" in report:
            logger.info(f"Judge avg score: {report['judge']['average_score']}")

        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed report written to {output_json}")

        return report

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize for loose comparison: lowercase, strip, collapse spaces."""
        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)
        # Remove common punctuation differences
        text = text.replace(",", "").replace("、", "").replace(".", "").replace("．", "")
        return text


# ───────────────────────────────────────────────
# 4. CLI ENTRY
# ───────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Data Aster RAG submissions")
    parser.add_argument("--submission", type=Path, required=True, help="Submission CSV")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Ground truth CSV")
    parser.add_argument("--questions", type=Path, default=None, help="Questions CSV (for judge)")
    parser.add_argument("--output", type=Path, default=Path("output/evaluation.json"))
    parser.add_argument("--use-judge", action="store_true", help="Enable LLM-as-a-judge scoring")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model name")

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    evaluator = CompetitionEvaluator(
        use_judge=args.use_judge,
        judge_model=args.judge_model,
    )
    report = evaluator.evaluate(
        submission_csv=args.submission,
        ground_truth_csv=args.ground_truth,
        questions_csv=args.questions,
        output_json=args.output,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
