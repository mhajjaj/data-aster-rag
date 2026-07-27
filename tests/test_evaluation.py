"""Unit tests for the evaluation harness."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.evaluator import FormatChecker, JudgeScorer, CompetitionEvaluator


class TestFormatChecker:

    def test_empty_answer(self):
        assert FormatChecker.is_empty("") is True
        assert FormatChecker.is_empty("   ") is True
        assert FormatChecker.is_empty("valid answer") is False

    def test_token_budget(self):
        short = "短い回答"
        assert FormatChecker.is_over_token_budget(short, max_tokens=1000) is False

        long_text = "あ" * 4000  # ~4000 chars, ~1200+ tokens
        assert FormatChecker.is_over_token_budget(long_text, max_tokens=1000) is True

    def test_trailing_jargon(self):
        assert FormatChecker.has_trailing_jargon("This answer has ABC, XYZ, QWE, RTY") is True
        assert FormatChecker.has_trailing_jargon("This answer is normal") is False

    def test_check_all(self):
        df = pd.DataFrame({
            "question_id": ["Q1", "Q2", "Q3"],
            "answer": ["", "該当するものがない", "A" * 4000],
        })
        issues = FormatChecker.check_all(df)
        assert "Q1" not in [str(i) for i in issues["empty_answer"]]  # indices, not IDs
        assert 0 in issues["empty_answer"]
        assert 2 in issues["over_token_budget"]


class TestCompetitionEvaluator:

    def test_evaluate_format_only(self):
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "submission.csv"
            df = pd.DataFrame({
                "question_id": ["Q1", "Q2", "Q3"],
                "answer": ["Answer 1", "", "A" * 4000],
            })
            df.to_csv(sub, index=False, encoding="utf-8")

            out = Path(td) / "eval.json"
            evaluator = CompetitionEvaluator(use_judge=False)
            report = evaluator.evaluate(submission_csv=sub, output_json=out)

            assert report["total_questions"] == 3
            assert report["format_check"]["failed"] >= 2
            assert out.exists()

    def test_evaluate_with_ground_truth(self):
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "submission.csv"
            gt = Path(td) / "ground_truth.csv"

            pd.DataFrame({
                "question_id": ["Q1", "Q2"],
                "answer": ["123", "ABC"],
            }).to_csv(sub, index=False, encoding="utf-8")

            pd.DataFrame({
                "question_id": ["Q1", "Q2"],
                "answer": ["123", "ABC"],
            }).to_csv(gt, index=False, encoding="utf-8")

            evaluator = CompetitionEvaluator(use_judge=False)
            report = evaluator.evaluate(submission_csv=sub, ground_truth_csv=gt)

            assert report["ground_truth"]["exact_match"] == 2
            assert report["ground_truth"]["exact_match_rate"] == 1.0

    def test_normalize(self):
        assert CompetitionEvaluator._normalize(" Abc ") == "abc"
        assert CompetitionEvaluator._normalize("A, B、C.") == "abc"


class TestJudgeScorer:

    @pytest.mark.skipif(True, reason="Requires OpenAI API key")
    def test_score_single_mock(self):
        scorer = JudgeScorer()
        result = scorer.score_single(
            question="What is 2+2?",
            answer="4",
            expected="4",
        )
        assert "score" in result
        assert 0 <= result["score"] <= 100
