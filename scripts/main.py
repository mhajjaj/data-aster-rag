"""
Main CLI entry point.
Usage:
    python scripts/build_index.py --data-dir data/raw
    python scripts/answer.py --questions questions.csv --output answers.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

from config.settings import settings
from src.ingestion.pipeline import IngestionPipeline
from src.indexing.processor import DocumentProcessor
from src.indexing.embeddings import EmbeddingClient
from src.indexing.qdrant_store import QdrantVectorStore
from src.agent.orchestrator import AgenticRAG
from src.evaluation.evaluator import CompetitionEvaluator
from src.utils.submission import json_to_submission, validate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_index(data_dir: Path, reset: bool = True):
    """Ingest documents, process chunks, embed, and index into Qdrant."""
    logger.info("=" * 60)
    logger.info("Building index...")

    # 1) Ingest
    ingestion = IngestionPipeline()
    docs = ingestion.load_directory(data_dir)
    logger.info(f"Loaded {len(docs)} documents")

    # 2) Process
    processor = DocumentProcessor()
    all_chunks: List[dict] = []
    for doc in docs:
        chunks = processor.process(doc)
        all_chunks.extend(chunks)
    logger.info(f"Total chunks before image description: {len(all_chunks)}")

    # 3) Image description via GPT-4o Vision
    all_chunks = processor.describe_images(all_chunks)
    logger.info("Image descriptions completed")

    # 4) Embed
    emb = EmbeddingClient(provider="openai")
    texts = [c["text"] for c in all_chunks]
    embeddings = emb.embed(texts)
    logger.info(f"Embeddings computed: {len(embeddings)}")

    # 5) Index
    vs = QdrantVectorStore(url=settings.vector_store_url)
    if reset:
        vs.create_collection(
            settings.vector_store_collection,
            dimension=settings.vector_dimension,
        )

    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        chunk_id = f"{chunk['metadata'].get('source_path', 'unknown')}_{i}"
        vs.upsert(
            settings.vector_store_collection,
            [{
                "id": chunk_id,
                "embedding": embedding,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }],
        )

    logger.info(f"Indexed {len(all_chunks)} chunks into {settings.vector_store_collection}")
    logger.info("Done.")


def answer_questions(questions_path: Path, output_path: Path):
    """Answer questions from a JSON/CSV file and write results."""
    logger.info("=" * 60)
    logger.info("Answering questions...")

    # Load questions
    if questions_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(questions_path)
        questions = df.to_dict("records")
    else:
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

    agent = AgenticRAG()
    results = agent.batch_answer(questions)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(results)} answers to {output_path}")


def submit(answers_json: Path, output_csv: Path):
    """Convert JSON answers to submission CSV and validate."""
    logger.info("=" * 60)
    logger.info("Formatting submission...")

    with open(answers_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Map to submission format
    submission = [
        {"question_id": r.get("question_id", ""), "answer": r.get("answer", "")}
        for r in results
    ]

    json_to_submission(submission, output_csv)
    is_valid = validate_submission(output_csv)
    if not is_valid:
        logger.error("Submission validation failed!")
        sys.exit(1)
    logger.info(f"Submission written to {output_csv}")


def evaluate(
    submission_csv: Path,
    ground_truth_csv: Path = None,
    questions_csv: Path = None,
    output_json: Path = None,
    use_judge: bool = False,
):
    """Run evaluation harness on a generated submission."""
    logger.info("=" * 60)
    logger.info("Running evaluation...")

    evaluator = CompetitionEvaluator(use_judge=use_judge)
    report = evaluator.evaluate(
        submission_csv=submission_csv,
        ground_truth_csv=ground_truth_csv,
        questions_csv=questions_csv,
        output_json=output_json,
    )

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total questions: {report['total_questions']}")
    fc = report['format_check']
    print(f"Format passed: {fc['passed']}/{report['total_questions']}")
    if 'ground_truth' in report:
        gt = report['ground_truth']
        print(f"Exact match: {gt['exact_match']} ({gt['exact_match_rate']*100:.1f}%)")
    if 'judge' in report:
        print(f"Judge avg score: {report['judge']['average_score']}")
    print("=========================\n")

    if output_json:
        logger.info(f"Detailed report: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Aster RAG System")
    subparsers = parser.add_subparsers(dest="command")

    # build-index
    build_parser = subparsers.add_parser("build-index", help="Build vector index from documents")
    build_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    build_parser.add_argument("--reset", action="store_true", default=True)

    # answer
    answer_parser = subparsers.add_parser("answer", help="Answer questions")
    answer_parser.add_argument("--questions", type=Path, required=True)
    answer_parser.add_argument("--output", type=Path, default=Path("output/answers.json"))

    # submit
    submit_parser = subparsers.add_parser("submit", help="Convert JSON answers to submission CSV")
    submit_parser.add_argument("--answers-json", type=Path, required=True)
    submit_parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a submission")
    eval_parser.add_argument("--submission", type=Path, required=True)
    eval_parser.add_argument("--ground-truth", type=Path, default=None)
    eval_parser.add_argument("--questions", type=Path, default=None)
    eval_parser.add_argument("--output", type=Path, default=Path("output/evaluation.json"))
    eval_parser.add_argument("--use-judge", action="store_true")

    args = parser.parse_args()

    if args.command == "build-index":
        build_index(args.data_dir, reset=args.reset)
    elif args.command == "answer":
        output_dir = args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        answer_questions(args.questions, args.output)
    elif args.command == "submit":
        submit(args.answers_json, args.output)
    elif args.command == "evaluate":
        evaluate(
            submission_csv=args.submission,
            ground_truth_csv=args.ground_truth,
            questions_csv=args.questions,
            output_json=args.output,
            use_judge=args.use_judge,
        )
    else:
        parser.print_help()
        sys.exit(1)
