"""Evaluate the RepoMind retriever against the ExpenseTracker golden set."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from src.rag.retriever import retrieve_documents
from src.rag.reranker import RerankingRetriever
from src.rag.vector_store import get_collection_name
from src.ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

load_dotenv()
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")

GOLDEN_PATH = Path(__file__).parents[1] / "golden test" / "retriever_goldens.json"
DEFAULT_GITHUB_URL = "https://github.com/adityakes11/ExpenseTracker_RemoteMCPServer"
DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", os.getenv("LLM_MODEL", "qwen2.5:7b"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_THRESHOLD = 0.7
DEFAULT_TOP_K = 5


def load_goldens(path=GOLDEN_PATH):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def retrieve_for_evaluation(collection_name, query, top_k, reranker=None):
    if reranker:
        return reranker.invoke(query)
    return retrieve_documents(query, collection_name, top_k)


def build_test_cases(collection_name, goldens, top_k, reranker=None):
    test_cases = []
    for golden in goldens:
        retrieved = retrieve_for_evaluation(collection_name, golden["query"], top_k, reranker)
        retrieval_context = [document.page_content for document in retrieved]
        retrieved_files = sorted({document.metadata.get("file", "unknown") for document in retrieved})
        print(f"{golden['id']}: {golden['query']}")
        print(f"  expected files: {golden['relevant_documents']}")
        print(f"  retrieved files: {retrieved_files}")
        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                expected_output=golden["ideal_answer"],
                actual_output="Retriever-only evaluation; generator output is not evaluated.",
                retrieval_context=retrieval_context,
            )
        )
    return test_cases


def run_deterministic(collection_name, top_k=DEFAULT_TOP_K, fetch_k=10, use_reranker=False):
    goldens = load_goldens()
    reranker = RerankingRetriever(collection_name, fetch_k=fetch_k, top_k=top_k) if use_reranker else None
    total_precision = 0.0
    total_recall = 0.0
    total_hits = 0

    print(f"Running deterministic retriever evaluation at k={top_k}")
    for golden in goldens:
        retrieved = retrieve_for_evaluation(collection_name, golden["query"], top_k, reranker)
        retrieved_files = {document.metadata.get("file", "unknown") for document in retrieved}
        expected_files = set(golden["relevant_documents"])
        relevant_files = retrieved_files & expected_files
        precision = len(relevant_files) / len(retrieved_files) if retrieved_files else 0.0
        recall = len(relevant_files) / len(expected_files) if expected_files else 0.0
        total_precision += precision
        total_recall += recall
        total_hits += bool(relevant_files)
        print(
            f"{golden['id']}: {'PASS' if relevant_files else 'FAIL'} "
            f"precision={precision:.2f} recall={recall:.2f} "
            f"retrieved={sorted(retrieved_files)}"
        )

    count = len(goldens)
    if count:
        print(f"\nHit Rate@{top_k}: {total_hits / count:.2%}")
        print(f"Mean Precision@{top_k}: {total_precision / count:.2%}")
        print(f"Mean Recall@{top_k}: {total_recall / count:.2%}")


def run(collection_name, judge_model=DEFAULT_JUDGE_MODEL, threshold=DEFAULT_THRESHOLD, top_k=DEFAULT_TOP_K, fetch_k=10, use_reranker=False):
    goldens = load_goldens()
    reranker = RerankingRetriever(collection_name, fetch_k=fetch_k, top_k=top_k) if use_reranker else None
    test_cases = build_test_cases(collection_name, goldens, top_k, reranker)
    local_judge = OllamaModel(
        model=judge_model,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={"think": False, "num_predict": 2048},
    )
    metrics = [
        ContextualRecallMetric(threshold=threshold, model=local_judge, include_reason=False, async_mode=False),
        ContextualPrecisionMetric(threshold=threshold, model=local_judge, include_reason=False, async_mode=False),
    ]
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
        hyperparameters={
            "retriever": "src.rag.reranker.RerankingRetriever" if use_reranker else "src.rag.retriever.retrieve_documents",
            "embedding_model": "nomic-embed-text",
            "chunk_size": int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
            "top_k": top_k,
            "fetch_k": fetch_k,
            "reranker": use_reranker,
            "judge_model": judge_model,
            "judge_provider": "ollama",
            "ollama_base_url": OLLAMA_BASE_URL,
            "threshold": threshold,
            "golden_set": str(GOLDEN_PATH),
            "collection_name": collection_name,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--collection-name")
    target.add_argument("--github-url", default=DEFAULT_GITHUB_URL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fetch-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true", help="Evaluate with the cross-encoder reranker instead of the vector baseline.")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Run DeepEval Ollama contextual metrics instead of deterministic retrieval metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collection = args.collection_name or get_collection_name(args.github_url)
    print(f"Running retriever evaluation for collection: {collection}")
    print(f"Golden set: {GOLDEN_PATH}")
    if args.semantic:
        try:
            run(collection, args.judge_model, args.threshold, args.top_k, args.fetch_k, args.rerank)
        except Exception as error:
            print("\nSemantic evaluation could not complete with the local Ollama judge.")
            print(f"Reason: {type(error).__name__}: {error}")
            print("Run without --semantic for deterministic Hit Rate, Precision@K, and Recall@K.")
    else:
        run_deterministic(collection, args.top_k, args.fetch_k, args.rerank)
