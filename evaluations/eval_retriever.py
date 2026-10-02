"""
Evaluate the RepoMind retriever against a golden set.

This module can be:
1. Run directly from the command line.
2. Imported and called from another Python script.

Examples:

CLI:
    python evaluations/eval_retriever.py \
        --collection-name repomind_e2af1fe6fdf92747

    python evaluations/eval_retriever.py \
        --github-url https://github.com/adityakes11/ExpenseTracker_RemoteMCPServer

    python evaluations/eval_retriever.py \
        --collection-name repomind_e2af1fe6fdf92747 \
        --top-k 5 \
        --rerank

Python:
    from evaluations.eval_retriever import evaluate_retriever

    result = evaluate_retriever(
        collection_name="repomind_e2af1fe6fdf92747",
        top_k=5,
    )

    print(result)

Or from outside the RepoMind project:

    from evaluations.eval_retriever import evaluate_retriever
"""

# ============================================================
# IMPORTS
# ============================================================

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# PROJECT ROOT SETUP
# ============================================================

# RepoMind/
# ├── src/
# ├── evaluations/
# │   └── eval_retriever.py
# └── ...

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from src.rag.retriever import retrieve_documents
from src.rag.reranker import RerankingRetriever
from src.rag.vector_store import get_collection_name
from src.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "300",
)


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = (
    PROJECT_ROOT
    / "golden test"
    / "retriever_goldens.json"
)

DEFAULT_GITHUB_URL = (
    "https://github.com/adityakes11/ExpenseTracker_RemoteMCPServer"
)

DEFAULT_JUDGE_MODEL = os.getenv(
    "EVAL_JUDGE_MODEL",
    os.getenv("LLM_MODEL", "qwen2.5:7b"),
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

DEFAULT_THRESHOLD = 0.7
DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 10


# ============================================================
# GOLDEN SET
# ============================================================

def load_goldens(
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load the retriever golden dataset.

    Parameters
    ----------
    path:
        Optional path to a golden JSON file.

    Returns
    -------
    list
        Loaded golden test cases.
    """

    golden_path = Path(path) if path else GOLDEN_PATH

    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {golden_path}"
        )

    with golden_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_for_evaluation(
    collection_name: str,
    query: str,
    top_k: int,
    reranker=None,
):
    """
    Retrieve documents for a single evaluation query.

    If reranker is provided, the reranker is used.
    Otherwise the normal vector retriever is used.
    """

    if reranker is not None:
        return reranker.invoke(query)

    return retrieve_documents(
        query,
        collection_name,
        top_k,
    )


# ============================================================
# TEST CASE CREATION
# ============================================================

def build_test_cases(
    collection_name: str,
    goldens: List[Dict[str, Any]],
    top_k: int,
    reranker=None,
) -> List[LLMTestCase]:
    """
    Build DeepEval test cases from the golden dataset.
    """

    test_cases = []

    for golden in goldens:

        retrieved = retrieve_for_evaluation(
            collection_name=collection_name,
            query=golden["query"],
            top_k=top_k,
            reranker=reranker,
        )

        retrieval_context = [
            document.page_content
            for document in retrieved
        ]

        retrieved_files = sorted(
            {
                document.metadata.get(
                    "file",
                    "unknown",
                )
                for document in retrieved
            }
        )

        print(
            f"{golden['id']}: {golden['query']}"
        )

        print(
            f"  expected files: "
            f"{golden['relevant_documents']}"
        )

        print(
            f"  retrieved files: "
            f"{retrieved_files}"
        )

        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                expected_output=golden["ideal_answer"],
                actual_output=(
                    "Retriever-only evaluation; "
                    "generator output is not evaluated."
                ),
                retrieval_context=retrieval_context,
            )
        )

    return test_cases


# ============================================================
# DETERMINISTIC EVALUATION
# ============================================================

def run_deterministic(
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    use_reranker: bool = False,
    golden_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run deterministic retriever evaluation.

    Calculates:

    - Hit Rate@K
    - Mean Precision@K
    - Mean Recall@K

    Returns
    -------
    dict
        Evaluation results.
    """

    goldens = load_goldens(golden_path)

    reranker = (
        RerankingRetriever(
            collection_name,
            fetch_k=fetch_k,
            top_k=top_k,
        )
        if use_reranker
        else None
    )

    total_precision = 0.0
    total_recall = 0.0
    total_hits = 0

    per_query = []

    print(
        f"Running deterministic retriever "
        f"evaluation at k={top_k}"
    )

    for golden in goldens:

        retrieved = retrieve_for_evaluation(
            collection_name=collection_name,
            query=golden["query"],
            top_k=top_k,
            reranker=reranker,
        )

        retrieved_files = {
            document.metadata.get(
                "file",
                "unknown",
            )
            for document in retrieved
        }

        expected_files = set(
            golden["relevant_documents"]
        )

        relevant_files = (
            retrieved_files & expected_files
        )

        precision = (
            len(relevant_files)
            / len(retrieved_files)
            if retrieved_files
            else 0.0
        )

        recall = (
            len(relevant_files)
            / len(expected_files)
            if expected_files
            else 0.0
        )

        hit = bool(relevant_files)

        total_precision += precision
        total_recall += recall
        total_hits += hit

        result = {
            "id": golden["id"],
            "query": golden["query"],
            "expected_files": sorted(
                expected_files
            ),
            "retrieved_files": sorted(
                retrieved_files
            ),
            "relevant_files": sorted(
                relevant_files
            ),
            "hit": hit,
            "precision": precision,
            "recall": recall,
        }

        per_query.append(result)

        print(
            f"{golden['id']}: "
            f"{'PASS' if hit else 'FAIL'} "
            f"precision={precision:.2f} "
            f"recall={recall:.2f} "
            f"retrieved={sorted(retrieved_files)}"
        )

    count = len(goldens)

    if count:
        hit_rate = total_hits / count
        mean_precision = total_precision / count
        mean_recall = total_recall / count
    else:
        hit_rate = 0.0
        mean_precision = 0.0
        mean_recall = 0.0

    print(
        f"\nHit Rate@{top_k}: "
        f"{hit_rate:.2%}"
    )

    print(
        f"Mean Precision@{top_k}: "
        f"{mean_precision:.2%}"
    )

    print(
        f"Mean Recall@{top_k}: "
        f"{mean_recall:.2%}"
    )

    return {
        "evaluation_type": "deterministic",
        "collection_name": collection_name,
        "top_k": top_k,
        "fetch_k": fetch_k,
        "reranker": use_reranker,
        "golden_set": str(
            golden_path or GOLDEN_PATH
        ),
        "total_queries": count,
        "hit_rate": hit_rate,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "per_query": per_query,
    }


# ============================================================
# SEMANTIC DEEPEVAL EVALUATION
# ============================================================

def run_semantic(
    collection_name: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    use_reranker: bool = False,
    golden_path: Optional[Path] = None,
):
    """
    Run DeepEval contextual precision and recall
    using a local Ollama judge.
    """

    goldens = load_goldens(golden_path)

    reranker = (
        RerankingRetriever(
            collection_name,
            fetch_k=fetch_k,
            top_k=top_k,
        )
        if use_reranker
        else None
    )

    test_cases = build_test_cases(
        collection_name=collection_name,
        goldens=goldens,
        top_k=top_k,
        reranker=reranker,
    )

    local_judge = OllamaModel(
        model=judge_model,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={
            "think": False,
            "num_predict": 2048,
        },
    )

    metrics = [
        ContextualRecallMetric(
            threshold=threshold,
            model=local_judge,
            include_reason=False,
            async_mode=False,
        ),
        ContextualPrecisionMetric(
            threshold=threshold,
            model=local_judge,
            include_reason=False,
            async_mode=False,
        ),
    ]

    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            max_concurrent=1,
        ),
        hyperparameters={
            "retriever": (
                "src.rag.reranker.RerankingRetriever"
                if use_reranker
                else "src.rag.retriever.retrieve_documents"
            ),
            "embedding_model": "nomic-embed-text",
            "chunk_size": int(
                os.getenv(
                    "CHUNK_SIZE",
                    DEFAULT_CHUNK_SIZE,
                )
            ),
            "chunk_overlap": int(
                os.getenv(
                    "CHUNK_OVERLAP",
                    DEFAULT_CHUNK_OVERLAP,
                )
            ),
            "top_k": top_k,
            "fetch_k": fetch_k,
            "reranker": use_reranker,
            "judge_model": judge_model,
            "judge_provider": "ollama",
            "ollama_base_url": OLLAMA_BASE_URL,
            "threshold": threshold,
            "golden_set": str(
                golden_path or GOLDEN_PATH
            ),
            "collection_name": collection_name,
        },
    )


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def evaluate_retriever(
    collection_name: Optional[str] = None,
    github_url: Optional[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    use_reranker: bool = False,
    semantic: bool = False,
    golden_path: Optional[Path] = None,
) -> Any:
    """
    Public API for running RepoMind retriever evaluation.

    This is the function you can call from another Python file.

    Parameters
    ----------
    collection_name:
        Existing Chroma collection name.

    github_url:
        GitHub URL used to derive the collection name.
        Ignored when collection_name is supplied.

    judge_model:
        Ollama model used for semantic evaluation.

    threshold:
        DeepEval metric threshold.

    top_k:
        Number of documents to retrieve.

    fetch_k:
        Number of documents fetched before reranking.

    use_reranker:
        Whether to use the cross-encoder reranker.

    semantic:
        False -> deterministic evaluation.
        True -> DeepEval contextual metrics.

    golden_path:
        Optional custom golden dataset path.

    Returns
    -------
    dict or DeepEval result
        Evaluation results.
    """

    # --------------------------------------------------------
    # Resolve collection
    # --------------------------------------------------------

    if collection_name:
        collection = collection_name

    elif github_url:
        collection = get_collection_name(
            github_url
        )

    else:
        collection = get_collection_name(
            DEFAULT_GITHUB_URL
        )

    print(
        f"Running retriever evaluation "
        f"for collection: {collection}"
    )

    print(
        f"Golden set: "
        f"{golden_path or GOLDEN_PATH}"
    )

    # --------------------------------------------------------
    # Semantic evaluation
    # --------------------------------------------------------

    if semantic:

        try:

            return run_semantic(
                collection_name=collection,
                judge_model=judge_model,
                threshold=threshold,
                top_k=top_k,
                fetch_k=fetch_k,
                use_reranker=use_reranker,
                golden_path=golden_path,
            )

        except Exception as error:

            print(
                "\nSemantic evaluation could not "
                "complete with the local Ollama judge."
            )

            print(
                f"Reason: "
                f"{type(error).__name__}: {error}"
            )

            print(
                "Run with semantic=False for "
                "deterministic Hit Rate, "
                "Precision@K, and Recall@K."
            )

            raise

    # --------------------------------------------------------
    # Deterministic evaluation
    # --------------------------------------------------------

    return run_deterministic(
        collection_name=collection,
        top_k=top_k,
        fetch_k=fetch_k,
        use_reranker=use_reranker,
        golden_path=golden_path,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    target = parser.add_mutually_exclusive_group()

    target.add_argument(
        "--collection-name",
        help="Existing Chroma collection name.",
    )

    target.add_argument(
        "--github-url",
        default=DEFAULT_GITHUB_URL,
        help="GitHub repository URL.",
    )

    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Ollama judge model.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--fetch-k",
        type=int,
        default=DEFAULT_FETCH_K,
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Evaluate using the cross-encoder "
            "reranker instead of the vector baseline."
        ),
    )

    parser.add_argument(
        "--semantic",
        action="store_true",
        help=(
            "Run DeepEval Ollama contextual "
            "metrics instead of deterministic metrics."
        ),
    )

    parser.add_argument(
        "--golden-path",
        type=Path,
        default=None,
        help="Optional custom golden JSON path.",
    )

    return parser.parse_args()


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """
    Command-line entry point.
    """

    args = parse_args()

    evaluate_retriever(
        collection_name=args.collection_name,
        github_url=args.github_url,
        judge_model=args.judge_model,
        threshold=args.threshold,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        use_reranker=args.rerank,
        semantic=args.semantic,
        golden_path=args.golden_path,
    )


if __name__ == "__main__":
    main()