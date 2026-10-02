"""
Evaluate the complete RepoMind RAG pipeline with the RAG Triad.

Metrics:
    1. Contextual Relevancy
    2. Faithfulness
    3. Answer Relevancy

Golden dataset:
    golden_tests/faithfulness_goldens.json

CLI examples:

    python evaluations/eval_rag_pipeline.py

    python evaluations/eval_rag_pipeline.py --all

    python evaluations/eval_rag_pipeline.py --limit 5

    python evaluations/eval_rag_pipeline.py --top-k 3

    python evaluations/eval_rag_pipeline.py \
        --collection-name repomind_e2af1fe6fdf92747

Python example:

    from evaluations.eval_rag_pipeline import evaluate_rag_pipeline

    result = evaluate_rag_pipeline(
        collection_name="repomind_e2af1fe6fdf92747",
        top_k=5,
        limit=5,
    )
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
# PROJECT ROOT
# ============================================================

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
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from deepeval.models import OllamaModel

from deepeval.test_case import LLMTestCase

from src.pipeline import RAGPipeline

from src.rag.vector_store import get_collection_name


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)

os.environ.setdefault(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "300",
)


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = (
    PROJECT_ROOT
    / "golden_tests"
    / "faithfulness_goldens.json"
)


DEFAULT_GITHUB_URL = (
    "https://github.com/adityakes11/ExpenseTracker_RemoteMCPServer"
)


DEFAULT_GENERATOR_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2.5:7b",
)


DEFAULT_JUDGE_MODEL = os.getenv(
    "EVAL_JUDGE_MODEL",
    DEFAULT_GENERATOR_MODEL,
)


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


DEFAULT_THRESHOLD = 0.7

DEFAULT_TOP_K = 5

DEFAULT_LIMIT = 3


# ============================================================
# LOAD GOLDENS
# ============================================================

def load_goldens(
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load the RAG pipeline golden dataset.
    """

    golden_path = (
        Path(path)
        if path
        else GOLDEN_PATH
    )


    if not golden_path.exists():

        raise FileNotFoundError(
            "RAG pipeline golden dataset "
            f"not found: {golden_path}"
        )


    with golden_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)


    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            "Golden dataset must contain "
            "a JSON list."
        )


    return data


# ============================================================
# BUILD TEST CASES
# ============================================================

def build_test_cases(
    pipeline: RAGPipeline,
    collection_name: str,
    goldens: List[Dict[str, Any]],
    top_k: int,
) -> List[LLMTestCase]:
    """
    Execute the complete RepoMind pipeline and create
    DeepEval test cases.

    Flow:

        Query
          ↓
        Retriever
          ↓
        Retrieved Documents
          ↓
        Generator
          ↓
        Final Answer
    """

    if not collection_name:

        raise ValueError(
            "collection_name is required."
        )


    test_cases = []


    for index, golden in enumerate(
        goldens,
        start=1,
    ):

        question = golden.get(
            "query",
            golden.get(
                "question"
            ),
        )


        if not question:

            raise ValueError(
                f"Golden case #{index} does not "
                "contain 'query' or 'question'."
            )


        golden_id = golden.get(
            "id",
            f"case_{index}",
        )


        print()

        print(
            f"[{index}/{len(goldens)}] "
            f"{golden_id}"
        )

        print(
            f"Query: {question}"
        )


        # ----------------------------------------------------
        # Run actual RepoMind pipeline
        # ----------------------------------------------------

        result = pipeline.run(
            question,
            collection_name,
            top_k,
        )


        # ----------------------------------------------------
        # Extract retrieved context
        # ----------------------------------------------------

        documents = result.get(
            "documents",
            [],
        )


        context = []


        for document in documents:

            page_content = getattr(
                document,
                "page_content",
                None,
            )


            if page_content:

                context.append(
                    page_content
                )


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )


        print(
            f"Retrieved documents: "
            f"{len(documents)}"
        )


        print(
            f"Sources: {sources}"
        )


        # ----------------------------------------------------
        # Create DeepEval test case
        # ----------------------------------------------------

        test_cases.append(
            LLMTestCase(

                input=question,

                actual_output=result[
                    "answer"
                ],

                retrieval_context=context,
            )
        )


    return test_cases


# ============================================================
# BUILD LOCAL JUDGE
# ============================================================

def build_judge_model(
    judge_model: str,
) -> OllamaModel:
    """
    Build the local Ollama judge used by DeepEval.
    """

    return OllamaModel(

        model=judge_model,

        base_url=OLLAMA_BASE_URL,

        temperature=0,

        generation_kwargs={
            "think": False,
            "num_predict": 2048,
        },
    )


# ============================================================
# BUILD METRICS
# ============================================================

def build_metrics(
    judge_model: str,
    threshold: float,
):
    """
    Create the three RAG Triad metrics.
    """

    local_judge = build_judge_model(
        judge_model
    )


    metrics = [

        ContextualRelevancyMetric(

            threshold=threshold,

            model=local_judge,

            include_reason=False,

            async_mode=False,
        ),


        FaithfulnessMetric(

            threshold=threshold,

            model=local_judge,

            include_reason=False,

            async_mode=False,
        ),


        AnswerRelevancyMetric(

            threshold=threshold,

            model=local_judge,

            include_reason=False,

            async_mode=False,
        ),
    ]


    return metrics


# ============================================================
# RUN EVALUATION
# ============================================================

def run(
    collection_name: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    limit: Optional[int] = None,
    golden_path: Optional[Path] = None,
    pipeline: Optional[RAGPipeline] = None,
):
    """
    Run the complete RAG Triad evaluation.

    Parameters
    ----------
    collection_name:
        Chroma collection used by RepoMind.

    judge_model:
        Local Ollama model used as the evaluator.

    threshold:
        DeepEval threshold.

    top_k:
        Number of retrieved documents.

    limit:
        Number of golden cases.
        None means all cases.

    golden_path:
        Optional custom golden dataset.

    pipeline:
        Optional existing RAGPipeline.

        Passing an existing pipeline is useful when this
        evaluator is called from run_suite.py so the suite
        can reuse one pipeline instance.
    """

    # --------------------------------------------------------
    # Validate collection
    # --------------------------------------------------------

    if not collection_name:

        raise ValueError(
            "collection_name is required."
        )


    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )


    # --------------------------------------------------------
    # Load goldens
    # --------------------------------------------------------

    goldens = load_goldens(
        golden_path
    )


    if limit is not None:

        if limit <= 0:

            raise ValueError(
                "limit must be greater than 0."
            )


        goldens = goldens[
            :limit
        ]


    if not goldens:

        raise ValueError(
            "No golden test cases were found."
        )


    # --------------------------------------------------------
    # Create/reuse pipeline
    # --------------------------------------------------------

    if pipeline is None:

        pipeline = RAGPipeline()


    # --------------------------------------------------------
    # Execute RepoMind
    # --------------------------------------------------------

    test_cases = build_test_cases(

        pipeline=pipeline,

        collection_name=collection_name,

        goldens=goldens,

        top_k=top_k,
    )


    # --------------------------------------------------------
    # Build metrics
    # --------------------------------------------------------

    metrics = build_metrics(

        judge_model=judge_model,

        threshold=threshold,
    )


    # --------------------------------------------------------
    # Run DeepEval
    # --------------------------------------------------------

    result = evaluate(

        test_cases=test_cases,

        metrics=metrics,

        async_config=AsyncConfig(

            run_async=False,

            max_concurrent=1,
        ),

        hyperparameters={

            "pipeline": (
                "src.pipeline.RAGPipeline"
            ),

            "generator_model": (
                DEFAULT_GENERATOR_MODEL
            ),

            "embedding_model": os.getenv(
                "EMBEDDING_MODEL",
                "nomic-embed-text",
            ),

            "judge_model": judge_model,

            "judge_provider": "ollama",

            "ollama_base_url": (
                OLLAMA_BASE_URL
            ),

            "top_k": top_k,

            "threshold": threshold,

            "golden_set": str(
                golden_path
                or GOLDEN_PATH
            ),

            "evaluation_mode": (
                "live_retrieval_and_generation"
            ),

            "case_count": len(
                goldens
            ),

            "metrics": (
                "contextual_relevancy,"
                "faithfulness,"
                "answer_relevancy"
            ),
        },
    )


    return result


# ============================================================
# PUBLIC API
# ============================================================

def evaluate_rag_pipeline(
    collection_name: Optional[str] = None,
    github_url: Optional[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    limit: Optional[int] = DEFAULT_LIMIT,
    run_all: bool = False,
    golden_path: Optional[Path] = None,
    pipeline: Optional[RAGPipeline] = None,
):
    """
    Public API for the complete RepoMind RAG pipeline
    evaluation.

    Collection resolution priority:

        1. collection_name
        2. github_url
        3. DEFAULT_GITHUB_URL
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


    # --------------------------------------------------------
    # --all overrides --limit
    # --------------------------------------------------------

    if run_all:

        limit = None


    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "RepoMind RAG PIPELINE EVALUATION"
    )

    print("=" * 70)

    print(
        f"Collection    : {collection}"
    )

    print(
        f"Golden set    : "
        f"{golden_path or GOLDEN_PATH}"
    )

    print(
        f"Generator     : "
        f"{DEFAULT_GENERATOR_MODEL}"
    )

    print(
        f"Judge         : "
        f"{judge_model}"
    )

    print(
        f"Judge provider: Ollama"
    )

    print(
        f"Top-k         : {top_k}"
    )

    print(
        f"Threshold     : {threshold}"
    )

    print(
        f"Cases         : "
        f"{'all' if limit is None else limit}"
    )

    print(
        "Metrics       : "
        "contextual_relevancy, "
        "faithfulness, "
        "answer_relevancy"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    return run(

        collection_name=collection,

        judge_model=judge_model,

        threshold=threshold,

        top_k=top_k,

        limit=limit,

        golden_path=golden_path,

        pipeline=pipeline,
    )


# ============================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================

def evaluate_rag_triad(
    collection_name: Optional[str] = None,
    github_url: Optional[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    limit: Optional[int] = DEFAULT_LIMIT,
    run_all: bool = False,
    golden_path: Optional[Path] = None,
    pipeline: Optional[RAGPipeline] = None,
):
    """
    Backward-compatible alias.

    Existing code importing:

        evaluate_rag_triad

    will continue to work.
    """

    return evaluate_rag_pipeline(

        collection_name=collection_name,

        github_url=github_url,

        judge_model=judge_model,

        threshold=threshold,

        top_k=top_k,

        limit=limit,

        run_all=run_all,

        golden_path=golden_path,

        pipeline=pipeline,
    )


# ============================================================
# CLI ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the complete RepoMind "
            "RAG pipeline with the RAG Triad."
        )
    )


    # --------------------------------------------------------
    # Collection / GitHub target
    # --------------------------------------------------------

    target = (
        parser
        .add_mutually_exclusive_group()
    )


    target.add_argument(

        "--collection-name",

        help=(
            "Existing Chroma collection name."
        ),
    )


    target.add_argument(

        "--github-url",

        default=None,

        help=(
            "GitHub repository URL used "
            "to derive the collection name."
        ),
    )


    # --------------------------------------------------------
    # Judge
    # --------------------------------------------------------

    parser.add_argument(

        "--judge-model",

        default=DEFAULT_JUDGE_MODEL,

        help=(
            "Ollama model used as the "
            "DeepEval judge."
        ),
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    parser.add_argument(

        "--threshold",

        type=float,

        default=DEFAULT_THRESHOLD,

        help=(
            "DeepEval metric threshold."
        ),
    )


    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    parser.add_argument(

        "--top-k",

        type=int,

        default=DEFAULT_TOP_K,

        help=(
            "Number of documents retrieved."
        ),
    )


    # --------------------------------------------------------
    # Limit
    # --------------------------------------------------------

    parser.add_argument(

        "--limit",

        type=int,

        default=DEFAULT_LIMIT,

        help=(
            "Number of golden cases to evaluate."
        ),
    )


    # --------------------------------------------------------
    # All
    # --------------------------------------------------------

    parser.add_argument(

        "--all",

        action="store_true",

        help=(
            "Evaluate all golden cases."
        ),
    )


    # --------------------------------------------------------
    # Golden path
    # --------------------------------------------------------

    parser.add_argument(

        "--golden-path",

        type=Path,

        default=None,

        help=(
            "Custom golden JSON path."
        ),
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    try:

        evaluate_rag_pipeline(

            collection_name=(
                args.collection_name
            ),

            github_url=(
                args.github_url
            ),

            judge_model=(
                args.judge_model
            ),

            threshold=(
                args.threshold
            ),

            top_k=(
                args.top_k
            ),

            limit=(
                args.limit
            ),

            run_all=(
                args.all
            ),

            golden_path=(
                args.golden_path
            ),
        )


    except Exception as error:

        print()

        print(
            "RAG pipeline evaluation failed."
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()