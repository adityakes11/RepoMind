"""
Evaluate application-level correctness, completeness, and teaching style.

This module can be:

1. Run directly from the command line.
2. Imported and called from another Python file.

Metrics:

- Correctness
- Completeness
- Style
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
# │   └── eval_correctness.py
# ├── golden_tests/
# │   └── correctness_goldens.json
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
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

from src.pipeline import RAGPipeline
from src.rag.vector_store import get_collection_name


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
    / "golden_tests"
    / "correctness_goldens.json"
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
DEFAULT_TOP_K = 3
DEFAULT_LIMIT = 3


# ============================================================
# GOLDEN DATASET
# ============================================================

def load_goldens(
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load the correctness golden dataset.

    Parameters
    ----------
    path:
        Optional custom golden JSON path.

    Returns
    -------
    list
        Golden test cases.
    """

    golden_path = Path(path) if path else GOLDEN_PATH

    if not golden_path.exists():
        raise FileNotFoundError(
            f"Correctness golden dataset not found: "
            f"{golden_path}"
        )

    with golden_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# TEST CASE CREATION
# ============================================================

def build_test_cases(
    pipeline: RAGPipeline,
    collection_name: str,
    goldens: List[Dict[str, Any]],
    top_k: int,
) -> List[LLMTestCase]:
    """
    Run the complete RepoMind pipeline and create
    DeepEval test cases.
    """

    cases = []

    for golden in goldens:

        result = pipeline.run(
            golden["question"],
            collection_name,
            top_k,
        )

        print(
            f"{golden['id']}: "
            f"{golden['question']}"
        )

        print(
            f"  retrieved sources: "
            f"{result['sources']}"
        )

        cases.append(
            LLMTestCase(
                input=golden["question"],
                actual_output=result["answer"],
                expected_output=golden["ideal_answer"],
            )
        )

    return cases


# ============================================================
# METRICS
# ============================================================

def build_metrics(
    judge_model: OllamaModel,
    threshold: float,
):
    """
    Build the application-level GEval metrics.

    Metrics:

    1. Correctness
    2. Completeness
    3. Style
    """

    common = {
        "evaluation_params": [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        "threshold": threshold,
        "model": judge_model,
        "strict_mode": False,
        "async_mode": False,
    }

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    correctness = GEval(
        name="Correctness",

        evaluation_steps=[
            (
                "Compare factual claims in the actual "
                "output with the expected output."
            ),
            (
                "Penalize contradictions and false claims, "
                "but do not penalize brevity or omitted details."
            ),
            (
                "Additional information must not lower "
                "the score unless it is false or contradictory."
            ),
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "All stated claims are factually correct "
                    "and consistent with the expected answer."
                ),
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "The answer is mostly correct but "
                    "contains a minor inaccuracy."
                ),
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "The answer contains a clear factual "
                    "error or contradiction."
                ),
            ),
        ],

        **common,
    )

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    completeness = GEval(
        name="Completeness",

        evaluation_steps=[
            (
                "Identify the key points in the "
                "expected output."
            ),
            (
                "Check how many key points the actual "
                "output addresses."
            ),
            (
                "Judge coverage only; do not penalize "
                "an answer for a factual error because "
                "correctness is separate."
            ),
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "The answer covers essentially "
                    "all key points."
                ),
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "The answer covers the main points "
                    "but misses one or more details."
                ),
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "The answer misses several "
                    "important points."
                ),
            ),
        ],

        **common,
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    style = GEval(
        name="Style",

        evaluation_steps=[
            (
                "Judge only whether the answer is clear, "
                "direct, conversational, and suitable "
                "for teaching."
            ),
            (
                "Reward plain-language explanations "
                "and briefly explained technical terms."
            ),
            (
                "Do not judge factual correctness or "
                "completeness in this metric."
            ),
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Clear, conversational, intuitive, "
                    "and well explained."
                ),
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "Understandable but somewhat flat, "
                    "formal, or list-heavy."
                ),
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "Unclear, robotic, overly stiff, "
                    "or jargon-heavy."
                ),
            ),
        ],

        # Style uses its own evaluation_params,
        # so don't pass the common evaluation_params.
        **{
            key: value
            for key, value in common.items()
            if key != "evaluation_params"
        },
    )

    return [
        correctness,
        completeness,
        style,
    ]


# ============================================================
# EVALUATION
# ============================================================

def run(
    collection_name: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    limit: Optional[int] = None,
    golden_path: Optional[Path] = None,
):
    """
    Run the application-level evaluation.

    Evaluates:

    - Correctness
    - Completeness
    - Style

    Parameters
    ----------
    collection_name:
        Chroma collection used by RepoMind.

    judge_model:
        Ollama model used by DeepEval.

    threshold:
        DeepEval threshold.

    top_k:
        Number of documents retrieved.

    limit:
        Number of golden cases.
        None means all cases.

    golden_path:
        Optional custom golden dataset.

    Returns
    -------
    DeepEval result
    """

    # --------------------------------------------------------
    # Load goldens
    # --------------------------------------------------------

    goldens = load_goldens(golden_path)

    if limit is not None:
        goldens = goldens[:limit]

    # --------------------------------------------------------
    # Create local judge
    # --------------------------------------------------------

    local_judge = OllamaModel(
        model=judge_model,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={
            "think": False,
            "num_predict": 2048,
        },
    )

    # --------------------------------------------------------
    # Build pipeline test cases
    # --------------------------------------------------------

    pipeline = RAGPipeline()

    cases = build_test_cases(
        pipeline=pipeline,
        collection_name=collection_name,
        goldens=goldens,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # Build metrics
    # --------------------------------------------------------

    metrics = build_metrics(
        judge_model=local_judge,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # DeepEval
    # --------------------------------------------------------

    return evaluate(
        test_cases=cases,

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

            "judge_model": judge_model,

            "judge_provider": "ollama",

            "ollama_base_url": OLLAMA_BASE_URL,

            "top_k": top_k,

            "threshold": threshold,

            "golden_set": str(
                golden_path or GOLDEN_PATH
            ),

            "evaluation_mode": (
                "application_reference_based"
            ),

            "case_count": len(goldens),

            "metrics": (
                "correctness,"
                "completeness,"
                "style"
            ),
        },
    )


# ============================================================
# PUBLIC API
# ============================================================

def evaluate_correctness(
    collection_name: Optional[str] = None,
    github_url: Optional[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    limit: Optional[int] = DEFAULT_LIMIT,
    run_all: bool = False,
    golden_path: Optional[Path] = None,
):
    """
    Public API for application-level evaluation.

    This function can be imported and called from another
    Python file.

    Parameters
    ----------
    collection_name:
        Existing Chroma collection.

    github_url:
        GitHub repository URL.

        collection_name takes priority if both are supplied.

    judge_model:
        Ollama judge model.

    threshold:
        DeepEval threshold.

    top_k:
        Number of retrieved documents.

    limit:
        Number of golden cases to evaluate.

    run_all:
        If True, evaluate all golden cases.

    golden_path:
        Optional custom correctness golden set.

    Returns
    -------
    DeepEval result
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
    # Resolve limit
    # --------------------------------------------------------

    if run_all:
        limit = None

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print(
        f"Application collection: "
        f"{collection}"
    )

    print(
        f"Application golden set: "
        f"{golden_path or GOLDEN_PATH}"
    )

    print(
        f"Generator model: "
        f"{DEFAULT_GENERATOR_MODEL}"
    )

    print(
        f"Judge model: "
        f"{judge_model}"
    )

    print(
        f"Cases: "
        f"{'all' if limit is None else limit}"
    )

    print(
        f"Top-k: "
        f"{top_k}"
    )

    print(
        "Metrics: "
        "correctness, completeness, style"
    )

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
        help="DeepEval threshold.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved documents.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of golden cases to run.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all golden cases.",
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

    try:

        evaluate_correctness(
            collection_name=args.collection_name,
            github_url=args.github_url,
            judge_model=args.judge_model,
            threshold=args.threshold,
            top_k=args.top_k,
            limit=args.limit,
            run_all=args.all,
            golden_path=args.golden_path,
        )

    except Exception as error:

        print(
            "\nApplication correctness evaluation "
            "could not complete."
        )

        print(
            f"Reason: "
            f"{type(error).__name__}: {error}"
        )

        print(
            "Check Ollama availability, the selected "
            "judge model, the pipeline, and the "
            "correctness golden dataset."
        )

        raise


# ============================================================
# SCRIPT ENTRY
# ============================================================

if __name__ == "__main__":
    main()