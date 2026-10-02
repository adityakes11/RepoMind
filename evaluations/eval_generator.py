"""
Evaluate RepoMind generator faithfulness and answer relevance in isolation.

This module can be:

1. Run directly from the command line.
2. Imported and called from another Python script.

CLI examples:

    python evaluations/eval_faithfulness.py

    python evaluations/eval_faithfulness.py --all

    python evaluations/eval_faithfulness.py --limit 5

    python evaluations/eval_faithfulness.py --metrics both

Python example:

    from evaluations.eval_faithfulness import evaluate_generator

    result = evaluate_generator(
        limit=5,
        metric_names={"faithfulness", "relevancy"},
    )

    print(result)
"""

# ============================================================
# IMPORTS
# ============================================================

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================
# PROJECT ROOT SETUP
# ============================================================

# RepoMind/
# ├── src/
# ├── evaluations/
# │   └── eval_faithfulness.py
# ├── golden_tests/
# │   └── faithfulness_goldens.json
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
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from langchain_core.documents import Document

from src.llm.generator import AnswerGenerator


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
    / "faithfulness_goldens.json"
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

DEFAULT_LIMIT = 3


# ============================================================
# GOLDEN DATASET
# ============================================================

def load_goldens(
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load the faithfulness golden dataset.

    Parameters
    ----------
    path:
        Optional custom path to the golden JSON file.

    Returns
    -------
    list
        Golden test cases.
    """

    golden_path = Path(path) if path else GOLDEN_PATH

    if not golden_path.exists():
        raise FileNotFoundError(
            f"Faithfulness golden dataset not found: "
            f"{golden_path}"
        )

    with golden_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# GOLDEN DOCUMENTS
# ============================================================

def golden_documents(
    golden: Dict[str, Any],
) -> List[Document]:
    """
    Convert golden-context strings into LangChain Documents.

    The golden set provides the retrieval context directly,
    allowing the generator to be evaluated independently
    from the retriever.
    """

    source_files = golden.get(
        "source_files",
        [],
    )

    documents = []

    for index, context in enumerate(
        golden["ideal_context"]
    ):

        file_name = (
            source_files[index]
            if index < len(source_files)
            else "golden_context"
        )

        documents.append(
            Document(
                page_content=context,
                metadata={
                    "file": file_name,
                },
            )
        )

    return documents


# ============================================================
# TEST CASE CREATION
# ============================================================

def build_test_cases(
    generator: AnswerGenerator,
    goldens: List[Dict[str, Any]],
) -> List[LLMTestCase]:
    """
    Generate answers from the golden contexts and create
    DeepEval test cases.
    """

    test_cases = []

    for golden in goldens:

        documents = golden_documents(golden)

        result = generator.generate(
            golden["query"],
            documents,
        )

        answer = result["answer"]

        print(
            f"{golden['id']}: "
            f"{golden['query']}"
        )

        print(
            f"  generated sources: "
            f"{result['sources']}"
        )

        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output=answer,
                retrieval_context=[
                    document.page_content
                    for document in documents
                ],
            )
        )

    return test_cases


# ============================================================
# SEMANTIC EVALUATION
# ============================================================

def run(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    limit: Optional[int] = DEFAULT_LIMIT,
    metric_names: Optional[Set[str]] = None,
    golden_path: Optional[Path] = None,
):
    """
    Run generator evaluation.

    Parameters
    ----------
    judge_model:
        Ollama model used by DeepEval to judge the answers.

    threshold:
        DeepEval pass threshold.

    limit:
        Number of golden cases to evaluate.
        None means all cases.

    metric_names:
        Set containing:
            "faithfulness"
            "relevancy"

    golden_path:
        Optional custom golden dataset.

    Returns
    -------
    DeepEval result
    """

    goldens = load_goldens(golden_path)

    # --------------------------------------------------------
    # Apply case limit
    # --------------------------------------------------------

    if limit is not None:
        goldens = goldens[:limit]

    # --------------------------------------------------------
    # Validate metrics
    # --------------------------------------------------------

    metric_names = (
        metric_names
        if metric_names
        else {
            "faithfulness",
            "relevancy",
        }
    )

    valid_metrics = {
        "faithfulness",
        "relevancy",
    }

    invalid_metrics = (
        metric_names - valid_metrics
    )

    if invalid_metrics:
        raise ValueError(
            f"Invalid metric names: "
            f"{sorted(invalid_metrics)}. "
            f"Allowed values: "
            f"{sorted(valid_metrics)}"
        )

    if not metric_names:
        raise ValueError(
            "At least one evaluation metric "
            "must be selected."
        )

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    generator = AnswerGenerator()

    # --------------------------------------------------------
    # Build DeepEval test cases
    # --------------------------------------------------------

    test_cases = build_test_cases(
        generator,
        goldens,
    )

    # --------------------------------------------------------
    # Local Ollama judge
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
    # Metrics
    # --------------------------------------------------------

    metrics = []

    if "faithfulness" in metric_names:

        metrics.append(
            FaithfulnessMetric(
                threshold=threshold,
                model=local_judge,
                include_reason=False,
                async_mode=False,
            )
        )

    if "relevancy" in metric_names:

        metrics.append(
            AnswerRelevancyMetric(
                threshold=threshold,
                model=local_judge,
                include_reason=False,
                async_mode=False,
            )
        )

    # --------------------------------------------------------
    # DeepEval
    # --------------------------------------------------------

    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            max_concurrent=1,
        ),
        hyperparameters={
            "generator": (
                "src.llm.generator.AnswerGenerator"
            ),
            "generator_model": os.getenv(
                "LLM_MODEL",
                "qwen2.5:7b",
            ),
            "judge_model": judge_model,
            "judge_provider": "ollama",
            "ollama_base_url": OLLAMA_BASE_URL,
            "threshold": threshold,
            "golden_set": str(
                golden_path or GOLDEN_PATH
            ),
            "evaluation_mode": (
                "golden_context_isolation"
            ),
            "case_count": len(goldens),
            "metrics": ",".join(
                sorted(metric_names)
            ),
        },
    )


# ============================================================
# MAIN PUBLIC API
# ============================================================

def evaluate_generator(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    limit: Optional[int] = DEFAULT_LIMIT,
    metric_names: Optional[Set[str]] = None,
    run_all: bool = False,
    golden_path: Optional[Path] = None,
):
    """
    Public API for evaluating the RepoMind generator.

    This is the function to import from another Python file.

    Parameters
    ----------
    judge_model:
        Ollama model used as the DeepEval judge.

    threshold:
        Evaluation threshold.

    limit:
        Number of golden cases to evaluate.

    metric_names:
        Example:

            {"faithfulness"}

        or:

            {"relevancy"}

        or:

            {"faithfulness", "relevancy"}

    run_all:
        If True, evaluate the complete golden dataset.
        This overrides limit.

    golden_path:
        Optional custom golden dataset path.

    Returns
    -------
    DeepEval result
    """

    if run_all:
        limit = None

    if metric_names is None:
        metric_names = {
            "faithfulness",
        }

    print(
        f"Golden set: "
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
        f"Metrics: "
        f"{', '.join(sorted(metric_names))}"
    )

    return run(
        judge_model=judge_model,
        threshold=threshold,
        limit=limit,
        metric_names=metric_names,
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

    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "Ollama model used by DeepEval "
            "as the judge."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="DeepEval metric threshold.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Number of golden cases to run. "
            "Use --all for the complete suite."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all golden cases.",
    )

    parser.add_argument(
        "--metrics",
        choices=[
            "faithfulness",
            "relevancy",
            "both",
        ],
        default="faithfulness",
        help=(
            "Metric to evaluate. "
            "Default: faithfulness."
        ),
    )

    parser.add_argument(
        "--golden-path",
        type=Path,
        default=None,
        help=(
            "Optional custom faithfulness "
            "golden JSON file."
        ),
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

    metric_names = (
        {
            "faithfulness",
            "relevancy",
        }
        if args.metrics == "both"
        else {
            args.metrics,
        }
    )

    try:

        evaluate_generator(
            judge_model=args.judge_model,
            threshold=args.threshold,
            limit=args.limit,
            metric_names=metric_names,
            run_all=args.all,
            golden_path=args.golden_path,
        )

    except Exception as error:

        print(
            "\nGenerator evaluation could not "
            "complete with the local Ollama judge."
        )

        print(
            f"Reason: "
            f"{type(error).__name__}: {error}"
        )

        print(
            "Check Ollama availability, "
            "the selected judge model, and "
            "the faithfulness golden dataset."
        )

        raise


if __name__ == "__main__":
    main()