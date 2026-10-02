"""
RepoMind Evaluation Suite

Runs:
    1. Retriever evaluation
    2. Generator evaluation
    3. RAG pipeline evaluation
    4. Application evaluation
    5. Safety evaluation
    6. Operational evaluation

The suite creates one shared RAGPipeline instance for
evaluators that explicitly require pipeline injection.
"""

# ============================================================
# IMPORTS
# ============================================================

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.pipeline import RAGPipeline

from evaluations import (
    eval_retriever,
    eval_generator,
    eval_rag_pipeline,
    eval_application,
    eval_safety,
    eval_ops,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_TOP_K = 3
DEFAULT_COLLECTION_NAME = "repomind_e2af1fe6fdf92747"

BASELINES_DIR = PROJECT_ROOT / "baselines"

BASELINES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def resolve_collection_name(collection_name=None):
    """
    Resolve target Chroma collection name.

    Order of preference:
    1. Explicit collection name if provided.
    2. collection_name recorded in baselines/baseline.json.
    3. DEFAULT_COLLECTION_NAME (repomind_e2af1fe6fdf92747).
    """
    if collection_name and str(collection_name).strip():
        return str(collection_name).strip()

    baseline_path = BASELINES_DIR / "baseline.json"
    if baseline_path.exists():
        try:
            with baseline_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                name = data.get("collection_name")
                if name:
                    return name
        except Exception:
            pass

    return DEFAULT_COLLECTION_NAME


# ============================================================
# HELPERS
# ============================================================

def make_json_serializable(value):
    """
    Convert evaluator results into JSON-safe structures.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    return str(value)


def print_result(
    result,
    verbose=False,
):
    """
    Print evaluator result when verbose mode is enabled.
    """

    if not verbose:
        return

    print(
        json.dumps(
            make_json_serializable(result),
            indent=2,
            ensure_ascii=False,
        )
    )


def failed_result(exc):
    """
    Create a standard failed-evaluator result.
    """

    return {
        "status": "failed",
        "exception": type(exc).__name__,
        "error": str(exc),
    }


# ============================================================
# BASELINE & CANDIDATE SNAPSHOTS
# ============================================================

def save_snapshot(
    results,
    collection_name,
    top_k,
    force_baseline=False,
    force_candidate=False,
):
    """
    Save the complete evaluation snapshot.

    Workflow:
        - If baselines/baseline.json does NOT exist (or force_baseline is True):
            Saves to baselines/baseline.json and baselines/baseline_YYYYMMDD_HHMMSS.json.
        - If baselines/baseline.json DOES exist:
            Saves to baselines/candidate.json and baselines/candidate_YYYYMMDD_HHMMSS.json.
            This candidate can immediately be compared using compare.py.
    """

    timestamp = datetime.now()

    baseline_path = BASELINES_DIR / "baseline.json"
    baseline_exists = baseline_path.exists() and baseline_path.stat().st_size > 0

    if force_baseline:
        target_name = "baseline"
    elif force_candidate:
        target_name = "candidate"
    elif not baseline_exists:
        target_name = "baseline"
    else:
        target_name = "candidate"

    snapshot = {
        "timestamp": timestamp.isoformat(),
        "snapshot_type": target_name,
        "collection_name": collection_name,
        "top_k": top_k,
        "project_root": str(PROJECT_ROOT),
        "results": make_json_serializable(results),
    }

    # --------------------------------------------------------
    # Stable path (baseline.json or candidate.json)
    # --------------------------------------------------------

    stable_path = (
        BASELINES_DIR / f"{target_name}.json"
    )

    with stable_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            snapshot,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Timestamped archive snapshot
    # --------------------------------------------------------

    timestamp_name = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    timestamp_path = (
        BASELINES_DIR
        / f"{target_name}_{timestamp_name}.json"
    )

    with timestamp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            snapshot,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return stable_path, timestamp_path


# Backwards compatibility alias
save_baseline = save_snapshot


# ============================================================
# 1. RETRIEVER
# ============================================================

def run_retriever_eval(
    collection_name,
    top_k,
    verbose=False,
):
    """
    Run deterministic retriever evaluation.

    Public API:

        eval_retriever.evaluate_retriever(...)
    """

    print()
    print("=" * 70)
    print("[1/6] RETRIEVER EVALUATION")
    print("=" * 70)

    try:

        result = eval_retriever.evaluate_retriever(
            collection_name=collection_name,
            top_k=top_k,
            semantic=False,
        )

        print(
            "Retriever evaluation completed."
        )

        print_result(
            result,
            verbose,
        )

        return result

    except Exception as exc:

        print(
            "Retriever evaluation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return failed_result(exc)


# ============================================================
# 2. GENERATOR
# ============================================================

def run_generator_eval(
    verbose=False,
):
    """
    Run generator evaluation.

    Current public API:

        evaluate_generator(
            limit=5,
            metric_names={"faithfulness", "relevancy"}
        )

    The generator evaluator manages its own pipeline
    and golden dataset.
    """

    print()
    print("=" * 70)
    print("[2/6] GENERATOR EVALUATION")
    print("=" * 70)

    try:

        result = eval_generator.evaluate_generator(
            limit=5,
            metric_names={
                "faithfulness",
                "relevancy",
            },
        )

        print(
            "Generator evaluation completed."
        )

        print_result(
            result,
            verbose,
        )

        return result

    except Exception as exc:

        print(
            "Generator evaluation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return failed_result(exc)


# ============================================================
# 3. RAG PIPELINE
# ============================================================

def run_rag_pipeline_eval(
    collection_name,
    top_k,
    verbose=False,
):
    """
    Run full RAG pipeline evaluation.

    Current public API:

        eval_rag_pipeline.evaluate_rag_triad(...)
    """

    print()
    print("=" * 70)
    print("[3/6] RAG PIPELINE EVALUATION")
    print("=" * 70)

    try:

        result = eval_rag_pipeline.evaluate_rag_triad(
            collection_name=collection_name,
            top_k=top_k,
        )

        print(
            "RAG pipeline evaluation completed."
        )

        print_result(
            result,
            verbose,
        )

        return result

    except Exception as exc:

        print(
            "RAG pipeline evaluation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return failed_result(exc)


# ============================================================
# 4. APPLICATION
# ============================================================

def run_application_eval(
    collection_name,
    top_k,
    verbose=False,
):
    """
    Run application-level evaluation.

    Current public API:

        eval_application.evaluate_correctness(...)
    """

    print()
    print("=" * 70)
    print("[4/6] APPLICATION EVALUATION")
    print("=" * 70)

    try:

        result = eval_application.evaluate_correctness(
            collection_name=collection_name,
            top_k=top_k,
            limit=3,
            run_all=False,
        )

        print(
            "Application evaluation completed."
        )

        print_result(
            result,
            verbose,
        )

        return result

    except Exception as exc:

        print(
            "Application evaluation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return failed_result(exc)


# ============================================================
# 5. SAFETY
# ============================================================

def run_safety_eval(
    rag,
    collection_name,
    verbose=False,
):
    """
    Run all safety evaluations.

    Public API:

        eval_safety.run_safety(
            rag=rag,
            collection_name=collection_name,
            verbose=verbose
        )
    """

    print()
    print("=" * 70)
    print("[5/6] SAFETY EVALUATION")
    print("=" * 70)

    try:

        result = eval_safety.run_safety(
            rag=rag,
            collection_name=collection_name,
            verbose=verbose,
        )

        print(
            "Safety evaluation completed."
        )

        print_result(
            result,
            verbose,
        )

        return result

    except Exception as exc:

        print(
            "Safety evaluation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return failed_result(exc)


# ============================================================
# 6. OPERATIONS
# ============================================================

def run_ops_eval(
    rag,
    collection_name,
    top_k,
    verbose=False,
):
    """
    Run operational evaluation.

    Current public API:

        eval_ops.run_ops(
            pipeline=...,
            collection_name=...,
            top_k=...
        )
    """

    print()
    print("=" * 70)
    print("[6/6] OPERATIONAL EVALUATION")
    print("=" * 70)

    try:

        result = eval_ops.run_ops(
            pipeline=rag,
            collection_name=collection_name,
            top_k=top_k,
            verbose=verbose,
        )

        print(
            "Operational evaluation completed."
        )

        print_result(
            result,
            verbose,
        )

        return result

    except Exception as exc:

        print(
            "Operational evaluation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return failed_result(exc)


# ============================================================
# MAIN SUITE
# ============================================================

def run_suite(
    collection_name=None,
    top_k=DEFAULT_TOP_K,
    verbose=False,
    force_baseline=False,
    force_candidate=False,
):
    """
    Run the complete RepoMind evaluation suite.
    """

    collection_name = resolve_collection_name(collection_name)

    if not collection_name:
        raise ValueError(
            "collection_name could not be determined."
        )

    if not isinstance(collection_name, str):
        raise TypeError(
            "collection_name must be a string."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    print()
    print("=" * 70)
    print("REPO-MIND EVALUATION SUITE")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Collection   : {collection_name}"
    )

    print(
        f"Top-K        : {top_k}"
    )

    print(
        f"Started      : {datetime.now().isoformat()}"
    )

    print("=" * 70)

    # ========================================================
    # SHARED PIPELINE
    # ========================================================

    print()
    print(
        "Creating ONE RAGPipeline instance..."
    )

    rag = RAGPipeline()

    print(
        "RAGPipeline created successfully."
    )

    results = {}

    # ========================================================
    # 1. RETRIEVER
    # ========================================================

    results["retriever"] = run_retriever_eval(
        collection_name=collection_name,
        top_k=top_k,
        verbose=verbose,
    )

    # ========================================================
    # 2. GENERATOR
    # ========================================================

    results["generator"] = run_generator_eval(
        verbose=verbose,
    )

    # ========================================================
    # 3. RAG PIPELINE
    # ========================================================

    results["rag_pipeline"] = run_rag_pipeline_eval(
        collection_name=collection_name,
        top_k=top_k,
        verbose=verbose,
    )

    # ========================================================
    # 4. APPLICATION
    # ========================================================

    results["application"] = run_application_eval(
        collection_name=collection_name,
        top_k=top_k,
        verbose=verbose,
    )

    # ========================================================
    # 5. SAFETY
    # ========================================================

    results["safety"] = run_safety_eval(
        rag=rag,
        collection_name=collection_name,
        verbose=verbose,
    )

    # ========================================================
    # 6. OPERATIONS
    # ========================================================

    results["ops"] = run_ops_eval(
        rag=rag,
        collection_name=collection_name,
        top_k=top_k,
        verbose=verbose,
    )

    # ========================================================
    # SAVE SNAPSHOT (BASELINE OR CANDIDATE)
    # ========================================================

    stable_path, timestamp_path = save_snapshot(
        results=results,
        collection_name=collection_name,
        top_k=top_k,
        force_baseline=force_baseline,
        force_candidate=force_candidate,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION SUITE COMPLETE")
    print("=" * 70)

    for name, result in results.items():

        if (
            isinstance(result, dict)
            and result.get("status") == "failed"
        ):

            print(
                f"{name:<20} FAILED"
            )

        else:

            print(
                f"{name:<20} COMPLETED"
            )

    print()
    target_name = stable_path.stem
    if target_name == "baseline":
        print("Created initial baseline snapshot:")
        print(f"  Stable baseline      : {stable_path}")
        print(f"  Timestamped baseline : {timestamp_path}")
        print()
        print("Next time you run run_suite.py, it will automatically save")
        print("results as candidate.json so you can compare them.")
    else:
        print("Existing baseline found (baseline.json). Created candidate snapshot:")
        print(f"  Candidate snapshot   : {stable_path}")
        print(f"  Timestamped archive  : {timestamp_path}")
        print()
        print("To compare candidate against baseline, run:")
        print("    python -m evaluations.compare")
        print("  or:")
        print("    python -m evaluations.compare --all")

    print("=" * 70)

    return results


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete RepoMind "
            "evaluation suite."
        )
    )

    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            f"Chroma collection name to evaluate. "
            f"Default: auto-detected from baseline.json or '{DEFAULT_COLLECTION_NAME}'"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"Number of documents to retrieve. "
            f"Default: {DEFAULT_TOP_K}"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print detailed evaluator output."
        ),
    )

    parser.add_argument(
        "--as-baseline",
        "--force-baseline",
        action="store_true",
        dest="force_baseline",
        help=(
            "Force saving results to baseline.json even if one exists."
        ),
    )

    parser.add_argument(
        "--as-candidate",
        "--force-candidate",
        action="store_true",
        dest="force_candidate",
        help=(
            "Force saving results to candidate.json."
        ),
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    args = parse_args()

    run_suite(
        collection_name=args.collection_name,
        top_k=args.top_k,
        verbose=args.verbose,
        force_baseline=args.force_baseline,
        force_candidate=args.force_candidate,
    )


if __name__ == "__main__":
    main()