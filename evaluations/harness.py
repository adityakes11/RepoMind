# evaluations/harness.py
"""
Shared harness for the RepoMind evaluation suite.

The evaluation files each run their own DeepEval evaluation and return results.
This module contains the common utilities they can share:

    load_goldens(path)
        Read a golden JSON file.

    summarize_by_metric(result)
        Convert a DeepEval EvaluationResult into a per-metric summary.

    print_summary(title, summary)
        Print a readable per-metric evaluation summary.

Why per-metric instead of pooled:

A single RepoMind evaluation may run several metrics at once, for example:

    - Correctness
    - Completeness
    - Style

or:

    - Contextual Precision
    - Contextual Recall

or:

    - Faithfulness
    - Answer Relevancy

Pooling all metrics into one number could hide a regression in one metric
behind improvements in another metric.

Keeping metrics separate makes the evaluation suite suitable for comparing
current results against future baselines.

The extractor is defensive across DeepEval versions:

    - Evaluation results may be stored in `.test_results`
      or returned as a bare list.

    - Per-test metrics may be stored in `.metrics_data`
      in newer versions or `.metrics` in older versions.

Pass/fail is determined from each metric's own `success` flag rather than
recomputing it from the score. This means the harness works regardless of
whether a metric considers a higher or lower score better.
"""

import json


def load_goldens(path):
    """
    Load a RepoMind golden dataset from JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the golden JSON file.

    Returns
    -------
    list
        Parsed golden dataset.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def summarize_by_metric(result):
    """
    Convert a DeepEval EvaluationResult into a per-metric summary.

    Parameters
    ----------
    result
        DeepEval EvaluationResult or a list of test results.

    Returns
    -------
    dict
        Dictionary keyed by metric name.

    Example
    -------
    {
        "Correctness": {
            "n": 15,
            "pass_rate": 93.33,
            "avg_score": 0.91,
            "min_score": 0.70,
            "max_score": 1.00
        },
        "Completeness": {
            "n": 15,
            "pass_rate": 86.67,
            "avg_score": 0.87,
            "min_score": 0.60,
            "max_score": 1.00
        }
    }
    """

    # Newer DeepEval versions expose test results through
    # result.test_results.
    test_results = getattr(result, "test_results", None)

    # Some versions/results may directly return a list.
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    # Metric name -> accumulated statistics
    buckets = {}

    for test_result in test_results:

        # Newer DeepEval versions use metrics_data.
        metrics = getattr(test_result, "metrics_data", None)

        # Older versions may use metrics.
        if not metrics:
            metrics = getattr(test_result, "metrics", None)

        if not metrics:
            metrics = []

        for metric in metrics:

            name = getattr(metric, "name", "unknown")

            bucket = buckets.setdefault(
                name,
                {
                    "scores": [],
                    "passed": 0,
                    "total": 0,
                },
            )

            bucket["total"] += 1

            # Store score when available.
            score = getattr(metric, "score", None)

            if score is not None:
                bucket["scores"].append(score)

            # IMPORTANT:
            # Use DeepEval's own success flag instead of comparing
            # score against a threshold ourselves.
            if getattr(metric, "success", False):
                bucket["passed"] += 1

    summary = {}

    for name, bucket in buckets.items():

        scores = bucket["scores"]
        total = bucket["total"]

        summary[name] = {
            "n": total,
            "pass_rate": (
                100 * bucket["passed"] / total
                if total
                else 0.0
            ),
            "avg_score": (
                sum(scores) / len(scores)
                if scores
                else float("nan")
            ),
            "min_score": (
                min(scores)
                if scores
                else float("nan")
            ),
            "max_score": (
                max(scores)
                if scores
                else float("nan")
            ),
        }

    return summary


def print_summary(title, summary):
    """
    Print a readable per-metric evaluation summary.

    This is intended to be printed after DeepEval's own report when
    an evaluation file is executed directly.
    """

    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)

    if not summary:
        print("  No metric results found.")
        print("=" * 60)
        return

    for name, stats in summary.items():

        avg_score = stats["avg_score"]

        # NaN is the only Python value that is not equal to itself.
        if avg_score == avg_score:
            avg = f"{avg_score:.2f}"
        else:
            avg = "nan"

        print(
            f"  {name:<26} "
            f"pass_rate={stats['pass_rate']:5.0f}%  "
            f"avg={avg}  "
            f"n={stats['n']}"
        )

    print("=" * 60)