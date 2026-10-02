# evaluations/compare.py

"""
compare.py

Regression decision function for the RepoMind evaluation suite.

Reads two evaluation snapshots:

    baseline.json
    candidate.json

The comparator supports both snapshot formats:

1. Preferred format:

    {
        "metrics": {
            "retriever.hit_rate": 1.0,
            "retriever.precision": 0.95,
            ...
        }
    }

2. RepoMind run_suite.py format:

    {
        "results": {
            "retriever": {...},
            "generator": {...},
            "rag_pipeline": {...},
            "application": {...},
            "safety": {...},
            "ops": {...}
        }
    }

Metrics are classified according to metric_registry.py.

Possible overall verdicts:

    PASS
        No gate regressed and no guardrail regressed beyond tolerance.

    REVIEW
        A guardrail regressed beyond tolerance.

    FAIL
        A gate regressed beyond tolerance.

Exit codes:

    PASS   = 0
    FAIL   = 1
    REVIEW = 2

Usage:

    python -m evaluations.compare

    python -m evaluations.compare \
        --baseline baselines/baseline.json \
        --candidate baselines/candidate.json

    python -m evaluations.compare --all
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. REPO MIND REGISTRY
# ============================================================

from evaluations.metric_registry import rule_for


# ============================================================
# 3. DEFAULT PATHS
# ============================================================

BASELINE_PATH = (
    PROJECT_ROOT
    / "baselines"
    / "baseline.json"
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "baselines"
    / "candidate.json"
)


# ============================================================
# 4. EXIT CODES
# ============================================================

EXIT_CODE = {
    "PASS": 0,
    "FAIL": 1,
    "REVIEW": 2,
}


# ============================================================
# 5. LOAD SNAPSHOT
# ============================================================

def load(path):
    """
    Load a RepoMind evaluation snapshot.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Snapshot must contain a JSON object: {path}"
        )

    return data


# ============================================================
# 6. NUMBER CHECK
# ============================================================

def _is_number(value):
    """
    Return True only for finite numeric values.

    Booleans are excluded because bool is a subclass of int.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


# ============================================================
# 7. FLATTEN DICTIONARY
# ============================================================

def _flatten_dict(
    value,
    prefix="",
):
    """
    Recursively flatten a nested dictionary.

    Example:

        {
            "latency": {
                "e2e": {
                    "p95_ms": 5000
                }
            }
        }

    becomes:

        {
            "latency.e2e.p95_ms": 5000
        }

    Lists are intentionally not flattened because evaluator
    test-case details are not regression metrics.
    """

    output = {}

    if not isinstance(value, dict):
        return output

    for key, item in value.items():

        key = str(key)

        full_key = (
            f"{prefix}.{key}"
            if prefix
            else key
        )

        if isinstance(item, dict):

            output.update(
                _flatten_dict(
                    item,
                    full_key,
                )
            )

        elif _is_number(item) or isinstance(item, bool):

            output[full_key] = item

    return output


# ============================================================
# 8. NORMALIZE METRIC NAME
# ============================================================

def _normalize_metric_name(metric_id):
    """
    Normalize common naming differences.

    This keeps comparison stable when evaluators use slightly
    different names such as:

        avg
        average
        avg_score

    The original metric ID is preserved whenever possible.
    """

    replacements = {
        ".average_score": ".avg_score",
        ".average": ".avg",
        ".mean": ".avg",
    }

    normalized = metric_id

    for old, new in replacements.items():

        normalized = normalized.replace(
            old,
            new,
        )

    return normalized


# ============================================================
# 9. EXTRACT METRICS
# ============================================================

def extract_metrics(snapshot):
    """
    Extract comparable metrics from a RepoMind snapshot.

    Preferred format:

        snapshot["metrics"]

    Fallback format:

        snapshot["results"]

    The fallback allows compare.py to work with the current
    run_suite.py snapshot structure.
    """

    # --------------------------------------------------------
    # Preferred format
    # --------------------------------------------------------

    metrics = snapshot.get("metrics")

    if isinstance(metrics, dict):

        return {
            str(key): value
            for key, value in metrics.items()
            if (
                _is_number(value)
                or isinstance(value, bool)
            )
        }

    # --------------------------------------------------------
    # Fallback to evaluator results
    # --------------------------------------------------------

    results = snapshot.get("results")

    if not isinstance(results, dict):

        raise ValueError(
            "Snapshot does not contain either "
            "'metrics' or 'results'."
        )

    extracted = {}

    # ========================================================
    # Evaluator-specific metric extraction
    # ========================================================

    for evaluator_name, evaluator_result in results.items():

        if not isinstance(evaluator_result, dict):
            continue

        # ----------------------------------------------------
        # 1. Direct metrics dictionary
        # ----------------------------------------------------

        direct_metrics = evaluator_result.get(
            "metrics"
        )

        if isinstance(direct_metrics, dict):

            for metric_name, metric_value in direct_metrics.items():

                if (
                    _is_number(metric_value)
                    or isinstance(metric_value, bool)
                ):

                    metric_id = (
                        f"{evaluator_name}.{metric_name}"
                    )

                    extracted[
                        _normalize_metric_name(metric_id)
                    ] = metric_value

        # ----------------------------------------------------
        # 2. Flatten evaluator result
        # ----------------------------------------------------

        flattened = _flatten_dict(
            evaluator_result
        )

        for metric_name, metric_value in flattened.items():

            # Ignore detailed test-case arrays and metadata.
            if metric_name.startswith(
                (
                    "cases.",
                    "test_cases.",
                    "details.",
                    "samples.",
                    "results.",
                )
            ):
                continue

            metric_id = (
                f"{evaluator_name}.{metric_name}"
            )

            metric_id = _normalize_metric_name(
                metric_id
            )

            # Avoid replacing an explicit metrics dictionary
            # entry with an inferred duplicate.
            if metric_id not in extracted:

                extracted[metric_id] = metric_value

    return extracted


# ============================================================
# 10. NUMBER CHECK FOR METRIC VALUES
# ============================================================

def _metric_value(
    metrics,
    metric_id,
):
    """
    Safely retrieve a metric value.
    """

    if metric_id not in metrics:
        return None

    return metrics[metric_id]


# ============================================================
# 11. CLASSIFY ONE METRIC
# ============================================================

def classify(
    metric_id,
    baseline_value,
    candidate_value,
    rule,
):
    """
    Compare one baseline metric against one candidate metric.
    """

    row = {
        "id": metric_id,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "kind": rule["kind"],
        "direction": rule["direction"],
        "delta": None,
        "status": None,
    }

    # ========================================================
    # METRIC EXISTS ONLY IN CANDIDATE
    # ========================================================

    if baseline_value is None:

        row["status"] = "new"

        return row

    # ========================================================
    # METRIC EXISTS ONLY IN BASELINE
    # ========================================================

    if candidate_value is None:

        row["status"] = "dropped"

        return row

    # ========================================================
    # INFO METRICS
    # ========================================================

    if rule["kind"] == "info":

        if (
            _is_number(baseline_value)
            and _is_number(candidate_value)
        ):

            row["delta"] = (
                candidate_value
                - baseline_value
            )

        row["status"] = "info"

        return row

    # ========================================================
    # BOOLEAN METRICS
    # ========================================================

    if (
        rule.get("bool")
        or isinstance(baseline_value, bool)
        or isinstance(candidate_value, bool)
    ):

        baseline_bool = bool(
            baseline_value
        )

        candidate_bool = bool(
            candidate_value
        )

        if baseline_bool == candidate_bool:

            row["status"] = "flat"

        elif (
            not baseline_bool
            and candidate_bool
        ):

            row["status"] = "improved"

        else:

            if rule["kind"] == "gate":

                row["status"] = "blocked"

            else:

                row["status"] = "regressed"

        return row

    # ========================================================
    # NON-NUMERIC VALUES
    # ========================================================

    if not (
        _is_number(baseline_value)
        and _is_number(candidate_value)
    ):

        row["status"] = "info"

        return row

    # ========================================================
    # NUMERIC METRICS
    # ========================================================

    delta = (
        candidate_value
        - baseline_value
    )

    row["delta"] = delta

    # ========================================================
    # DIRECTION
    # ========================================================

    higher_is_better = (
        rule["direction"] == "higher"
    )

    if higher_is_better:

        improved = delta > 0

    else:

        improved = delta < 0

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    if improved:

        row["status"] = "improved"

        return row

    # ========================================================
    # TOLERANCE
    # ========================================================

    worsening = abs(delta)

    absolute_tolerance = rule.get(
        "tol",
        0.0,
    )

    relative_tolerance = (
        rule.get("rel_tol", 0.0)
        * abs(baseline_value)
    )

    tolerance = max(
        absolute_tolerance,
        relative_tolerance,
    )

    # ========================================================
    # WITHIN TOLERANCE
    # ========================================================

    if worsening <= tolerance:

        row["status"] = "flat"

        return row

    # ========================================================
    # REAL REGRESSION
    # ========================================================

    if rule["kind"] == "gate":

        row["status"] = "blocked"

    else:

        row["status"] = "regressed"

    return row


# ============================================================
# 12. COMPARE TWO SNAPSHOTS
# ============================================================

def compare(
    baseline,
    candidate,
):
    """
    Compare two RepoMind snapshots.

    Returns:

        verdict, rows
    """

    baseline_metrics = extract_metrics(
        baseline
    )

    candidate_metrics = extract_metrics(
        candidate
    )

    # --------------------------------------------------------
    # Union of metric IDs
    # --------------------------------------------------------

    metric_ids = sorted(
        set(baseline_metrics)
        | set(candidate_metrics)
    )

    rows = []

    for metric_id in metric_ids:

        baseline_value = _metric_value(
            baseline_metrics,
            metric_id,
        )

        candidate_value = _metric_value(
            candidate_metrics,
            metric_id,
        )

        rule = rule_for(
            metric_id
        )

        row = classify(
            metric_id=metric_id,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            rule=rule,
        )

        rows.append(row)

    # ========================================================
    # OVERALL VERDICT
    # ========================================================

    if any(
        row["status"] == "blocked"
        for row in rows
    ):

        verdict = "FAIL"

    elif any(
        row["status"] == "regressed"
        for row in rows
    ):

        verdict = "REVIEW"

    else:

        verdict = "PASS"

    return verdict, rows


# ============================================================
# 13. VALUE FORMATTER
# ============================================================

def _fmt(value):
    """
    Format a metric value for terminal output.
    """

    if isinstance(value, bool):

        return str(value)

    if isinstance(value, float):

        if math.isnan(value):

            return "nan"

        if math.isinf(value):

            return "inf"

        return f"{value:.4g}"

    if value is None:

        return "-"

    return str(value)


# ============================================================
# 14. PRINT COMPARISON REPORT
# ============================================================

def print_report(
    verdict,
    rows,
    show_all=False,
):
    """
    Print baseline/candidate comparison.
    """

    order = {
        "blocked": 0,
        "regressed": 1,
        "dropped": 2,
        "new": 3,
        "improved": 4,
        "flat": 5,
        "info": 6,
    }

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    shown = [
        row
        for row in rows
        if (
            show_all
            or row["kind"] != "info"
        )
    ]

    shown.sort(
        key=lambda row: (
            order.get(
                row["status"],
                9,
            ),
            row["id"],
        )
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 115)
    print(
        "RepoMind REGRESSION COMPARISON"
    )
    print("=" * 115)

    print(
        f"{'metric':<55}"
        f"{'baseline':>12}"
        f"{'candidate':>12}"
        f"{'delta':>12}"
        f"  status"
    )

    print("-" * 115)

    # ========================================================
    # ROWS
    # ========================================================

    for row in shown:

        delta = row["delta"]

        if _is_number(delta):

            delta_string = (
                f"{delta:+.4g}"
            )

        else:

            delta_string = ""

        if row["status"] in {
            "blocked",
            "regressed",
        }:

            marker = "  <<<"

        else:

            marker = ""

        print(
            f"{row['id']:<55}"
            f"{_fmt(row['baseline']):>12}"
            f"{_fmt(row['candidate']):>12}"
            f"{delta_string:>12}"
            f"  {row['status']}"
            f"{marker}"
        )

    # ========================================================
    # COUNTS
    # ========================================================

    counts = Counter(
        row["status"]
        for row in rows
    )

    statuses = [
        "blocked",
        "regressed",
        "improved",
        "flat",
        "new",
        "dropped",
        "info",
    ]

    count_line = "  ".join(
        f"{status}={counts[status]}"
        for status in statuses
        if counts[status]
    )

    print("-" * 115)

    if count_line:

        print(count_line)

    info_count = counts["info"]

    if (
        not show_all
        and info_count
    ):

        print(
            f"({info_count} info metrics hidden; "
            f"use --all to show them)"
        )

    # ========================================================
    # VERDICT
    # ========================================================

    print("=" * 115)

    banners = {
        "PASS": (
            "PASS   -- no gate blocked and no "
            "guardrail regressed beyond tolerance."
        ),

        "REVIEW": (
            "REVIEW -- a guardrail regressed "
            "beyond tolerance. Human review required."
        ),

        "FAIL": (
            "FAIL   -- a gate regressed "
            "beyond tolerance."
        ),
    }

    print(
        f"VERDICT: {banners[verdict]}"
    )

    print("=" * 115)


# ============================================================
# 15. METADATA REPORT
# ============================================================

def print_metadata(
    baseline,
    candidate,
    baseline_path,
    candidate_path,
):
    """
    Print provenance information for both snapshots.
    """

    baseline_meta = baseline.get(
        "metadata",
        {},
    )

    candidate_meta = candidate.get(
        "metadata",
        {},
    )

    # --------------------------------------------------------
    # Handle current run_suite.py format
    # --------------------------------------------------------

    baseline_collection = (
        baseline_meta.get(
            "collection_name"
        )
        or baseline.get(
            "collection_name",
            "?",
        )
    )

    candidate_collection = (
        candidate_meta.get(
            "collection_name"
        )
        or candidate.get(
            "collection_name",
            "?",
        )
    )

    baseline_top_k = (
        baseline_meta.get(
            "top_k"
        )
        or baseline.get(
            "top_k",
            "?",
        )
    )

    candidate_top_k = (
        candidate_meta.get(
            "top_k"
        )
        or candidate.get(
            "top_k",
            "?",
        )
    )

    baseline_label = (
        baseline_meta.get(
            "label"
        )
        or baseline_path
    )

    candidate_label = (
        candidate_meta.get(
            "label"
        )
        or candidate_path
    )

    baseline_git = (
        baseline_meta.get(
            "git_sha",
            "?",
        )
    )

    candidate_git = (
        candidate_meta.get(
            "git_sha",
            "?",
        )
    )

    baseline_prompt = (
        baseline_meta.get(
            "prompt_hash",
            "?",
        )
    )

    candidate_prompt = (
        candidate_meta.get(
            "prompt_hash",
            "?",
        )
    )

    baseline_model = (
        baseline_meta.get(
            "generator_model",
            "?",
        )
    )

    candidate_model = (
        candidate_meta.get(
            "generator_model",
            "?",
        )
    )

    print(
        f"baseline  : {baseline_label}"
    )

    print(
        f"candidate : {candidate_label}"
    )

    print(
        f"baseline git       : {baseline_git}"
    )

    print(
        f"candidate git      : {candidate_git}"
    )

    print(
        f"baseline prompt    : {baseline_prompt}"
    )

    print(
        f"candidate prompt   : {candidate_prompt}"
    )

    print(
        f"collection baseline : "
        f"{baseline_collection}"
    )

    print(
        f"collection candidate: "
        f"{candidate_collection}"
    )

    print(
        f"generator baseline : "
        f"{baseline_model}"
    )

    print(
        f"generator candidate: "
        f"{candidate_model}"
    )

    print(
        f"top_k baseline     : "
        f"{baseline_top_k}"
    )

    print(
        f"top_k candidate    : "
        f"{candidate_top_k}"
    )


# ============================================================
# 16. SNAPSHOT VALIDATION
# ============================================================

def validate_snapshot(
    snapshot,
    label,
):
    """
    Validate basic snapshot structure before comparison.
    """

    if not isinstance(snapshot, dict):

        raise ValueError(
            f"{label} snapshot must be a JSON object."
        )

    if (
        "metrics" not in snapshot
        and "results" not in snapshot
    ):

        raise ValueError(
            f"{label} snapshot must contain "
            "'metrics' or 'results'."
        )


# ============================================================
# 17. CLI
# ============================================================

def main():
    """
    Command-line entrypoint.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compare two RepoMind evaluation "
            "snapshots and return a regression verdict."
        )
    )

    parser.add_argument(
        "--baseline",
        default=str(
            BASELINE_PATH
        ),
        help=(
            "Baseline snapshot path. "
            "Default: baselines/baseline.json"
        ),
    )

    parser.add_argument(
        "--candidate",
        default=str(
            CANDIDATE_PATH
        ),
        help=(
            "Candidate snapshot path. "
            "Default: baselines/candidate.json"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Also display informational metrics."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD
    # ========================================================

    try:

        baseline = load(
            args.baseline
        )

        candidate = load(
            args.candidate
        )

        validate_snapshot(
            baseline,
            "Baseline",
        )

        validate_snapshot(
            candidate,
            "Candidate",
        )

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as error:

        print(
            f"Could not load evaluation snapshot:"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        sys.exit(1)

    # ========================================================
    # METADATA
    # ========================================================

    print(
        "=" * 115
    )

    print(
        "RepoMind REGRESSION COMPARISON"
    )

    print(
        "=" * 115
    )

    print_metadata(
        baseline,
        candidate,
        args.baseline,
        args.candidate,
    )

    # ========================================================
    # EXTRACT METRICS
    # ========================================================

    try:

        baseline_metrics = extract_metrics(
            baseline
        )

        candidate_metrics = extract_metrics(
            candidate
        )

    except Exception as error:

        print()
        print(
            "Metric extraction failed:"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        sys.exit(1)

    print()

    print(
        f"Baseline metrics  : "
        f"{len(baseline_metrics)}"
    )

    print(
        f"Candidate metrics : "
        f"{len(candidate_metrics)}"
    )

    # ========================================================
    # COMPARE
    # ========================================================

    try:

        verdict, rows = compare(
            baseline,
            candidate,
        )

    except Exception as error:

        print()
        print(
            "Comparison failed:"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        sys.exit(1)

    # ========================================================
    # REPORT
    # ========================================================

    print_report(
        verdict,
        rows,
        show_all=args.all,
    )

    # ========================================================
    # CI EXIT CODE
    # ========================================================

    sys.exit(
        EXIT_CODE[verdict]
    )


# ============================================================
# 18. ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()