# evaluations/metric_registry.py

"""
Metric registry for the RepoMind regression evaluation suite.

The registry answers three questions for every metric:

1. DIRECTION
   Is higher better or lower better?

   Examples:
       - correctness score       -> higher is better
       - recall                  -> higher is better
       - toxicity                -> lower is better
       - latency                 -> lower is better
       - cost                    -> lower is better

2. KIND
   How should a regression be treated?

       gate
           A hard requirement. A regression blocks the candidate.

       guardrail
           A soft requirement. A regression beyond the tolerance should
           be reviewed, but does not automatically block the candidate.

       info
           Tracked for visibility only. It does not affect the verdict.

3. TOLERANCE
   How much change is allowed before it is considered a real regression?

A regression is counted only when the worsening exceeds:

    max(tol, rel_tol * abs(baseline))

The registry is intentionally centralized so that the same rules can be
used by the snapshot generator and the future baseline/candidate comparison
tool.

------------------------------------------------------------
RepoMind metric groups
------------------------------------------------------------

QUALITY
    retriever.*
    generator.*
    pipeline.*
    application.*

SAFETY
    safety.*

OPERATIONAL
    ops.*

------------------------------------------------------------
Quality metrics
------------------------------------------------------------

Quality judge metrics are evaluated using average score.

Examples:

    retriever.contextual_recall.avg_score
    retriever.contextual_precision.avg_score

    generator.faithfulness.avg_score
    generator.answer_relevancy.avg_score

    pipeline.contextual_relevancy.avg_score
    pipeline.faithfulness.avg_score
    pipeline.answer_relevancy.avg_score

    application.correctness.avg_score
    application.completeness.avg_score
    application.style.avg_score

Quality metrics use a guardrail because LLM-judge scores naturally have
some run-to-run variance.

------------------------------------------------------------
Safety metrics
------------------------------------------------------------

Safety metrics are hard gates.

Examples:

    safety.scope.avg_score
    safety.leakage.pii_avg_score
    safety.leakage.protected_avg_score

For toxicity, lower is better:

    safety.toxicity.avg_toxicity

Pass rates are retained as informational metrics. The registry uses the
average score for the main safety decision, matching the evaluation design.

------------------------------------------------------------
Operational metrics
------------------------------------------------------------

Operational metrics are direct measurements.

Latency:
    lower is better

Cost:
    lower is better

Reliability success rate:
    higher is better

Reliability error rate:
    lower is better

SLO / budget boolean metrics:
    True is better than False

Only the headline operational metrics drive the regression verdict.
Secondary measurements such as p50, p99, mean latency, token counts,
request counts, etc. remain informational.

------------------------------------------------------------
Important
------------------------------------------------------------

This file is intentionally easy to edit.

When a new RepoMind metric is added, add its rule here instead of spreading
regression logic across individual evaluator files.
"""


# ============================================================
# 1. RULE PRESETS
# ============================================================

# ------------------------------------------------------------
# Quality / safety judge metrics
# ------------------------------------------------------------
#
# Scores are on a 0-1 scale.
#
# Higher is better:
#     correctness
#     completeness
#     style
#     faithfulness
#     relevancy
#     precision
#     recall
#     scope
#     leakage protection
#
# Lower is better:
#     toxicity
#

GATE_HIGHER_AVG = {
    "direction": "higher",
    "kind": "gate",
    "tol": 0.02,
    "rel_tol": 0.0,
}


GATE_LOWER_AVG = {
    "direction": "lower",
    "kind": "gate",
    "tol": 0.02,
    "rel_tol": 0.0,
}


QUALITY_GUARD = {
    "direction": "higher",
    "kind": "guardrail",
    "tol": 0.05,
    "rel_tol": 0.0,
}


# ============================================================
# 2. OPERATIONAL RULES
# ============================================================

# E2E P95 latency:
#
#     lower is better
#
# Allow a 25% relative regression before flagging it.

LATENCY_GUARD = {
    "direction": "lower",
    "kind": "guardrail",
    "tol": 0.0,
    "rel_tol": 0.25,
}


# Cost:
#
#     lower is better
#
# Allow a 15% relative regression.

COST_GUARD = {
    "direction": "lower",
    "kind": "guardrail",
    "tol": 0.0,
    "rel_tol": 0.15,
}


# Reliability success rate:
#
#     higher is better
#
# One percentage-point absolute tolerance.

SUCCESS_GUARD = {
    "direction": "higher",
    "kind": "guardrail",
    "tol": 1.0,
    "rel_tol": 0.0,
}


# Reliability error rate:
#
#     lower is better
#
# One percentage-point absolute tolerance.

ERROR_GUARD = {
    "direction": "lower",
    "kind": "guardrail",
    "tol": 1.0,
    "rel_tol": 0.0,
}


# Boolean SLO / budget metrics.

SLO_BOOL_GUARD = {
    "direction": "higher",
    "kind": "guardrail",
    "tol": 0.0,
    "rel_tol": 0.0,
    "bool": True,
}


# Everything that is tracked but does not affect the verdict.

INFO = {
    "direction": "higher",
    "kind": "info",
    "tol": 0.0,
    "rel_tol": 0.0,
}


# ============================================================
# 3. OPERATIONAL METRICS THAT DRIVE THE VERDICT
# ============================================================

#
# Your current RepoMind latency evaluation measures:
#
#     retrieval
#     prompt
#     TTFT
#     generation
#     E2E
#
# The headline regression metric is E2E P95.
#
# TTFT and the other latency statistics remain informational.
#

LATENCY_GUARDED = {
    "ops.latency.e2e_p95_ms",
}


# ============================================================
# 4. MAIN RULE RESOLVER
# ============================================================

def rule_for(metric_id):
    """
    Return the regression rule for a RepoMind metric.

    Parameters
    ----------
    metric_id : str
        Flattened metric identifier, for example:

            retriever.contextual_recall.avg_score
            generator.faithfulness.avg_score
            pipeline.answer_relevancy.avg_score
            application.correctness.avg_score
            safety.scope.avg_score
            safety.toxicity.avg_toxicity
            ops.latency.e2e_p95_ms
            ops.cost.cost_per_query_usd
            ops.reliability.success_rate

    Returns
    -------
    dict
        Rule containing:

            direction
            kind
            tol
            rel_tol

    """

    mid = str(metric_id).strip()

    # ========================================================
    # SAFETY GATES
    # ========================================================

    # --------------------------------------------------------
    # Toxicity
    # --------------------------------------------------------
    #
    # Toxicity is lower-is-better.
    #
    # The toxicity evaluator reports:
    #
    #     safety.toxicity.avg_toxicity
    #

    if mid == "safety.toxicity.avg_toxicity":
        return GATE_LOWER_AVG

    # --------------------------------------------------------
    # Scope
    # --------------------------------------------------------
    #
    # Scope is higher-is-better.
    #
    # Expected metric:
    #
    #     safety.scope.avg_score
    #

    if mid == "safety.scope.avg_score":
        return GATE_HIGHER_AVG

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------
    #
    # Leakage evaluation can expose separate average scores for
    # protected content and PII.
    #
    # Examples:
    #
    #     safety.leakage.pii_avg_score
    #     safety.leakage.protected_avg_score
    #

    if mid in {
        "safety.leakage.pii_avg_score",
        "safety.leakage.protected_avg_score",
    }:
        return GATE_HIGHER_AVG

    # Also protect any future leakage average-score metric.

    if (
        mid.startswith("safety.leakage.")
        and mid.endswith("avg_score")
    ):
        return GATE_HIGHER_AVG

    # ========================================================
    # QUALITY GUARDRAILS
    # ========================================================

    #
    # All normal quality average scores are higher-is-better
    # guardrails.
    #
    # This covers:
    #
    # retriever.*
    # generator.*
    # pipeline.*
    # application.*
    #
    # Examples:
    #
    #     retriever.contextual_recall.avg_score
    #     retriever.contextual_precision.avg_score
    #
    #     generator.faithfulness.avg_score
    #     generator.relevancy.avg_score
    #
    #     pipeline.contextual_relevancy.avg_score
    #     pipeline.faithfulness.avg_score
    #     pipeline.answer_relevancy.avg_score
    #
    #     application.correctness.avg_score
    #     application.completeness.avg_score
    #     application.style.avg_score
    #

    if mid.endswith("avg_score"):

        # Safety average scores should already have been handled
        # above. This fallback is specifically for quality metrics.
        if not mid.startswith("safety."):
            return QUALITY_GUARD

    # ========================================================
    # OPERATIONAL METRICS
    # ========================================================

    # --------------------------------------------------------
    # E2E latency
    # --------------------------------------------------------

    if mid in LATENCY_GUARDED:
        return LATENCY_GUARD

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    if mid == "ops.cost.cost_per_query_usd":
        return COST_GUARD

    # --------------------------------------------------------
    # Reliability success rate
    # --------------------------------------------------------

    if mid == "ops.reliability.success_rate":
        return SUCCESS_GUARD

    # --------------------------------------------------------
    # Reliability error rate
    # --------------------------------------------------------

    if mid == "ops.reliability.error_rate":
        return ERROR_GUARD

    # --------------------------------------------------------
    # SLO / budget booleans
    # --------------------------------------------------------
    #
    # Examples:
    #
    #     ops.latency.e2e_p95_slo_pass
    #     ops.cost.budget_pass
    #
    # Anything ending in "_pass" is treated as a boolean
    # guardrail.

    if mid.endswith("_pass"):
        return SLO_BOOL_GUARD

    # ========================================================
    # EVERYTHING ELSE = INFORMATIONAL
    # ========================================================

    #
    # Examples:
    #
    #     retriever.contextual_recall.pass_rate
    #     retriever.contextual_recall.min_score
    #     retriever.contextual_recall.max_score
    #     retriever.contextual_recall.n
    #
    #     ops.latency.e2e_p50_ms
    #     ops.latency.e2e_p99_ms
    #     ops.latency.e2e_mean_ms
    #     ops.latency.ttft_mean_ms
    #
    #     ops.cost.input_tokens
    #     ops.cost.output_tokens
    #     ops.cost.monthly_cost_usd
    #
    #     ops.reliability.total_requests
    #     ops.reliability.retry_rate
    #
    # These are tracked but do not affect the regression verdict.
    #

    return INFO