"""
RepoMind Operational Evaluation
================================

Runs three operational evaluations:

1. LATENCY
   - End-to-end latency
   - TTFT
   - Retrieval latency
   - Generation latency
   - P50 / P95 / P99
   - SLO checks

2. COST
   - Input tokens
   - Output tokens
   - Cached tokens
   - Estimated hosted-model cost
   - Daily/monthly projection
   - Cost budget check

3. RELIABILITY
   - Success rate
   - Error rate
   - Retry rate
   - Exponential-backoff retries

run_ops() returns a flat operational snapshot.

Example:

    python evaluations/eval_ops.py \
        --collection repomind_e2af1fe6fdf92747 \
        --top-k 3

The RepoMind application currently uses local Ollama.

Therefore cost numbers are theoretical hosted-model
projections, not actual Ollama billing.
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import math
import sys
import time
from pathlib import Path


# ============================================================
# 2. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 3. PROJECT IMPORTS
# ============================================================

from src.pipeline import RAGPipeline
from src.llm.prompts import build_prompt


# ============================================================
# 4. QUESTION SET
# ============================================================

QUESTIONS = [
    "What is the overall architecture of this repository?",
    "How does the RAG pipeline retrieve documents?",
    "How is the answer generated after retrieval?",
    "How does the application connect to PostgreSQL?",
    "How are embeddings stored and retrieved?",
    "How does the application create or use Chroma collections?",
    "How does repository ingestion work?",
    "How does the system handle multiple GitHub repositories?",
    "What is the purpose of the Streamlit application?",
    "How does the application process a GitHub repository URL?",
    "What files are responsible for retrieval?",
    "What files are responsible for answer generation?",
    "How does the system identify source files?",
    "How does the RAG pipeline handle missing documents?",
    "What technologies are used in this project?",
]


# ============================================================
# 5. DEFAULT CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 3

# Latency
DEFAULT_LAT_REPEATS = 5
DEFAULT_LAT_WARMUP_RUNS = 2

# Cost
DEFAULT_COST_REPEATS = 3

# Reliability
DEFAULT_REL_REPEATS = 5
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_S = 0.5

# Latency SLOs
SLO_P95_MS = 3000
SLO_TTFT_P95_MS = 1200


# ============================================================
# 6. HOSTED COST CONFIGURATION
# ============================================================

# Hypothetical hosted-model pricing.
#
# RepoMind currently uses local Ollama, so these numbers
# represent a hypothetical hosted deployment.

PRICE_INPUT_PER_1M = 0.15

PRICE_CACHED_INPUT_PER_1M = 0.075

PRICE_OUTPUT_PER_1M = 0.60

QUERIES_PER_DAY = 2000

USD_TO_INR = 88.0

COST_BUDGET_PER_QUERY_USD = 0.0015


# ============================================================
# 7. BASIC HELPERS
# ============================================================

def average(values):
    """Return arithmetic mean."""

    if not values:
        return 0.0

    return sum(values) / len(values)


def percentile(values, p):
    """
    Calculate percentile using linear interpolation.
    """

    clean = [
        value
        for value in values
        if value is not None
        and not math.isnan(value)
    ]

    if not clean:
        return float("nan")

    clean = sorted(clean)

    position = (
        (len(clean) - 1)
        * (p / 100.0)
    )

    lower = math.floor(position)

    upper = math.ceil(position)

    if lower == upper:
        return clean[int(position)]

    return (
        clean[lower]
        * (upper - position)
        +
        clean[upper]
        * (position - lower)
    )


def column_average(rows, key):
    """Average one dictionary field."""

    values = [
        row[key]
        for row in rows
        if key in row
    ]

    return average(values)


def safe_float(value):
    """Convert a value to float where possible."""

    try:
        return float(value)
    except Exception:
        return float("nan")


# ============================================================
# 8. PIPELINE ADAPTERS
# ============================================================

def retrieve_documents(
    pipeline,
    question,
    collection_name,
    top_k,
):
    """
    RepoMind retriever adapter.

    Current RepoMind API:

        pipeline.retriever(
            question,
            collection_name,
            top_k,
        )
    """

    return pipeline.retriever(
        question,
        collection_name,
        top_k,
    )


def generate_answer(
    pipeline,
    question,
    documents,
):
    """
    RepoMind non-streaming generator adapter.
    """

    result = pipeline.generator.generate(
        question,
        documents,
    )

    return result["answer"]


def generate_stream(
    pipeline,
    question,
    documents,
):
    """
    RepoMind streaming generator adapter.
    """

    return pipeline.generator.generate_stream(
        question,
        documents,
    )


# ============================================================
# ============================================================
# LATENCY EVALUATION
# ============================================================
# ============================================================


# ============================================================
# 9. END-TO-END LATENCY
# ============================================================

def latency_end_to_end(
    pipeline,
    question,
    collection_name,
    top_k,
):
    """
    Measure complete retrieval + generation latency.
    """

    request_start = time.perf_counter()

    documents = retrieve_documents(
        pipeline,
        question,
        collection_name,
        top_k,
    )

    generate_answer(
        pipeline,
        question,
        documents,
    )

    return (
        time.perf_counter()
        - request_start
    ) * 1000


# ============================================================
# 10. STREAMING STAGE LATENCY
# ============================================================

def latency_stages_streaming(
    pipeline,
    question,
    collection_name,
    top_k,
):
    """
    Measure:

        retrieval
        generation
        TTFT

    TTFT is measured from the beginning of the request until
    the first non-empty generated chunk.

    Therefore:

        TTFT includes retrieval latency.
    """

    request_start = time.perf_counter()

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    documents = retrieve_documents(
        pipeline,
        question,
        collection_name,
        top_k,
    )

    retrieval_end = time.perf_counter()

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation_start = retrieval_end

    first_token_time = None

    pieces = []

    stream = generate_stream(
        pipeline,
        question,
        documents,
    )

    for piece in stream:

        if piece:

            if first_token_time is None:

                first_token_time = (
                    time.perf_counter()
                )

            pieces.append(piece)

    generation_end = time.perf_counter()

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    retrieval_ms = (
        retrieval_end
        - request_start
    ) * 1000

    generation_ms = (
        generation_end
        - generation_start
    ) * 1000

    if first_token_time is None:

        ttft_ms = float("nan")

    else:

        ttft_ms = (
            first_token_time
            - request_start
        ) * 1000

    answer = "".join(pieces)

    return answer, {
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
    }


# ============================================================
# 11. LATENCY BENCHMARK
# ============================================================

def latency_benchmark(
    pipeline,
    collection_name,
    top_k,
    repeats,
    warmup_runs,
):
    """
    Run the latency benchmark.
    """

    print()
    print("=" * 78)
    print("LATENCY EVALUATION")
    print("=" * 78)

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        f"[latency] warming up "
        f"({warmup_runs} runs, discarded)..."
    )

    for i in range(warmup_runs):

        question = QUESTIONS[
            i % len(QUESTIONS)
        ]

        try:

            latency_end_to_end(
                pipeline,
                question,
                collection_name,
                top_k,
            )

        except Exception as exc:

            print(
                f"[latency] warmup failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    total_ms = []

    retrieval_ms = []

    generation_ms = []

    ttft_ms = []

    answer_lengths = []

    total_requests = 0

    failed_requests = 0

    print(
        "[latency] measuring..."
    )

    for question in QUESTIONS:

        for _ in range(repeats):

            total_requests += 1

            request_start = (
                time.perf_counter()
            )

            try:

                answer, stages = (
                    latency_stages_streaming(
                        pipeline,
                        question,
                        collection_name,
                        top_k,
                    )
                )

                elapsed_ms = (
                    time.perf_counter()
                    - request_start
                ) * 1000

                total_ms.append(
                    elapsed_ms
                )

                retrieval_ms.append(
                    stages["retrieval"]
                )

                generation_ms.append(
                    stages["generation"]
                )

                ttft_ms.append(
                    stages["ttft"]
                )

                answer_lengths.append(
                    len(answer or "")
                )

            except Exception as exc:

                failed_requests += 1

                print(
                    "[latency] FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
        "total_requests": total_requests,
        "failed_requests": failed_requests,
    }


# ============================================================
# 12. LATENCY SUMMARY
# ============================================================

def latency_summary(samples):

    clean = [
        value
        for value in samples
        if value is not None
        and not math.isnan(value)
    ]

    if not clean:

        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(clean),

        "mean": average(clean),

        "p50": percentile(
            clean,
            50,
        ),

        "p95": percentile(
            clean,
            95,
        ),

        "p99": percentile(
            clean,
            99,
        ),

        "min": min(clean),

        "max": max(clean),
    }


# ============================================================
# 13. LATENCY REPORT
# ============================================================

def latency_print_row(
    label,
    stats,
):

    print(
        f"{label:<14} | "
        f"n={stats['n']:<4} "
        f"mean={stats['mean']:9.1f} "
        f"p50={stats['p50']:9.1f} "
        f"p95={stats['p95']:9.1f} "
        f"p99={stats['p99']:9.1f} "
        f"min={stats['min']:9.1f} "
        f"max={stats['max']:9.1f}"
    )


def latency_slo_line(
    label,
    p95,
    budget,
):

    if math.isnan(p95):

        verdict = "NO DATA"

    elif p95 <= budget:

        verdict = "PASS"

    else:

        verdict = "FAIL"

    print(
        f"SLO: {label:<25} "
        f"p95 <= {budget:>5} ms  "
        f"actual={p95:>9.0f} ms "
        f"[{verdict}]"
    )


def latency_report(results):

    total = latency_summary(
        results["total"]
    )

    retrieval = latency_summary(
        results["retrieval"]
    )

    generation = latency_summary(
        results["generation"]
    )

    ttft = latency_summary(
        results["ttft"]
    )

    print()
    print("-" * 78)

    print(
        f"{'stage':<14} | "
        f"{'n':<5} "
        f"{'mean':>10} "
        f"{'p50':>10} "
        f"{'p95':>10} "
        f"{'p99':>10} "
        f"{'min':>10} "
        f"{'max':>10}"
    )

    print("-" * 78)

    latency_print_row(
        "end-to-end",
        total,
    )

    latency_print_row(
        "ttft",
        ttft,
    )

    latency_print_row(
        "retrieval",
        retrieval,
    )

    latency_print_row(
        "generation",
        generation,
    )

    print("-" * 78)

    print(
        f"successful runs : "
        f"{len(results['total'])}"
    )

    print(
        f"failed runs     : "
        f"{results['failed_requests']}"
    )

    print(
        f"avg answer chars: "
        f"{average(results['answer_len']):.0f}"
    )

    print("-" * 78)

    latency_slo_line(
        "end-to-end latency",
        total["p95"],
        SLO_P95_MS,
    )

    latency_slo_line(
        "time to first token",
        ttft["p95"],
        SLO_TTFT_P95_MS,
    )

    print("-" * 78)


# ============================================================
# 14. RUN LATENCY
# ============================================================

def run_latency(
    pipeline,
    collection_name,
    top_k=DEFAULT_TOP_K,
    repeats=DEFAULT_LAT_REPEATS,
    warmup_runs=DEFAULT_LAT_WARMUP_RUNS,
    verbose=True,
):
    """
    Run latency evaluation and return flat metrics.
    """

    results = latency_benchmark(
        pipeline=pipeline,
        collection_name=collection_name,
        top_k=top_k,
        repeats=repeats,
        warmup_runs=warmup_runs,
    )

    if verbose:

        latency_report(
            results
        )

    total = latency_summary(
        results["total"]
    )

    retrieval = latency_summary(
        results["retrieval"]
    )

    generation = latency_summary(
        results["generation"]
    )

    ttft = latency_summary(
        results["ttft"]
    )

    return {
        "e2e_mean_ms": total["mean"],
        "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"],
        "e2e_p99_ms": total["p99"],
        "e2e_min_ms": total["min"],
        "e2e_max_ms": total["max"],

        "ttft_mean_ms": ttft["mean"],
        "ttft_p50_ms": ttft["p50"],
        "ttft_p95_ms": ttft["p95"],
        "ttft_p99_ms": ttft["p99"],

        "retrieval_mean_ms": (
            retrieval["mean"]
        ),

        "retrieval_p50_ms": (
            retrieval["p50"]
        ),

        "retrieval_p95_ms": (
            retrieval["p95"]
        ),

        "retrieval_p99_ms": (
            retrieval["p99"]
        ),

        "generation_mean_ms": (
            generation["mean"]
        ),

        "generation_p50_ms": (
            generation["p50"]
        ),

        "generation_p95_ms": (
            generation["p95"]
        ),

        "generation_p99_ms": (
            generation["p99"]
        ),

        "avg_answer_len": average(
            results["answer_len"]
        ),

        "successful_runs": len(
            results["total"]
        ),

        "failed_runs": (
            results["failed_requests"]
        ),

        "slo_e2e_pass": (
            total["p95"] <= SLO_P95_MS
            if not math.isnan(
                total["p95"]
            )
            else False
        ),

        "slo_ttft_pass": (
            ttft["p95"] <= SLO_TTFT_P95_MS
            if not math.isnan(
                ttft["p95"]
            )
            else False
        ),
    }


# ============================================================
# ============================================================
# COST EVALUATION
# ============================================================
# ============================================================


# ============================================================
# 15. TOKEN USAGE EXTRACTION
# ============================================================

def extract_usage(message):
    """
    Extract token usage from a LangChain AIMessage.

    Supports:

        usage_metadata

    and:

        response_metadata
    """

    input_tokens = 0

    output_tokens = 0

    cached_tokens = 0

    # --------------------------------------------------------
    # LangChain usage_metadata
    # --------------------------------------------------------

    usage = getattr(
        message,
        "usage_metadata",
        None,
    )

    if usage:

        input_tokens = (
            usage.get(
                "input_tokens",
                0,
            )
            or 0
        )

        output_tokens = (
            usage.get(
                "output_tokens",
                0,
            )
            or 0
        )

        input_details = (
            usage.get(
                "input_token_details",
                {},
            )
            or {}
        )

        cached_tokens = (
            input_details.get(
                "cache_read",
                0,
            )
            or input_details.get(
                "cached_tokens",
                0,
            )
            or 0
        )

    # --------------------------------------------------------
    # Ollama metadata fallback
    # --------------------------------------------------------

    metadata = getattr(
        message,
        "response_metadata",
        None,
    )

    if metadata:

        if input_tokens == 0:

            input_tokens = (
                metadata.get(
                    "prompt_eval_count",
                    0,
                )
                or 0
            )

        if output_tokens == 0:

            output_tokens = (
                metadata.get(
                    "eval_count",
                    0,
                )
                or 0
            )

        cached_tokens = (
            metadata.get(
                "cache_read",
                cached_tokens,
            )
            or metadata.get(
                "cached_tokens",
                cached_tokens,
            )
            or cached_tokens
        )

    return {
        "input": int(
            input_tokens
        ),

        "output": int(
            output_tokens
        ),

        "cached": int(
            cached_tokens
        ),
    }


# ============================================================
# 16. COST CALCULATION
# ============================================================

def cost_usd(
    input_tokens,
    output_tokens,
    cached_tokens,
):
    """
    Calculate theoretical hosted-model cost.
    """

    uncached_input = max(
        input_tokens - cached_tokens,
        0,
    )

    input_cost = (
        uncached_input
        / 1_000_000
        * PRICE_INPUT_PER_1M
    )

    cached_cost = (
        cached_tokens
        / 1_000_000
        * PRICE_CACHED_INPUT_PER_1M
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * PRICE_OUTPUT_PER_1M
    )

    return {
        "input": input_cost,

        "cached": cached_cost,

        "output": output_cost,

        "total": (
            input_cost
            + cached_cost
            + output_cost
        ),
    }


# ============================================================
# 17. COST TOKEN MEASUREMENT
# ============================================================

def cost_measure_tokens(
    pipeline,
    question,
    collection_name,
    top_k,
):
    """
    Retrieve context and invoke the actual RepoMind LLM.

    The direct LLM invocation is intentional because it lets
    us inspect token usage metadata.
    """

    documents = retrieve_documents(
        pipeline,
        question,
        collection_name,
        top_k,
    )

    prompt = build_prompt(
        question,
        documents,
    )

    llm = pipeline.generator.llm

    message = llm.invoke(
        prompt
    )

    return extract_usage(
        message
    )


# ============================================================
# 18. COST BENCHMARK
# ============================================================

def cost_benchmark(
    pipeline,
    collection_name,
    top_k,
    repeats,
):
    """
    Run token/cost benchmark.
    """

    rows = []

    print()
    print("=" * 78)
    print("COST EVALUATION")
    print("=" * 78)

    print(
        "[cost] measuring token usage..."
    )

    for question in QUESTIONS:

        for _ in range(repeats):

            try:

                tokens = (
                    cost_measure_tokens(
                        pipeline,
                        question,
                        collection_name,
                        top_k,
                    )
                )

                cost = cost_usd(
                    tokens["input"],
                    tokens["output"],
                    tokens["cached"],
                )

                rows.append(
                    {
                        **tokens,

                        "cost_input": (
                            cost["input"]
                        ),

                        "cost_cached": (
                            cost["cached"]
                        ),

                        "cost_output": (
                            cost["output"]
                        ),

                        "cost_total": (
                            cost["total"]
                        ),
                    }
                )

            except Exception as exc:

                print(
                    "[cost] FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    return rows


# ============================================================
# 19. COST REPORT
# ============================================================

def cost_report(rows):

    if not rows:

        print(
            "[cost] No successful measurements."
        )

        return

    avg_input = column_average(
        rows,
        "input",
    )

    avg_output = column_average(
        rows,
        "output",
    )

    avg_cached = column_average(
        rows,
        "cached",
    )

    avg_cost = column_average(
        rows,
        "cost_total",
    )

    min_cost = min(
        row["cost_total"]
        for row in rows
    )

    max_cost = max(
        row["cost_total"]
        for row in rows
    )

    avg_output_cost = (
        column_average(
            rows,
            "cost_output",
        )
    )

    output_share = (
        100
        * avg_output_cost
        / avg_cost
        if avg_cost
        else 0.0
    )

    daily = (
        avg_cost
        * QUERIES_PER_DAY
    )

    monthly = (
        daily
        * 30
    )

    budget_pass = (
        avg_cost
        <= COST_BUDGET_PER_QUERY_USD
    )

    print()
    print("-" * 78)

    print(
        "HOSTED COST ESTIMATE"
    )

    print("-" * 78)

    print(
        f"samples            : "
        f"{len(rows)}"
    )

    print(
        f"avg input tokens   : "
        f"{avg_input:.0f}"
    )

    print(
        f"avg output tokens  : "
        f"{avg_output:.0f}"
    )

    print(
        f"avg cached tokens  : "
        f"{avg_cached:.0f}"
    )

    print("-" * 78)

    print(
        f"avg cost/query     : "
        f"${avg_cost:.6f}"
    )

    print(
        f"cost/query in INR  : "
        f"Rs {avg_cost * USD_TO_INR:.4f}"
    )

    print(
        f"min cost           : "
        f"${min_cost:.6f}"
    )

    print(
        f"max cost           : "
        f"${max_cost:.6f}"
    )

    print(
        f"output cost share  : "
        f"{output_share:.1f}%"
    )

    print("-" * 78)

    print(
        f"daily projection   : "
        f"${daily:.2f} "
        f"(Rs {daily * USD_TO_INR:.2f})"
    )

    print(
        f"monthly projection : "
        f"${monthly:.2f} "
        f"(Rs {monthly * USD_TO_INR:.2f})"
    )

    print("-" * 78)

    print(
        f"budget/query       : "
        f"${COST_BUDGET_PER_QUERY_USD:.6f}"
    )

    print(
        f"budget status      : "
        f"{'PASS' if budget_pass else 'FAIL'}"
    )

    print("-" * 78)

    print(
        "NOTE: RepoMind currently uses local Ollama."
    )

    print(
        "These are theoretical hosted-model costs."
    )


# ============================================================
# 20. RUN COST
# ============================================================

def run_cost(
    pipeline,
    collection_name,
    top_k=DEFAULT_TOP_K,
    repeats=DEFAULT_COST_REPEATS,
    verbose=True,
):
    """
    Run cost evaluation and return flat metrics.
    """

    rows = cost_benchmark(
        pipeline=pipeline,
        collection_name=collection_name,
        top_k=top_k,
        repeats=repeats,
    )

    if verbose:

        cost_report(
            rows
        )

    if not rows:

        return {
            "samples": 0,
            "cost_per_query_usd": float("nan"),
            "cost_per_query_inr": float("nan"),
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_cached_tokens": 0.0,
            "output_cost_share_pct": 0.0,
            "daily_usd": float("nan"),
            "monthly_usd": float("nan"),
            "budget_pass": False,
        }

    avg_cost = column_average(
        rows,
        "cost_total",
    )

    avg_output_cost = column_average(
        rows,
        "cost_output",
    )

    output_share = (
        100
        * avg_output_cost
        / avg_cost
        if avg_cost
        else 0.0
    )

    daily = (
        avg_cost
        * QUERIES_PER_DAY
    )

    monthly = (
        daily
        * 30
    )

    return {
        "samples": len(rows),

        "cost_per_query_usd": avg_cost,

        "cost_per_query_inr": (
            avg_cost
            * USD_TO_INR
        ),

        "avg_input_tokens": (
            column_average(
                rows,
                "input",
            )
        ),

        "avg_output_tokens": (
            column_average(
                rows,
                "output",
            )
        ),

        "avg_cached_tokens": (
            column_average(
                rows,
                "cached",
            )
        ),

        "output_cost_share_pct": (
            output_share
        ),

        "daily_usd": daily,

        "monthly_usd": monthly,

        "budget_pass": (
            avg_cost
            <= COST_BUDGET_PER_QUERY_USD
        ),
    }


# ============================================================
# ============================================================
# RELIABILITY EVALUATION
# ============================================================
# ============================================================


# ============================================================
# 21. RELIABILITY TRACKER
# ============================================================

class ReliabilityTracker:

    def __init__(self):

        self.calls = 0

        self.successes = 0

        self.failures = 0

        self.retries = 0

        self.recovered = 0

        self.first_attempt_successes = 0

        self.total_attempts = 0

        self.errors = []


# ============================================================
# 22. RETRY WRAPPER
# ============================================================

def call_with_retries(
    fn,
    tracker,
    max_retries,
    backoff_base_s,
):
    """
    Execute a request with exponential backoff.

    Retry delays:

        retry 1 -> base
        retry 2 -> base * 2
        retry 3 -> base * 4
    """

    tracker.calls += 1

    for attempt in range(
        max_retries + 1
    ):

        tracker.total_attempts += 1

        try:

            result = fn()

            tracker.successes += 1

            if attempt == 0:

                tracker.first_attempt_successes += 1

            else:

                tracker.recovered += 1

            return result

        except Exception as exc:

            if attempt < max_retries:

                tracker.retries += 1

                delay = (
                    backoff_base_s
                    * (2 ** attempt)
                )

                time.sleep(
                    delay
                )

            else:

                tracker.failures += 1

                tracker.errors.append(
                    str(exc)
                )

                print(
                    "[reliability] FAILED "
                    f"after {max_retries} "
                    f"retries: {exc}"
                )

                return None


# ============================================================
# 23. RELIABILITY BENCHMARK
# ============================================================

def reliability_benchmark(
    pipeline,
    collection_name,
    top_k,
    repeats,
    max_retries,
    backoff_base_s,
):
    """
    Run reliability benchmark.
    """

    tracker = ReliabilityTracker()

    print()
    print("=" * 78)
    print("RELIABILITY EVALUATION")
    print("=" * 78)

    print(
        "[reliability] measuring..."
    )

    for question in QUESTIONS:

        for _ in range(repeats):

            def request(
                q=question,
            ):

                documents = (
                    retrieve_documents(
                        pipeline,
                        q,
                        collection_name,
                        top_k,
                    )
                )

                return pipeline.generator.generate(
                    q,
                    documents,
                )

            call_with_retries(
                request,
                tracker,
                max_retries,
                backoff_base_s,
            )

    return tracker


# ============================================================
# 24. RELIABILITY REPORT
# ============================================================

def reliability_report(
    tracker,
):

    calls = tracker.calls

    if calls == 0:

        print(
            "[reliability] No requests executed."
        )

        return

    success_rate = (
        100
        * tracker.successes
        / calls
    )

    error_rate = (
        100
        * tracker.failures
        / calls
    )

    retry_rate = (
        100
        * tracker.retries
        / calls
    )

    first_attempt_rate = (
        100
        * tracker.first_attempt_successes
        / calls
    )

    recovery_rate = (
        100
        * tracker.recovered
        / calls
    )

    average_attempts = (
        tracker.total_attempts
        / calls
    )

    print()
    print("-" * 78)

    print(
        f"total requests          : "
        f"{tracker.calls}"
    )

    print(
        f"successful              : "
        f"{tracker.successes}"
    )

    print(
        f"failed                  : "
        f"{tracker.failures}"
    )

    print(
        f"first-attempt success   : "
        f"{tracker.first_attempt_successes}"
    )

    print(
        f"recovered after retry   : "
        f"{tracker.recovered}"
    )

    print(
        f"retry attempts          : "
        f"{tracker.retries}"
    )

    print(
        f"total attempts          : "
        f"{tracker.total_attempts}"
    )

    print("-" * 78)

    print(
        f"success rate            : "
        f"{success_rate:.2f}%"
    )

    print(
        f"error rate              : "
        f"{error_rate:.2f}%"
    )

    print(
        f"retry rate              : "
        f"{retry_rate:.2f}%"
    )

    print(
        f"first-attempt success   : "
        f"{first_attempt_rate:.2f}%"
    )

    print(
        f"retry recovery rate     : "
        f"{recovery_rate:.2f}%"
    )

    print(
        f"average attempts/request: "
        f"{average_attempts:.2f}"
    )

    if tracker.errors:

        print("-" * 78)

        print(
            "FINAL ERRORS:"
        )

        for error in tracker.errors:

            print(
                f"  - {error}"
            )

    else:

        print(
            "No final request failures."
        )


# ============================================================
# 25. RUN RELIABILITY
# ============================================================

def run_reliability(
    pipeline,
    collection_name,
    top_k=DEFAULT_TOP_K,
    repeats=DEFAULT_REL_REPEATS,
    max_retries=DEFAULT_MAX_RETRIES,
    backoff_base_s=DEFAULT_BACKOFF_BASE_S,
    verbose=True,
):
    """
    Run reliability evaluation and return flat metrics.
    """

    tracker = reliability_benchmark(
        pipeline=pipeline,
        collection_name=collection_name,
        top_k=top_k,
        repeats=repeats,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
    )

    if verbose:

        reliability_report(
            tracker
        )

    calls = tracker.calls

    if calls == 0:

        return {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "retry_rate": 0.0,
            "first_attempt_success_rate": 0.0,
            "recovered_after_retry": 0,
            "retry_recovery_rate": 0.0,
            "total_retry_attempts": 0,
            "total_attempts": 0,
            "average_attempts_per_request": 0.0,
        }

    return {
        "total_requests": tracker.calls,

        "successful": tracker.successes,

        "failed": tracker.failures,

        "success_rate": (
            100
            * tracker.successes
            / calls
        ),

        "error_rate": (
            100
            * tracker.failures
            / calls
        ),

        "retry_rate": (
            100
            * tracker.retries
            / calls
        ),

        "first_attempt_success_rate": (
            100
            * tracker.first_attempt_successes
            / calls
        ),

        "recovered_after_retry": (
            tracker.recovered
        ),

        "retry_recovery_rate": (
            100
            * tracker.recovered
            / calls
        ),

        "total_retry_attempts": (
            tracker.retries
        ),

        "total_attempts": (
            tracker.total_attempts
        ),

        "average_attempts_per_request": (
            tracker.total_attempts
            / calls
        ),
    }


# ============================================================
# ============================================================
# RUN ALL OPS
# ============================================================
# ============================================================


def run_ops(
    pipeline=None,
    collection_name=None,
    top_k=DEFAULT_TOP_K,
    lat_repeats=DEFAULT_LAT_REPEATS,
    lat_warmup_runs=DEFAULT_LAT_WARMUP_RUNS,
    cost_repeats=DEFAULT_COST_REPEATS,
    rel_repeats=DEFAULT_REL_REPEATS,
    max_retries=DEFAULT_MAX_RETRIES,
    backoff_base_s=DEFAULT_BACKOFF_BASE_S,
    verbose=True,
):
    """
    Run latency, cost and reliability evaluations.

    Returns one flat operational snapshot.
    """

    if not collection_name:

        raise ValueError(
            "collection_name is required."
        )

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )

    if pipeline is None:

        pipeline = RAGPipeline()

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency = run_latency(
        pipeline=pipeline,
        collection_name=collection_name,
        top_k=top_k,
        repeats=lat_repeats,
        warmup_runs=lat_warmup_runs,
        verbose=verbose,
    )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    cost = run_cost(
        pipeline=pipeline,
        collection_name=collection_name,
        top_k=top_k,
        repeats=cost_repeats,
        verbose=verbose,
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    reliability = run_reliability(
        pipeline=pipeline,
        collection_name=collection_name,
        top_k=top_k,
        repeats=rel_repeats,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        verbose=verbose,
    )

    # --------------------------------------------------------
    # Flat snapshot
    # --------------------------------------------------------

    snapshot = {}

    for key, value in latency.items():

        snapshot[
            f"latency.{key}"
        ] = value

    for key, value in cost.items():

        snapshot[
            f"cost.{key}"
        ] = value

    for key, value in reliability.items():

        snapshot[
            f"reliability.{key}"
        ] = value

    return snapshot


# ============================================================
# 27. SUITE-FRIENDLY OPERATIONAL SNAPSHOT
# ============================================================

def run_ops_for_suite(
    pipeline,
    collection_name,
    top_k=DEFAULT_TOP_K,
    lat_repeats=DEFAULT_LAT_REPEATS,
    lat_warmup_runs=DEFAULT_LAT_WARMUP_RUNS,
    cost_repeats=DEFAULT_COST_REPEATS,
    rel_repeats=DEFAULT_REL_REPEATS,
    max_retries=DEFAULT_MAX_RETRIES,
    backoff_base_s=DEFAULT_BACKOFF_BASE_S,
    verbose=False,
):
    """
    Explicit suite wrapper.

    This exists so run_suite.py can call one predictable
    function without needing to know internal details.
    """

    return run_ops(
        pipeline=pipeline,
        collection_name=collection_name,
        top_k=top_k,
        lat_repeats=lat_repeats,
        lat_warmup_runs=lat_warmup_runs,
        cost_repeats=cost_repeats,
        rel_repeats=rel_repeats,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        verbose=verbose,
    )


# ============================================================
# 28. SNAPSHOT REPORT
# ============================================================

def print_operational_snapshot(
    snapshot,
):

    print()
    print()
    print("=" * 78)
    print(
        "REPO MIND OPERATIONAL SNAPSHOT"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    print()
    print("[LATENCY]")

    latency_keys = [
        "latency.e2e_mean_ms",
        "latency.e2e_p50_ms",
        "latency.e2e_p95_ms",
        "latency.e2e_p99_ms",
        "latency.ttft_p95_ms",
        "latency.retrieval_p95_ms",
        "latency.generation_p95_ms",
        "latency.slo_e2e_pass",
        "latency.slo_ttft_pass",
    ]

    for key in latency_keys:

        if key not in snapshot:
            continue

        value = snapshot[key]

        if isinstance(
            value,
            float,
        ):

            print(
                f"  {key:<38}"
                f"{value:.2f}"
            )

        else:

            print(
                f"  {key:<38}"
                f"{value}"
            )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    print()
    print("[COST]")

    cost_keys = [
        "cost.cost_per_query_usd",
        "cost.cost_per_query_inr",
        "cost.avg_input_tokens",
        "cost.avg_output_tokens",
        "cost.avg_cached_tokens",
        "cost.output_cost_share_pct",
        "cost.daily_usd",
        "cost.monthly_usd",
        "cost.budget_pass",
    ]

    for key in cost_keys:

        if key not in snapshot:
            continue

        value = snapshot[key]

        if isinstance(
            value,
            float,
        ):

            print(
                f"  {key:<38}"
                f"{value:.6f}"
            )

        else:

            print(
                f"  {key:<38}"
                f"{value}"
            )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    print()
    print("[RELIABILITY]")

    reliability_keys = [
        "reliability.total_requests",
        "reliability.successful",
        "reliability.failed",
        "reliability.success_rate",
        "reliability.error_rate",
        "reliability.retry_rate",
        "reliability.first_attempt_success_rate",
        "reliability.recovered_after_retry",
        "reliability.average_attempts_per_request",
    ]

    for key in reliability_keys:

        if key not in snapshot:
            continue

        value = snapshot[key]

        if isinstance(
            value,
            float,
        ):

            print(
                f"  {key:<38}"
                f"{value:.2f}"
            )

        else:

            print(
                f"  {key:<38}"
                f"{value}"
            )

    print()
    print("=" * 78)


# ============================================================
# 29. ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run RepoMind latency, cost "
            "and reliability evaluations."
        )
    )

    # --------------------------------------------------------
    # Collection
    #
    # Support both:
    #
    # --collection
    # --collection-name
    #
    # so old commands and run-suite conventions work.
    # --------------------------------------------------------

    parser.add_argument(
        "--collection",
        dest="collection",
        default=None,
        help=(
            "Chroma collection name."
        ),
    )

    parser.add_argument(
        "--collection-name",
        dest="collection_name",
        default=None,
        help=(
            "Alias for --collection."
        ),
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    parser.add_argument(
        "--lat-repeats",
        type=int,
        default=DEFAULT_LAT_REPEATS,
    )

    parser.add_argument(
        "--lat-warmup",
        type=int,
        default=DEFAULT_LAT_WARMUP_RUNS,
    )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    parser.add_argument(
        "--cost-repeats",
        type=int,
        default=DEFAULT_COST_REPEATS,
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    parser.add_argument(
        "--rel-repeats",
        type=int,
        default=DEFAULT_REL_REPEATS,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
    )

    parser.add_argument(
        "--backoff",
        type=float,
        default=DEFAULT_BACKOFF_BASE_S,
    )

    return parser.parse_args()


# ============================================================
# 30. MAIN
# ============================================================

def main():

    args = parse_args()

    collection_name = (
        args.collection_name
        or args.collection
    )

    if not collection_name:

        print(
            "ERROR: collection name is required."
        )

        print()

        print(
            "Use:"
        )

        print(
            "python evaluations/eval_ops.py "
            "--collection "
            "repomind_e2af1fe6fdf92747"
        )

        raise SystemExit(2)

    print()
    print("=" * 78)
    print(
        "REPO MIND - OPERATIONAL EVALUATION"
    )
    print("=" * 78)

    print(
        f"Collection       : "
        f"{collection_name}"
    )

    print(
        f"Top-k            : "
        f"{args.top_k}"
    )

    print(
        f"Questions        : "
        f"{len(QUESTIONS)}"
    )

    print(
        f"Latency repeats  : "
        f"{args.lat_repeats}"
    )

    print(
        f"Cost repeats     : "
        f"{args.cost_repeats}"
    )

    print(
        f"Reliability reps : "
        f"{args.rel_repeats}"
    )

    print(
        f"Max retries      : "
        f"{args.max_retries}"
    )

    print("=" * 78)

    # --------------------------------------------------------
    # ONE PIPELINE
    # --------------------------------------------------------

    pipeline = RAGPipeline()

    # --------------------------------------------------------
    # ALL OPS
    # --------------------------------------------------------

    snapshot = run_ops(
        pipeline=pipeline,

        collection_name=collection_name,

        top_k=args.top_k,

        lat_repeats=args.lat_repeats,

        lat_warmup_runs=args.lat_warmup,

        cost_repeats=args.cost_repeats,

        rel_repeats=args.rel_repeats,

        max_retries=args.max_retries,

        backoff_base_s=args.backoff,

        verbose=True,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print_operational_snapshot(
        snapshot
    )


# ============================================================
# 31. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()