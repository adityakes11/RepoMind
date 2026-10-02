"""
Operational evaluation: RELIABILITY

Reliability measures whether the RepoMind RAG application can
successfully serve requests without failing.

We measure:

    - success rate
    - error rate
    - retry rate
    - first-attempt success rate
    - recovered request rate
    - average attempts per request
    - maximum attempts used

A request is considered successful if the complete RepoMind request
eventually succeeds within the configured retry limit.

Retries are important because a system may eventually succeed while
still failing on its first attempt. Therefore this evaluator keeps
first-attempt failures separate from final request failures.

RepoMind request flow:

    Question
        |
        v
    Retriever
        |
        v
    Retrieved documents
        |
        v
    AnswerGenerator
        |
        v
    Ollama LLM
        |
        v
    Final answer

The retry wrapper covers the complete retrieval + generation operation.

This means a failure in either retrieval or generation is treated as
a failed request attempt.
"""

# ============================================================
# 1. IMPORTS & PROJECT PATH
# ============================================================

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


# ------------------------------------------------------------
# Make the RepoMind project root importable.
#
# This allows:
#
#     python evaluations/eval_reliability.py
#
# to work without manually setting PYTHONPATH.
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline import RAGPipeline


load_dotenv()


# ============================================================
# 2. CONFIG
# ============================================================

QUESTIONS = [
    "What is RepoMind?",
    "Explain the architecture of RepoMind.",
    "What are the main components of RepoMind?",
    "How does the RAG pipeline work in RepoMind?",
    "How does document retrieval work?",
    "How are documents processed before being stored?",
    "How are embeddings generated and used?",
    "Why does RepoMind use PostgreSQL?",
    "How does RepoMind process a GitHub repository URL?",
    "How can another repository be added to RepoMind?",
    "What is the role of the retriever?",
    "What happens when there is not enough information in the repository?",
    "How does RepoMind restrict answers to repository context?",
    "How is RepoMind evaluated?",
    "Which files implement the main RepoMind functionality?",
]


# Number of benchmark requests per question.
REPEATS = 5


# Maximum number of retries after the initial attempt.
#
# MAX_RETRIES = 2 means:
#
#   attempt 1 -> initial request
#   attempt 2 -> retry 1
#   attempt 3 -> retry 2
#
MAX_RETRIES = 2


# Exponential backoff:
#
# retry 1 -> 0.5 seconds
# retry 2 -> 1.0 seconds
#
BACKOFF_BASE_S = 0.5


# Same retrieval configuration used by your other
# RepoMind operational evaluators.
DEFAULT_TOP_K = 3


# ============================================================
# 3. RELIABILITY TRACKER
# ============================================================

class Reliability:
    """
    Track reliability statistics for the benchmark.
    """

    def __init__(self):

        # Number of logical user requests.
        self.calls = 0

        # Requests that eventually succeeded.
        self.successes = 0

        # Requests that failed even after all retries.
        self.failures = 0

        # Number of retry attempts performed.
        self.retries = 0

        # Number of requests that failed initially but recovered.
        self.recovered = 0

        # Number of requests that succeeded on their first attempt.
        self.first_attempt_successes = 0

        # Number of individual attempts made.
        self.total_attempts = 0

        # Track attempts used by each logical request.
        self.attempts_per_request = []

        # Store failure messages for debugging.
        self.errors = []


# ============================================================
# 4. COMPLETE REPO MIND REQUEST
# ============================================================

def run_request(
    pipeline,
    question,
    collection_name,
    top_k,
):
    """
    Execute one complete RepoMind request.

    This deliberately does not use pipeline.run() so that the
    evaluator follows the actual current RepoMind architecture.

    RAGPipeline.run() internally performs:

        retriever(...)
        generator.generate(...)

    We reproduce that operation here so that exceptions from
    retrieval or generation can be caught by the reliability
    evaluator.
    """

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    documents = pipeline.retriever(
        question,
        collection_name,
        top_k,
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    result = pipeline.generator.generate(
        question,
        documents,
    )

    # --------------------------------------------------------
    # Basic validation
    #
    # A successful model call should return a dictionary
    # containing the expected answer field.
    # --------------------------------------------------------

    if not isinstance(result, dict):
        raise RuntimeError(
            "RAGPipeline generator returned a non-dictionary result."
        )

    if "answer" not in result:
        raise RuntimeError(
            "RAGPipeline generator result does not contain 'answer'."
        )

    return result


# ============================================================
# 5. RETRY WRAPPER
# ============================================================

def call_with_retries(
    fn,
    reliability,
):
    """
    Execute a logical request with exponential-backoff retries.

    Returns:

        {
            "success": bool,
            "attempts": int,
            "retries_used": int,
            "error": str | None,
            "recovered": bool
        }
    """

    reliability.calls += 1

    attempts = 0
    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        attempts += 1

        reliability.total_attempts += 1

        try:

            result = fn()

            # ------------------------------------------------
            # Successful request
            # ------------------------------------------------

            reliability.successes += 1

            # First-attempt success.
            if attempt == 0:

                reliability.first_attempt_successes += 1

            # Request initially failed but succeeded later.
            elif attempt > 0:

                reliability.recovered += 1

            reliability.attempts_per_request.append(
                attempts
            )

            return {
                "success": True,
                "attempts": attempts,
                "retries_used": attempt,
                "error": None,
                "recovered": attempt > 0,
                "result": result,
            }

        except Exception as exc:

            last_error = exc

            # ------------------------------------------------
            # There are still retries available.
            # ------------------------------------------------

            if attempt < MAX_RETRIES:

                reliability.retries += 1

                sleep_time = (
                    BACKOFF_BASE_S
                    * (2 ** attempt)
                )

                print(
                    f"    attempt {attempt + 1} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"    retrying in {sleep_time:.2f}s..."
                )

                time.sleep(sleep_time)

            # ------------------------------------------------
            # All attempts exhausted.
            # ------------------------------------------------

            else:

                reliability.failures += 1

                reliability.attempts_per_request.append(
                    attempts
                )

                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )

                reliability.errors.append(
                    error_message
                )

                print(
                    f"    FAILED after "
                    f"{MAX_RETRIES} retries: "
                    f"{error_message}"
                )

                return {
                    "success": False,
                    "attempts": attempts,
                    "retries_used": MAX_RETRIES,
                    "error": error_message,
                    "recovered": False,
                    "result": None,
                }

    # This should never be reached.
    reliability.failures += 1

    reliability.attempts_per_request.append(
        attempts
    )

    return {
        "success": False,
        "attempts": attempts,
        "retries_used": MAX_RETRIES,
        "error": str(last_error),
        "recovered": False,
        "result": None,
    }


# ============================================================
# 6. BENCHMARK
# ============================================================

def benchmark(
    pipeline,
    collection_name,
    top_k,
    repeats,
):
    """
    Run the reliability benchmark.
    """

    reliability = Reliability()

    total_requests = (
        len(QUESTIONS)
        * repeats
    )

    print("=" * 70)
    print("RepoMind RELIABILITY Evaluation")
    print("=" * 70)

    print(
        f"Questions       : {len(QUESTIONS)}"
    )

    print(
        f"Repeats         : {repeats}"
    )

    print(
        f"Total requests  : {total_requests}"
    )

    print(
        f"Collection      : {collection_name}"
    )

    print(
        f"Top-k           : {top_k}"
    )

    print(
        f"Max retries     : {MAX_RETRIES}"
    )

    print(
        f"Backoff base    : {BACKOFF_BASE_S}s"
    )

    print()
    print("Measuring reliability...")
    print()

    request_number = 0

    for question in QUESTIONS:

        for repeat in range(repeats):

            request_number += 1

            print(
                f"[{request_number:02d}/{total_requests}] "
                f"{question}"
            )

            outcome = call_with_retries(
                lambda q=question: run_request(
                    pipeline=pipeline,
                    question=q,
                    collection_name=collection_name,
                    top_k=top_k,
                ),
                reliability,
            )

            if outcome["success"]:

                if outcome["recovered"]:

                    print(
                        f"    RECOVERED "
                        f"after {outcome['attempts']} attempts"
                    )

                else:

                    print(
                        "    SUCCESS on first attempt"
                    )

            else:

                print(
                    "    FINAL FAILURE"
                )

            print()

    return reliability


# ============================================================
# 7. REPORT
# ============================================================

def report(rel):
    """
    Print aggregate reliability statistics.
    """

    total_requests = rel.calls

    # --------------------------------------------------------
    # Success rate
    # --------------------------------------------------------

    success_rate = (
        100 * rel.successes / total_requests
        if total_requests
        else 0
    )

    # --------------------------------------------------------
    # Final error rate
    #
    # This measures requests that remained failed after all
    # configured retries.
    # --------------------------------------------------------

    error_rate = (
        100 * rel.failures / total_requests
        if total_requests
        else 0
    )

    # --------------------------------------------------------
    # Retry rate
    #
    # Percentage of logical requests that required at least
    # one retry.
    #
    # We derive this from recovered + final failures because
    # both categories required retries.
    # --------------------------------------------------------

    requests_requiring_retry = (
        rel.recovered
        + (
            rel.failures
            if rel.failures
            else 0
        )
    )

    retry_rate = (
        100
        * requests_requiring_retry
        / total_requests
        if total_requests
        else 0
    )

    # --------------------------------------------------------
    # First attempt success rate
    # --------------------------------------------------------

    first_attempt_success_rate = (
        100
        * rel.first_attempt_successes
        / total_requests
        if total_requests
        else 0
    )

    # --------------------------------------------------------
    # Recovery rate
    #
    # Of all requests that needed retries, how many eventually
    # recovered?
    # --------------------------------------------------------

    requests_that_needed_retry = (
        requests_requiring_retry
    )

    recovery_rate = (
        100
        * rel.recovered
        / requests_that_needed_retry
        if requests_that_needed_retry
        else 0
    )

    # --------------------------------------------------------
    # Average attempts
    # --------------------------------------------------------

    average_attempts = (
        sum(rel.attempts_per_request)
        / len(rel.attempts_per_request)
        if rel.attempts_per_request
        else 0
    )

    # --------------------------------------------------------
    # Maximum attempts
    # --------------------------------------------------------

    max_attempts_used = (
        max(rel.attempts_per_request)
        if rel.attempts_per_request
        else 0
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RELIABILITY")
    print("=" * 70)

    print(
        f"total requests          : "
        f"{total_requests}"
    )

    print(
        f"successful              : "
        f"{rel.successes}"
    )

    print(
        f"failed                  : "
        f"{rel.failures}"
    )

    print(
        f"first-attempt success   : "
        f"{rel.first_attempt_successes}"
    )

    print(
        f"recovered after retry   : "
        f"{rel.recovered}"
    )

    print(
        f"total retry attempts    : "
        f"{rel.retries}"
    )

    print(
        f"total attempts          : "
        f"{rel.total_attempts}"
    )

    print("-" * 70)

    print(
        f"success rate             : "
        f"{success_rate:.2f}%"
    )

    print(
        f"final error rate        : "
        f"{error_rate:.2f}%"
    )

    print(
        f"retry rate              : "
        f"{retry_rate:.2f}%"
    )

    print(
        f"first-attempt success   : "
        f"{first_attempt_success_rate:.2f}%"
    )

    print(
        f"retry recovery rate     : "
        f"{recovery_rate:.2f}%"
    )

    print(
        f"average attempts/request: "
        f"{average_attempts:.2f}"
    )

    print(
        f"maximum attempts used   : "
        f"{max_attempts_used}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Failure details
    # --------------------------------------------------------

    if rel.errors:

        print()
        print("FAILURE DETAILS")
        print("-" * 70)

        for index, error in enumerate(
            rel.errors,
            start=1,
        ):

            print(
                f"{index}. {error}"
            )

        print("=" * 70)

    else:

        print()
        print(
            "No final request failures recorded."
        )

        print("=" * 70)


# ============================================================
# 8. ARGUMENT PARSER
# ============================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Measure RepoMind request reliability "
            "with retries."
        )
    )

    parser.add_argument(
        "--collection",
        required=True,
        help="Chroma collection name.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"Number of retrieved documents. "
            f"Default: {DEFAULT_TOP_K}"
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help=(
            f"Number of requests per question. "
            f"Default: {REPEATS}"
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=(
            f"Maximum retries after an initial failure. "
            f"Default: {MAX_RETRIES}"
        ),
    )

    parser.add_argument(
        "--backoff",
        type=float,
        default=BACKOFF_BASE_S,
        help=(
            f"Base exponential backoff in seconds. "
            f"Default: {BACKOFF_BASE_S}"
        ),
    )

    return parser.parse_args()


# ============================================================
# 9. ENTRYPOINT
# ============================================================

def main():

    global MAX_RETRIES
    global BACKOFF_BASE_S

    args = parse_args()

    MAX_RETRIES = args.max_retries

    BACKOFF_BASE_S = args.backoff

    print()
    print("Initializing RepoMind pipeline...")

    pipeline = RAGPipeline()

    reliability = benchmark(
        pipeline=pipeline,
        collection_name=args.collection,
        top_k=args.top_k,
        repeats=args.repeats,
    )

    report(reliability)


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":
    main()