"""
Stage-level latency evaluation for RepoMind.

Measures:

1. Retrieval latency
2. TTFT (Time To First Token)
3. Generation latency
4. End-to-end latency

The evaluator uses the actual RAGPipeline interface:

    pipeline.retriever(question, collection_name, top_k)

and the existing AnswerGenerator / Ollama streaming interface.
"""

import argparse
import statistics
import sys
import time
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
from src.llm.prompts import build_prompt


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = [
    "What is RepoMind?",
    "Explain the architecture of the repository.",
    "What are the main components of the project?",
    "Explain the RAG pipeline.",
    "How does document retrieval work?",
    "How are documents processed?",
    "How are embeddings generated?",
    "How is PostgreSQL used in the project?",
    "How does the application process a GitHub URL?",
    "How can another repository be added?",
    "What retriever is used?",
    "What happens when there is not enough information?",
    "How does the system restrict answers to repository context?",
    "How is the RAG system evaluated?",
    "Which files contain the main RAG pipeline implementation?",
]


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * (p / 100)

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_stats(name, values):

    if not values:
        print(f"{name:<15} no data")
        return

    print(
        f"{name:<15} "
        f"n={len(values):<3} "
        f"mean={statistics.mean(values):>9.2f} ms "
        f"p50={percentile(values, 50):>9.2f} ms "
        f"p95={percentile(values, 95):>9.2f} ms "
        f"p99={percentile(values, 99):>9.2f} ms "
        f"min={min(values):>9.2f} ms "
        f"max={max(values):>9.2f} ms"
    )


# ============================================================
# MEASURE ONE REQUEST
# ============================================================

def measure_request(
    pipeline,
    question,
    collection_name,
    top_k,
):
    """
    Measure:

        Retrieval
        Prompt construction
        TTFT
        Generation
        E2E
    """

    # --------------------------------------------------------
    # START E2E TIMER
    # --------------------------------------------------------

    e2e_start = time.perf_counter()

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    # IMPORTANT:
    #
    # Your actual RAGPipeline defines:
    #
    # self.retriever = retriever
    #
    # and run() calls:
    #
    # self.retriever(question, collection_name, top_k)
    #
    documents = pipeline.retriever(
        question,
        collection_name,
        top_k,
    )

    retrieval_end = time.perf_counter()

    retrieval_ms = (
        retrieval_end - retrieval_start
    ) * 1000

    # --------------------------------------------------------
    # NO DOCUMENTS
    # --------------------------------------------------------

    if not documents:

        answer = (
            "I couldn't find enough information in the repository."
        )

        e2e_end = time.perf_counter()

        e2e_ms = (
            e2e_end - e2e_start
        ) * 1000

        return {
            "retrieval_ms": retrieval_ms,
            "prompt_ms": 0.0,
            "ttft_ms": None,
            "generation_ms": 0.0,
            "e2e_ms": e2e_ms,
            "answer": answer,
            "num_documents": 0,
        }

    # --------------------------------------------------------
    # PROMPT CONSTRUCTION
    # --------------------------------------------------------

    prompt_start = time.perf_counter()

    prompt = build_prompt(
        question,
        documents,
    )

    prompt_end = time.perf_counter()

    prompt_ms = (
        prompt_end - prompt_start
    ) * 1000

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    generator = pipeline.generator

    llm = generator.llm

    # --------------------------------------------------------
    # GENERATION START
    # --------------------------------------------------------

    generation_start = time.perf_counter()

    first_token_time = None
    last_token_time = None

    chunks = []

    # --------------------------------------------------------
    # STREAM LLM RESPONSE
    # --------------------------------------------------------

    for chunk in llm.stream(prompt):

        now = time.perf_counter()

        content = ""

        if hasattr(chunk, "content"):
            content = chunk.content

        elif isinstance(chunk, str):
            content = chunk

        if not content:
            continue

        # First non-empty chunk
        if first_token_time is None:
            first_token_time = now

        # Latest chunk
        last_token_time = now

        chunks.append(content)

    # --------------------------------------------------------
    # GENERATION END
    # --------------------------------------------------------

    generation_end = time.perf_counter()

    # --------------------------------------------------------
    # TTFT
    # --------------------------------------------------------

    if first_token_time is not None:

        ttft_ms = (
            first_token_time - generation_start
        ) * 1000

    else:

        ttft_ms = None

    # --------------------------------------------------------
    # GENERATION LATENCY
    #
    # From first generation request until final chunk.
    # --------------------------------------------------------

    if last_token_time is not None:

        generation_ms = (
            last_token_time - generation_start
        ) * 1000

    else:

        generation_ms = (
            generation_end - generation_start
        ) * 1000

    # --------------------------------------------------------
    # E2E END
    # --------------------------------------------------------

    e2e_end = time.perf_counter()

    e2e_ms = (
        e2e_end - e2e_start
    ) * 1000

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    answer = "".join(chunks)

    return {
        "retrieval_ms": retrieval_ms,
        "prompt_ms": prompt_ms,
        "ttft_ms": ttft_ms,
        "generation_ms": generation_ms,
        "e2e_ms": e2e_ms,
        "answer": answer,
        "num_documents": len(documents),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="RepoMind stage-level latency evaluation"
    )

    parser.add_argument(
        "--collection",
        default="repomind_e2af1fe6fdf92747",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--slo",
        type=float,
        default=3000,
        help="E2E P95 SLO in milliseconds",
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 80)
    print("RepoMind Stage-Level Latency Evaluation")
    print("=" * 80)

    print(f"Questions       : {len(QUESTIONS)}")
    print(f"Repeats         : {args.repeats}")
    print(f"Warmup          : {args.warmup}")
    print(f"Top-K           : {args.top_k}")
    print(f"Collection      : {args.collection}")
    print(f"E2E SLO P95     : {args.slo:.0f} ms")
    print()

    # ========================================================
    # PIPELINE
    # ========================================================

    print("Initializing RAG pipeline...")

    pipeline = RAGPipeline()

    print("Pipeline initialized.")
    print()

    # ========================================================
    # WARMUP
    # ========================================================

    print("=" * 80)
    print("WARMUP")
    print("=" * 80)

    for i in range(args.warmup):

        print(
            f"Warmup {i + 1}/{args.warmup}...",
            end=" ",
            flush=True,
        )

        try:

            result = measure_request(
                pipeline,
                QUESTIONS[0],
                args.collection,
                args.top_k,
            )

            print(
                f"{result['e2e_ms']:.1f} ms"
            )

        except Exception as e:

            print(
                f"FAILED: {e}"
            )

    print()

    # ========================================================
    # STORAGE
    # ========================================================

    retrieval_times = []
    prompt_times = []
    ttft_times = []
    generation_times = []
    e2e_times = []

    successful = 0
    failed = 0

    per_question = {}

    total_runs = (
        len(QUESTIONS)
        * args.repeats
    )

    run_number = 0

    # ========================================================
    # BENCHMARK
    # ========================================================

    print("=" * 80)
    print("BENCHMARK")
    print("=" * 80)

    for question_index, question in enumerate(
        QUESTIONS,
        start=1,
    ):

        print()
        print(
            f"Q{question_index:02d}: {question}"
        )

        question_results = []

        for repeat in range(
            1,
            args.repeats + 1,
        ):

            run_number += 1

            print(
                f"  Run {repeat}/{args.repeats} "
                f"[{run_number}/{total_runs}] ... ",
                end="",
                flush=True,
            )

            try:

                result = measure_request(
                    pipeline,
                    question,
                    args.collection,
                    args.top_k,
                )

                retrieval_ms = result[
                    "retrieval_ms"
                ]

                prompt_ms = result[
                    "prompt_ms"
                ]

                ttft_ms = result[
                    "ttft_ms"
                ]

                generation_ms = result[
                    "generation_ms"
                ]

                e2e_ms = result[
                    "e2e_ms"
                ]

                # Store
                retrieval_times.append(
                    retrieval_ms
                )

                prompt_times.append(
                    prompt_ms
                )

                generation_times.append(
                    generation_ms
                )

                e2e_times.append(
                    e2e_ms
                )

                if ttft_ms is not None:
                    ttft_times.append(
                        ttft_ms
                    )

                successful += 1

                question_results.append(
                    result
                )

                ttft_display = (
                    f"{ttft_ms:.1f}"
                    if ttft_ms is not None
                    else "N/A"
                )

                print(
                    f"retrieval={retrieval_ms:.1f} ms | "
                    f"prompt={prompt_ms:.1f} ms | "
                    f"TTFT={ttft_display} ms | "
                    f"generation={generation_ms:.1f} ms | "
                    f"E2E={e2e_ms:.1f} ms"
                )

            except Exception as e:

                failed += 1

                print(
                    f"FAILED: {e}"
                )

        per_question[question] = (
            question_results
        )

    # ========================================================
    # AGGREGATE
    # ========================================================

    print()
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    print()

    print_stats(
        "Retrieval",
        retrieval_times,
    )

    print_stats(
        "Prompt",
        prompt_times,
    )

    print_stats(
        "TTFT",
        ttft_times,
    )

    print_stats(
        "Generation",
        generation_times,
    )

    print_stats(
        "E2E",
        e2e_times,
    )

    # ========================================================
    # RUN COUNTS
    # ========================================================

    print()
    print("-" * 80)

    print(
        f"Successful runs : {successful}"
    )

    print(
        f"Failed runs     : {failed}"
    )

    print(
        f"Total runs      : {total_runs}"
    )

    # ========================================================
    # SLO
    # ========================================================

    if e2e_times:

        e2e_p95 = percentile(
            e2e_times,
            95,
        )

        print()
        print(
            f"E2E P95         : "
            f"{e2e_p95:.2f} ms"
        )

        print(
            f"E2E P95 SLO     : "
            f"{args.slo:.2f} ms"
        )

        if e2e_p95 <= args.slo:

            print(
                "SLO STATUS      : PASS"
            )

        else:

            print(
                "SLO STATUS      : FAIL"
            )

    # ========================================================
    # PER QUESTION
    # ========================================================

    print()
    print("=" * 80)
    print("PER-QUESTION SUMMARY")
    print("=" * 80)

    for index, question in enumerate(
        QUESTIONS,
        start=1,
    ):

        results = per_question.get(
            question,
            [],
        )

        if not results:
            continue

        q_retrieval = [
            r["retrieval_ms"]
            for r in results
        ]

        q_prompt = [
            r["prompt_ms"]
            for r in results
        ]

        q_ttft = [
            r["ttft_ms"]
            for r in results
            if r["ttft_ms"] is not None
        ]

        q_generation = [
            r["generation_ms"]
            for r in results
        ]

        q_e2e = [
            r["e2e_ms"]
            for r in results
        ]

        print()
        print(
            f"Q{index:02d}: {question}"
        )

        print(
            f"  Retrieval : "
            f"{statistics.mean(q_retrieval):.2f} ms"
        )

        print(
            f"  Prompt    : "
            f"{statistics.mean(q_prompt):.2f} ms"
        )

        if q_ttft:

            print(
                f"  TTFT      : "
                f"{statistics.mean(q_ttft):.2f} ms"
            )

        else:

            print(
                "  TTFT      : N/A"
            )

        print(
            f"  Generation: "
            f"{statistics.mean(q_generation):.2f} ms"
        )

        print(
            f"  E2E       : "
            f"{statistics.mean(q_e2e):.2f} ms"
        )

    # ========================================================
    # BREAKDOWN
    # ========================================================

    print()
    print("=" * 80)
    print("LATENCY BREAKDOWN")
    print("=" * 80)

    if e2e_times:

        avg_retrieval = statistics.mean(
            retrieval_times
        )

        avg_prompt = statistics.mean(
            prompt_times
        )

        avg_ttft = (
            statistics.mean(ttft_times)
            if ttft_times
            else 0
        )

        avg_generation = statistics.mean(
            generation_times
        )

        avg_e2e = statistics.mean(
            e2e_times
        )

        print()
        print(
            f"Average retrieval : "
            f"{avg_retrieval:.2f} ms"
        )

        print(
            f"Average prompt    : "
            f"{avg_prompt:.2f} ms"
        )

        print(
            f"Average TTFT      : "
            f"{avg_ttft:.2f} ms"
        )

        print(
            f"Average generation: "
            f"{avg_generation:.2f} ms"
        )

        print(
            f"Average E2E       : "
            f"{avg_e2e:.2f} ms"
        )

        print()

        if avg_e2e > 0:

            print(
                f"Retrieval share   : "
                f"{avg_retrieval / avg_e2e * 100:.2f}%"
            )

            print(
                f"Prompt share      : "
                f"{avg_prompt / avg_e2e * 100:.2f}%"
            )

            print(
                f"Generation share  : "
                f"{avg_generation / avg_e2e * 100:.2f}%"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()