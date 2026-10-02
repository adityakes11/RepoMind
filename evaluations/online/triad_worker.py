"""Score recent RepoMind LangSmith traces with a local Ollama RAG Triad judge.

Run one polling pass:
    python -m evaluations.online.triad_worker --once

Run continuously:
    python -m evaluations.online.triad_worker

The worker reads the ``RagPipeline`` root traces produced by RepoMind and
attaches DeepEval feedback to those same LangSmith runs. It is intentionally
idempotent per metric key: an existing feedback key is never rescored.
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langsmith import Client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PROJECT = os.getenv("LANGSMITH_PROJECT", "RepoMind")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", os.getenv("LLM_MODEL", "qwen2.5:7b"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
THRESHOLD = float(os.getenv("LANGSMITH_TRIAD_THRESHOLD", "0.7"))
SAMPLE_RATE = float(os.getenv("LANGSMITH_TRIAD_SAMPLE_RATE", "0.3"))
POLL_SECONDS = int(os.getenv("LANGSMITH_TRIAD_POLL_SECONDS", "60"))


client = Client()


def _sampled(run):
    """Use a stable sample decision across worker restarts."""
    digest = hashlib.sha256(str(run.id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < SAMPLE_RATE


def _existing_keys(run):
    feedback = client.list_feedback(run_ids=[run.id])
    return {item.key for item in feedback}


def _document_text(document):
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        return document.get("page_content") or document.get("content") or ""
    return getattr(document, "page_content", "") or ""


def _context_from_run(run):
    outputs = run.outputs or {}
    raw_documents = outputs.get("documents") or outputs.get("context") or []
    if not raw_documents and run.name == "ChatStream":
        for child in client.list_runs(parent_run_id=run.id):
            if child.name == "Retriever":
                raw_documents = (child.outputs or {}).get("output") or []
                break
    if isinstance(raw_documents, str):
        return [raw_documents]
    return [text for text in (_document_text(item) for item in raw_documents) if text]


def _question_from_run(run):
    inputs = run.inputs or {}
    return inputs.get("question") or inputs.get("query") or ""


def _answer_from_run(run):
    outputs = run.outputs or {}
    if outputs.get("answer"):
        return outputs["answer"]
    raw_output = outputs.get("output") or ""
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, list):
        return "".join(item if isinstance(item, str) else _document_text(item) for item in raw_output)
    return ""


def _build_judge():
    return OllamaModel(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={"think": False, "num_predict": 2048},
    )


def _jobs(judge):
    return [
        (
            "faithfulness",
            FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
            lambda question, answer, context: dict(
                input=question,
                actual_output=answer,
                retrieval_context=context,
            ),
        ),
        (
            "answer_relevancy",
            AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
            lambda question, answer, context: dict(input=question, actual_output=answer),
        ),
        (
            "contextual_relevancy",
            ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
            lambda question, answer, context: dict(
                input=question,
                actual_output=answer,
                retrieval_context=context,
            ),
        ),
    ]


def score_recent_traces():
    """Score missing RAG Triad metrics on recent ``RagPipeline`` root traces."""
    scored = 0
    skipped = 0
    runs = client.list_runs(project_name=PROJECT, is_root=True, run_type="chain")
    judge = None

    for run in runs:
        if run.name not in {"RagPipeline", "ChatStream"} or not _sampled(run):
            skipped += 1
            continue

        outputs = run.outputs or {}
        answer = _answer_from_run(run)
        context = _context_from_run(run)
        question = _question_from_run(run)
        if not answer or not question or not context:
            skipped += 1
            continue

        if judge is None:
            judge = _build_judge()
        existing = _existing_keys(run)

        for key, metric, make_case in _jobs(judge):
            if key in existing:
                continue
            try:
                metric.measure(LLMTestCase(**make_case(question, answer, context)))
                if metric.score is None:
                    raise ValueError("DeepEval returned no score")
                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=float(metric.score),
                    comment=metric.reason or "",
                )
                scored += 1
                print(f"[{key}] run={run.id} score={metric.score:.3f}")
            except Exception as exc:
                print(f"[{key}] failed for run {run.id}: {exc}")

    print(f"Triad pass complete: scored={scored} skipped={skipped} project={PROJECT}")
    return {"scored": scored, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one pass instead of polling continuously")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args()

    if args.once:
        score_recent_traces()
        return

    while True:
        score_recent_traces()
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()