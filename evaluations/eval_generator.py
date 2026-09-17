"""Evaluate generator faithfulness and answer relevance in isolation."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document

from src.llm.generator import AnswerGenerator

load_dotenv()
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")

GOLDEN_PATH = Path(__file__).parents[1] / "golden test" / "faithfulness_goldens.json"
DEFAULT_GENERATOR_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_GENERATOR_MODEL)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_THRESHOLD = 0.7


def load_goldens(path=GOLDEN_PATH):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def golden_documents(golden):
    source_files = golden.get("source_files", [])
    return [
        Document(
            page_content=context,
            metadata={"file": source_files[index] if index < len(source_files) else "golden_context"},
        )
        for index, context in enumerate(golden["ideal_context"])
    ]


def build_test_cases(generator, goldens):
    test_cases = []
    for golden in goldens:
        documents = golden_documents(golden)
        result = generator.generate(golden["query"], documents)
        answer = result["answer"]
        print(f"{golden['id']}: {golden['query']}")
        print(f"  generated sources: {result['sources']}")
        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output=answer,
                retrieval_context=[document.page_content for document in documents],
            )
        )
    return test_cases


def run(judge_model=DEFAULT_JUDGE_MODEL, threshold=DEFAULT_THRESHOLD):
    goldens = load_goldens()
    generator = AnswerGenerator()
    test_cases = build_test_cases(generator, goldens)
    local_judge = OllamaModel(
        model=judge_model,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={"think": False, "num_predict": 2048},
    )
    metrics = [
        FaithfulnessMetric(threshold=threshold, model=local_judge, include_reason=False, async_mode=False),
        AnswerRelevancyMetric(threshold=threshold, model=local_judge, include_reason=False, async_mode=False),
    ]
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
        hyperparameters={
            "generator": "src.llm.generator.AnswerGenerator",
            "generator_model": os.getenv("LLM_MODEL", "qwen2.5:7b"),
            "judge_model": judge_model,
            "judge_provider": "ollama",
            "ollama_base_url": OLLAMA_BASE_URL,
            "threshold": threshold,
            "golden_set": str(GOLDEN_PATH),
            "evaluation_mode": "golden_context_isolation",
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Golden set: {GOLDEN_PATH}")
    print(f"Generator model: {DEFAULT_GENERATOR_MODEL}")
    print(f"Judge model: {args.judge_model}")
    try:
        evaluate_result = run(args.judge_model, args.threshold)
    except Exception as error:
        print("\nGenerator evaluation could not complete with the local Ollama judge.")
        print(f"Reason: {type(error).__name__}: {error}")
        print("Check Ollama availability and use the deterministic retriever evaluation separately.")
