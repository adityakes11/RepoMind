from langsmith import traceable

from src.llm.ollama import get_llm
from src.llm.prompts import build_prompt


class AnswerGenerator:

    def __init__(self):
        self.llm = get_llm()

    # Normal generation
    @traceable(run_type="chain", name="AnswerGenerator")
    def generate(self, question, documents):
        if not documents:
            return {
                "answer": "I couldn't find enough information in the repository.",
                "sources": []
            }

        prompt = build_prompt(question, documents)

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": self._extract_sources(documents)
        }

    # Streaming generation
    # Useful for measuring TTFT (Time To First Token/Chunk)
    @traceable(run_type="chain", name="AnswerGeneratorStream")
    def generate_stream(self, question, documents):

        if not documents:
            yield "I couldn't find enough information in the repository."
            return

        prompt = build_prompt(question, documents)

        for chunk in self.llm.stream(prompt):

            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def _extract_sources(self, documents):
        seen = []

        for document in documents:
            file_name = document.metadata.get("file", "unknown")

            if file_name not in seen:
                seen.append(file_name)

        return seen