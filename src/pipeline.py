"""End-to-end retrieval-augmented generation pipeline."""

from langsmith import traceable

from src.llm.generator import AnswerGenerator
from src.rag.retriever import retrieve_documents


class RAGPipeline:
    """Connect repository retrieval with answer generation."""

    def __init__(self, generator=None, retriever=retrieve_documents):
        self.generator = generator or AnswerGenerator()
        self.retriever = retriever

    @traceable(run_type="chain", name="RagPipeline")
    def run(self, question, collection_name, top_k=3):
        documents = self.retriever(question, collection_name, top_k)
        result = self.generator.generate(question, documents)
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "documents": documents,
        }
