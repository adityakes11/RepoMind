"""Cross-encoder reranking for repository retrieval."""

from sentence_transformers import CrossEncoder

from src.rag.vector_store import load_vector_store


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(self, collection_name, fetch_k=10, top_k=5, model_name=CROSS_ENCODER_MODEL):
        if fetch_k < top_k:
            raise ValueError("fetch_k must be greater than or equal to top_k")
        self.store = load_vector_store(collection_name)
        self.reranker = CrossEncoder(model_name)
        self.fetch_k = fetch_k
        self.top_k = top_k

    def invoke(self, query):
        candidates = self.store.similarity_search(query, k=self.fetch_k)
        if not candidates:
            return []

        pairs = [(query, document.page_content) for document in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        return [document for document, _ in ranked[: self.top_k]]
