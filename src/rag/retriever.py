from langsmith import traceable

from src.rag.vector_store import load_vector_store


@traceable(run_type="chain", name="Retriever")
def retrieve_documents(question,collection_name,k=5):
    return load_vector_store(collection_name).similarity_search(question,k=k)
