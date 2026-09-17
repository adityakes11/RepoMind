from src.rag.vector_store import load_vector_store
def retrieve_documents(question,collection_name,k=5):
    return load_vector_store(collection_name).similarity_search(question,k=k)
