import hashlib
import os
from langchain_chroma import Chroma
from src.rag.embeddings import get_embeddings
CHROMA_PATH="data/chroma"
def get_collection_name(github_url):
    url=github_url.strip().lower().rstrip("/")
    if url.endswith(".git"):url=url[:-4]
    return "repomind_"+hashlib.sha256(url.encode()).hexdigest()[:16]
def create_vector_store(github_url, documents, progress_callback=None):
    name=get_collection_name(github_url)
    store=Chroma(persist_directory=CHROMA_PATH,embedding_function=get_embeddings(),collection_name=name)
    if store._collection.count()==0 and documents:
        batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
        total = len(documents)
        for start in range(0, total, batch_size):
            batch = documents[start:start + batch_size]
            store.add_documents(batch)
            if progress_callback:
                progress_callback(min(start + len(batch), total), total)
    return store
def load_vector_store(collection_name):
    return Chroma(persist_directory=CHROMA_PATH,embedding_function=get_embeddings(),collection_name=collection_name)
def collection_exists(github_url):
    try:return load_vector_store(get_collection_name(github_url))._collection.count()>0
    except Exception:return False
