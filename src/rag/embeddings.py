import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langsmith import traceable
from src.config import settings

load_dotenv()
@traceable(run_type="chain", name="CreateEmbeddings")
def get_embeddings():
	timeout = float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "120"))
	return OllamaEmbeddings(
		model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
		base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
		client_kwargs={"timeout": timeout},
	)


@traceable(run_type="chain", name="CreateEmbeddings")
def get_embeddings(model=None, base_url=None, timeout=None):
    return OllamaEmbeddings(
        model=model or settings.ollama.embedding_model,
        base_url=base_url or settings.ollama.base_url,
        client_kwargs={"timeout": timeout or settings.ollama.embedding_timeout},
    )
