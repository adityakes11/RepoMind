import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
load_dotenv()
def get_embeddings():
	timeout = float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "120"))
	return OllamaEmbeddings(
		model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
		base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
		client_kwargs={"timeout": timeout},
	)
