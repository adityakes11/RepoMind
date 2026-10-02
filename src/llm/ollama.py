import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langsmith import traceable
from src.config import settings

load_dotenv()
@traceable(run_type="chain", name="CreateOllamaLLM")
def get_llm():
    return ChatOllama(model=os.getenv("LLM_MODEL","qwen2.5:7b"),temperature=0,base_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))


@traceable(run_type="chain", name="CreateOllamaLLM")
def get_llm(model=None, base_url=None, temperature=0):
    return ChatOllama(
        model=model or settings.ollama.llm_model,
        temperature=temperature,
        base_url=base_url or settings.ollama.base_url,
    )
