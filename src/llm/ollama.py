import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()
def get_llm():
    return ChatOllama(model=os.getenv("LLM_MODEL","qwen2.5:7b"),temperature=0,base_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))
