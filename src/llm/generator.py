from src.llm.ollama import get_llm
from src.llm.prompts import build_prompt
class AnswerGenerator:
    def __init__(self): self.llm=get_llm()
    def generate(self,question,documents):
        if not documents:return {"answer":"I couldn't find enough information in the repository.","sources":[]}
        response=self.llm.invoke(build_prompt(question,documents))
        return {"answer":response.content,"sources":self._extract_sources(documents)}
    def _extract_sources(self,documents):
        seen=[]
        for d in documents:
            f=d.metadata.get("file","unknown")
            if f not in seen:seen.append(f)
        return seen
