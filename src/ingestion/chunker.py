import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from src.config import settings
from src.ingestion.code_parser import read_file,get_language
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=150

DEFAULT_CHUNK_SIZE = settings.ingestion.chunk_size
DEFAULT_CHUNK_OVERLAP = settings.ingestion.chunk_overlap
FUNCTION_BOUNDARY=re.compile(
    r"(?m)^(?:\s*@[^\n]+\n)*\s*(?:async\s+def|def|class|function|export\s+(?:default\s+)?(?:function|class))\b"
)

@traceable(run_type="chain", name="SplitCodeUnits")
def split_code_units(content):
    boundaries=[match.start() for match in FUNCTION_BOUNDARY.finditer(content)]
    if not boundaries or boundaries[0] != 0:
        boundaries=[0]+boundaries
    return [content[start:end] for start,end in zip(boundaries,boundaries[1:]+[len(content)]) if content[start:end].strip()]

@traceable(run_type="chain", name="CreateCodeChunks")
def create_chunks(repo_path,files,chunk_size=None,chunk_overlap=None):
    chunk_size=chunk_size or int(os.getenv("CHUNK_SIZE",DEFAULT_CHUNK_SIZE))
    chunk_overlap=chunk_overlap or int(os.getenv("CHUNK_OVERLAP",DEFAULT_CHUNK_OVERLAP))
    chunk_size=chunk_size or settings.ingestion.chunk_size
    chunk_overlap=chunk_overlap or settings.ingestion.chunk_overlap
    if chunk_size <= chunk_overlap:
        raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\nclass ","\ndef ","\nfunction ","\n\n","\n"," ",""])
    docs=[]
    for path in files:
        content=read_file(path)
        if not content.strip():continue
        rel=os.path.relpath(path,repo_path); lang=get_language(path)
        chunk_id=0
        for unit in split_code_units(content):
            for chunk in splitter.split_text(unit):
                docs.append(Document(page_content=chunk,metadata={"file":rel,"language":lang,"chunk_id":chunk_id}))
                chunk_id+=1
    return docs
