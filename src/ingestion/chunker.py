import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.code_parser import read_file,get_language
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=150

def create_chunks(repo_path,files,chunk_size=None,chunk_overlap=None):
    chunk_size=chunk_size or int(os.getenv("CHUNK_SIZE",DEFAULT_CHUNK_SIZE))
    chunk_overlap=chunk_overlap or int(os.getenv("CHUNK_OVERLAP",DEFAULT_CHUNK_OVERLAP))
    if chunk_size <= chunk_overlap:
        raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\nclass ","\ndef ","\nfunction ","\n\n","\n"," ",""])
    docs=[]
    for path in files:
        content=read_file(path)
        if not content.strip():continue
        rel=os.path.relpath(path,repo_path); lang=get_language(path)
        for i,chunk in enumerate(splitter.split_text(content)):
            docs.append(Document(page_content=chunk,metadata={"file":rel,"language":lang,"chunk_id":i}))
    return docs
