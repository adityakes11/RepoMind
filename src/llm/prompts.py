SYSTEM_PROMPT="""You are RepoMind, an AI codebase assistant. Answer questions using only the supplied repository context. Do not invent files, functions, classes, APIs, or behavior. If the context is insufficient, say: I couldn't find enough information in the repository. Mention relevant files and functions/classes when possible. Keep answers focused."""
def build_prompt(question,documents):
    context="\n\n".join(f"FILE: {d.metadata.get('file')}\nLANGUAGE: {d.metadata.get('language')}\nCODE:\n{d.page_content}" for d in documents)
    return f"{SYSTEM_PROMPT}\n\nREPOSITORY CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nAnswer with a clear explanation. End with Sources listing the relevant file names."
