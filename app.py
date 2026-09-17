import os
import streamlit as st
from src.database import init_db,create_thread,get_threads,get_thread,save_message,update_thread_repository
from src.ingestion.github_loader import clone_repository,get_repository_files
from src.ingestion.chunker import create_chunks
from src.rag.vector_store import create_vector_store
from src.rag.retriever import retrieve_documents
from src.llm.generator import AnswerGenerator

init_db()
st.set_page_config(page_title="RepoMind",page_icon="🧠",layout="wide")
if "thread_id" not in st.session_state:st.session_state.thread_id=None
st.markdown("""
<style>
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.18);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] .stButton > button {
    min-height: 2.7rem;
    border: 1px solid transparent;
    border-radius: 0.55rem;
    text-align: left;
    padding: 0.55rem 0.75rem;
    font-size: 0.92rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(128, 128, 128, 0.35);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #1f6feb;
    color: white;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    margin-top: 0.5rem;
    margin-bottom: 0.65rem;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.68;
}
[data-testid="stSidebar"] hr {
    margin: 1.15rem 0;
}
</style>
""",unsafe_allow_html=True)
with st.sidebar:
    st.title("🧠 RepoMind")
    st.subheader("New thread")
    if st.button("＋  New chat",use_container_width=True,type="primary"):
        st.session_state.thread_id=create_thread();st.rerun()
    current_thread=get_thread(st.session_state.thread_id) if st.session_state.thread_id else None
    if current_thread and current_thread["repo_name"]:
        st.caption(f"Current repository  ·  {current_thread['repo_name']}")
    st.divider()
    st.subheader("Old conversations")
    for t in get_threads():
        label=t["title"] or t["repo_name"] or "New conversation"
        if st.button(f"▸  {label}",key=f"thread_{t['id']}",use_container_width=True):
            st.session_state.thread_id=t["id"]
            st.rerun()
if st.session_state.thread_id is None:st.session_state.thread_id=create_thread();st.rerun()
thread=get_thread(st.session_state.thread_id)
st.title("🧠 RepoMind")
st.caption("AI Codebase Intelligence using RAG + PostgreSQL + ChromaDB + Ollama")
st.info(f"Current thread: **{thread['title']}**")
st.subheader("📂 Repository")
url=st.text_input("GitHub Repository URL",value=thread["github_url"] or "",placeholder="https://github.com/user/project")
if st.button("🚀 Analyze / Load Repository"):
    if not url.strip():st.error("Please enter a GitHub repository URL.");st.stop()
    try:
        repo_name=url.rstrip("/").split("/")[-1].removesuffix(".git")
        with st.status("Analyzing repository...",expanded=True) as status:
            st.write("📥 Downloading repository...");path=clone_repository(url,repo_name)
            st.write("🔍 Finding source files...");source_files=get_repository_files(path);st.write(f"Found {len(source_files)} source files.")
            st.write("✂️ Creating code chunks...");docs=create_chunks(path,source_files);st.write(f"Created {len(docs)} chunks.")
            st.write("🧠 Creating/loading repository-specific Chroma collection...")
            embedding_progress = st.empty()
            def report_embedding_progress(completed, total):
                embedding_progress.write(f"Embedded {completed}/{total} code chunks...")
            store=create_vector_store(url,docs,report_embedding_progress)
            embedding_progress.empty()
            status.update(label="✅ Repository indexed successfully!",state="complete")
        update_thread_repository(thread["id"],url,repo_name,store._collection.name);st.rerun()
    except Exception as e:st.error(f"❌ Error: {e}")
thread=get_thread(st.session_state.thread_id)
if thread["collection_name"]:
    st.success(f"Repository loaded: `{thread['repo_name']}`")
    st.divider();st.subheader("💬 Ask Your Codebase")
    for m in thread["messages"]:
        with st.chat_message(m["role"]):st.write(m["content"])
    question=st.chat_input("Ask something about the repository...")
    if question:
        save_message(thread["id"],"user",question)
        with st.chat_message("user"):st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching codebase..."):docs=retrieve_documents(question,thread["collection_name"],5)
            if not docs:
                answer="I couldn't find enough relevant information in the repository.";st.warning(answer);save_message(thread["id"],"assistant",answer);st.stop()
            with st.spinner("🤖 Generating answer..."):result=AnswerGenerator().generate(question,docs)
            st.write(result["answer"]);save_message(thread["id"],"assistant",result["answer"])
            st.divider();st.subheader("📚 Sources")
            for source in result["sources"]:st.write(f"📄 `{source}`")
            st.subheader("🔍 Retrieved Code")
            for d in docs:
                with st.expander(f"📄 {d.metadata.get('file','unknown')}"):st.code(d.page_content,language=d.metadata.get("language"))
else:st.info("Paste a GitHub repository URL above and analyze it to start asking questions.")
