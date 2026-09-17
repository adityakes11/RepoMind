# RepoMind 🧠

RepoMind is a codebase-intelligence RAG application with **PostgreSQL for threads/chat history**, **ChromaDB for code embeddings**, and **Ollama/Qwen for local generation**.

## Key design

Each repository URL gets a deterministic Chroma collection:

```text
Thread A → Repo A → Chroma collection repomind_xxx
Thread B → Repo B → Chroma collection repomind_yyy
```

## Project structure

```text
RepoMind/
├── app.py
├── src/
│   ├── database.py
│   ├── ingestion/
│   ├── llm/
│   ├── rag/
├── evaluations/
│   ├── evaluate.py
│   └── eval_retriever.py
├── data/
├── golden test/
├── requirements.txt
└── pyproject.toml
```

So indexing a new URL **does not delete old embeddings**. PostgreSQL keeps the mapping between each thread and its repository/collection.

PostgreSQL stores threads, repository metadata, and messages. ChromaDB stores code chunks and embeddings.

Code chunking defaults to `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=150`. Override these values in `.env` when needed. Existing Chroma collections must be rebuilt after changing chunk settings because stored embeddings keep their original chunks.

## Setup

1. Install PostgreSQL and create the database:

```sql
CREATE DATABASE repomind;
```

2. Copy `.env.example` to `.env` and set your PostgreSQL password.

3. Start Ollama and pull models:

```bash
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run:

```bash
streamlit run app.py
```

## Important behavior

Click **New Thread** before working with another independent conversation. You can paste another GitHub URL into a thread to associate that thread with a different repository; for clean separation, create a new thread for a new repository.

The same normalized repository URL reuses its existing Chroma collection, avoiding duplicate embeddings.

## Evaluation

Install the package in editable mode once:

```bash
pip install -e .
```

Then use a collection name such as `repomind_...`:

```bash
python -m evaluations.evaluate repomind_xxxxxxxxxxxxxxxx
```

Update `evaluations/dataset.json` to match the repository being tested.
