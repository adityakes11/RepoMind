# RepoMind → FastAPI + Frontend Conversion Plan

## Current Architecture (What Exists)

Your backend logic is already well-structured. Here's what we're working with:

| Layer | Files | What It Does |
|-------|-------|--------------|
| **Config** | `src/config.py` | Central settings (PostgreSQL, Ollama, paths) |
| **Database** | `src/database.py` | PostgreSQL CRUD – `threads` & `messages` tables already exist |
| **Ingestion** | `src/ingestion/github_loader.py`, `code_parser.py`, `chunker.py` | Clone repo → parse files → chunk code |
| **RAG** | `src/rag/vector_store.py`, `retriever.py`, `reranker.py`, `embeddings.py` | Chroma vector store, similarity search, cross-encoder reranking |
| **LLM** | `src/llm/ollama.py`, `generator.py`, `prompts.py` | Ollama ChatLLM with streaming support |
| **Pipeline** | `src/pipeline.py` | `RAGPipeline.run(question, collection, top_k)` → answer + sources |
| **Evaluations** | `evaluations/*.py` | 6-pillar eval suite, baseline/candidate comparison |

> [!IMPORTANT]
> Your `database.py` already has thread/message CRUD and `generator.py` already has `generate_stream()`. This means ~60% of the backend logic is already written.

---

## Target Architecture

```
RepoMind/
├── backend/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point, CORS, lifespan
│   ├── dependencies.py               # Shared dependencies (DB, pipeline instances)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── repos.py                  # POST /ingest, GET /status
│   │   ├── threads.py                # CRUD for threads
│   │   ├── chat.py                   # POST /chat, GET /chat/stream (SSE)
│   │   └── evaluations.py           # Optional: trigger evals via API
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── repos.py                  # Pydantic models for repo requests/responses
│   │   ├── threads.py                # Pydantic models for thread requests/responses
│   │   └── chat.py                   # Pydantic models for chat requests/responses
│   └── services/
│       ├── __init__.py
│       ├── ingestion_service.py      # Wraps src/ingestion/* (background task)
│       └── chat_service.py           # Wraps src/pipeline.py for streaming
│
├── frontend/                         # React + Vite + Tailwind CSS
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api/
│       │   └── client.js             # Axios/fetch wrapper for backend API
│       ├── components/
│       │   ├── Layout.jsx            # App shell with sidebar
│       │   ├── Sidebar.jsx           # Thread list + new thread button
│       │   ├── ChatWindow.jsx        # Message list + input box
│       │   ├── MessageBubble.jsx     # Single message (user/assistant)
│       │   ├── IngestForm.jsx        # GitHub URL input + progress bar
│       │   ├── SourceFiles.jsx       # Collapsible source file list
│       │   └── MarkdownRenderer.jsx  # Render assistant markdown responses
│       ├── pages/
│       │   ├── HomePage.jsx          # Landing page / repo ingestion
│       │   └── ChatPage.jsx          # Main chat interface
│       └── hooks/
│           ├── useThreads.js         # Fetch/manage threads
│           ├── useChat.js            # Send messages, handle SSE streaming
│           └── useIngest.js          # Trigger ingestion, track progress
│
├── src/                              # Existing core logic (UNCHANGED)
│   ├── config.py
│   ├── database.py
│   ├── pipeline.py
│   ├── ingestion/
│   ├── llm/
│   └── rag/
│
├── evaluations/                      # Existing eval suite (UNCHANGED)
├── baselines/
├── golden_tests/
├── .env
└── requirements.txt                  # Add: fastapi, uvicorn, sse-starlette
```

---

## Phase 1: FastAPI Backend

### 1.1 — API Endpoints

#### Repositories
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/repos/ingest` | Start ingesting a GitHub repo (runs as background task) |
| `GET` | `/api/repos/{collection_name}/status` | Check if collection exists + doc count |

#### Threads
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/threads` | Create a new empty thread |
| `GET` | `/api/threads` | List all threads (with last message preview) |
| `GET` | `/api/threads/{thread_id}` | Get thread details + full message history |
| `DELETE` | `/api/threads/{thread_id}` | Delete a thread and its messages |

#### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/threads/{thread_id}/chat` | Send a question, get full answer back |
| `GET` | `/api/threads/{thread_id}/chat/stream` | SSE endpoint — streams answer token-by-token |

#### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check (DB + Ollama connectivity) |

### 1.2 — Key Backend Design Decisions

**Streaming via SSE (Server-Sent Events)**
- Uses `sse-starlette` package
- Your `generator.py` already has `generate_stream()` that yields chunks
- The `/chat/stream` endpoint wraps this as an SSE event stream
- Frontend consumes via `EventSource` API (native browser support, no WebSocket complexity)

**Background Ingestion**
- `POST /api/repos/ingest` returns immediately with a task ID
- Ingestion runs via FastAPI `BackgroundTasks`
- Progress tracked in-memory (or optionally in a DB table)
- Frontend polls `/api/repos/{collection}/status` for progress

**No Code Duplication**
- Backend routers import directly from your existing `src/` modules
- `chat_service.py` wraps `RAGPipeline` — no rewrite needed
- `ingestion_service.py` wraps `github_loader` + `chunker` + `vector_store`

### 1.3 — New Dependencies

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.0.0
python-multipart>=0.0.9
```

---

## Phase 2: React Frontend

### 2.1 — Tech Stack

| Tool | Purpose |
|------|---------|
| **React 18** | UI framework |
| **Vite** | Fast dev server + bundler |
| **Tailwind CSS v3** | Utility-first styling |
| **React Router v6** | Client-side routing |
| **Lucide React** | Icons |
| **react-markdown + remark-gfm** | Render assistant markdown/code |

### 2.2 — Pages & Components

#### Page 1: Home / Ingest (`/`)
```
┌──────────────────────────────────────────────┐
│  🧠 RepoMind                                │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  Paste a GitHub URL to get started     │  │
│  │  [https://github.com/user/repo    ] ▶  │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ████████████████░░░░░░  67% Ingesting...    │
│                                              │
│  Recent Repositories:                        │
│  • Smart_AI_Driving_Assistant  [Chat →]      │
│  • another-repo                [Chat →]      │
└──────────────────────────────────────────────┘
```

#### Page 2: Chat (`/chat/:threadId`)
```
┌─────────────┬────────────────────────────────┐
│  Threads    │  Smart_AI_Driving_Assistant     │
│             │                                 │
│  + New      │  ┌─ USER ──────────────────┐    │
│             │  │ How does auth work?      │    │
│  ● Thread 1 │  └─────────────────────────┘    │
│  ○ Thread 2 │                                 │
│  ○ Thread 3 │  ┌─ ASSISTANT ─────────────┐    │
│             │  │ The authentication uses  │    │
│             │  │ Firebase Auth with...    │    │
│             │  │                          │    │
│             │  │ Sources: auth.dart,      │    │
│             │  │   login_screen.dart      │    │
│             │  └─────────────────────────┘    │
│             │                                 │
│             │  ┌──────────────────────[Send]┐  │
│             │  │ Ask about this repo...     │  │
│             │  └───────────────────────────┘  │
└─────────────┴────────────────────────────────┘
```

### 2.3 — Streaming UX

1. User types question → clicks Send
2. Frontend `POST`s to `/api/threads/{id}/chat` to save the user message
3. Frontend opens `EventSource` to `/api/threads/{id}/chat/stream?question=...`
4. Tokens arrive one-by-one → assistant bubble updates in real-time (typing effect)
5. On stream end → full response is saved to DB

---

## Phase 3: Integration & Polish

### 3.1 — CORS Configuration
```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3.2 — Development Workflow

```bash
# Terminal 1: Backend
cd RepoMind
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd RepoMind/frontend
npm run dev     # → http://localhost:5173
```

### 3.3 — Proxy in Development
Vite proxies `/api/*` to `http://localhost:8000` so no CORS issues during dev.

---

## Implementation Order

| Step | Task | Effort |
|------|------|--------|
| **1** | Create `backend/main.py` + lifespan (init DB) | 15 min |
| **2** | Create Pydantic schemas (`schemas/`) | 20 min |
| **3** | Create `routers/threads.py` — wraps existing `database.py` | 20 min |
| **4** | Create `services/ingestion_service.py` — wraps existing ingestion pipeline | 25 min |
| **5** | Create `routers/repos.py` — ingest endpoint with BackgroundTasks | 20 min |
| **6** | Create `services/chat_service.py` + `routers/chat.py` — SSE streaming | 30 min |
| **7** | Create `routers/evaluations.py` (optional) | 15 min |
| **8** | Scaffold React app (`npm create vite@latest frontend`) | 5 min |
| **9** | Build `api/client.js` — API wrapper | 15 min |
| **10** | Build `Sidebar` + `ChatWindow` + `MessageBubble` components | 45 min |
| **11** | Build `IngestForm` with progress polling | 25 min |
| **12** | Implement SSE streaming in `useChat` hook | 25 min |
| **13** | Add markdown rendering + code syntax highlighting | 20 min |
| **14** | Polish UI — loading states, error handling, responsive layout | 30 min |

**Total estimated: ~5 hours of implementation**

---

## What Stays Untouched

> [!NOTE]
> The entire `src/` directory and `evaluations/` directory remain **completely unchanged**. The FastAPI backend is a thin wrapper around your existing modules. No core logic is rewritten.

| Directory | Changes? |
|-----------|----------|
| `src/config.py` | ❌ No changes |
| `src/database.py` | ❌ No changes (already has all CRUD) |
| `src/pipeline.py` | ❌ No changes |
| `src/ingestion/*` | ❌ No changes |
| `src/llm/*` | ❌ No changes |
| `src/rag/*` | ❌ No changes |
| `evaluations/*` | ❌ No changes |
| `baselines/` | ❌ No changes |
| `golden_tests/` | ❌ No changes |
