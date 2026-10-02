
# RepoMind

RepoMind is a local-first codebase intelligence workspace. Give it a public GitHub repository, let it index the source, and ask questions with answers grounded in the retrieved files.

It combines a React and Vite frontend, a FastAPI backend, PostgreSQL for accounts and conversation history, ChromaDB for code embeddings, and Ollama for local generation and embeddings.

## What it does

- Sign up and log in with an isolated user workspace.
- Create and manage repository-focused conversation threads.
- Clone supported source files from a GitHub URL in a background ingestion task.
- Track indexing status and document counts from the UI.
- Ask questions with normal responses or token-by-token SSE streaming.
- Show retrieved source file names alongside answers.
- Evaluate retrieval, generation, RAG quality, application correctness, and safety with golden datasets.
- Optionally trace ingestion and RAG runs in LangSmith while keeping model inference local through Ollama.

## Architecture

```text
Browser (React/Vite)
	|
	| REST + Server-Sent Events
	v
FastAPI (backend/)
   |              |
   |              +--> Ollama: chat generation and embeddings
   |
   +--> PostgreSQL: users, sessions, threads, messages
   |
   +--> Ingestion: clone -> parse -> chunk -> embed
			 |
			 v
		    ChromaDB (data/chroma)
```

Repository URLs are normalized and hashed into deterministic collection names such as `repomind_0123456789abcdef`. Re-indexing the same URL reuses its collection rather than creating duplicate embeddings. PostgreSQL stores the thread-to-repository mapping; ChromaDB stores the chunks and vectors.

## Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer
- PostgreSQL 14 or newer
- Ollama
- Git, with network access to the repositories you want to index

Create the database once:

```sql
CREATE DATABASE repomind;
```

Pull the default local models:

```bash
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## Local development

From the repository root, create a virtual environment and install the backend dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Copy the environment template and set the PostgreSQL password:

```bash
copy .env.example .env
```

Start the backend in one terminal:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the frontend in another:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>, create an account, create a thread, and paste a public GitHub URL. The backend health endpoint is available at <http://localhost:8000/api/health>.

The API intentionally does not expose Swagger, ReDoc, or an OpenAPI document in this deployment.

## Configuration

The complete template is in `.env.example`. The main settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `repomind` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | empty | Database password |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `LLM_MODEL` | `qwen2.5:7b` | Answer-generation model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per code chunk |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `EMBEDDING_BATCH_SIZE` | `16` | Embeddings per batch |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |

Changing chunk settings affects new indexing runs. Existing Chroma collections must be rebuilt if their chunk boundaries need to change.

## API surface

Protected application routes require the bearer token returned by signup or login. Signup, login, the root endpoint, and the health check are public.

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Service metadata |
| `GET` | `/api/health` | PostgreSQL and Ollama health |
| `POST` | `/api/auth/signup` | Create an account |
| `POST` | `/api/auth/login` | Start a 30-day session |
| `GET` | `/api/auth/me` | Get the current user |
| `POST` | `/api/auth/logout` | Revoke the current session |
| `POST` | `/api/threads` | Create a thread |
| `GET` | `/api/threads` | List the user's threads |
| `GET` | `/api/threads/{thread_id}` | Get a thread and messages |
| `DELETE` | `/api/threads/{thread_id}` | Delete a thread |
| `POST` | `/api/repos/ingest` | Queue GitHub repository ingestion |
| `GET` | `/api/repos/{collection_name}/status` | Read ingestion status |
| `POST` | `/api/threads/{thread_id}/chat` | Return a complete answer |
| `GET` | `/api/threads/{thread_id}/chat/stream` | Stream answer chunks over SSE |

## Docker

Build the backend and frontend images from the repository root:

```bash
docker build -f Dockerfile.backend -t repomind-backend .
docker build -f Dockerfile.frontend -t repomind-frontend .
```

Run the containers only after PostgreSQL and Ollama are reachable from the container network. The backend listens on port `8000`. The frontend image serves the production build through Nginx on port `80` and proxies `/api` to a host named `backend`:

```bash
docker run --rm -p 8000:8000 --env-file .env repomind-backend
docker run --rm -p 5173:80 repomind-frontend
```

For a multi-container deployment, provide a network alias named `backend`, configure the backend environment variables for PostgreSQL and Ollama, and persist `data/chroma` and `data/repos`. No Docker Compose file is included in this repository.

## AWS EC2 deployment

RepoMind has been deployed successfully on a single Ubuntu EC2 instance with Docker. This is the simplest deployment topology for a small demo:

```text
EC2
|-- frontend     Nginx on port 80
|-- backend      FastAPI on the private Docker network
|-- postgres     PostgreSQL 16
`-- ollama       Local CPU inference and embeddings
```

### Recommended EC2 settings

- Use an Ubuntu LTS AMI.
- Enable a public IPv4 address.
- Allow inbound TCP `22` from **My IP** and TCP `80` from the internet.
- Do not expose PostgreSQL `5432`, backend `8000`, or Ollama `11434` publicly.
- Use one `gp3` root volume of at least 30 GiB and no additional file system.

The free-tier `t3.micro` has only 1 GiB RAM. For that instance size, add a 2 GiB swap file and use the smaller `qwen2.5:0.5b` model. Larger repositories and the default `qwen2.5:7b` model require a larger instance and may incur AWS charges.

### Prepare the server

Connect to the instance, install Docker and Git, and clone the repository:

```bash
sudo apt update
sudo apt install -y docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
git clone https://github.com/adityakes11/RepoMind.git
cd RepoMind
```

For a 1 GiB instance, create swap before starting the containers:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Create `.env.aws` and use Docker service names for internal connections:

```env
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=repomind
POSTGRES_USER=postgres
POSTGRES_PASSWORD=replace_with_a_strong_password
OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=qwen2.5:0.5b
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
EMBEDDING_BATCH_SIZE=4
LANGSMITH_TRACING=false
```

Create a shared network and persistent volumes:

```bash
docker network create repomind-net
docker volume create repomind-postgres
docker volume create repomind-ollama
docker volume create repomind-chroma
docker volume create repomind-repos
```

Start PostgreSQL and Ollama:

```bash
docker run -d --name postgres --network repomind-net --restart unless-stopped \
	-e POSTGRES_DB=repomind -e POSTGRES_USER=postgres \
	-e POSTGRES_PASSWORD=replace_with_a_strong_password \
	-v repomind-postgres:/var/lib/postgresql/data postgres:16

docker run -d --name ollama --network repomind-net --restart unless-stopped \
	-p 127.0.0.1:11434:11434 -v repomind-ollama:/root/.ollama ollama/ollama

docker exec ollama ollama pull qwen2.5:0.5b
docker exec ollama ollama pull nomic-embed-text
```

Build and start the application containers:

```bash
docker build -f Dockerfile.backend -t repomind-backend .
docker build -f Dockerfile.frontend -t repomind-frontend .

docker run -d --name backend --network repomind-net --restart unless-stopped \
	--env-file .env.aws -p 127.0.0.1:8000:8000 \
	-v repomind-chroma:/app/data/chroma \
	-v repomind-repos:/app/data/repos repomind-backend

docker run -d --name frontend --network repomind-net --restart unless-stopped \
	-p 80:80 repomind-frontend
```

Open `http://<EC2-public-ip>`. The frontend proxies `/api` requests to the backend container. Check the deployment with `curl http://localhost:8000/api/health` and `docker ps`.

The production image uses `requirements.runtime.txt`, which excludes the CUDA-heavy evaluation dependencies. Evaluation commands should run in a development environment, not inside the small production container.

Stopping the EC2 instance makes the application unavailable but preserves Docker volumes. Starting it again may assign a different public IP unless an Elastic IP is configured. HTTP is suitable for a private demo only; use a domain and HTTPS before handling sensitive data.

## Testing and evaluation

Run the end-to-end authenticated workflow with PostgreSQL, Ollama, and the backend running:

```bash
python tests/e2e_test.py
```

The test signs up or logs in, creates a thread, indexes `YoutubeRAGChatbot` by default, polls ingestion, asks a streamed question, verifies persistence, and checks logout. Override `E2E_REPO_URL`, `E2E_EMAIL`, `E2E_PASSWORD`, or `E2E_INGEST_TIMEOUT` as needed.

Run the basic deterministic retrieval check:

```bash
python -m evaluations.evaluate <collection_name>
```

Run the full regression suite and compare later candidates with the saved baseline:

```bash
python -m evaluations.run_suite --collection-name <collection_name>
python -m evaluations.compare
```

Focused evaluation examples:

```bash
python -m evaluations.eval_retriever --collection-name <collection_name> --top-k 5
python -m evaluations.eval_generator --judge-model qwen2.5:3b --limit 3 --metrics faithfulness
python -m evaluations.eval_rag_pipeline --collection-name <collection_name> --judge-model qwen2.5:3b --limit 3 --top-k 5
python -m evaluations.eval_toxicity --collection-name <collection_name> --judge-model qwen2.5:3b --limit 3
```

See [evaluations/README.md](evaluations/README.md) for the complete evaluation matrix, reranking options, golden datasets, and baseline behavior.

## LangSmith tracing

Tracing is opt-in. Add the following to `.env` when a LangSmith project is configured:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=RepoMind
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Traces may contain questions, prompts, and retrieved code. Configure project access accordingly. Authenticated chat and ingestion runs attach the PostgreSQL `thread_id` as trace metadata.

The online triad worker can attach DeepEval faithfulness, answer relevancy, and contextual relevancy feedback to matching traces:

```bash
python -m evaluations.online.triad_worker --once
python -m evaluations.online.triad_worker
```

## Data and operational notes

- PostgreSQL tables are created or migrated by `backend.main` at startup.
- Passwords use salted `scrypt` hashes; bearer sessions are revocable and expire after 30 days.
- Repository clones are stored under `data/repos`; Chroma persists under `data/chroma`.
- Supported ingestion extensions and ignored directories are defined in `src/ingestion/github_loader.py`.
- Ingestion task status is kept in backend process memory. Use one backend process for reliable polling, or move task state to shared storage before scaling horizontally.
- A failed or interrupted ingestion can leave a cloned repository on disk; retrying the same URL replaces that clone and reuses the deterministic collection.

## Repository layout

```text
backend/       FastAPI app, routers, schemas, and services
src/           ingestion, RAG, Ollama, and PostgreSQL core logic
frontend/     React/Vite client and production Nginx configuration
evaluations/   Golden-set runners, regression suite, and online feedback worker
golden_tests/  Evaluation cases
baselines/     Evaluation baseline and candidate results
data/          Local repository clones and Chroma persistence
tests/         End-to-end validation
```

## CI

GitHub Actions runs Python compilation and API contract checks, builds the frontend, and builds container images. The workflow targets Python 3.11 and Node.js 20. Images published to GitHub Container Registry use the repository's backend and frontend image names with commit-SHA tags.
