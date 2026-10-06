from contextlib import asynccontextmanager
from urllib.request import urlopen

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import auth, chat, repos, threads
from src.config import settings
from src.database import get_connection, init_db


@asynccontextmanager
async def lifespan(_app):
    try:
        init_db()
    except Exception as exc:
        print(f"Database startup check failed: {exc}")
    yield


app = FastAPI(
    title="RepoMind API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",
        "http://3.25.230.158",
        "http://3.25.230.158:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(threads.router)
app.include_router(repos.router)
app.include_router(chat.router)
app.include_router(auth.router)


@app.get("/")
def root():
    return {
        "name": "RepoMind API",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    database = "ok"
    ollama = "ok"
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1")
    except Exception:
        database = "unavailable"
    try:
        with urlopen(f"{settings.ollama.base_url}/api/tags", timeout=2):
            pass
    except Exception:
        ollama = "unavailable"
    overall = "ok" if database == ollama == "ok" else "degraded"
    return {"status": overall, "database": database, "ollama": ollama}