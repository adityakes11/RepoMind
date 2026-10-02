import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

from langsmith import traceable
from src.database import get_thread, update_thread_repository
from src.ingestion.chunker import create_chunks
from src.ingestion.github_loader import clone_repository, get_repository_files
from src.rag.vector_store import create_vector_store, get_collection_name, load_vector_store


_tasks = {}
_tasks_lock = threading.Lock()


def _repo_name(github_url):
    name = Path(urlparse(github_url).path.rstrip("/")).name
    return name.removesuffix(".git") or "repository"


def start_ingestion(github_url, thread_id, user_id):
    if get_thread(thread_id, user_id) is None:
        raise ValueError("Thread not found")

    task_id = uuid.uuid4()
    collection_name = get_collection_name(github_url)
    with _tasks_lock:
        _tasks[task_id] = {
            "collection_name": collection_name,
            "status": "queued",
            "progress": 0,
            "document_count": 0,
            "error": None,
        }
    return task_id, collection_name


@traceable(run_type="chain", name="RepositoryIngestion")
def run_ingestion(task_id, github_url, thread_id, user_id):
    collection_name = get_collection_name(github_url)
    try:
        _update_task(task_id, status="running")
        repo_name = _repo_name(github_url)
        trace_metadata = {"metadata": {"thread_id": str(thread_id)}}
        path = clone_repository(github_url, repo_name, langsmith_extra=trace_metadata)
        files = get_repository_files(path, langsmith_extra=trace_metadata)
        documents = create_chunks(path, files, langsmith_extra=trace_metadata)

        def report_progress(completed, total):
            _update_task(task_id, progress=round(completed * 100 / total) if total else 100, document_count=total)

        store = create_vector_store(
            github_url,
            documents,
            report_progress,
            langsmith_extra=trace_metadata,
        )
        update_thread_repository(thread_id, user_id, github_url, repo_name, collection_name)
        _update_task(task_id, status="completed", progress=100, document_count=store._collection.count())
    except Exception as exc:
        _update_task(task_id, status="failed", error=str(exc))


def _update_task(task_id, **updates):
    with _tasks_lock:
        if task_id in _tasks:
            _tasks[task_id].update(updates)


def get_repository_status(collection_name):
    with _tasks_lock:
        matching = [task for task in _tasks.values() if task["collection_name"] == collection_name]
        task = matching[-1].copy() if matching else None

    if task:
        return {"collection_name": collection_name, **task}

    try:
        count = load_vector_store(collection_name)._collection.count()
    except Exception:
        count = 0
    return {
        "collection_name": collection_name,
        "status": "ready" if count else "not_found",
        "document_count": count,
        "progress": 100 if count else 0,
        "error": None,
    }