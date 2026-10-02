from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from backend.dependencies import get_current_user
from backend.schemas.repos import IngestRequest, IngestResponse, RepositoryStatus
from backend.services.ingestion_service import get_repository_status, run_ingestion, start_ingestion


router = APIRouter(prefix="/api/repos", tags=["repositories"])


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_repository(request: IngestRequest, background_tasks: BackgroundTasks, user=Depends(get_current_user)):
    try:
        task_id, collection_name = start_ingestion(request.github_url, request.thread_id, user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    background_tasks.add_task(
        run_ingestion,
        task_id,
        request.github_url,
        request.thread_id,
        user["id"],
        langsmith_extra={"metadata": {"thread_id": str(request.thread_id)}},
    )
    return {"task_id": task_id, "collection_name": collection_name, "status": "queued"}


@router.get("/{collection_name}/status", response_model=RepositoryStatus)
def repository_status(collection_name: str, user=Depends(get_current_user)):
    return get_repository_status(collection_name)