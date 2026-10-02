from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_current_user
from src.database import create_thread, delete_thread, get_all_threads, get_thread
from backend.schemas.threads import ThreadCreate, ThreadResponse, ThreadSummary


router = APIRouter(prefix="/api/threads", tags=["threads"])


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
def create_thread_route(request: ThreadCreate, user=Depends(get_current_user)):
    return get_thread(create_thread(user["id"], request.title), user["id"])


@router.get("", response_model=list[ThreadSummary])
def list_threads(user=Depends(get_current_user)):
    return get_all_threads(user["id"])


@router.get("/{thread_id}", response_model=ThreadResponse)
def read_thread(thread_id: UUID, user=Depends(get_current_user)):
    thread = get_thread(thread_id, user["id"])
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_thread(thread_id: UUID, user=Depends(get_current_user)):
    if not delete_thread(thread_id, user["id"]):
        raise HTTPException(status_code=404, detail="Thread not found")