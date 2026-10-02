import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from backend.dependencies import get_current_user
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.chat_service import answer_question, stream_question


router = APIRouter(prefix="/api/threads/{thread_id}/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(thread_id: UUID, request: ChatRequest, user=Depends(get_current_user)):
    try:
        return answer_question(
            thread_id,
            request.question,
            user["id"],
            langsmith_extra={"metadata": {"thread_id": str(thread_id)}},
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/stream")
def stream_chat(thread_id: UUID, question: str = Query(min_length=1, max_length=10000), user=Depends(get_current_user)):
    try:
        events = stream_question(
            thread_id,
            question,
            user["id"],
            langsmith_extra={"metadata": {"thread_id": str(thread_id)}},
        )
        return EventSourceResponse(
            ({"event": "token", "data": json.dumps({"content": chunk})} for chunk in events)
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc