from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ThreadCreate(BaseModel):
    title: str = Field(default="New Repository", min_length=1, max_length=200)


class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime


class ThreadSummary(BaseModel):
    id: UUID
    title: str
    repo_name: Optional[str] = None
    github_url: Optional[str] = None
    collection_name: Optional[str] = None
    updated_at: datetime
    last_message: Optional[str] = None


class ThreadResponse(ThreadSummary):
    messages: list[MessageResponse]