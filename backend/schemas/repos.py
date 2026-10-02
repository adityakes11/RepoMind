from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    github_url: str = Field(min_length=1)
    thread_id: UUID


class IngestResponse(BaseModel):
    task_id: UUID
    collection_name: str
    status: str


class RepositoryStatus(BaseModel):
    collection_name: str
    status: str
    document_count: int = 0
    progress: int = 0
    error: Optional[str] = None