"""
RepoMind Application Configuration.

Centralized settings management with automatic environment variable
loading, type casting, sensible defaults, and path resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# PROJECT ROOT & PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file from project root
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class PostgresConfig:
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "repomind"))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
    judge_model: str = field(
        default_factory=lambda: os.getenv("EVAL_JUDGE_MODEL", os.getenv("LLM_MODEL", "qwen2.5:7b"))
    )
    embedding_timeout: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "120.0"))
    )


@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))


@dataclass(frozen=True)
class PathConfig:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    repos_dir: Path = PROJECT_ROOT / "data" / "repos"
    baselines_dir: Path = PROJECT_ROOT / "baselines"
    golden_tests_dir: Path = PROJECT_ROOT / "golden_tests"


@dataclass(frozen=True)
class Settings:
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Global singleton instance
settings = Settings()
