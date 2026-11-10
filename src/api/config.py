"""Scoring service configuration, read from the environment and .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact import DEFAULT_ARTIFACT_PATH


class Settings(BaseSettings):
    """Service settings. Every field has a default, so .env is optional.

    Attributes:
        model_path: Path to the joblib artifact loaded at startup.
        score_threshold: Decision threshold override. `None` (the default)
            means take the threshold from the artifact.
        redis_url: Cache server. `None` disables the cache; the service then
            answers every request uncached and reports `X-Cache: BYPASS`.
        cache_ttl_seconds: Expiry applied to every cached score.
        batch_concurrency: Applications scored at once by `/score/batch`.
    """

    # protected_namespaces=(): pydantic reserves the `model_` prefix and would
    # otherwise warn about `model_path`.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CS_",
        extra="ignore",
        protected_namespaces=(),
    )

    model_path: Path = DEFAULT_ARTIFACT_PATH
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    redis_url: str | None = None
    cache_ttl_seconds: int = Field(default=300, gt=0)
    batch_concurrency: int = Field(default=8, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return the settings, built once per process."""
    return Settings()
