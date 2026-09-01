"""Konfiguracja serwisu scoringowego, czytana ze środowiska i `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact import DEFAULT_ARTIFACT_PATH


class Settings(BaseSettings):
    """Ustawienia serwisu. Każde pole ma wartość domyślną, więc `.env` jest opcjonalny.

    Attributes:
        model_path: Ścieżka do artefaktu joblib ładowanego przy starcie.
        score_threshold: Nadpisanie progu decyzyjnego. `None` (domyślnie)
            oznacza "weź próg z artefaktu" — metryki zapisane przy treningu
            opisują decyzję podjętą przy TAMTYM progu, więc serwis ma stosować
            tę samą regułę, którą zmierzono. Ustaw jawnie tylko po to, żeby
            świadomie odejść od wartości z artefaktu.
    """

    # protected_namespaces=(): pydantic domyślnie rezerwuje przedrostek `model_`
    # dla własnych atrybutów i ostrzega przy `model_path`. Nazwa jest tu
    # właściwa, więc wyłączamy ochronę zamiast zmieniać nazwę pola.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CS_",
        extra="ignore",
        protected_namespaces=(),
    )

    model_path: Path = DEFAULT_ARTIFACT_PATH
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Zwraca ustawienia, tworzone raz na proces."""
    return Settings()
