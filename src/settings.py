"""Konfiguracja strony treningowej, czytana ze środowiska i `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from config import DATA_PATH
from strategies import ESTIMATOR_STRATEGIES, IMBALANCE_STRATEGIES


class TrainingSettings(BaseSettings):
    """Parametry treningu. Każde pole ma wartość domyślną.

    Attributes:
        data_path: Ścieżka do zbioru treningowego.
        test_size: Udział zbioru testowego.
        cv_splits: Liczba foldów przy wyznaczaniu progu.
        models: Algorytmy w siatce.
        arms: Ramiona niezbalansowania w siatce.
        engineered: Czy liczyć cechy pochodne.
        track: Czy zapisywać biegi do MLflow.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CS_", extra="ignore")

    data_path: Path = DATA_PATH
    test_size: float = 0.2
    cv_splits: int = 5
    models: tuple[str, ...] = tuple(ESTIMATOR_STRATEGIES)
    arms: tuple[str, ...] = tuple(IMBALANCE_STRATEGIES)
    engineered: bool = True
    track: bool = True


@lru_cache
def get_training_settings() -> TrainingSettings:
    """Zwraca ustawienia treningu, tworzone raz na proces."""
    return TrainingSettings()
