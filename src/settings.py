"""Training-side configuration, read from the environment and .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from config import DATA_PATH
from strategies import ESTIMATOR_STRATEGIES, IMBALANCE_STRATEGIES


class TrainingSettings(BaseSettings):
    """Training parameters. Every field has a default.

    Attributes:
        data_path: Path to the training set.
        test_size: Test set fraction.
        cv_splits: Number of folds used to choose the threshold.
        models: Algorithms in the grid.
        arms: Imbalance arms in the grid.
        engineered: Whether to compute derived features.
        track: Whether to record runs in MLflow.
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
    """Return the training settings, built once per process."""
    return TrainingSettings()
