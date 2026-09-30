"""Training flow: step order, retries and caching.

Prefect owns execution: order, retries, caching and scheduling. MLflow owns
recording the results. Neither replaces the other.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import mlflow
import pandas as pd
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from sklearn.pipeline import Pipeline

from config import ENGINEERED_FEATURES, MLFLOW_TRACKING_URI, FeatureGroups
from data_loader import DataLoader
from preprocessor import Preprocessor
from settings import TrainingSettings, get_training_settings
from train import GRID_EXPERIMENT_NAME, promote_champion
from trainer import ModelTrainer

logger = logging.getLogger(__name__)


def _features(settings: TrainingSettings) -> tuple[str, ...]:
    """Return the derived features implied by the configuration."""
    return ENGINEERED_FEATURES if settings.engineered else ()


def _trainer(settings: TrainingSettings, groups: FeatureGroups) -> ModelTrainer:
    """Build a trainer for already-computed feature groups."""
    return ModelTrainer.from_settings(
        settings, Preprocessor(groups, _features(settings))
    )


@task
def contracts_task(
    X_train: pd.DataFrame, settings: TrainingSettings
) -> tuple[FeatureGroups, FeatureGroups]:
    """Compute both column contracts once per run.

    Args:
        X_train: Raw training frame.
        settings: Training configuration.

    Returns:
        `(model groups, input contract)`.
    """
    loader = DataLoader.from_settings(settings)
    return (
        loader.model_groups(X_train, _features(settings)),
        loader.input_contract(X_train),
    )


@task(retries=2, retry_delay_seconds=10)
def load_task(settings: TrainingSettings) -> pd.DataFrame:
    """Load the data; retries because the read can fail transiently.

    Args:
        settings: Training configuration.

    Returns:
        The full data frame.
    """
    return DataLoader.from_settings(settings).load()


@task
def split_task(
    df: pd.DataFrame, settings: TrainingSettings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and test sets.

    Args:
        df: Full frame.
        settings: Training configuration.

    Returns:
        `(X_train, X_test, y_train, y_test)`.
    """
    return DataLoader.from_settings(settings).split(df)


# INPUTS + TASK_SOURCE, not INPUTS alone: the key must see the task code,
# or the grid returns the previous run's result after a hyperparameter change.
@task(cache_policy=INPUTS + TASK_SOURCE, cache_expiration=timedelta(days=7))
def grid_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: FeatureGroups,
    settings: TrainingSettings,
) -> dict[tuple[str, str], dict[str, float]]:
    """Train the grid; the result is cached on the input values.

    The cache key comes from the data, not the file path: after the dataset
    changes the step must recompute, or training would run on stale data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        groups: Model feature groups.
        settings: Training configuration.

    Returns:
        Metrics for every grid cell.
    """
    return _trainer(settings, groups).run_grid(X_train, y_train, X_test, y_test)


@task
def champion_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results: dict[tuple[str, str], dict[str, float]],
    groups: FeatureGroups,
    settings: TrainingSettings,
) -> tuple[Pipeline, float, dict[str, float], str, str]:
    """Select the champion, fit it, and choose its threshold.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        results: Grid metrics.
        groups: Model feature groups.
        settings: Training configuration.

    Returns:
        `(pipeline, threshold, metrics, model name, arm)`.
    """
    trainer = _trainer(settings, groups)
    model_name, arm = trainer.select_champion(results)
    pipeline, threshold, metrics = trainer.fit_champion(
        X_train, y_train, X_test, y_test, model_name, arm
    )
    for key, value in sorted(metrics.items()):
        logger.info("  %-14s %.4f", key, value)
    return pipeline, threshold, metrics, model_name, arm


@task
def promote_task(
    pipeline: Pipeline,
    threshold: float,
    metrics: dict[str, float],
    model_name: str,
    arm: str,
    groups: FeatureGroups,
    inputs: FeatureGroups,
    settings: TrainingSettings,
    rows_train: int,
    rows_test: int,
) -> bool:
    """Run the champion through the quality gates and promote it if it passes.

    Args:
        pipeline: Fitted champion pipeline.
        threshold: Decision threshold.
        metrics: Metrics on the test set.
        model_name: Algorithm name.
        arm: Imbalance arm.
        groups: Model feature groups.
        inputs: Input contract.
        settings: Training configuration.
        rows_train: Number of training rows.
        rows_test: Number of test rows.

    Returns:
        `True` when the model was promoted.
    """
    return promote_champion(
        pipeline=pipeline,
        threshold=threshold,
        metrics=metrics,
        model_name=model_name,
        arm=arm,
        groups=groups,
        inputs=inputs,
        engineered=settings.engineered,
        loader=DataLoader.from_settings(settings),
        rows_train=rows_train,
        rows_test=rows_test,
    )


@flow(name="training_pipeline")
def training_pipeline(
    register_champion: bool = True,
    settings: TrainingSettings | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Run the full training path.

    Args:
        register_champion: `False` stops after the comparison table and does
            not touch the registry.
        settings: Configuration; `None` uses the process settings.

    Returns:
        Metrics for every grid cell.
    """
    settings = settings or get_training_settings()

    if settings.track:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(GRID_EXPERIMENT_NAME)

    df = load_task(settings)
    X_train, X_test, y_train, y_test = split_task(df, settings)
    groups, inputs = contracts_task(X_train, settings)
    results = grid_task(X_train, y_train, X_test, y_test, groups, settings)

    logger.info(
        "Comparison table:\n%s",
        _trainer(settings, groups).comparison_table(results).to_string(index=False),
    )

    if register_champion:
        pipeline, threshold, metrics, model_name, arm = champion_task(
            X_train, y_train, X_test, y_test, results, groups, settings
        )
        promote_task(
            pipeline,
            threshold,
            metrics,
            model_name,
            arm,
            groups,
            inputs,
            settings,
            int(len(X_train)),
            int(len(X_test)),
        )

    return results
