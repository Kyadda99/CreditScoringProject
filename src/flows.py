"""Przepływ treningowy: kolejność kroków, ponawianie i cache.

Prefect odpowiada za wykonanie — kolejność, ponowienia, cache i harmonogram.
MLflow odpowiada za zapis wyników. Żaden z nich nie zastępuje drugiego.
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
    """Zwraca listę cech pochodnych wynikającą z konfiguracji."""
    return ENGINEERED_FEATURES if settings.engineered else ()


def _trainer(settings: TrainingSettings, groups: FeatureGroups) -> ModelTrainer:
    """Buduje trenera dla policzonych juz grup cech."""
    return ModelTrainer.from_settings(
        settings, Preprocessor(groups, _features(settings))
    )


@task
def contracts_task(
    X_train: pd.DataFrame, settings: TrainingSettings
) -> tuple[FeatureGroups, FeatureGroups]:
    """Liczy oba kontrakty kolumn raz na przebieg.

    Args:
        X_train: Surowa ramka treningowa.
        settings: Konfiguracja treningu.

    Returns:
        `(grupy modelu, kontrakt wejsciowy)`.
    """
    loader = DataLoader.from_settings(settings)
    return (
        loader.model_groups(X_train, _features(settings)),
        loader.input_contract(X_train),
    )


@task(retries=2, retry_delay_seconds=10)
def load_task(settings: TrainingSettings) -> pd.DataFrame:
    """Wczytuje dane; ponawia, bo odczyt bywa przejściowo zawodny.

    Args:
        settings: Konfiguracja treningu.

    Returns:
        Pełna ramka danych.
    """
    return DataLoader.from_settings(settings).load()


@task
def split_task(
    df: pd.DataFrame, settings: TrainingSettings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dzieli dane na treningowe i testowe.

    Args:
        df: Pełna ramka.
        settings: Konfiguracja treningu.

    Returns:
        `(X_train, X_test, y_train, y_test)`.
    """
    return DataLoader.from_settings(settings).split(df)


# INPUTS + TASK_SOURCE, a nie samo INPUTS: klucz musi widziec kod zadania,
# inaczej po zmianie hiperparametrow siatka wraca z poprzedniego przebiegu.
@task(cache_policy=INPUTS + TASK_SOURCE, cache_expiration=timedelta(days=7))
def grid_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: FeatureGroups,
    settings: TrainingSettings,
) -> dict[tuple[str, str], dict[str, float]]:
    """Trenuje siatkę; wynik jest cache'owany po wartościach wejściowych.

    Cache liczony z danych, a nie ze ścieżki pliku: po zmianie zbioru krok
    musi policzyć się od nowa, inaczej trening przeszedłby na starych danych.

    Args:
        X_train: Cechy treningowe.
        y_train: Etykiety treningowe.
        X_test: Cechy testowe.
        y_test: Etykiety testowe.
        groups: Grupy cech modelu.
        settings: Konfiguracja treningu.

    Returns:
        Metryki każdej komórki siatki.
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
    """Wybiera championa, dopasowuje go i wyznacza jego próg.

    Args:
        X_train: Cechy treningowe.
        y_train: Etykiety treningowe.
        X_test: Cechy testowe.
        y_test: Etykiety testowe.
        results: Metryki siatki.
        groups: Grupy cech modelu.
        settings: Konfiguracja treningu.

    Returns:
        `(pipeline, próg, metryki, nazwa modelu, ramię)`.
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
    """Przepuszcza championa przez bramki jakości i promuje go, gdy przejdzie.

    Args:
        pipeline: Dopasowany potok championa.
        threshold: Próg decyzyjny.
        metrics: Metryki na zbiorze testowym.
        model_name: Nazwa algorytmu.
        arm: Ramię niezbalansowania.
        groups: Grupy cech modelu.
        inputs: Kontrakt wejściowy.
        settings: Konfiguracja treningu.
        rows_train: Liczba wierszy treningowych.
        rows_test: Liczba wierszy testowych.

    Returns:
        `True`, gdy model został promowany.
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
    """Uruchamia pełną ścieżkę treningu.

    Args:
        register_champion: `False` kończy po tabeli porównawczej i nie dotyka
            rejestru.
        settings: Konfiguracja; `None` bierze ustawienia procesu.

    Returns:
        Metryki wszystkich komórek siatki.
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
        "Tabela porownawcza:\n%s",
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
