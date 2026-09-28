"""Kontrakt przepływu treningowego: kolejność kroków, ponawianie, cache."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from prefect.testing.utilities import prefect_test_harness

from config import ID_COLUMN, TARGET
from flows import grid_task, load_task, split_task, training_pipeline
from settings import TrainingSettings
from test_characterisation import synthetic_frame


@pytest.fixture(scope="module", autouse=True)
def prefect_harness() -> Iterator[None]:
    """Uruchamia przepływy na tymczasowej bazie, nie na lokalnej instalacji."""
    with prefect_test_harness():
        yield


def _labelled_frame(rows: int = 120) -> pd.DataFrame:
    frame, target = synthetic_frame(rows=rows)
    frame[TARGET] = target
    frame[ID_COLUMN] = range(100000, 100000 + len(frame))
    return frame


def test_tasks_are_callable_without_prefect_via_fn() -> None:
    """`.fn` daje czystą funkcję — logikę testujemy bez silnika przepływów."""
    frame = _labelled_frame(rows=40)
    X_train, X_test, _y_train, _y_test = split_task.fn(frame, TrainingSettings())
    assert len(X_train) + len(X_test) == len(frame)
    assert TARGET not in X_train.columns


def test_load_task_retries_on_failure() -> None:
    """Pobranie danych bywa przejściowo zawodne, więc krok ma ponawiać."""
    assert load_task.retries >= 1


def test_grid_task_cache_key_depends_on_inputs() -> None:
    """Cache liczony z wartości wejściowych, nie z samego faktu wywołania.

    Polityka oparta na czymkolwiek innym zwracałaby wynik poprzedniego
    przebiegu po zmianie danych — trening "przechodziłby" na starym zbiorze
    i nikt by tego nie zauważył.
    """
    frame_a, _ = synthetic_frame(rows=20, seed=0)
    frame_b, _ = synthetic_frame(rows=20, seed=7)
    assert not frame_a.equals(frame_b)

    policy = grid_task.cache_policy
    assert policy is not None

    key_a = policy.compute_key(
        task_ctx=None, inputs={"X_train": frame_a}, flow_parameters={}
    )
    key_b = policy.compute_key(
        task_ctx=None, inputs={"X_train": frame_b}, flow_parameters={}
    )
    assert key_a != key_b


def test_flow_runs_the_grid_end_to_end(tmp_path: Path) -> None:
    """Przepływ przechodzi od pliku do tabeli porównawczej."""
    csv = tmp_path / "application_train.csv"
    _labelled_frame().to_csv(csv, index=False)

    settings = TrainingSettings(
        data_path=csv,
        models=("logistic_regression",),
        arms=("none",),
        cv_splits=2,
        track=False,
    )
    results = training_pipeline(register_champion=False, settings=settings)
    assert set(results) == {("logistic_regression", "none")}


def test_grid_cache_is_invalidated_by_a_code_change() -> None:
    """Cache musi widzieć kod zadania, nie tylko dane.

    Sama polityka INPUTS trzyma wynik między przebiegami także wtedy, gdy
    zmienią się hiperparametry — siatka wróciłaby z metrykami konfiguracji,
    która już nie istnieje, a champion dopasowałby się do nowej.
    """
    kinds = {type(policy).__name__ for policy in grid_task.cache_policy.policies}
    assert {"Inputs", "TaskSource"} <= kinds


def test_grid_cache_expires() -> None:
    """Cache bez wygaśnięcia żyje wiecznie w lokalnym magazynie wyników."""
    assert grid_task.cache_expiration is not None
