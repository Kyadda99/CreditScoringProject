"""Orkiestracja treningu: dane -> podział -> pipeline -> metryki -> artefakt.

Ten moduł niczego nie wymyśla — składa gotowe elementy z pozostałych modułów
i zapisuje wynik. Cała logika mieszka tam, gdzie da się ją testować bez 158 MB
danych na dysku.

Usage:
    uv run cs-train
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from api import FEATURE_SCHEMA_PATH as _API_FEATURE_SCHEMA_PATH
from artifact import DEFAULT_ARTIFACT_PATH, save_bundle
from config import (
    ID_COLUMN,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET,
    FeatureGroups,
    split_feature_groups,
)
from data import load_data
from evaluation import DEFAULT_THRESHOLD, evaluate
from models import get_models
from preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
MODEL_NAME = "logistic_regression"
EXPERIMENT_NAME = "credit-scoring-baseline"
# Importowana, nie wyprowadzana ponownie — patrz docstring `api/__init__.py`.
# Wcześniejsza wersja liczyła tę ścieżkę z PROJECT_ROOT i rozjeżdżała się
# z API wszędzie poza katalogiem repozytorium.
FEATURE_SCHEMA_PATH: Path = _API_FEATURE_SCHEMA_PATH
# MLflow 3.x odrzuca backend plikowy ("./mlruns") — jest w trybie utrzymaniowym
# i `set_experiment` rzuca wyjątkiem, dopóki nie ustawi się MLFLOW_ALLOW_FILE_STORE.
# Zamiast wchodzić w wycofywany backend, bierzemy SQLite: Faza 3 i tak go
# wymaga pod rejestr modeli i alias @production. `mlflow.db` jest w .gitignore.
# as_posix(): SQLAlchemy oczekuje ukośników, a nie windowsowych backslashy.
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


def split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dzieli ramkę na train/test w proporcji 80/20, ze stratyfikacją.

    Args:
        df: Pełna ramka z kolumną celu i identyfikatorem.

    Returns:
        `(X_train, X_test, y_train, y_test)`. Cel i identyfikator są usunięte
        z cech — identyfikator nie niesie sygnału, a model potrafi go zapamiętać.
    """
    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COLUMN])
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_pipeline(groups: FeatureGroups) -> Pipeline:
    """Składa preprocessing i estymator w jeden obiekt.

    Jeden `Pipeline` oznacza, że imputacja i skalowanie uczą się wyłącznie na
    foldzie treningowym, a serwowanie dostaje dokładnie te same przekształcenia
    co trening — nie ma miejsca na train/serve skew.

    Args:
        groups: Grupy cech wyprowadzone na zbiorze treningowym.

    Returns:
        Niedopasowany `Pipeline`.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(groups)),
            ("model", get_models()[MODEL_NAME]),
        ]
    )


def write_feature_schema(
    groups: FeatureGroups, path: Path = FEATURE_SCHEMA_PATH
) -> Path:
    """Zapisuje grupy cech jako **commitowany** kontrakt dla schematu API.

    API musi znać nazwy i typy pól, żeby zbudować model żądania, ale artefakt
    jest gitignorowany — nie ma go ani w CI, ani w świeżym klonie. Ten plik
    jest tym, z czego `api/schemas.py` generuje `ScoreRequest`.

    Args:
        groups: Grupy cech z treningu.
        path: Ścieżka docelowa.

    Returns:
        Ścieżka zapisanego pliku.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(groups.to_dict(), indent=2), encoding="utf-8")
    logger.info("Zapisano kontrakt cech: %s", path)
    return path


def main() -> None:
    """Trenuje model bazowy i zapisuje wszystko, czego potrzebuje serwowanie."""
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # Grupy wyprowadzamy WYŁĄCZNIE ze zbioru treningowego. Policzone na pełnej
    # ramce pozwoliłyby zbiorowi testowemu wpłynąć na to, która kolumna uchodzi
    # za flagę — subtelny, ale prawdziwy przeciek.
    groups = split_feature_groups(X_train)
    logger.info(
        "Cechy: %d numerycznych, %d kategorycznych, %d binarnych.",
        len(groups.numeric),
        len(groups.categorical),
        len(groups.binary),
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MODEL_NAME):
        pipeline = build_pipeline(groups)
        logger.info("Trenuję %s na %d wierszach ...", MODEL_NAME, len(X_train))
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), y_proba, threshold=DEFAULT_THRESHOLD)

        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_numeric": len(groups.numeric),
                "n_categorical": len(groups.categorical),
                "n_binary": len(groups.binary),
            }
        )
        mlflow.log_metrics(metrics)

        metadata = {
            **metrics,
            "model": MODEL_NAME,
            "trained_at": datetime.now(UTC).isoformat(),
            "sklearn_version": sklearn.__version__,
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
        }
        save_bundle(pipeline, groups, metadata, DEFAULT_ARTIFACT_PATH)
        write_feature_schema(groups)

    logger.info("BAZOWY ROC-AUC: %.4f", metrics["roc_auc"])
    for name, value in sorted(metrics.items()):
        logger.info("  %-10s %.4f", name, value)


if __name__ == "__main__":
    main()
