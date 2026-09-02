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
    DROPPED_COLUMNS,
    ENGINEERED_FEATURES,
    FEATURE_SOURCE_COLUMNS,
    ID_COLUMN,
    INFORMATIVE_MISSING,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET,
    FeatureGroups,
    split_feature_groups,
)
from data import load_data
from evaluation import DEFAULT_THRESHOLD, evaluate
from features import FeatureEngineer
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


def build_pipeline(
    groups: FeatureGroups, features: tuple[str, ...] = ENGINEERED_FEATURES
) -> Pipeline:
    """Składa inżynierię cech, preprocessing i estymator w jeden obiekt.

    `FeatureEngineer` jest **pierwszym** krokiem (spec D1): dzięki temu cała
    derywacja jedzie w artefakcie, a `/score` przyjmuje surowe kolumny. Jeden
    `Pipeline` oznacza też, że imputacja i skalowanie uczą się wyłącznie na
    foldzie treningowym — nie ma miejsca na train/serve skew.

    Args:
        groups: Grupy cech wyprowadzone na ramce treningowej **po** inżynierii.
        features: Cechy pochodne do policzenia. `()` daje transzę 1 z D7 —
            samo czyszczenie, bez ilorazów.

    Returns:
        Niedopasowany `Pipeline`.
    """
    return Pipeline(
        [
            ("features", FeatureEngineer(features=features)),
            ("preprocessor", build_preprocessor(groups)),
            ("model", get_models()[MODEL_NAME]),
        ]
    )


def input_contract(
    X_train: pd.DataFrame, dropped: tuple[str, ...] = DROPPED_COLUMNS
) -> FeatureGroups:
    """Wyprowadza kontrakt wejściowy `/score` z surowej ramki treningowej.

    Reguła przynależności (spec D2): kolumna jest przyjmowana, jeśli **albo**
    zostaje cechą modelu, **albo** karmi cechę pochodną. Drugi warunek jest
    istotny — kolumna usunięta z modelu, ale licząca się do ilorazu, musi
    nadal być przyjmowana, inaczej iloraz zawsze wychodzi NaN.

    Args:
        X_train: Surowa ramka treningowa, bez celu i identyfikatora.
        dropped: Kolumny usunięte decyzją z EDA.

    Returns:
        Grupy **surowych** kolumn — bez cech pochodnych.
    """
    kept = set(X_train.columns) - set(dropped) | set(FEATURE_SOURCE_COLUMNS)
    accepted = [c for c in X_train.columns if c in kept]
    return split_feature_groups(X_train[accepted])


def write_feature_schema(
    groups: FeatureGroups, path: Path = FEATURE_SCHEMA_PATH
) -> Path:
    """Zapisuje **kontrakt wejściowy** jako commitowany plik dla schematu API.

    API musi znać nazwy i typy pól, żeby zbudować model żądania, ale artefakt
    jest gitignorowany — nie ma go ani w CI, ani w świeżym klonie. Ten plik
    jest tym, z czego `api/schemas.py` generuje `ScoreRequest`.

    Uwaga (spec D2): od Fazy 2 dostaje **kontrakt wejściowy** (surowe kolumny),
    a nie grupy modelu. Grupy modelu zawierają cechy pochodne, których klient
    nie ma jak przysłać — zapisanie ich tutaj wygenerowałoby `/score`
    przyjmujące `CREDIT_INCOME_RATIO`.

    Args:
        groups: Kontrakt wejściowy z `input_contract()`.
        path: Ścieżka docelowa.

    Returns:
        Ścieżka zapisanego pliku.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(groups.to_dict(), indent=2), encoding="utf-8")
    logger.info("Zapisano kontrakt cech: %s", path)
    return path


def main(engineered: bool = True) -> None:
    """Trenuje model bazowy i zapisuje wszystko, czego potrzebuje serwowanie.

    Args:
        engineered: Gdy `False`, cechy pochodne są wyłączone — to transza 1
            ze spec D7, mierząca sam efekt czyszczenia i obsługi braków.
    """
    features = ENGINEERED_FEATURES if engineered else ()
    run_name = MODEL_NAME if engineered else f"{MODEL_NAME}_no_engineered"

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # Grupy modelu wyprowadzamy z ramki PO inżynierii — inaczej cechy pochodne
    # nie trafiłyby do żadnej grupy, a `remainder="drop"` cicho by je wyrzucił,
    # i cała Faza 2 nie zmieniłaby ani jednej liczby.
    engineered_train = FeatureEngineer(features=features).fit_transform(X_train)
    groups = split_feature_groups(engineered_train, dropped=DROPPED_COLUMNS)
    inputs = input_contract(X_train)
    logger.info(
        "Model: %d num / %d kat / %d bin. Kontrakt wejściowy: %d kolumn.",
        len(groups.numeric),
        len(groups.categorical),
        len(groups.binary),
        len(inputs.all_features),
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        pipeline = build_pipeline(groups, features=features)
        logger.info("Trenuję %s na %d wierszach ...", run_name, len(X_train))
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), y_proba, threshold=DEFAULT_THRESHOLD)

        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "engineered": engineered,
                "n_engineered": len(features),
                "n_dropped": len(DROPPED_COLUMNS),
                "n_informative_missing": len(INFORMATIVE_MISSING),
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
            "engineered": engineered,
            "trained_at": datetime.now(UTC).isoformat(),
            "sklearn_version": sklearn.__version__,
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
        }
        save_bundle(pipeline, groups, inputs, metadata, DEFAULT_ARTIFACT_PATH)
        write_feature_schema(inputs)

    logger.info("ROC-AUC (%s): %.4f", run_name, metrics["roc_auc"])
    for name, value in sorted(metrics.items()):
        logger.info("  %-10s %.4f", name, value)


def cli() -> None:
    """Wejście `uv run cs-train`. `--no-engineered` daje transzę 1 z D7."""
    import argparse

    parser = argparse.ArgumentParser(description="Trenuje model scoringowy.")
    parser.add_argument(
        "--no-engineered",
        action="store_true",
        help="Wyłącza cechy pochodne — mierzy sam efekt czyszczenia (spec D7).",
    )
    args = parser.parse_args()
    main(engineered=not args.no_engineered)


if __name__ == "__main__":
    cli()
