"""Orkiestracja treningu: dane -> podział -> pipeline -> metryki -> artefakt.

Ten moduł niczego nie wymyśla — składa gotowe elementy z pozostałych modułów
i zapisuje wynik. Cała logika mieszka tam, gdzie da się ją testować bez 158 MB
danych na dysku.

Usage:
    uv run cs-train
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import sklearn
from sklearn.pipeline import Pipeline

from api import FEATURE_SCHEMA_PATH as _API_FEATURE_SCHEMA_PATH
from artifact import DEFAULT_ARTIFACT_PATH, save_bundle
from config import (
    COST_FN,
    COST_FP,
    DROPPED_COLUMNS,
    INFORMATIVE_MISSING,
    FeatureGroups,
)
from data_loader import DataLoader
from quality_gate import beats_incumbent, quality_gate
from registry import (
    BUNDLE_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    SERIALIZATION_FORMAT,
    export_champion,
    production_metrics,
    promote,
    register,
)
from settings import TrainingSettings, get_training_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRID_EXPERIMENT_NAME = "credit-scoring-phase3"
"""Osobny eksperyment: biegi Fazy 3 nie mają mieszać się z baseline'em."""
# Importowana, nie wyprowadzana ponownie — patrz docstring `api/__init__.py`.
# Wcześniejsza wersja liczyła tę ścieżkę z PROJECT_ROOT i rozjeżdżała się
# z API wszędzie poza katalogiem repozytorium.
FEATURE_SCHEMA_PATH: Path = _API_FEATURE_SCHEMA_PATH


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


def champion_metadata(
    metrics: dict[str, float],
    model_name: str,
    arm: str,
    engineered: bool,
    loader: DataLoader,
    rows_train: int,
    rows_test: int,
) -> dict[str, object]:
    """Buduje metadane artefaktu opisujące ten konkretny przebieg.

    Podział bierzemy z `loader`, a nie ze stałej: metadane mają opisywać to,
    co faktycznie się wydarzyło, bo to one trafiają do rejestru i do artefaktu.

    Args:
        metrics: Metryki championa na zbiorze testowym.
        model_name: Nazwa algorytmu.
        arm: Ramię niezbalansowania.
        engineered: Czy liczono cechy pochodne.
        loader: Loader, który wykonał podział.
        rows_train: Liczba wierszy treningowych.
        rows_test: Liczba wierszy testowych.

    Returns:
        Słownik metadanych.
    """
    return {
        **metrics,
        "model": model_name,
        "imbalance": arm,
        "engineered": engineered,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "n_dropped": len(DROPPED_COLUMNS),
        "n_informative_missing": len(INFORMATIVE_MISSING),
        "test_size": loader.test_size,
        "random_state": loader.random_state,
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "rows_train": rows_train,
        "rows_test": rows_test,
    }


def build_parser(settings: TrainingSettings) -> argparse.ArgumentParser:
    """Buduje parser CLI, którego domyślne wartości pochodzą z konfiguracji.

    Args:
        settings: Ustawienia treningu.

    Returns:
        Gotowy parser.
    """
    parser = argparse.ArgumentParser(description="Trenuje siatkę modeli scoringowych.")
    parser.add_argument(
        "--engineered",
        action=argparse.BooleanOptionalAction,
        default=settings.engineered,
        help="Cechy pochodne; --no-engineered mierzy sam efekt czyszczenia.",
    )
    parser.add_argument(
        "--models",
        default=",".join(settings.models),
        help="Lista algorytmów po przecinku.",
    )
    parser.add_argument(
        "--imbalance",
        default=",".join(settings.arms),
        help="Lista ramion niezbalansowania po przecinku.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Zatrzymuje się na tabeli porównawczej, nie dotyka rejestru.",
    )
    return parser


def promote_champion(
    pipeline: Pipeline,
    threshold: float,
    metrics: dict[str, float],
    model_name: str,
    arm: str,
    groups: FeatureGroups,
    inputs: FeatureGroups,
    engineered: bool,
    loader: DataLoader,
    rows_train: int,
    rows_test: int,
) -> bool:
    """Przepuszcza championa przez obie bramki i, gdy przejdzie, promuje go.

    Args:
        pipeline: Dopasowany potok championa.
        threshold: Próg decyzyjny.
        metrics: Metryki na zbiorze testowym.
        model_name: Nazwa algorytmu.
        arm: Ramię niezbalansowania.
        groups: Grupy cech modelu.
        inputs: Kontrakt wejściowy.
        engineered: Czy liczono cechy pochodne.
        loader: Loader, który wykonał podział.
        rows_train: Liczba wierszy treningowych.
        rows_test: Liczba wierszy testowych.

    Returns:
        `True`, gdy model został promowany.
    """
    # Bramka PRZED promocją: progi bezwzględne odpowiadają na pytanie
    # "czy wystarczająco dobry".
    if not quality_gate(metrics):
        logger.warning(
            "Champion nie przeszedł bramki — NIE promuję. Model zostaje "
            "w MLflow jako bieg, ale alias @production pozostaje bez zmian."
        )
        return False

    # Druga bramka: model może spełnić progi i JEDNOCZEŚNIE być gorszy
    # od tego, który już jest na produkcji.
    if not beats_incumbent(metrics, production_metrics()):
        logger.warning(
            "Kandydat nie bije obecnego championa — NIE promuję. "
            "Bieg zostaje w MLflow, alias @production bez zmian."
        )
        return False

    metadata = champion_metadata(
        metrics=metrics,
        model_name=model_name,
        arm=arm,
        engineered=engineered,
        loader=loader,
        rows_train=rows_train,
        rows_test=rows_test,
    )

    with mlflow.start_run(run_name=f"champion__{model_name}__{arm}") as run:
        mlflow.log_params(
            {"model": model_name, "imbalance": arm, "threshold": threshold}
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_PATH,
            serialization_format=SERIALIZATION_FORMAT,
        )
        # Bundle jedzie jako artefakt TEGO biegu, żeby eksport mógł go później
        # odnaleźć po aliasie -> wersji -> run_id.
        local = save_bundle(pipeline, groups, inputs, metadata, DEFAULT_ARTIFACT_PATH)
        mlflow.log_artifact(str(local), artifact_path=BUNDLE_ARTIFACT_PATH)
        run_id = run.info.run_id

    promote(register(run_id))
    export_champion()
    write_feature_schema(inputs)
    logger.info("Champion promowany i wyeksportowany.")
    return True


def cli() -> None:
    """Wejście `uv run cs-train` — uruchamia przepływ Prefect."""
    # Import lokalny: `flows` importuje ten moduł, więc import na górze
    # zamknąłby cykl.
    from flows import training_pipeline

    settings = get_training_settings()
    args = build_parser(settings).parse_args()

    training_pipeline(
        register_champion=not args.no_register,
        settings=settings.model_copy(
            update={
                "engineered": args.engineered,
                "models": tuple(args.models.split(",")),
                "arms": tuple(args.imbalance.split(",")),
            }
        ),
    )


if __name__ == "__main__":
    cli()
