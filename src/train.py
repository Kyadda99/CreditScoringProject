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
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from api import FEATURE_SCHEMA_PATH as _API_FEATURE_SCHEMA_PATH
from artifact import DEFAULT_ARTIFACT_PATH, save_bundle
from config import (
    COST_FN,
    COST_FP,
    DROPPED_COLUMNS,
    ENGINEERED_FEATURES,
    FEATURE_SOURCE_COLUMNS,
    ID_COLUMN,
    INFORMATIVE_MISSING,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    TARGET,
    FeatureGroups,
    split_feature_groups,
)
from data import load_data
from evaluation import DEFAULT_THRESHOLD, choose_threshold, evaluate
from features import FeatureEngineer
from models import IMBALANCE_ARMS, get_models, get_sampler
from preprocessing import build_preprocessor
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
MODEL_NAME = "logistic_regression"
EXPERIMENT_NAME = "credit-scoring-baseline"
GRID_EXPERIMENT_NAME = "credit-scoring-phase3"
"""Osobny eksperyment: biegi Fazy 3 nie mają mieszać się z baseline'em."""
# Importowana, nie wyprowadzana ponownie — patrz docstring `api/__init__.py`.
# Wcześniejsza wersja liczyła tę ścieżkę z PROJECT_ROOT i rozjeżdżała się
# z API wszędzie poza katalogiem repozytorium.
FEATURE_SCHEMA_PATH: Path = _API_FEATURE_SCHEMA_PATH


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
    groups: FeatureGroups,
    estimator: BaseEstimator | None = None,
    sampler: BaseEstimator | None = None,
    features: tuple[str, ...] = ENGINEERED_FEATURES,
) -> Pipeline:
    """Składa inżynierię cech, preprocessing, opcjonalny sampler i estymator.

    `FeatureEngineer` jest **pierwszym** krokiem (spec Fazy 2, D1): dzięki temu
    cała derywacja jedzie w artefakcie, a `/score` przyjmuje surowe kolumny.
    Jeden `Pipeline` oznacza też, że imputacja i skalowanie uczą się wyłącznie
    na foldzie treningowym — nie ma miejsca na train/serve skew.

    Gdy `sampler` jest podany, zwracany jest `imblearn.pipeline.Pipeline`, który
    **stosuje sampler wyłącznie w `fit`** — w `transform` i `predict` krok jest
    bezczynny. To jest strukturalna gwarancja ze spec Fazy 3 D3: nie istnieje
    ścieżka kodu, którą resampling mógłby dotknąć foldu walidacyjnego albo
    żądania scoringowego.

    Args:
        groups: Grupy cech wyprowadzone na ramce treningowej **po** inżynierii.
        estimator: Estymator do wpięcia. `None` daje baseline z Fazy 2.
            Wartość domyślna istnieje celowo — `build_pipeline(groups)` ma pięć
            istniejących wywołań, w tym fixture `synthetic_artifact`, od której
            zależy cały zestaw testów HTTP.
        sampler: Sampler `imblearn` albo `None` dla pozostałych ramion.
        features: Cechy pochodne do policzenia. `()` daje transzę 1 z D7 —
            samo czyszczenie, bez ilorazów.

    Returns:
        Niedopasowany pipeline: `sklearn` bez samplera, `imblearn` z samplerem.
    """
    if estimator is None:
        estimator = get_models()[MODEL_NAME]

    steps = [
        ("features", FeatureEngineer(features=features)),
        ("preprocessor", build_preprocessor(groups)),
    ]
    if sampler is None:
        return Pipeline([*steps, ("model", estimator)])
    # Sampler ZA preprocessorem: SMOTE potrzebuje macierzy numerycznej,
    # więc nie może stanąć przed one-hot encodingiem.
    return ImbPipeline([*steps, ("sampler", sampler), ("model", estimator)])


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


# --- Faza 3: siatka eksperymentu (spec D2) ---------------------------------

ALL_MODELS: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")
"""Algorytmy siatki. PLAN.md wymaga co najmniej trzech."""

TABLE_COLUMNS: tuple[str, ...] = (
    "model",
    "imbalance",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "threshold",
    "expected_cost",
)
"""Kolumny tabeli porównawczej, w kolejności raportowania."""


def positive_ratio(y: pd.Series) -> float:
    """Zwraca stosunek negatywów do pozytywów — `scale_pos_weight` dla XGBoosta.

    Liczony z faktycznych etykiet, a nie ze stałej: gdyby higiena wierszy
    kiedyś zmieniła bazową częstość, model ma iść za danymi.

    Args:
        y: Etykiety 0/1.

    Returns:
        `neg / pos`.
    """
    positives = int(y.sum())
    return float((len(y) - positives) / positives)


def run_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: FeatureGroups,
    models: tuple[str, ...] = ALL_MODELS,
    arms: tuple[str, ...] = IMBALANCE_ARMS,
    features: tuple[str, ...] = ENGINEERED_FEATURES,
    track: bool = True,
) -> dict[tuple[str, str], dict[str, float]]:
    """Trenuje siatkę algorytm x ramię niezbalansowania na jednym podziale.

    Każda komórka to jeden bieg MLflow. Metryki liczone są przy progu 0.5 —
    strojenie progu dotyczy dopiero championa (spec D4), bo próg dobrany
    osobno dla każdej komórki uczyniłby kolumny tabeli nieporównywalnymi.

    Args:
        X_train: Cechy treningowe.
        y_train: Etykiety treningowe.
        X_test: Cechy testowe.
        y_test: Etykiety testowe.
        groups: Grupy cech modelu.
        models: Nazwy algorytmów z `get_models()`.
        arms: Ramiona z `IMBALANCE_ARMS`.
        features: Cechy pochodne.
        track: `False` wyłącza MLflow — używane w testach.

    Returns:
        `{(nazwa_modelu, ramię): metryki}`.
    """
    ratio = positive_ratio(y_train)
    logger.info("scale_pos_weight z danych: %.2f", ratio)

    results: dict[tuple[str, str], dict[str, float]] = {}
    for arm in arms:
        for name in models:
            estimator = get_models(arm, scale_pos_weight=ratio)[name]
            # Świeży sampler na komórkę — dopasowany obiekt nie może wyciec
            # do kolejnego biegu.
            pipeline = build_pipeline(groups, estimator, get_sampler(arm), features)

            logger.info("Trenuję %s / %s ...", name, arm)
            started = time.perf_counter()
            pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - started

            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test.to_numpy(), y_proba, threshold=DEFAULT_THRESHOLD)
            metrics["fit_seconds"] = float(elapsed)
            results[(name, arm)] = metrics

            if track:
                with mlflow.start_run(run_name=f"{name}__{arm}"):
                    mlflow.log_params(
                        {
                            "model": name,
                            "imbalance": arm,
                            "scale_pos_weight": ratio if arm == "class_weight" else 1.0,
                            "random_state": RANDOM_STATE,
                            "n_features": len(groups.all_features),
                            "n_engineered": len(features),
                        }
                    )
                    mlflow.log_metrics(metrics)

            logger.info(
                "  %-20s %-13s PR-AUC=%.4f ROC-AUC=%.4f recall=%.4f (%.1fs)",
                name,
                arm,
                metrics["pr_auc"],
                metrics["roc_auc"],
                metrics["recall"],
                elapsed,
            )
    return results


def comparison_table(
    results: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    """Zamienia wynik `run_grid` w tabelę porównawczą posortowaną po PR-AUC.

    Args:
        results: Wynik `run_grid`.

    Returns:
        Ramka z jedną linią na komórkę siatki, malejąco po PR-AUC (spec D1).
    """
    rows = [
        {"model": model, "imbalance": arm, **{c: m[c] for c in TABLE_COLUMNS[2:]}}
        for (model, arm), m in results.items()
    ]
    return (
        pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )


CV_SPLITS = 5
"""Foldy do wyznaczenia progu poza zbiorem testowym (spec D4)."""


def select_champion(
    results: dict[tuple[str, str], dict[str, float]],
) -> tuple[str, str]:
    """Wybiera komórkę siatki o najwyższym PR-AUC.

    PR-AUC, nie ROC-AUC (spec D1): przy 8.07% klasy dodatniej ROC-AUC jest
    zawyżane przez łatwą większość, a champion ma być dobry przy granicy
    decyzyjnej, bo to tam zapada decyzja kredytowa.

    Args:
        results: Wynik `run_grid`.

    Returns:
        `(nazwa_modelu, ramię)`.
    """
    model, arm = max(results, key=lambda key: results[key]["pr_auc"])
    logger.info(
        "Champion: %s / %s (PR-AUC %.4f)", model, arm, results[(model, arm)]["pr_auc"]
    )
    return model, arm


def fit_champion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: FeatureGroups,
    model_name: str,
    arm: str,
    features: tuple[str, ...] = ENGINEERED_FEATURES,
    cv_splits: int = CV_SPLITS,
) -> tuple[Pipeline, float, dict[str, float]]:
    """Dopasowuje championa i wyznacza jego próg decyzyjny.

    Kolejność jest istotna (spec D4):

    1. `cross_val_predict` na zbiorze **treningowym** daje prawdopodobieństwa
       out-of-fold — każdy wiersz oceniony przez model, który go nie widział.
    2. `choose_threshold` minimalizuje na nich koszt `10*FN + 1*FP`.
    3. Próg jedzie **niezmieniony** na zbiór testowy.

    Dobranie progu na teście i zaraportowanie kosztu przy tym progu byłoby
    dopasowaniem do zbioru testowego i zaraportowaniem tego dopasowania.

    Args:
        X_train: Cechy treningowe.
        y_train: Etykiety treningowe.
        X_test: Cechy testowe.
        y_test: Etykiety testowe.
        groups: Grupy cech modelu.
        model_name: Nazwa algorytmu.
        arm: Ramię niezbalansowania.
        features: Cechy pochodne.
        cv_splits: Liczba foldów do predykcji OOF.

    Returns:
        `(dopasowany pipeline, próg, metryki na teście przy tym progu)`.
    """
    ratio = positive_ratio(y_train)
    estimator = get_models(arm, scale_pos_weight=ratio)[model_name]

    logger.info("Wyznaczam próg z predykcji OOF (%d foldów) ...", cv_splits)
    oof_proba = cross_val_predict(
        build_pipeline(groups, estimator, get_sampler(arm), features),
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, oof_cost = choose_threshold(y_train.to_numpy(), oof_proba)
    logger.info(
        "Próg z OOF: %.4f (koszt %.0f). Analityczne optimum dla %.0f:1 to %.4f "
        "— różnica jest sygnałem kalibracji, nie błędem.",
        threshold,
        oof_cost,
        COST_FN / COST_FP,
        1.0 / (1.0 + COST_FN / COST_FP),
    )

    # Świeży pipeline: ten z cross_val_predict został dopasowany per fold,
    # a championa chcemy nauczyć na CAŁYM zbiorze treningowym.
    pipeline = build_pipeline(groups, estimator, get_sampler(arm), features)
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), y_proba, threshold=threshold)
    return pipeline, threshold, metrics


def main(
    engineered: bool = True,
    models: tuple[str, ...] = ALL_MODELS,
    arms: tuple[str, ...] = IMBALANCE_ARMS,
    register_champion: bool = True,
) -> None:
    """Pełna ścieżka Fazy 3: siatka -> champion -> próg -> bramka -> rejestr.

    Args:
        engineered: Gdy `False`, cechy pochodne są wyłączone — to transza 1
            ze spec Fazy 2 D7, mierząca sam efekt czyszczenia i obsługi braków.
        models: Algorytmy do przetrenowania.
        arms: Ramiona niezbalansowania.
        register_champion: `False` przerywa po tabeli porównawczej — używane,
            gdy chcemy przemierzyć siatkę bez dotykania rejestru.
    """
    features = ENGINEERED_FEATURES if engineered else ()

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # Grupy modelu wyprowadzamy z ramki PO inżynierii — inaczej cechy pochodne
    # nie trafiłyby do żadnej grupy, a `remainder="drop"` cicho by je wyrzucił.
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
    mlflow.set_experiment(GRID_EXPERIMENT_NAME)

    results = run_grid(X_train, y_train, X_test, y_test, groups, models, arms, features)
    table = comparison_table(results)
    logger.info("Tabela porównawcza:\n%s", table.to_string(index=False))

    if not register_champion:
        return

    model_name, arm = select_champion(results)
    pipeline, threshold, metrics = fit_champion(
        X_train, y_train, X_test, y_test, groups, model_name, arm, features
    )
    for key, value in sorted(metrics.items()):
        logger.info("  %-14s %.4f", key, value)

    # Bramka PRZED promocją. W projekcie referencyjnym ten moduł istnieje,
    # ale nie jest wołany, i promocja dzieje się bezwarunkowo (spec D5).
    if not quality_gate(metrics):
        logger.warning(
            "Champion nie przeszedł bramki — NIE promuję. Model zostaje "
            "w MLflow jako bieg, ale alias @production pozostaje bez zmian."
        )
        return

    # Druga bramka: progi bezwzględne nie wystarczają, bo model może je
    # spełnić i JEDNOCZEŚNIE być gorszy od tego, który już jest na produkcji.
    if not beats_incumbent(metrics, production_metrics()):
        logger.warning(
            "Kandydat nie bije obecnego championa — NIE promuję. "
            "Bieg zostaje w MLflow, alias @production bez zmian."
        )
        return

    metadata = {
        **metrics,
        "model": model_name,
        "imbalance": arm,
        "engineered": engineered,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "n_dropped": len(DROPPED_COLUMNS),
        "n_informative_missing": len(INFORMATIVE_MISSING),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
    }

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
        # Bundle jedzie jako artefakt TEGO biegu, żeby export_champion mógł
        # go później odnaleźć po aliasie -> wersji -> run_id (spec D7).
        local = save_bundle(pipeline, groups, inputs, metadata, DEFAULT_ARTIFACT_PATH)
        mlflow.log_artifact(str(local), artifact_path=BUNDLE_ARTIFACT_PATH)
        run_id = run.info.run_id

    promote(register(run_id))
    export_champion()
    write_feature_schema(inputs)
    logger.info("Champion promowany i wyeksportowany.")


def cli() -> None:
    """Wejście `uv run cs-train`."""
    import argparse

    parser = argparse.ArgumentParser(description="Trenuje siatkę modeli scoringowych.")
    parser.add_argument(
        "--no-engineered",
        action="store_true",
        help="Wyłącza cechy pochodne — mierzy sam efekt czyszczenia (spec D7).",
    )
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="Lista algorytmów po przecinku.",
    )
    parser.add_argument(
        "--imbalance",
        default=",".join(IMBALANCE_ARMS),
        help="Lista ramion niezbalansowania po przecinku.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Zatrzymuje się na tabeli porównawczej, nie dotyka rejestru.",
    )
    args = parser.parse_args()

    main(
        engineered=not args.no_engineered,
        models=tuple(args.models.split(",")),
        arms=tuple(args.imbalance.split(",")),
        register_champion=not args.no_register,
    )


if __name__ == "__main__":
    cli()
