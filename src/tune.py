"""Przeszukiwanie hiperparametrów championa przez Optunę.

Budżet jest **zadeklarowany z góry**, a nie odkrywany w trakcie (spec D8):
30 prób, 3 foldy, 25% stratyfikowanej podpróbki zbioru treningowego.
`PLAN.md` wskazuje Optunę na 300k wierszach z walidacją krzyżową jako główne
ryzyko czasowe fazy; tak wygląda ograniczenie tego ryzyka.

Najważniejsza różnica wobec projektu referencyjnego instruktora: najlepsze
parametry są dopasowywane ponownie na **pełnym zbiorze treningowym**, nigdy
na całości danych. Referencyjne `best_pipe.fit(_X, _y)` zjadłoby zbiór
testowy, na którym opiera się każde porównanie międzyfazowe tego projektu —
od baseline'u z Fazy 1 począwszy.

Uwaga o `MLflowCallback`: jest **deprecated** od optuna-integration 4.9.0
i zniknie w 6.0.0. Zostaje, bo `CLAUDE.md` wymienia `optuna-integration[mlflow]`
w stosie, a projekt referencyjny instruktora używa dokładnie tego wywołania —
zgodność z materiałem kursu jest tu więcej warta niż wyprzedzanie migracji.
Gdy przyjdzie 6.0.0: logować metryki próby ręcznie w `objective`.

Usage:
    uv run cs-tune --trials 30 --subsample 0.25
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import mlflow
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from artifact import DEFAULT_ARTIFACT_PATH, save_bundle
from config import (
    ENGINEERED_FEATURES,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    FeatureGroups,
)
from data_loader import DataLoader
from evaluation import choose_threshold, evaluate
from preprocessor import Preprocessor
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
from settings import get_training_settings
from strategies import get_estimator_strategy, get_imbalance_strategy
from train import write_feature_schema
from trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_TRIALS = 30
"""Budżet prób — zadeklarowany, nie odkrywany w trakcie biegu."""

CV_SPLITS = 3
"""Mniej foldów niż przy wyznaczaniu progu: strojenie ma być tanie."""

SUBSAMPLE_FRACTION = 0.25
"""Ułamek zbioru treningowego używany w przeszukiwaniu (~61.5k wierszy)."""

TUNE_EXPERIMENT_NAME = "credit-scoring-optuna"
"""Osobny eksperyment, żeby próby nie zaśmiecały tabeli siatki."""

TUNED_MODEL = "xgboost"
"""Strojony jest tylko lider siatki — trzy przeszukiwania dla jednego
championa byłyby wydatkiem bez pokrycia w dowodach."""


def subsample(
    X: pd.DataFrame, y: pd.Series, fraction: float, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series]:
    """Zwraca stratyfikowaną podpróbkę zbioru treningowego.

    Stratyfikacja jest konieczna: przy 8% klasy dodatniej losowa podpróbka
    potrafi zmienić bazową częstość na tyle, że strojenie optymalizuje inny
    problem niż ten, który potem trenujemy.

    Args:
        X: Cechy.
        y: Etykiety.
        fraction: Ułamek do zachowania; `>= 1.0` zwraca wszystko bez kopiowania.
        seed: Ziarno losowania.

    Returns:
        `(X_podpróbka, y_podpróbka)`.
    """
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=fraction, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def build_objective(
    X: pd.DataFrame,
    y: pd.Series,
    groups: FeatureGroups,
    arm: str,
    features: tuple[str, ...] = ENGINEERED_FEATURES,
    cv_splits: int = CV_SPLITS,
) -> Callable[[optuna.Trial], float]:
    """Buduje funkcję celu domkniętą nad danymi.

    Domknięcie zamiast globali modułu (tak robi projekt referencyjny): funkcja
    celu jest wtedy testowalna bez ustawiania stanu modułu, a dwa równoległe
    przeszukiwania nie mogą sobie nadpisać danych.

    Cel to **średnie CV PR-AUC** (`average_precision`), zgodnie ze spec D1 —
    ta sama metryka, którą wybieramy championa w siatce.

    Args:
        X: Cechy przeszukiwania (zwykle podpróbka).
        y: Etykiety przeszukiwania.
        groups: Grupy cech modelu.
        arm: Ramię niezbalansowania championa.
        features: Cechy pochodne.
        cv_splits: Liczba foldów.

    Returns:
        Funkcja `objective(trial) -> średnie CV PR-AUC`.
    """
    ratio = ModelTrainer.positive_ratio(y)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        imbalance = get_imbalance_strategy(arm)
        estimator = get_estimator_strategy(TUNED_MODEL).build(
            class_weighted=imbalance.uses_class_weights, scale_pos_weight=ratio
        )
        # n_jobs=1 na estymatorze: cross_val_score bierze -1, a zagnieżdżona
        # równoległość na 211 kolumnach walczy sama ze sobą o rdzenie i potrafi
        # być wolniejsza niż wariant jednowątkowy.
        estimator.set_params(**params, n_jobs=1)

        pipeline = Preprocessor(groups, features).build_pipeline(
            estimator, imbalance.make_sampler()
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=StratifiedKFold(
                n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
            ),
            scoring="average_precision",
            n_jobs=-1,
        )
        return float(scores.mean())

    return objective


def make_study() -> optuna.Study:
    """Tworzy powtarzalne badanie maksymalizujące PR-AUC.

    Returns:
        Puste `optuna.Study` z zasianym `TPESampler`.
    """
    return optuna.create_study(
        direction="maximize",
        study_name="xgboost_hpo",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )


def tune(
    n_trials: int = N_TRIALS,
    fraction: float = SUBSAMPLE_FRACTION,
    arm: str = "class_weight",
    engineered: bool = True,
) -> optuna.Study:
    """Przeszukuje hiperparametry i promuje wynik, jeśli przejdzie bramkę.

    Args:
        n_trials: Budżet prób.
        fraction: Ułamek zbioru treningowego dla przeszukiwania.
        arm: Ramię niezbalansowania wygrane przez championa w siatce.
        engineered: Czy liczyć cechy pochodne.

    Returns:
        Zakończone badanie Optuny.
    """
    features = ENGINEERED_FEATURES if engineered else ()
    loader = DataLoader.from_settings(get_training_settings())
    X_train, X_test, y_train, y_test = loader.split(loader.load())
    groups = loader.model_groups(X_train, features)
    inputs = loader.input_contract(X_train)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(TUNE_EXPERIMENT_NAME)

    X_search, y_search = subsample(X_train, y_train, fraction)
    logger.info(
        "Przeszukiwanie na %d wierszach (%.0f%% treningu), %d prób, %d foldów.",
        len(X_search),
        fraction * 100,
        n_trials,
        CV_SPLITS,
    )

    study = make_study()
    study.optimize(
        build_objective(X_search, y_search, groups, arm, features),
        n_trials=n_trials,
        callbacks=[
            # create_experiment=False jest konieczne, a nie kosmetyczne:
            # domyślnie callback zakłada WŁASNY eksperyment nazwany jak study
            # ("xgboost_hpo") i ignoruje ten ustawiony wyżej. Próby lądowałyby
            # wtedy gdzie indziej niż bieg championa i porównanie ich w UI
            # wymagałoby przeskakiwania między eksperymentami.
            MLflowCallback(
                tracking_uri=MLFLOW_TRACKING_URI,
                metric_name="pr_auc",
                create_experiment=False,
            )
        ],
        n_jobs=1,
    )
    logger.info("Najlepsze CV PR-AUC: %.4f", study.best_value)
    logger.info("Najlepsze parametry: %s", study.best_params)

    # Refit na PEŁNYM zbiorze treningowym — nigdy na całości danych (spec D8).
    ratio = ModelTrainer.positive_ratio(y_train)
    imbalance = get_imbalance_strategy(arm)
    estimator = get_estimator_strategy(TUNED_MODEL).build(
        class_weighted=imbalance.uses_class_weights, scale_pos_weight=ratio
    )
    estimator.set_params(**study.best_params)

    # Próg z predykcji out-of-fold na treningu — nigdy z testu (spec D4).
    oof_proba = cross_val_predict(
        Preprocessor(groups, features).build_pipeline(
            estimator, imbalance.make_sampler()
        ),
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, _oof_cost = choose_threshold(y_train.to_numpy(), oof_proba)

    pipeline = Preprocessor(groups, features).build_pipeline(
        estimator, imbalance.make_sampler()
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate(
        y_test.to_numpy(), pipeline.predict_proba(X_test)[:, 1], threshold=threshold
    )
    for key, value in sorted(metrics.items()):
        logger.info("  %-14s %.4f", key, value)

    if not quality_gate(metrics):
        logger.warning(
            "Strojony model nie przeszedł bramki — NIE promuję. "
            "Alias @production pozostaje przy poprzednim championie."
        )
        return study

    # Druga bramka: progi bezwzględne nie wystarczają, bo model może je
    # spełnić i JEDNOCZEŚNIE być gorszy od tego, który już jest na produkcji.
    if not beats_incumbent(metrics, production_metrics()):
        logger.warning(
            "Kandydat nie bije obecnego championa — NIE promuję. "
            "Bieg zostaje w MLflow, alias @production bez zmian."
        )
        return study

    metadata = {
        **metrics,
        "model": TUNED_MODEL,
        "imbalance": arm,
        "tuned": True,
        "n_trials": n_trials,
        "subsample_fraction": fraction,
        "cv_pr_auc": float(study.best_value),
        **{f"param_{k}": v for k, v in study.best_params.items()},
    }
    with mlflow.start_run(run_name=f"champion__{TUNED_MODEL}__tuned") as run:
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_PATH,
            serialization_format=SERIALIZATION_FORMAT,
        )
        local = save_bundle(pipeline, groups, inputs, metadata, DEFAULT_ARTIFACT_PATH)
        mlflow.log_artifact(str(local), artifact_path=BUNDLE_ARTIFACT_PATH)
        run_id = run.info.run_id

    promote(register(run_id))
    export_champion()
    write_feature_schema(inputs)
    logger.info("Strojony champion promowany i wyeksportowany.")
    return study


def cli() -> None:
    """Wejście `uv run cs-tune`."""
    import argparse

    parser = argparse.ArgumentParser(description="Stroi hiperparametry championa.")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--subsample", type=float, default=SUBSAMPLE_FRACTION)
    parser.add_argument(
        "--arm",
        default="class_weight",
        help="Ramię niezbalansowania wygrane w siatce.",
    )
    parser.add_argument("--no-engineered", action="store_true")
    args = parser.parse_args()

    tune(
        n_trials=args.trials,
        fraction=args.subsample,
        arm=args.arm,
        engineered=not args.no_engineered,
    )
