"""Hyperparameter search for the champion, using Optuna.

The budget is fixed up front: 30 trials, 3 folds, a 25% stratified subsample
of the training set.

The best parameters are refit on the full training set, never on all the data.
Refitting on everything would consume the test set that every cross-run
comparison in this project rests on.

`MLflowCallback` is deprecated from optuna-integration 4.9.0 and is removed in
6.0.0. When that lands, log trial metrics directly in `objective`.

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
"""Trial budget, fixed up front rather than discovered during the run."""

CV_SPLITS = 3
"""Fewer folds than threshold selection uses: tuning must stay cheap."""

SUBSAMPLE_FRACTION = 0.25
"""Fraction of the training set used for the search (~61.5k rows)."""

TUNE_EXPERIMENT_NAME = "credit-scoring-optuna"
"""A separate experiment so trials do not clutter the grid table."""

TUNED_MODEL = "xgboost"
"""Only the grid leader is tuned; three searches for one champion would
cost more than the evidence supports."""


def subsample(
    X: pd.DataFrame, y: pd.Series, fraction: float, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a stratified subsample of the training set.

    Stratification is required: at 8% positives a random subsample can shift
    the base rate enough that tuning optimises a different problem from the
    one later trained.

    Args:
        X: Features.
        y: Labels.
        fraction: Fraction to keep; `>= 1.0` returns everything uncopied.
        seed: Sampling seed.

    Returns:
        `(X_subsample, y_subsample)`.
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
    """Build an objective function closed over the data.

    A closure rather than module globals: the objective is testable without
    setting module state, and two parallel searches cannot overwrite each
    other's data.

    The objective is mean CV PR-AUC (`average_precision`), the same metric
    that picks the champion in the grid.

    Args:
        X: Search features, usually a subsample.
        y: Search labels.
        groups: Model feature groups.
        arm: The champion's imbalance arm.
        features: Derived features.
        cv_splits: Number of folds.

    Returns:
        A function `objective(trial) -> mean CV PR-AUC`.
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
        # n_jobs=1 on the estimator: cross_val_score already takes -1, and
        # nested parallelism on 211 columns contends for cores.
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
    """Create a reproducible study maximising PR-AUC.

    Returns:
        An empty `optuna.Study` with a seeded `TPESampler`.
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
    """Search hyperparameters and promote the result if it clears the gate.

    Args:
        n_trials: Trial budget.
        fraction: Fraction of the training set used for the search.
        arm: Imbalance arm the champion won the grid with.
        engineered: Whether to compute derived features.

    Returns:
        The completed Optuna study.
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
        "Searching on %d rows (%.0f%% of train), %d trials, %d folds.",
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
            # create_experiment=False is required: by default the callback
            # creates its own experiment named after the study and ignores the
            # one set above, scattering trials away from the champion run.
            MLflowCallback(
                tracking_uri=MLFLOW_TRACKING_URI,
                metric_name="pr_auc",
                create_experiment=False,
            )
        ],
        n_jobs=1,
    )
    logger.info("Best CV PR-AUC: %.4f", study.best_value)
    logger.info("Best parameters: %s", study.best_params)

    # Refit on the full training set, never on all the data.
    ratio = ModelTrainer.positive_ratio(y_train)
    imbalance = get_imbalance_strategy(arm)
    estimator = get_estimator_strategy(TUNED_MODEL).build(
        class_weighted=imbalance.uses_class_weights, scale_pos_weight=ratio
    )
    estimator.set_params(**study.best_params)

    # Threshold from out-of-fold predictions on train, never from test.
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
            "Tuned model did not clear the gate - not promoting. "
            "The @production alias stays on the previous champion."
        )
        return study

    # Second gate: absolute thresholds are not enough, because a model can
    # meet them and still be worse than the one already in production.
    if not beats_incumbent(metrics, production_metrics()):
        logger.warning(
            "Candidate does not beat the current champion - not promoting. "
            "The run stays in MLflow, the @production alias is unchanged."
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
    logger.info("Tuned champion promoted and exported.")
    return study


def cli() -> None:
    """Entry point for `uv run cs-tune`."""
    import argparse

    parser = argparse.ArgumentParser(description="Tune the champion hyperparameters.")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--subsample", type=float, default=SUBSAMPLE_FRACTION)
    parser.add_argument(
        "--arm",
        default="class_weight",
        help="Imbalance arm that won the grid.",
    )
    parser.add_argument("--no-engineered", action="store_true")
    args = parser.parse_args()

    tune(
        n_trials=args.trials,
        fraction=args.subsample,
        arm=args.arm,
        engineered=not args.no_engineered,
    )
