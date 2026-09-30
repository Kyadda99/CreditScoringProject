"""Training and champion selection.

The algorithm x imbalance-arm grid runs on one split. The decision threshold is
chosen only for the champion, from out-of-fold predictions on the training set;
a threshold computed on the test set would be fitted to it.
"""

from __future__ import annotations

import logging
import time

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from config import COST_FN, COST_FP, RANDOM_STATE
from evaluation import DEFAULT_THRESHOLD, choose_threshold, evaluate
from preprocessor import Preprocessor
from settings import TrainingSettings
from strategies import (
    ESTIMATOR_STRATEGIES,
    IMBALANCE_STRATEGIES,
    get_estimator_strategy,
    get_imbalance_strategy,
)

logger = logging.getLogger(__name__)

ALL_MODELS: tuple[str, ...] = tuple(ESTIMATOR_STRATEGIES)
"""Grid algorithms, taken from the strategy registry to avoid a second list."""

ALL_ARMS: tuple[str, ...] = tuple(IMBALANCE_STRATEGIES)
"""Imbalance arms, taken from the strategy registry."""

CV_SPLITS = 5
"""Folds used to choose the threshold outside the test set."""

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
"""Comparison table columns, in reporting order."""


class ModelTrainer:
    """Runs the experiment: grid, champion, threshold.

    Attributes:
        preprocessor: Pipeline builder for the fixed feature groups.
        models: Algorithms to train.
        arms: Imbalance arms.
        cv_splits: Number of folds used to choose the threshold.
        track: Whether to log runs to MLflow.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        models: tuple[str, ...] = ALL_MODELS,
        arms: tuple[str, ...] = ALL_ARMS,
        cv_splits: int = CV_SPLITS,
        track: bool = True,
    ) -> None:
        """Store the scope of the experiment.

        Args:
            preprocessor: Pipeline builder.
            models: Algorithm names.
            arms: Imbalance arm names.
            cv_splits: Number of folds for out-of-fold predictions.
            track: `False` disables MLflow; used in tests.
        """
        self.preprocessor = preprocessor
        self.models = models
        self.arms = arms
        self.cv_splits = cv_splits
        self.track = track

    @classmethod
    def from_settings(
        cls, settings: TrainingSettings, preprocessor: Preprocessor
    ) -> ModelTrainer:
        """Build a trainer from the configuration.

        Args:
            settings: Training settings.
            preprocessor: Pipeline builder for the feature groups.

        Returns:
            A trainer scoped from the configuration.
        """
        return cls(
            preprocessor,
            models=settings.models,
            arms=settings.arms,
            cv_splits=settings.cv_splits,
            track=settings.track,
        )

    @staticmethod
    def positive_ratio(y: pd.Series) -> float:
        """Return the ratio of negatives to positives.

        Computed from the actual labels, not a constant: if row hygiene changes
        the base rate, the model follows the data.

        Args:
            y: 0/1 labels.

        Returns:
            `neg / pos`.
        """
        positives = int(y.sum())
        return float((len(y) - positives) / positives)

    def _pipeline_for(self, name: str, arm: str, ratio: float) -> Pipeline:
        """Assemble the pipeline for one grid cell.

        Args:
            name: Algorithm name.
            arm: Imbalance arm.
            ratio: Class ratio for estimators that weight by number.

        Returns:
            An unfitted pipeline.
        """
        imbalance = get_imbalance_strategy(arm)
        estimator = get_estimator_strategy(name).build(
            class_weighted=imbalance.uses_class_weights,
            scale_pos_weight=ratio,
        )
        return self.preprocessor.build_pipeline(estimator, imbalance.make_sampler())

    def run_params(self, name: str, arm: str, ratio: float) -> dict[str, object]:
        """Build the run parameters describing one grid cell.

        The weight comes from the same strategy as the model, not from
        comparing the arm name, so the logged run cannot describe a different
        model from the one built.

        Args:
            name: Algorithm name.
            arm: Imbalance arm.
            ratio: Ratio of negatives to positives.

        Returns:
            The parameter dictionary to log with the run.
        """
        imbalance = get_imbalance_strategy(arm)
        return {
            "model": name,
            "imbalance": arm,
            "scale_pos_weight": ratio if imbalance.uses_class_weights else 1.0,
            "random_state": RANDOM_STATE,
            "n_features": len(self.preprocessor.groups.all_features),
            "n_engineered": len(self.preprocessor.features),
        }

    def run_grid(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Train the algorithm x arm grid on one split.

        Each cell is one MLflow run. Metrics use a 0.5 threshold; the threshold
        is tuned only for the champion, because a per-cell threshold would make
        the table columns incomparable.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            `{(model_name, arm): metrics}`.
        """
        ratio = self.positive_ratio(y_train)
        logger.info("scale_pos_weight from the data: %.2f", ratio)

        results: dict[tuple[str, str], dict[str, float]] = {}
        for arm in self.arms:
            for name in self.models:
                pipeline = self._pipeline_for(name, arm, ratio)

                logger.info("Training %s / %s ...", name, arm)
                started = time.perf_counter()
                pipeline.fit(X_train, y_train)
                elapsed = time.perf_counter() - started

                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics = evaluate(
                    y_test.to_numpy(), y_proba, threshold=DEFAULT_THRESHOLD
                )
                metrics["fit_seconds"] = float(elapsed)
                results[(name, arm)] = metrics

                if self.track:
                    with mlflow.start_run(run_name=f"{name}__{arm}"):
                        mlflow.log_params(self.run_params(name, arm, ratio))
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

    @staticmethod
    def comparison_table(
        results: dict[tuple[str, str], dict[str, float]],
    ) -> pd.DataFrame:
        """Turn `run_grid` output into a comparison table, descending by PR-AUC.

        Args:
            results: Output of `run_grid`.

        Returns:
            A frame with one row per grid cell.
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

    @staticmethod
    def select_champion(
        results: dict[tuple[str, str], dict[str, float]],
    ) -> tuple[str, str]:
        """Select the grid cell with the highest PR-AUC.

        PR-AUC, not ROC-AUC: at 8.07% positives ROC-AUC is inflated by the easy
        majority, and the champion must be good at the decision boundary, where
        the lending decision is made.

        Args:
            results: Output of `run_grid`.

        Returns:
            `(model_name, arm)`.
        """
        model, arm = max(results, key=lambda key: results[key]["pr_auc"])
        logger.info(
            "Champion: %s / %s (PR-AUC %.4f)",
            model,
            arm,
            results[(model, arm)]["pr_auc"],
        )
        return model, arm

    def fit_champion(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        arm: str,
    ) -> tuple[Pipeline, float, dict[str, float]]:
        """Fit the champion and choose its decision threshold.

        The order matters:

        1. `cross_val_predict` on the training set gives out-of-fold
           probabilities: every row scored by a model that did not see it.
        2. `choose_threshold` minimises `10*FN + 1*FP` on those.
        3. The threshold carries over unchanged to the test set.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            model_name: Algorithm name.
            arm: Imbalance arm.

        Returns:
            `(fitted pipeline, threshold, test metrics at that threshold)`.
        """
        ratio = self.positive_ratio(y_train)

        logger.info(
            "Choosing threshold from OOF predictions (%d folds) ...", self.cv_splits
        )
        oof_proba = cross_val_predict(
            self._pipeline_for(model_name, arm, ratio),
            X_train,
            y_train,
            cv=StratifiedKFold(
                n_splits=self.cv_splits, shuffle=True, random_state=RANDOM_STATE
            ),
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold, oof_cost = choose_threshold(y_train.to_numpy(), oof_proba)
        logger.info(
            "OOF threshold: %.4f (cost %.0f). Analytic optimum for %.0f:1 is "
            "%.4f; the gap reflects calibration, not an error.",
            threshold,
            oof_cost,
            COST_FN / COST_FP,
            1.0 / (1.0 + COST_FN / COST_FP),
        )

        # Fresh pipeline: the one from cross_val_predict was fitted per
        # fold, while the champion trains on the whole training set.
        pipeline = self._pipeline_for(model_name, arm, ratio)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), y_proba, threshold=threshold)
        return pipeline, threshold, metrics
