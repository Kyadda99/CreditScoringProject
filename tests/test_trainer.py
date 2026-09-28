"""Kontrakt ModelTrainera: siatka, champion i próg."""

from __future__ import annotations

import pandas as pd
import pytest

from config import DROPPED_COLUMNS, split_feature_groups
from preprocessor import Preprocessor
from test_characterisation import synthetic_frame
from trainer import ModelTrainer

_Fixture = tuple[ModelTrainer, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]

_METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "threshold",
    "expected_cost",
)


@pytest.fixture(scope="module")
def trainer_and_data() -> _Fixture:
    frame, target = synthetic_frame(rows=120)
    groups = split_feature_groups(
        Preprocessor(split_feature_groups(frame)).engineer(frame),
        dropped=DROPPED_COLUMNS,
    )
    trainer = ModelTrainer(
        Preprocessor(groups),
        models=("logistic_regression",),
        arms=("none", "class_weight"),
        cv_splits=2,
        track=False,
    )
    half = len(frame) // 2
    return (
        trainer,
        frame.iloc[:half],
        pd.Series(target[:half]),
        frame.iloc[half:],
        pd.Series(target[half:]),
    )


def test_positive_ratio_is_negatives_over_positives() -> None:
    frame, _ = synthetic_frame(rows=20)
    trainer = ModelTrainer(Preprocessor(split_feature_groups(frame)))
    assert trainer.positive_ratio(pd.Series([0, 0, 0, 1])) == pytest.approx(3.0)


def test_run_grid_returns_one_entry_per_cell(trainer_and_data: _Fixture) -> None:
    trainer, X_train, y_train, X_test, y_test = trainer_and_data
    results = trainer.run_grid(X_train, y_train, X_test, y_test)
    assert set(results) == {
        ("logistic_regression", "none"),
        ("logistic_regression", "class_weight"),
    }
    for metrics in results.values():
        assert {"roc_auc", "pr_auc", "recall", "threshold"} <= set(metrics)


def test_comparison_table_is_sorted_by_pr_auc(trainer_and_data: _Fixture) -> None:
    trainer = trainer_and_data[0]
    results = {
        ("a", "none"): {c: 0.1 for c in _METRIC_KEYS},
        ("b", "none"): {c: 0.2 for c in _METRIC_KEYS},
    }
    table = trainer.comparison_table(results)
    assert table.iloc[0]["model"] == "b"


def test_select_champion_picks_highest_pr_auc(trainer_and_data: _Fixture) -> None:
    trainer = trainer_and_data[0]
    results = {
        ("a", "none"): {"pr_auc": 0.10},
        ("b", "resample"): {"pr_auc": 0.30},
    }
    assert trainer.select_champion(results) == ("b", "resample")


def test_fit_champion_threshold_comes_from_out_of_fold_predictions(
    trainer_and_data: _Fixture,
) -> None:
    trainer, X_train, y_train, X_test, y_test = trainer_and_data
    pipeline, threshold, metrics = trainer.fit_champion(
        X_train, y_train, X_test, y_test, "logistic_regression", "none"
    )
    assert 0.0 <= threshold <= 1.0
    assert metrics["threshold"] == pytest.approx(threshold)
    assert hasattr(pipeline, "predict_proba")


def test_grid_lists_are_derived_from_the_strategy_registries() -> None:
    """Jedno źródło prawdy: nowa strategia ma wchodzić do siatki sama."""
    from strategies import ESTIMATOR_STRATEGIES, IMBALANCE_STRATEGIES
    from trainer import ALL_ARMS, ALL_MODELS

    assert ALL_MODELS == tuple(ESTIMATOR_STRATEGIES)
    assert ALL_ARMS == tuple(IMBALANCE_STRATEGIES)


def test_from_settings_maps_every_field() -> None:
    from settings import TrainingSettings

    frame, _ = synthetic_frame(rows=20)
    settings = TrainingSettings(models=("xgboost",), arms=("none",), cv_splits=3)
    trainer = ModelTrainer.from_settings(
        settings, Preprocessor(split_feature_groups(frame))
    )
    assert trainer.models == ("xgboost",)
    assert trainer.arms == ("none",)
    assert trainer.cv_splits == 3


def test_run_params_take_the_weight_from_the_strategy_not_a_string() -> None:
    """Ramię ważone ma trafić do zapisu biegu z tej samej decyzji co model."""
    frame, _ = synthetic_frame(rows=20)
    trainer = ModelTrainer(Preprocessor(split_feature_groups(frame)))

    weighted = trainer.run_params("xgboost", "class_weight", 11.39)
    plain = trainer.run_params("xgboost", "none", 11.39)
    resampled = trainer.run_params("xgboost", "resample", 11.39)

    assert weighted["scale_pos_weight"] == pytest.approx(11.39)
    assert plain["scale_pos_weight"] == pytest.approx(1.0)
    assert resampled["scale_pos_weight"] == pytest.approx(1.0)
