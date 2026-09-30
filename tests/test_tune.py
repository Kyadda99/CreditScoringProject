"""Hyperparameter search tests.

All on small synthetic frames with two trials: this checks the function
contract and reproducibility, not tuning quality.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from config import ID_COLUMN, TARGET, FeatureGroups

# Optuna logs every trial at INFO; in tests that is only noise.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _frame(n: int = 240, rate: float = 0.25) -> tuple[pd.DataFrame, pd.Series]:
    """A frame with every source column FeatureEngineer.fit requires."""
    rng = np.random.default_rng(42)
    target = pd.Series(rng.binomial(1, rate, n), name=TARGET)
    frame = pd.DataFrame(
        {
            ID_COLUMN: range(n),
            "AMT_INCOME_TOTAL": rng.random(n) * 1000 + 1.0,
            "AMT_CREDIT": rng.random(n) * 2000 + 1.0,
            "AMT_ANNUITY": rng.random(n) * 100 + 1.0,
            "AMT_GOODS_PRICE": rng.random(n) * 1800 + 1.0,
            "CNT_FAM_MEMBERS": rng.integers(1, 4, size=n).astype(float),
            "DAYS_BIRTH": -rng.integers(7000, 20000, size=n).astype(float),
            "DAYS_EMPLOYED": -rng.integers(100, 5000, size=n).astype(float),
            # signal, so PR-AUC is not pure noise
            "EXT_SOURCE_2": rng.random(n) * 0.5 + target * 0.4,
        }
    ).drop(columns=[ID_COLUMN])
    return frame, target


def _groups() -> FeatureGroups:
    return FeatureGroups(
        numeric=("AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_2"),
        categorical=(),
        binary=(),
    )


def test_subsample_keeps_the_class_ratio() -> None:
    """The subsample must be stratified, or tuning sees a different problem.

    At 8% positives a random subsample can shift the base rate enough that the
    optimisation targets a different distribution from the one later trained.
    """
    from tune import subsample

    frame, target = _frame(1000)
    X, y = subsample(frame, target, fraction=0.25, seed=42)
    assert len(X) == len(y) == 250
    assert abs(y.mean() - target.mean()) < 0.02


def test_subsample_returns_everything_when_fraction_is_one() -> None:
    from tune import subsample

    frame, target = _frame(100)
    X, y = subsample(frame, target, fraction=1.0, seed=42)
    assert len(X) == 100
    assert X is frame


def test_objective_returns_a_float_in_unit_range() -> None:
    from tune import build_objective

    frame, target = _frame()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        build_objective(frame, target, _groups(), arm="none", cv_splits=2), n_trials=2
    )
    assert isinstance(study.best_value, float)
    assert 0.0 <= study.best_value <= 1.0


def test_study_is_reproducible() -> None:
    """TPESampler(seed=42): two identical runs give the same parameters."""
    from tune import build_objective, make_study

    frame, target = _frame()
    runs = []
    for _ in range(2):
        study = make_study()
        study.optimize(
            build_objective(frame, target, _groups(), arm="none", cv_splits=2),
            n_trials=2,
        )
        runs.append(study.best_params)
    assert runs[0] == runs[1]


def test_objective_searches_the_declared_parameters() -> None:
    """The budget and the space are declared, so a test pins them."""
    from tune import build_objective

    frame, target = _frame()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        build_objective(frame, target, _groups(), arm="none", cv_splits=2), n_trials=1
    )
    assert set(study.trials[0].params) == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
    }


def test_declared_budget_matches_the_spec() -> None:
    """30 trials, 3 folds, a 25% subsample: declared, not discovered."""
    from tune import CV_SPLITS, N_TRIALS, SUBSAMPLE_FRACTION

    assert N_TRIALS == 30
    assert CV_SPLITS == 3
    assert SUBSAMPLE_FRACTION == 0.25


@pytest.mark.parametrize("arm", ["none", "class_weight"])
def test_objective_works_for_each_imbalance_arm(arm: str) -> None:
    from tune import build_objective

    frame, target = _frame()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        build_objective(frame, target, _groups(), arm=arm, cv_splits=2), n_trials=1
    )
    assert study.best_value >= 0.0
