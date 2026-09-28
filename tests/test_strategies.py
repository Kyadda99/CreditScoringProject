"""Kontrakt strategii estymatora i strategii niezbalansowania."""

from __future__ import annotations

import pytest
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE
from strategies import (
    ESTIMATOR_STRATEGIES,
    IMBALANCE_STRATEGIES,
    get_estimator_strategy,
    get_imbalance_strategy,
)


def test_registries_cover_the_documented_names() -> None:
    assert set(ESTIMATOR_STRATEGIES) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert set(IMBALANCE_STRATEGIES) == {"none", "class_weight", "resample"}


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Nieznany"):
        get_estimator_strategy("magic_forest")
    with pytest.raises(ValueError, match="Nieznane"):
        get_imbalance_strategy("undersample")


def test_class_weight_arm_sets_estimator_specific_parameter() -> None:
    weighted = {
        name: s.build(class_weighted=True, scale_pos_weight=11.39)
        for name, s in ESTIMATOR_STRATEGIES.items()
    }
    assert weighted["logistic_regression"].class_weight == "balanced"
    assert weighted["random_forest"].class_weight == "balanced"
    assert weighted["xgboost"].scale_pos_weight == pytest.approx(11.39)

    plain = {
        name: s.build(class_weighted=False, scale_pos_weight=11.39)
        for name, s in ESTIMATOR_STRATEGIES.items()
    }
    assert plain["logistic_regression"].class_weight is None
    assert plain["random_forest"].class_weight is None
    assert plain["xgboost"].scale_pos_weight == pytest.approx(1.0)


def test_only_the_resample_arm_produces_a_sampler() -> None:
    assert get_imbalance_strategy("none").make_sampler() is None
    assert get_imbalance_strategy("class_weight").make_sampler() is None
    sampler = get_imbalance_strategy("resample").make_sampler()
    assert isinstance(sampler, SMOTE)
    assert sampler.random_state == RANDOM_STATE


def test_only_the_class_weight_arm_asks_for_weights() -> None:
    assert get_imbalance_strategy("class_weight").uses_class_weights is True
    assert get_imbalance_strategy("none").uses_class_weights is False
    assert get_imbalance_strategy("resample").uses_class_weights is False


def test_each_build_returns_a_fresh_instance() -> None:
    strategy = get_estimator_strategy("xgboost")
    first = strategy.build(class_weighted=False, scale_pos_weight=1.0)
    second = strategy.build(class_weighted=False, scale_pos_weight=1.0)
    assert first is not second


def test_estimator_parameters_are_pinned() -> None:
    """Parametry bazowe championa — zmiana tutaj zmienia metryki modelu."""
    logreg = get_estimator_strategy("logistic_regression").build(
        class_weighted=False, scale_pos_weight=1.0
    )
    assert logreg.max_iter == 1000
    assert logreg.random_state == RANDOM_STATE

    forest = get_estimator_strategy("random_forest").build(
        class_weighted=False, scale_pos_weight=1.0
    )
    assert (forest.n_estimators, forest.max_depth, forest.min_samples_leaf) == (
        200,
        12,
        20,
    )

    xgb = get_estimator_strategy("xgboost").build(
        class_weighted=False, scale_pos_weight=1.0
    )
    assert (xgb.n_estimators, xgb.max_depth) == (300, 5)
    assert xgb.learning_rate == pytest.approx(0.1)
    assert xgb.tree_method == "hist"
    assert xgb.eval_metric == "aucpr"


def test_every_family_is_seeded() -> None:
    """Powtarzalność stoi na jednym ziarnie w każdej rodzinie estymatorów."""
    for name in ESTIMATOR_STRATEGIES:
        estimator = get_estimator_strategy(name).build(
            class_weighted=False, scale_pos_weight=1.0
        )
        assert estimator.random_state == RANDOM_STATE, name


def test_each_family_builds_its_own_estimator_type() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    build = {
        name: strategy.build(class_weighted=False, scale_pos_weight=1.0)
        for name, strategy in ESTIMATOR_STRATEGIES.items()
    }
    assert isinstance(build["logistic_regression"], LogisticRegression)
    assert isinstance(build["random_forest"], RandomForestClassifier)
    assert isinstance(build["xgboost"], XGBClassifier)


def test_registry_order_is_the_grid_order() -> None:
    """Kolejność rejestru steruje siatką i rozstrzyga remisy przy championie."""
    assert list(ESTIMATOR_STRATEGIES) == [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]
    assert list(IMBALANCE_STRATEGIES) == ["none", "class_weight", "resample"]


def test_samplers_are_fresh_instances() -> None:
    strategy = get_imbalance_strategy("resample")
    assert strategy.make_sampler() is not strategy.make_sampler()
