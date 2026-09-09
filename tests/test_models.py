"""Testy rejestru modeli.

Faza 3 rozszerza rejestr z jednego modelu do trzech algorytmów w trzech
wariantach obsługi niezbalansowania (spec D2).
"""

from __future__ import annotations

import pytest
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import RANDOM_STATE
from models import IMBALANCE_ARMS, SCALE_POS_WEIGHT, get_models, get_sampler

EXPECTED = ["logistic_regression", "random_forest", "xgboost"]


def test_registry_holds_the_three_planned_algorithms() -> None:
    """PLAN.md wymaga co najmniej trzech: liniowy, las, gradient boosting."""
    assert list(get_models()) == EXPECTED


def test_registry_covers_every_algorithm_in_every_arm() -> None:
    """Siatka 3x3 wymaga kompletu — brak komórki zafałszowałby porównanie."""
    for arm in IMBALANCE_ARMS:
        assert list(get_models(arm)) == EXPECTED


def test_estimators_are_unfitted_sklearn_estimators() -> None:
    for model in get_models().values():
        assert isinstance(model, BaseEstimator)
        assert not hasattr(model, "coef_"), "model nie może być już dopasowany"


def test_every_estimator_is_seeded() -> None:
    """Reprodukowalność: jedno ziarno dla całego projektu."""
    for model in get_models().values():
        assert model.random_state == RANDOM_STATE


def test_each_call_returns_a_fresh_estimator() -> None:
    """Dwa wywołania nie mogą dzielić obiektu — inaczej fit jednego psuje drugi."""
    assert (
        get_models()["logistic_regression"] is not get_models()["logistic_regression"]
    )


def test_none_arm_leaves_weights_untouched() -> None:
    models = get_models("none")
    assert models["logistic_regression"].class_weight is None
    assert models["random_forest"].class_weight is None
    assert models["xgboost"].scale_pos_weight == 1.0


def test_class_weight_arm_sets_the_right_parameter_per_family() -> None:
    """Każda rodzina ma inne pole: class_weight w sklearn, scale_pos_weight w XGB.

    Ustawienie class_weight na XGBClassifier nie zrobiłoby nic i ramię
    "class_weight" byłoby po cichu identyczne z "none".
    """
    models = get_models("class_weight")
    assert models["logistic_regression"].class_weight == "balanced"
    assert models["random_forest"].class_weight == "balanced"
    assert models["xgboost"].scale_pos_weight == pytest.approx(SCALE_POS_WEIGHT)


def test_scale_pos_weight_is_injectable() -> None:
    """Grid liczy stosunek klas z faktycznego y_train, a nie ze stałej."""
    assert (
        get_models("class_weight", scale_pos_weight=7.5)["xgboost"].scale_pos_weight
        == 7.5
    )


def test_resample_arm_does_not_touch_estimator_weights() -> None:
    """Resampling i wagi to alternatywy — łączenie ich liczyłoby imbalans dwa razy."""
    models = get_models("resample")
    assert models["logistic_regression"].class_weight is None
    assert models["random_forest"].class_weight is None
    assert models["xgboost"].scale_pos_weight == 1.0


def test_sampler_only_exists_for_the_resample_arm() -> None:
    assert get_sampler("none") is None
    assert get_sampler("class_weight") is None
    assert isinstance(get_sampler("resample"), SMOTE)


def test_sampler_is_seeded() -> None:
    assert get_sampler("resample").random_state == RANDOM_STATE


def test_sampler_instances_are_fresh() -> None:
    """Dopasowany sampler nie może wyciec do kolejnej komórki siatki."""
    assert get_sampler("resample") is not get_sampler("resample")


@pytest.mark.parametrize("factory", [get_models, get_sampler])
def test_unknown_arm_is_rejected_loudly(factory) -> None:  # noqa: ANN001
    """Literówka w --imbalance ma się wysypać, a nie cicho dać wariant `none`."""
    with pytest.raises(ValueError, match="undersample"):
        factory("undersample")


def test_estimator_families_are_what_we_claim() -> None:
    models = get_models()
    assert isinstance(models["logistic_regression"], LogisticRegression)
    assert isinstance(models["random_forest"], RandomForestClassifier)
    assert isinstance(models["xgboost"], XGBClassifier)
