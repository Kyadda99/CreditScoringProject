"""Testy rejestru modeli."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE
from models import get_models


def test_returns_logistic_regression_only() -> None:
    """Faza 1 celowo ma jeden model. Kolejne dokłada Faza 3."""
    assert list(get_models()) == ["logistic_regression"]


def test_estimator_is_an_unfitted_sklearn_estimator() -> None:
    model = get_models()["logistic_regression"]
    assert isinstance(model, BaseEstimator)
    assert isinstance(model, LogisticRegression)
    assert not hasattr(model, "coef_"), "model nie może być już dopasowany"


def test_seed_comes_from_config() -> None:
    """Reprodukowalność: jedno ziarno dla całego projektu."""
    assert get_models()["logistic_regression"].random_state == RANDOM_STATE


def test_each_call_returns_a_fresh_estimator() -> None:
    """Dwa wywołania nie mogą dzielić obiektu — inaczej fit jednego psuje drugi."""
    first = get_models()["logistic_regression"]
    second = get_models()["logistic_regression"]
    assert first is not second
