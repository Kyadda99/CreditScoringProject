"""Testy metryk — na ręcznie policzonych przypadkach, nie na prawdziwych danych."""

from __future__ import annotations

import numpy as np

from evaluation import evaluate

EXPECTED_KEYS = {"roc_auc", "accuracy", "precision", "recall", "f1", "threshold"}


def test_returns_the_expected_metric_keys() -> None:
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
    assert set(metrics) == EXPECTED_KEYS


def test_perfect_separation_scores_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluate(y_true, y_proba)["roc_auc"] == 1.0


def test_inverted_predictions_score_zero() -> None:
    """Odwrócone etykiety dają 0.0 — łapie pomylony indeks predict_proba."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])
    assert evaluate(y_true, y_proba)["roc_auc"] == 0.0


def test_random_predictions_score_near_half() -> None:
    """Zasiane, nie losowe: bez seeda ten test jest z definicji niestabilny."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=5000)
    y_proba = rng.random(5000)
    assert abs(evaluate(y_true, y_proba)["roc_auc"] - 0.5) < 0.05


def test_threshold_changes_recall_but_not_roc_auc() -> None:
    """ROC-AUC nie zależy od progu; metryki decyzyjne owszem."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    low = evaluate(y_true, y_proba, threshold=0.5)
    high = evaluate(y_true, y_proba, threshold=0.95)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["recall"] == 1.0
    assert high["recall"] == 0.0


def test_threshold_is_echoed_back() -> None:
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.3)
    assert metrics["threshold"] == 0.3


def test_every_value_is_a_plain_float() -> None:
    """MLflow i JSON nie przyjmują np.float64 bez konwersji."""
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
    assert all(type(v) is float for v in metrics.values())
