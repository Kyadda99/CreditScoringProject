"""Metric tests, on hand-computed cases rather than real data."""

from __future__ import annotations

import numpy as np

from evaluation import choose_threshold, evaluate

EXPECTED_KEYS = {
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "expected_cost",
    "threshold",
}
"""The metrics contract. The same dictionary feeds MLflow, the quality gate
and champion selection, so its shape is pinned by a test."""


def test_returns_the_expected_metric_keys() -> None:
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
    assert set(metrics) == EXPECTED_KEYS


def test_perfect_separation_scores_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluate(y_true, y_proba)["roc_auc"] == 1.0


def test_inverted_predictions_score_zero() -> None:
    """Inverted labels give 0.0, catching a swapped predict_proba index."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])
    assert evaluate(y_true, y_proba)["roc_auc"] == 0.0


def test_random_predictions_score_near_half() -> None:
    """Seeded, not random: without a seed this test is unstable by definition."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=5000)
    y_proba = rng.random(5000)
    assert abs(evaluate(y_true, y_proba)["roc_auc"] - 0.5) < 0.05


def test_threshold_changes_recall_but_not_roc_auc() -> None:
    """ROC-AUC does not depend on the threshold; decision metrics do."""
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
    """MLflow and JSON do not accept np.float64 without conversion."""
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
    assert all(type(v) is float for v in metrics.values())


# --- PR-AUC, confusion matrix, cost ---

_Y_TRUE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
_Y_PROBA = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.2, 0.9, 0.95])
"""Ten rows, three positives. At a 0.5 threshold the predicted positives are
0.6, 0.7, 0.8 (negatives) plus 0.9, 0.95 (positives)."""


def test_evaluate_reports_confusion_counts_that_sum_to_n() -> None:
    """tn+fp+fn+tp must equal the row count, or samples are being lost."""
    m = evaluate(_Y_TRUE, _Y_PROBA, threshold=0.5)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 10.0
    assert m["tp"] == 2.0
    assert m["fp"] == 3.0
    assert m["fn"] == 1.0
    assert m["tn"] == 4.0


def test_evaluate_expected_cost_is_hand_computable() -> None:
    """10*FN + 1*FP: arithmetic, not magic."""
    m = evaluate(_Y_TRUE, _Y_PROBA, threshold=0.5)
    assert m["expected_cost"] == 10.0 * 1 + 1.0 * 3


def test_evaluate_expected_cost_follows_injected_costs() -> None:
    """Costs are a parameter, not a hardcoded constant."""
    m = evaluate(_Y_TRUE, _Y_PROBA, threshold=0.5, cost_fn=3.0, cost_fp=2.0)
    assert m["expected_cost"] == 3.0 * 1 + 2.0 * 3


def test_evaluate_reports_pr_auc() -> None:
    """Perfect ranking gives PR-AUC 1.0."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluate(y_true, y_proba)["pr_auc"] == 1.0


def test_pr_auc_is_harsher_than_roc_auc_on_imbalance() -> None:
    """Why PR-AUC is the champion metric.

    With a rare positive class ROC-AUC is inflated by the easy majority; PR-AUC
    looks only at what happens near the boundary.
    """
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.05, size=2000)
    y_proba = np.clip(rng.normal(0.05, 0.05, 2000) + 0.15 * y_true, 0.001, 0.999)
    m = evaluate(y_true, y_proba)
    assert m["pr_auc"] < m["roc_auc"]


def test_evaluate_survives_degenerate_predictions() -> None:
    """At a high threshold the model predicts no positive at all.

    Without labels=[0, 1], confusion_matrix would return a 1x1 matrix and
    .ravel() would raise ValueError. This is a real trap, not a hypothetical.
    """
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    m = evaluate(y_true, y_proba, threshold=0.99)
    assert m["tp"] == 0.0
    assert m["fn"] == 2.0
    assert m["recall"] == 0.0


# --- Decision threshold selection ---


def test_threshold_boundary_is_inclusive() -> None:
    """A probability equal to the threshold classifies positive (`>=`, not `>`).

    With an exclusive threshold, a row whose probability equals the threshold
    would fall on the other side of the decision from the one measured when the
    threshold was chosen, and the metrics would describe a different model.
    """
    metrics = evaluate(np.array([0, 1]), np.array([0.3, 0.7]), threshold=0.7)
    assert metrics["tp"] == 1.0


def _separable(seed: int, n: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Labels and probabilities with real but incomplete signal."""
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, rate, size=n)
    y_proba = np.clip(rng.normal(rate, 0.08, n) + 0.25 * y_true, 0.001, 0.999)
    return y_true, y_proba


def test_choose_threshold_falls_when_missing_defaults_costs_more() -> None:
    """The costlier a false negative, the lower the threshold must go."""
    y_true, y_proba = _separable(42, 2000, 0.1)
    cheap_t, _ = choose_threshold(y_true, y_proba, cost_fn=2.0, cost_fp=1.0)
    dear_t, _ = choose_threshold(y_true, y_proba, cost_fn=50.0, cost_fp=1.0)
    assert dear_t < cheap_t


def test_choose_threshold_returns_the_cost_it_minimised() -> None:
    """The returned cost must match what evaluate computes at that threshold."""
    y_true, y_proba = _separable(7, 500, 0.2)
    threshold, cost = choose_threshold(y_true, y_proba)
    assert cost == evaluate(y_true, y_proba, threshold=threshold)["expected_cost"]


def test_choose_threshold_beats_the_default_half() -> None:
    """0.5 is worse on cost than a threshold chosen against the costs."""
    y_true, y_proba = _separable(1, 3000, 0.08)
    _, cost = choose_threshold(y_true, y_proba)
    assert cost <= evaluate(y_true, y_proba, threshold=0.5)["expected_cost"]


def test_choose_threshold_is_the_global_minimum() -> None:
    """The sweep must find the minimum, not the first acceptable point."""
    y_true, y_proba = _separable(3, 400, 0.15)
    threshold, cost = choose_threshold(y_true, y_proba)
    for candidate in np.unique(y_proba):
        assert (
            cost
            <= evaluate(y_true, y_proba, threshold=float(candidate))["expected_cost"]
        )


def test_choose_threshold_handles_tied_probabilities() -> None:
    """On ties, `>=` takes the whole group or none, and the threshold must respect it.

    Three rows sit at exactly 0.4. The returned threshold must give a cost that is
    actually reachable, not one that assumes cutting a tie in half.
    """
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.4, 0.4, 0.4, 0.9, 0.1, 0.8])
    threshold, cost = choose_threshold(y_true, y_proba)
    assert cost == evaluate(y_true, y_proba, threshold=threshold)["expected_cost"]
