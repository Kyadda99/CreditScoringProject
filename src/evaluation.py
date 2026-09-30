"""Binary classification metrics.

The dataset is imbalanced (8.07% positive), so accuracy alone says nothing: a
model that always predicts "no default" scores 91.9%. Decision metrics are
computed at an explicit threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import COST_FN, COST_FP

DEFAULT_THRESHOLD = 0.5


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> dict[str, float]:
    """Compute metrics from positive-class probabilities.

    The returned dictionary is the single metrics contract: MLflow logging, the
    quality gate and champion selection all read the same object.

    Args:
        y_true: True 0/1 labels.
        y_proba: Positive-class probability, `predict_proba(...)[:, 1]`, not
            `predict()`.
        threshold: Decision threshold, chosen outside the test set.
        cost_fn: Cost of a missed default.
        cost_fp: Cost of a false alarm.

    Returns:
        Metrics as plain floats, not `np.float64`; MLflow and JSON
        serialisation do not accept numpy types.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    # labels=[0, 1] keeps confusion_matrix 2x2; a degenerate prediction
    # otherwise returns 1x1 and ravel() raises.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        # average_precision, not auc(recall, precision): the latter
        # interpolates linearly and inflates the score on a rare class.
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: at a high threshold the model may predict no
        # positive class, which otherwise warns and returns NaN.
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "expected_cost": float(cost_fn * fn + cost_fp * fp),
        "threshold": float(threshold),
    }


def choose_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, float]:
    """Choose the threshold minimising `cost_fn*FN + cost_fp*FP`.

    Searches only reachable thresholds, the observed probability values, in one
    pass over the sorted array rather than a grid loop.

    The "flag nobody" option (a threshold above the maximum) is not considered.

    Args:
        y_true: True 0/1 labels.
        y_proba: Positive-class probabilities. Must come from outside the test
            set, in practice from `cross_val_predict` on the training set.
            Choosing the threshold on test and reporting the cost at that
            threshold would fit the test set and then report the fit.
        cost_fn: Cost of a missed default.
        cost_fp: Cost of a false alarm.

    Returns:
        `(threshold, cost at that threshold)`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    # Descending: moving right lowers the threshold, so rows enter the
    # positive class one at a time. mergesort is stable, which keeps ties in
    # order and makes the last_of_tie mask below correct.
    order = np.argsort(-y_proba, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_proba[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = int(y_true.sum()) - tp
    cost = cost_fn * fn + cost_fp * fp

    # A threshold cannot split a group of tied probabilities: >= takes the
    # whole group or none. Only positions at the end of a tie run are
    # reachable; the rest get infinite cost so argmin cannot select them.
    last_of_tie = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    cost = np.where(last_of_tie, cost, np.inf)

    best = int(np.argmin(cost))
    return float(p_sorted[best]), float(cost[best])
