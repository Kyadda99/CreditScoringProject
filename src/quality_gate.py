"""Hard thresholds a model must clear before it reaches the @production alias.

The gate is called from `train.py` and `tune.py` before `promote()`.

A failed gate is not an exception: it is a normal training outcome. The run
finishes, logs every violation, and does not promote.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_ROC_AUC = 0.75
"""No regression against the 0.7515 baseline.

A model that ranks risk worse than the baseline does not belong in production,
however well it scores elsewhere.
"""

MIN_RECALL = 0.50
"""The baseline caught 1.19% of defaults.

At FN = 10 x FP, a model that misses half the defaults cannot be justified on
cost, and this is already a lenient threshold.
"""

MIN_PRECISION = 0.12
"""1.5x the 8.07% base rate.

Flagging applications at random would land near 0.08, so this threshold
requires real signal. Without it the recall criterion could be met by
flagging everyone.
"""

_CRITERIA: tuple[tuple[str, float], ...] = (
    ("roc_auc", MIN_ROC_AUC),
    ("recall", MIN_RECALL),
    ("precision", MIN_PRECISION),
)
"""Criteria checked in order, all of them, without short-circuiting."""


def quality_gate(metrics: dict[str, float]) -> bool:
    """Check whether the metrics allow promoting the model.

    Args:
        metrics: Dictionary from `evaluate()`. Keys outside `_CRITERIA` are
            ignored. A missing key disqualifies: an incomplete measurement is
            not consent to promote.

    Returns:
        `True` when every criterion is met.
    """
    passed = True
    for key, minimum in _CRITERIA:
        # -inf, not 0: a missing key must lose to any threshold, including
        # a negative one.
        value = metrics.get(key, float("-inf"))
        if value >= minimum:
            logger.info("ACCEPTED  %-10s %.4f >= %.4f", key, value, minimum)
        else:
            logger.warning("REJECTED  %-10s %.4f <  %.4f", key, value, minimum)
            passed = False

    logger.info("Quality gate: %s", "PASSED" if passed else "FAILED")
    return passed


CHAMPION_METRIC = "pr_auc"
"""Metric that decides the champion."""


def beats_incumbent(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    metric: str = CHAMPION_METRIC,
) -> bool:
    """Check whether the candidate beats the model currently on @production.

    The absolute thresholds in `quality_gate` are not enough: a model can meet
    them and still be worse than the one already in production. A tuned
    XGBoost once passed all three criteria and displaced an untuned model with
    a higher PR-AUC (0.2564 vs 0.2626).

    The comparison is strict (`>`): on a tie the incumbent stays, because
    replacing the production model costs an image rebuild and verification.

    Args:
        candidate: Candidate metrics from `evaluate()`.
        incumbent: Metrics of the aliased model, or `None` when no alias
            exists yet, in which case the candidate wins by default.
        metric: Key of the deciding metric.

    Returns:
        `True` when the candidate should be promoted.
    """
    if incumbent is None:
        logger.info("No model on @production - promoting candidate by default.")
        return True

    new = candidate.get(metric, float("-inf"))
    old = incumbent.get(metric, float("-inf"))
    if new > old:
        logger.info("ACCEPTED  %s %.4f >  %.4f (current champion)", metric, new, old)
        return True

    logger.warning(
        "REJECTED  %s %.4f <= %.4f (current champion) - alias unchanged",
        metric,
        new,
        old,
    )
    return False
