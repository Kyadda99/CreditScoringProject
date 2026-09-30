"""Quality gate tests.

The gate is the only thing between a weak model and the @production alias, so
every criterion must be able to reject on its own.
"""

from __future__ import annotations

import pytest

from quality_gate import MIN_PRECISION, MIN_RECALL, MIN_ROC_AUC, quality_gate


def good_metrics() -> dict[str, float]:
    """Metrics clearing all three criteria with room to spare."""
    return {"roc_auc": 0.78, "recall": 0.62, "precision": 0.16}


def test_gate_accepts_a_good_model() -> None:
    assert quality_gate(good_metrics()) is True


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("roc_auc", MIN_ROC_AUC - 0.01),
        ("recall", MIN_RECALL - 0.01),
        ("precision", MIN_PRECISION - 0.01),
    ],
)
def test_each_criterion_rejects_independently(key: str, bad_value: float) -> None:
    """One violation is enough; the other two criteria are met."""
    metrics = good_metrics()
    metrics[key] = bad_value
    assert quality_gate(metrics) is False


@pytest.mark.parametrize("missing", ["roc_auc", "recall", "precision"])
def test_missing_key_fails_closed(missing: str) -> None:
    """An incomplete metrics dictionary is a failure, not consent to promote."""
    metrics = good_metrics()
    del metrics[missing]
    assert quality_gate(metrics) is False


def test_empty_metrics_fail_closed() -> None:
    """An empty dictionary must not pass by accident."""
    assert quality_gate({}) is False


def test_gate_reports_every_violation_not_just_the_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No short-circuit: the full list of problems must appear at once."""
    caplog.set_level("INFO")
    quality_gate({"roc_auc": 0.10, "recall": 0.10, "precision": 0.01})
    failures = [r for r in caplog.records if "REJECTED" in r.getMessage()]
    assert len(failures) == 3


def test_boundary_value_passes() -> None:
    """The threshold is inclusive: exactly at the boundary the model passes."""
    assert (
        quality_gate(
            {
                "roc_auc": MIN_ROC_AUC,
                "recall": MIN_RECALL,
                "precision": MIN_PRECISION,
            }
        )
        is True
    )


def test_extra_metric_keys_are_ignored() -> None:
    """The gate receives the full evaluate() dictionary; extras do not disturb it."""
    metrics = good_metrics() | {"pr_auc": 0.2, "tn": 1.0, "expected_cost": 999.0}
    assert quality_gate(metrics) is True


def test_thresholds_are_not_accidentally_trivial() -> None:
    """A gate that admits the baseline gates nothing.

    The baseline had ROC-AUC 0.7515 but recall 0.0119, and must be rejected.
    """
    baseline = {"roc_auc": 0.7515, "recall": 0.0119, "precision": 0.5514}
    assert quality_gate(baseline) is False


# --- Comparison against the current champion ---


def test_first_champion_wins_by_walkover() -> None:
    """Empty registry: nothing to compare against, so the candidate passes."""
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.20}, None) is True


def test_strictly_better_candidate_is_promoted() -> None:
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.2626}, {"pr_auc": 0.2564}) is True


def test_worse_candidate_is_rejected() -> None:
    """A real case: the tuned model came out worse than the default one.

    It cleared all three absolute thresholds and still displaced a better
    champion, because `promote()` overwrites the alias unconditionally.
    """
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.2564}, {"pr_auc": 0.2626}) is False


def test_tie_keeps_the_incumbent() -> None:
    """A tie does not justify replacement: rebuild and verification cost."""
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.25}, {"pr_auc": 0.25}) is False


def test_missing_candidate_metric_fails_closed() -> None:
    from quality_gate import beats_incumbent

    assert beats_incumbent({}, {"pr_auc": 0.25}) is False


def test_comparison_metric_is_configurable() -> None:
    """PR-AUC by default, but the comparison is not hardwired to it."""
    from quality_gate import CHAMPION_METRIC, beats_incumbent

    assert CHAMPION_METRIC == "pr_auc"
    assert beats_incumbent({"roc_auc": 0.9}, {"roc_auc": 0.8}, metric="roc_auc") is True
