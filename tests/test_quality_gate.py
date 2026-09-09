"""Testy bramki jakości.

Bramka jest jedyną rzeczą stojącą między słabym modelem a aliasem
@production, więc każde kryterium musi umieć odrzucić samodzielnie.
"""

from __future__ import annotations

import pytest

from quality_gate import MIN_PRECISION, MIN_RECALL, MIN_ROC_AUC, quality_gate


def good_metrics() -> dict[str, float]:
    """Metryki spełniające wszystkie trzy kryteria z zapasem."""
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
    """Jedno naruszenie wystarczy — pozostałe dwa kryteria są spełnione."""
    metrics = good_metrics()
    metrics[key] = bad_value
    assert quality_gate(metrics) is False


@pytest.mark.parametrize("missing", ["roc_auc", "recall", "precision"])
def test_missing_key_fails_closed(missing: str) -> None:
    """Niekompletny słownik metryk to awaria, a nie zgoda na promocję."""
    metrics = good_metrics()
    del metrics[missing]
    assert quality_gate(metrics) is False


def test_empty_metrics_fail_closed() -> None:
    """Pusty słownik nie może przypadkiem przejść."""
    assert quality_gate({}) is False


def test_gate_reports_every_violation_not_just_the_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bez short-circuitu: chcemy zobaczyć pełną listę problemów naraz."""
    caplog.set_level("INFO")
    quality_gate({"roc_auc": 0.10, "recall": 0.10, "precision": 0.01})
    failures = [r for r in caplog.records if "ODRZUCONE" in r.getMessage()]
    assert len(failures) == 3


def test_boundary_value_passes() -> None:
    """Próg jest włączny: dokładnie na granicy model przechodzi."""
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
    """Bramka dostaje pełny słownik z evaluate() — nadmiar jej nie rusza."""
    metrics = good_metrics() | {"pr_auc": 0.2, "tn": 1.0, "expected_cost": 999.0}
    assert quality_gate(metrics) is True


def test_thresholds_are_not_accidentally_trivial() -> None:
    """Bramka, która przepuszcza baseline z Fazy 2, nie bramkuje niczego.

    Baseline miał ROC-AUC 0.7515, ale recall 0.0119 — musi zostać odrzucony,
    bo dokładnie po to ta faza istnieje.
    """
    baseline = {"roc_auc": 0.7515, "recall": 0.0119, "precision": 0.5514}
    assert quality_gate(baseline) is False


# --- Porównanie z obecnym championem --------------------------------------


def test_first_champion_wins_by_walkover() -> None:
    """Pusty rejestr: nie ma z czym porównywać, więc kandydat przechodzi."""
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.20}, None) is True


def test_strictly_better_candidate_is_promoted() -> None:
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.2626}, {"pr_auc": 0.2564}) is True


def test_worse_candidate_is_rejected() -> None:
    """Realny przypadek z Fazy 3: strojony model wypadł GORZEJ niż domyślny.

    Przeszedł wszystkie trzy progi bezwzględne i mimo to wyparł lepszego
    championa, bo `promote()` nadpisuje alias bezwarunkowo.
    """
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.2564}, {"pr_auc": 0.2626}) is False


def test_tie_keeps_the_incumbent() -> None:
    """Remis nie uzasadnia wymiany: rebuild i weryfikacja kosztują."""
    from quality_gate import beats_incumbent

    assert beats_incumbent({"pr_auc": 0.25}, {"pr_auc": 0.25}) is False


def test_missing_candidate_metric_fails_closed() -> None:
    from quality_gate import beats_incumbent

    assert beats_incumbent({}, {"pr_auc": 0.25}) is False


def test_comparison_metric_is_configurable() -> None:
    """Domyślnie PR-AUC (spec D1), ale porównanie nie jest w nią wmurowane."""
    from quality_gate import CHAMPION_METRIC, beats_incumbent

    assert CHAMPION_METRIC == "pr_auc"
    assert beats_incumbent({"roc_auc": 0.9}, {"roc_auc": 0.8}, metric="roc_auc") is True
