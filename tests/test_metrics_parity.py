"""Strażnik rejestru: model na @production zgadza się z zapisaną bazą.

Ten test pilnuje **rejestru**, nie kodu — sprawdza, że aliasem nie ruszył nikt
ręcznie i że zapisana baza dalej opisuje to, co serwis wysyła. Odtworzenie
liczb przez sam kod pilnuje `test_training_golden.py`, który nie potrzebuje
danych i działa w CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASELINE_PATH = Path(__file__).parent / "fixtures" / "phase3_champion_metrics.json"
PINNED = ("roc_auc", "pr_auc", "precision", "recall", "threshold")


@pytest.mark.requires_data
def test_production_metrics_match_the_recorded_baseline() -> None:
    from registry import production_metrics

    current = production_metrics()
    if current is None:
        pytest.skip("Brak lokalnego rejestru MLflow — uruchom `uv run cs-train`.")

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    drift = {
        key: (baseline[key], current[key])
        for key in PINNED
        if repr(baseline[key]) != repr(current[key])
    }
    assert not drift, f"Metryki championa odjechały od zapisanej bazy: {drift}"
