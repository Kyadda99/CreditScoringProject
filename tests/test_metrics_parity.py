"""Registry guard: the model on @production matches the recorded baseline.

This pins the registry, not the code: it checks that nobody moved the alias by
hand and that the recorded baseline still describes what the service serves.
Reproducing the numbers from code is `test_training_golden.py`, which needs no
data and runs in CI.
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
        pytest.skip("No local MLflow registry - run `uv run cs-train`.")

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    drift = {
        key: (baseline[key], current[key])
        for key in PINNED
        if repr(baseline[key]) != repr(current[key])
    }
    assert not drift, f"Champion metrics drifted from the recorded baseline: {drift}"
