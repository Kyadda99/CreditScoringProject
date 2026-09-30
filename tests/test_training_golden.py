"""Golden sample: metrics on a fixed synthetic dataset.

This does not judge model quality; the data is random, so a ROC-AUC near 0.5
is the correct result here. It pins something else: that the whole training
path (strategies, pipeline, threshold from out-of-fold predictions) computes
exactly what it computed when these values were recorded. A typo in a
hyperparameter or a reordered pipeline step fails this test in CI, without the
real data.

Values recorded 2026-09-28 from working code. If the model configuration is
changed deliberately, update them and record the change.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import DROPPED_COLUMNS, split_feature_groups
from preprocessor import Preprocessor
from test_characterisation import synthetic_frame
from trainer import ModelTrainer

GOLDEN_ROC_AUC = 0.4486863711001642
GOLDEN_PR_AUC = 0.5501484701972957
GOLDEN_THRESHOLD = 0.01408308558166027

ROWS = 400
TRAIN_ROWS = 300
SEED = 11


def test_champion_path_reproduces_the_golden_metrics() -> None:
    frame, target = synthetic_frame(rows=ROWS, seed=SEED)
    y = pd.Series(target)
    groups = split_feature_groups(
        Preprocessor(split_feature_groups(frame)).engineer(frame),
        dropped=DROPPED_COLUMNS,
    )
    trainer = ModelTrainer(
        Preprocessor(groups),
        models=("xgboost",),
        arms=("none",),
        cv_splits=2,
        track=False,
    )

    _pipeline, threshold, metrics = trainer.fit_champion(
        frame.iloc[:TRAIN_ROWS],
        y[:TRAIN_ROWS],
        frame.iloc[TRAIN_ROWS:],
        y[TRAIN_ROWS:],
        "xgboost",
        "none",
    )

    assert metrics["roc_auc"] == pytest.approx(GOLDEN_ROC_AUC, rel=1e-9)
    assert metrics["pr_auc"] == pytest.approx(GOLDEN_PR_AUC, rel=1e-9)
    assert threshold == pytest.approx(GOLDEN_THRESHOLD, rel=1e-9)
