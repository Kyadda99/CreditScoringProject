"""A tiny but real model artifact, built without the Kaggle dataset.

The columns come from the committed `feature_schema.json`, so the pipeline has
exactly the production input contract, trained on a handful of invented rows.
The test suite and the CI service stack both serve this artifact.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from api.schemas import load_feature_groups
from artifact import save_bundle
from config import DROPPED_COLUMNS, split_feature_groups
from features import FeatureEngineer
from preprocessor import Preprocessor

ROWS = 40  # enough for every imputer and encoder to see each category


def build_synthetic_artifact(path: Path, seed: int = 0) -> Path:
    """Fit the production pipeline on random rows and save it as a bundle.

    Args:
        path: Destination of the `.joblib` file; parent directories are created.
        seed: Seed for the random rows and labels.

    Returns:
        Path of the written file.
    """
    inputs = load_feature_groups()
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    for name in inputs.numeric:
        # Positive and away from zero: a ratio denominator cannot be zero, or the
        # whole column comes out NaN and the imputer has nothing to learn from.
        data[name] = rng.uniform(1.0, 100.0, size=ROWS)
    for name in inputs.binary:
        data[name] = rng.integers(0, 2, size=ROWS).astype(float)
    for name in inputs.categorical:
        data[name] = rng.choice(["A", "B"], size=ROWS)
    frame = pd.DataFrame(data, columns=list(inputs.all_features))
    target = rng.integers(0, 2, size=ROWS)

    # Model groups come from the frame after engineering, as training does.
    engineered = FeatureEngineer().fit_transform(frame)
    groups = split_feature_groups(engineered, dropped=DROPPED_COLUMNS)

    pipeline = Preprocessor(groups).build_pipeline().fit(frame, target)
    return save_bundle(
        pipeline,
        groups,
        inputs,
        {
            "threshold": 0.5,
            "roc_auc": 0.5,
            "sklearn_version": sklearn.__version__,
            "synthetic": True,
        },
        path,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `uv run cs-synthetic-artifact`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Write a synthetic model artifact, for tests and CI only."
    )
    # Required: a default would sit next to the real champion, models/pipeline.joblib.
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    build_synthetic_artifact(args.output, seed=args.seed)
