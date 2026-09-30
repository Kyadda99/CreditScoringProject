"""Shared pytest configuration.

The data contract tests need the ~160 MB CSV, which CI does not have. The
`requires_data` marker skips them cleanly without disabling the whole file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import sklearn

from config import DATA_PATH


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked `requires_data` when the dataset is not on disk."""
    if DATA_PATH.exists():
        return
    skip = pytest.mark.skip(reason=f"Missing {DATA_PATH}; run `uv run cs-download`")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _write_column_manifest() -> None:
    """Write the column manifest from the real file when CS_WRITE_MANIFEST=1.

    Run once after the first download; the result is committed, and from then on
    any column rename fails the contract test.
    """
    if os.environ.get("CS_WRITE_MANIFEST") != "1" or not DATA_PATH.exists():
        return

    from data import load_data
    from test_data_contract import MANIFEST_PATH

    columns = list(load_data().columns)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(columns, indent=2), encoding="utf-8")
    print(f"\nWrote a manifest of {len(columns)} columns -> {MANIFEST_PATH}")


@pytest.fixture(scope="session")
def synthetic_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny but real model artifact, without the 158 MB CSV.

    These HTTP tests once hung on the `requires_data` marker, so CI skipped them
    all and a regression in /score went green. The columns come from the committed
    `feature_schema.json`, so the pipeline has exactly the production contract,
    trained on a handful of invented rows.
    """
    import numpy as np
    import pandas as pd

    from api.schemas import load_feature_groups
    from artifact import save_bundle
    from config import DROPPED_COLUMNS, split_feature_groups
    from features import FeatureEngineer
    from preprocessor import Preprocessor

    inputs = load_feature_groups()  # input contract: raw columns
    rng = np.random.default_rng(0)
    rows = 40
    data: dict[str, object] = {}
    for name in inputs.numeric:
        # Positive and away from zero: a ratio denominator cannot be zero, or the
        # whole column comes out NaN and the imputer has nothing to learn from.
        data[name] = rng.uniform(1.0, 100.0, size=rows)
    for name in inputs.binary:
        data[name] = rng.integers(0, 2, size=rows).astype(float)
    for name in inputs.categorical:
        data[name] = rng.choice(["A", "B"], size=rows)
    frame = pd.DataFrame(data, columns=list(inputs.all_features))
    target = rng.integers(0, 2, size=rows)

    # Model groups come from the frame after engineering, as training does.
    engineered = FeatureEngineer().fit_transform(frame)
    groups = split_feature_groups(engineered, dropped=DROPPED_COLUMNS)

    pipeline = Preprocessor(groups).build_pipeline().fit(frame, target)
    path = tmp_path_factory.mktemp("artifact") / "pipeline.joblib"
    return save_bundle(
        pipeline,
        groups,
        inputs,
        {"threshold": 0.5, "roc_auc": 0.5, "sklearn_version": sklearn.__version__},
        path,
    )
