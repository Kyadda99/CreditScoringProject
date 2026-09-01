"""Testy orkiestracji treningu — logika, nie pełny przebieg na 158 MB."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from config import ID_COLUMN, TARGET, FeatureGroups
from train import build_pipeline, split, write_feature_schema


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 200
    return pd.DataFrame(
        {
            ID_COLUMN: range(size),
            TARGET: rng.integers(0, 2, size=size),
            "AMT_INCOME_TOTAL": rng.random(size) * 1000,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash", "Revolving"], size=size),
            "FLAG_MOBIL": rng.integers(0, 2, size=size),
        }
    )


@pytest.fixture
def groups() -> FeatureGroups:
    return FeatureGroups(
        numeric=("AMT_INCOME_TOTAL",),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )


def test_split_excludes_target_and_id_from_features(frame: pd.DataFrame) -> None:
    X_train, _X_test, _y_train, _y_test = split(frame)
    assert TARGET not in X_train.columns
    assert ID_COLUMN not in X_train.columns


def test_split_is_eighty_twenty(frame: pd.DataFrame) -> None:
    X_train, X_test, _y_train, _y_test = split(frame)
    assert len(X_test) == pytest.approx(len(frame) * 0.2, abs=1)
    assert len(X_train) + len(X_test) == len(frame)


def test_split_is_stratified(frame: pd.DataFrame) -> None:
    """Przy 8% klasy pozytywnej niestratyfikowany podział potrafi ją zgubić."""
    _X_train, _X_test, y_train, y_test = split(frame)
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.05


def test_split_is_reproducible(frame: pd.DataFrame) -> None:
    """Cały projekt stoi na tym, że RANDOM_STATE daje ten sam podział."""
    first = split(frame)[1].index.tolist()
    second = split(frame)[1].index.tolist()
    assert first == second


def test_pipeline_has_preprocessor_then_model(groups: FeatureGroups) -> None:
    pipeline = build_pipeline(groups)
    assert isinstance(pipeline, Pipeline)
    assert list(dict(pipeline.steps)) == ["preprocessor", "model"]


def test_pipeline_fits_and_predicts_in_unit_range(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    X_train, X_test, y_train, _y_test = split(frame)
    pipeline = build_pipeline(groups).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_write_feature_schema_round_trips(tmp_path: Path) -> None:
    groups = FeatureGroups(numeric=("a",), categorical=("b",), binary=("c",))
    path = write_feature_schema(groups, tmp_path / "feature_schema.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert FeatureGroups.from_dict(payload) == groups
