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
            "AMT_INCOME_TOTAL": rng.random(size) * 1000 + 1.0,
            "AMT_CREDIT": rng.random(size) * 2000 + 1.0,
            "AMT_ANNUITY": rng.random(size) * 100 + 1.0,
            "AMT_GOODS_PRICE": rng.random(size) * 1800 + 1.0,
            "CNT_FAM_MEMBERS": rng.integers(1, 4, size=size).astype(float),
            "DAYS_BIRTH": -rng.integers(7000, 20000, size=size).astype(float),
            "DAYS_EMPLOYED": -rng.integers(100, 5000, size=size).astype(float),
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


def test_pipeline_has_features_preprocessor_then_model(
    groups: FeatureGroups,
) -> None:
    pipeline = build_pipeline(groups)
    assert isinstance(pipeline, Pipeline)
    assert list(dict(pipeline.steps)) == ["features", "preprocessor", "model"]


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


# --- Faza 2: kontrakt wejściowy i inżynieria cech (spec D1, D2, D7) --------


def test_input_contract_is_raw_columns_only() -> None:
    """Kontrakt wejściowy nie może zawierać cech pochodnych (spec D2)."""
    from train import input_contract

    frame = pd.DataFrame(
        {
            "AMT_CREDIT": [1.0, 2.0],
            "AMT_INCOME_TOTAL": [3.0, 4.0],
            "NAME_CONTRACT_TYPE": ["a", "b"],
        }
    )
    contract = input_contract(frame)
    assert "CREDIT_INCOME_RATIO" not in contract.all_features


def test_input_contract_keeps_a_dropped_column_that_feeds_a_feature() -> None:
    """Reguła przynależności z D2 — bez tego iloraz zawsze wychodzi NaN."""
    from config import FEATURE_SOURCE_COLUMNS
    from train import input_contract

    frame = pd.DataFrame({name: [1.0, 2.0] for name in FEATURE_SOURCE_COLUMNS})
    frame["JUNK"] = [9.0, 9.0]
    contract = input_contract(frame, dropped=(*FEATURE_SOURCE_COLUMNS, "JUNK"))
    assert set(FEATURE_SOURCE_COLUMNS) <= set(contract.all_features)
    assert "JUNK" not in contract.all_features


def test_build_pipeline_puts_the_engineer_first() -> None:
    from config import FeatureGroups
    from train import build_pipeline

    groups = FeatureGroups(numeric=("AMT_CREDIT",), categorical=(), binary=())
    assert list(build_pipeline(groups).named_steps) == [
        "features",
        "preprocessor",
        "model",
    ]


def test_build_pipeline_can_disable_engineered_features() -> None:
    """Transza 1 ze spec D7."""
    from config import FeatureGroups
    from train import build_pipeline

    groups = FeatureGroups(numeric=("AMT_CREDIT",), categorical=(), binary=())
    pipeline = build_pipeline(groups, features=())
    assert pipeline.named_steps["features"].features == ()
