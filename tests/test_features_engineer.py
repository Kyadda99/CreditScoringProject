"""Testy transformera cech."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from config import ENGINEERED_FEATURES
from features import FLAG_NOT_EMPLOYED, FeatureEngineer
from features.ratios import RATIO_FUNCTIONS


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [202500.0, 100000.0],
            "AMT_CREDIT": [406597.5, 200000.0],
            "AMT_ANNUITY": [24700.5, 10000.0],
            "AMT_GOODS_PRICE": [351000.0, 180000.0],
            "CNT_FAM_MEMBERS": [2.0, 1.0],
            "DAYS_BIRTH": [-9461.0, -12000.0],
            "DAYS_EMPLOYED": [-637.0, 365243.0],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
        }
    )


def test_registry_matches_the_config_contract() -> None:
    """`RATIO_FUNCTIONS` i `ENGINEERED_FEATURES` nie mogą się rozjechać."""
    assert set(RATIO_FUNCTIONS) == set(ENGINEERED_FEATURES)


def test_transform_adds_every_engineered_column(frame: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(frame)
    for name in ENGINEERED_FEATURES:
        assert name in result.columns


def test_transform_keeps_the_raw_columns(frame: pd.DataFrame) -> None:
    """Kolumny surowe zostają — o wyborze decyduje `ColumnTransformer`."""
    result = FeatureEngineer().fit_transform(frame)
    assert "NAME_CONTRACT_TYPE" in result.columns
    assert "AMT_CREDIT" in result.columns


def test_transform_decodes_the_sentinel_in_place(frame: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(frame)
    assert np.isnan(result["DAYS_EMPLOYED"].iloc[1])
    assert result[FLAG_NOT_EMPLOYED].tolist() == [0.0, 1.0]


def test_flag_is_added_even_with_no_engineered_features(frame: pd.DataFrame) -> None:
    """Transza 1 (spec D7): czyszczenie tak, ilorazy nie."""
    result = FeatureEngineer(features=()).fit_transform(frame)
    assert FLAG_NOT_EMPLOYED in result.columns
    assert "CREDIT_INCOME_RATIO" not in result.columns


def test_input_frame_is_not_mutated(frame: pd.DataFrame) -> None:
    FeatureEngineer().fit_transform(frame)
    assert frame["DAYS_EMPLOYED"].iloc[1] == 365243.0
    assert "CREDIT_INCOME_RATIO" not in frame.columns


def test_transform_is_total_on_a_partial_row(frame: pd.DataFrame) -> None:
    """Klient /score przysyła podzbiór pól — brak kolumny to NaN, nie KeyError."""
    engineer = FeatureEngineer().fit(frame)
    partial = pd.DataFrame({"AMT_CREDIT": [1000.0]})
    result = engineer.transform(partial)
    assert np.isnan(result["CREDIT_INCOME_RATIO"].iloc[0])


def test_fit_rejects_a_training_frame_missing_a_source_column(
    frame: pd.DataFrame,
) -> None:
    """Przy treningu brak kolumny źródłowej to błąd pipeline'u, nie żądania."""
    with pytest.raises(ValueError, match="AMT_INCOME_TOTAL"):
        FeatureEngineer().fit(frame.drop(columns=["AMT_INCOME_TOTAL"]))


def test_get_feature_names_out_matches_transform(frame: pd.DataFrame) -> None:
    engineer = FeatureEngineer().fit(frame)
    result = engineer.transform(frame)
    assert list(engineer.get_feature_names_out()) == list(result.columns)


def test_transformer_survives_a_joblib_round_trip(
    frame: pd.DataFrame, tmp_path: Path
) -> None:
    """Transformer jedzie w artefakcie — musi się piklować."""
    engineer = FeatureEngineer().fit(frame)
    path = tmp_path / "engineer.joblib"
    joblib.dump(engineer, path)
    restored = joblib.load(path)
    pd.testing.assert_frame_equal(restored.transform(frame), engineer.transform(frame))


def test_clone_preserves_the_features_parameter() -> None:
    """`clone` wymaga, żeby __init__ zapisywał parametry bez zmian."""
    engineer = FeatureEngineer(features=("CREDIT_TERM",))
    assert clone(engineer).features == ("CREDIT_TERM",)
