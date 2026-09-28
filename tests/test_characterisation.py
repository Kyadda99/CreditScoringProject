"""Utrwalone zachowanie ścieżki treningowej.

Te testy nie opisują nowego wymagania — pilnują zachowania, które już istnieje,
żeby przenoszenie kodu między modułami nie zmieniło go po cichu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.schemas import load_feature_groups
from config import (
    DROPPED_COLUMNS,
    ENGINEERED_FEATURES,
    FeatureGroups,
    split_feature_groups,
)
from data_loader import DataLoader
from features import FeatureEngineer
from preprocessor import Preprocessor
from strategies import get_estimator_strategy, get_imbalance_strategy


def synthetic_frame(rows: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Buduje małą ramkę o kontrakcie produkcyjnym i etykiety 0/1.

    Args:
        rows: Liczba wierszy.
        seed: Ziarno generatora.

    Returns:
        `(ramka surowych kolumn, etykiety)`.
    """
    inputs = load_feature_groups()
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    for name in inputs.numeric:
        data[name] = rng.uniform(1.0, 100.0, size=rows)
    for name in inputs.binary:
        data[name] = rng.integers(0, 2, size=rows).astype(float)
    for name in inputs.categorical:
        data[name] = rng.choice(["A", "B"], size=rows)
    frame = pd.DataFrame(data, columns=list(inputs.all_features))
    target = rng.integers(0, 2, size=rows)
    return frame, target


@pytest.fixture(scope="module")
def frame_and_target() -> tuple[pd.DataFrame, np.ndarray]:
    return synthetic_frame()


def test_pipeline_step_names_are_stable(
    frame_and_target: tuple[pd.DataFrame, np.ndarray],
) -> None:
    frame, _ = frame_and_target
    groups = split_feature_groups(
        FeatureEngineer().fit_transform(frame), dropped=DROPPED_COLUMNS
    )
    assert [name for name, _ in Preprocessor(groups).build_pipeline().steps] == [
        "features",
        "preprocessor",
        "model",
    ]


def test_sampler_arm_inserts_a_sampler_step(
    frame_and_target: tuple[pd.DataFrame, np.ndarray],
) -> None:
    frame, _ = frame_and_target
    groups = split_feature_groups(
        FeatureEngineer().fit_transform(frame), dropped=DROPPED_COLUMNS
    )
    pipeline = Preprocessor(groups).build_pipeline(
        get_estimator_strategy("xgboost").build(
            class_weighted=False, scale_pos_weight=1.0
        ),
        get_imbalance_strategy("resample").make_sampler(),
    )
    assert [name for name, _ in pipeline.steps] == [
        "features",
        "preprocessor",
        "sampler",
        "model",
    ]


def test_engineered_feature_values_are_stable(
    frame_and_target: tuple[pd.DataFrame, np.ndarray],
) -> None:
    frame, _ = frame_and_target
    engineered = FeatureEngineer().fit_transform(frame)
    expected = frame["AMT_CREDIT"] / frame["AMT_INCOME_TOTAL"]
    pd.testing.assert_series_equal(
        engineered["CREDIT_INCOME_RATIO"], expected, check_names=False
    )
    assert set(ENGINEERED_FEATURES) <= set(engineered.columns)


def test_input_contract_keeps_feature_source_columns(
    frame_and_target: tuple[pd.DataFrame, np.ndarray],
) -> None:
    frame, _ = frame_and_target
    contract = DataLoader().input_contract(frame)
    assert "AMT_INCOME_TOTAL" in contract.all_features
    assert not set(contract.all_features) & set(ENGINEERED_FEATURES)


def test_fitted_pipeline_probabilities_are_reproducible(
    frame_and_target: tuple[pd.DataFrame, np.ndarray],
) -> None:
    frame, target = frame_and_target
    first = (
        Preprocessor(_groups(frame))
        .build_pipeline()
        .fit(frame, target)
        .predict_proba(frame)[:, 1]
    )
    second = (
        Preprocessor(_groups(frame))
        .build_pipeline()
        .fit(frame, target)
        .predict_proba(frame)[:, 1]
    )
    np.testing.assert_array_equal(first, second)


def _groups(frame: pd.DataFrame) -> FeatureGroups:
    """Zwraca grupy cech modelu dla ramki po inżynierii cech."""
    return split_feature_groups(
        FeatureEngineer().fit_transform(frame), dropped=DROPPED_COLUMNS
    )
