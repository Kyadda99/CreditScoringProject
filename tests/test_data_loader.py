"""Kontrakt DataLoadera: wczytanie, podział i dwa kontrakty kolumn."""

from __future__ import annotations

import pandas as pd
import pytest

from config import DATA_PATH, ENGINEERED_FEATURES, ID_COLUMN, TARGET
from data_loader import DataLoader
from test_characterisation import synthetic_frame


@pytest.fixture
def frame() -> pd.DataFrame:
    data, target = synthetic_frame(rows=80)
    data[TARGET] = target
    data[ID_COLUMN] = range(100000, 100000 + len(data))
    return data


def test_split_is_stratified_and_drops_target_and_id(frame: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = DataLoader().split(frame)
    assert TARGET not in X_train.columns
    assert ID_COLUMN not in X_train.columns
    assert len(X_test) == pytest.approx(len(frame) * 0.2, abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)


def test_split_is_deterministic(frame: pd.DataFrame) -> None:
    first = DataLoader().split(frame)[0].index.tolist()
    second = DataLoader().split(frame)[0].index.tolist()
    assert first == second


def test_model_groups_include_engineered_features(frame: pd.DataFrame) -> None:
    loader = DataLoader()
    X_train = loader.split(frame)[0]
    groups = loader.model_groups(X_train)
    assert set(ENGINEERED_FEATURES) <= set(groups.all_features)


def test_input_contract_excludes_engineered_and_keeps_sources(
    frame: pd.DataFrame,
) -> None:
    loader = DataLoader()
    X_train = loader.split(frame)[0]
    contract = loader.input_contract(X_train)
    assert not set(contract.all_features) & set(ENGINEERED_FEATURES)
    assert "AMT_INCOME_TOTAL" in contract.all_features


def test_default_path_is_the_project_dataset() -> None:
    assert DataLoader().path == DATA_PATH


@pytest.mark.requires_data
def test_load_returns_the_real_frame() -> None:
    df = DataLoader().load()
    assert df.shape[1] == 122
    assert TARGET in df.columns
