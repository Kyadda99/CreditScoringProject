"""Testy dekodowania wartości sentinelowych."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.cleaning import days_employed_cleaned, not_employed_flag


def test_sentinel_becomes_nan() -> None:
    frame = pd.DataFrame({"DAYS_EMPLOYED": [-637.0, 365243.0, -1000.0]})
    result = days_employed_cleaned(frame)
    assert result.tolist()[0] == -637.0
    assert np.isnan(result.tolist()[1])
    assert result.tolist()[2] == -1000.0


def test_flag_marks_exactly_the_sentinel_rows() -> None:
    frame = pd.DataFrame({"DAYS_EMPLOYED": [-637.0, 365243.0, np.nan]})
    assert not_employed_flag(frame).tolist() == [0.0, 1.0, 0.0]


def test_missing_input_gives_nan_not_keyerror() -> None:
    """`transform` musi być totalna — pola /score są opcjonalne."""
    empty = pd.DataFrame(index=[0, 1])
    assert days_employed_cleaned(empty).isna().all()
    assert not_employed_flag(empty).tolist() == [0.0, 0.0]


def test_input_frame_is_not_mutated() -> None:
    frame = pd.DataFrame({"DAYS_EMPLOYED": [365243.0]})
    days_employed_cleaned(frame)
    assert frame["DAYS_EMPLOYED"].iloc[0] == 365243.0


def test_flag_is_a_binary_float_column() -> None:
    """`split_feature_groups` klasyfikuje flagę po wartościach — muszą być {0,1}."""
    frame = pd.DataFrame({"DAYS_EMPLOYED": [-1.0, 365243.0]})
    assert set(not_employed_flag(frame).unique()) <= {0.0, 1.0}
