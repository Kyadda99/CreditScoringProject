"""Sentinel value decoding.

`DAYS_EMPLOYED == 365243` is ~1000 years of service: how the dataset encodes
"not employed". Left as a number it enters `StandardScaler` hundreds of
deviations out and would dominate the feature in a linear model.

Decoding lives in the pipeline, not in `data.py`, because a `/score` client can
send 365243 and cleaning done only at load time would not cover production.

All functions are pure: they take a frame, return a new series, and leave the
input untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DAYS_EMPLOYED_SENTINEL

_DAYS_EMPLOYED = "DAYS_EMPLOYED"


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return the column as float, or a NaN series when it is absent.

    A missing column is not an error: `/score` fields are optional and the
    fitted imputer is where missing values are handled.

    Args:
        frame: Input frame.
        name: Column name.

    Returns:
        A `float64` series on the frame's index.
    """
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def days_employed_cleaned(frame: pd.DataFrame) -> pd.Series:
    """`DAYS_EMPLOYED` with the sentinel value replaced by NaN.

    Args:
        frame: Frame that may contain `DAYS_EMPLOYED`.

    Returns:
        A `float64` series; sentinel becomes NaN, the rest is unchanged.
    """
    series = _numeric_column(frame, _DAYS_EMPLOYED)
    return series.mask(series == DAYS_EMPLOYED_SENTINEL)


def not_employed_flag(frame: pd.DataFrame) -> pd.Series:
    """Explicit 0/1 flag: whether the row held the sentinel value.

    The "not employed" signal is real: 5.40% default rate in that group against
    8.66% elsewhere. Without this flag it disappears with the sentinel.

    Args:
        frame: Frame that may contain `DAYS_EMPLOYED`.

    Returns:
        A `float64` series of 0.0/1.0. NaN yields 0.0: an absent value is not
        evidence of unemployment.
    """
    series = _numeric_column(frame, _DAYS_EMPLOYED)
    return (series == DAYS_EMPLOYED_SENTINEL).astype("float64")
