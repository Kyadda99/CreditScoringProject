"""Derived features: one pure function per feature.

Each takes a frame and returns a series on the same index. Nothing holds state.

The governing rule: no ratio returns +/-inf. The request model sets
`allow_inf_nan=False`, and `SimpleImputer` treats `inf` as an ordinary value,
which corrupts the mean and variance of the whole column in `StandardScaler`.
A zero or missing denominator yields NaN, which the fitted imputer handles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from features.cleaning import _numeric_column, days_employed_cleaned

BUREAU_INQUIRY_COLUMNS: tuple[str, ...] = (
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)

EXT_SOURCE_COLUMNS: tuple[str, ...] = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide so the result is never +/-inf.

    A zero denominator becomes NaN before the division, not after: afterwards
    `0 / 0` gives NaN while `1 / 0` gives inf, which would need two cases. The
    extra cleanup catches float overflow on extreme amounts.

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        A `float64` series; NaN wherever the denominator is zero or missing, or
        the result overflowed float range.
    """
    denom = denominator.astype("float64").mask(denominator == 0)
    result = numerator.astype("float64") / denom
    return result.replace([np.inf, -np.inf], np.nan)


def credit_income_ratio(frame: pd.DataFrame) -> pd.Series:
    """`AMT_CREDIT / AMT_INCOME_TOTAL`: leverage."""
    return _safe_divide(
        _numeric_column(frame, "AMT_CREDIT"),
        _numeric_column(frame, "AMT_INCOME_TOTAL"),
    )


def annuity_income_ratio(frame: pd.DataFrame) -> pd.Series:
    """`AMT_ANNUITY / AMT_INCOME_TOTAL`: debt-to-income."""
    return _safe_divide(
        _numeric_column(frame, "AMT_ANNUITY"),
        _numeric_column(frame, "AMT_INCOME_TOTAL"),
    )


def credit_term(frame: pd.DataFrame) -> pd.Series:
    """`AMT_ANNUITY / AMT_CREDIT`: inverse of the implied loan term."""
    return _safe_divide(
        _numeric_column(frame, "AMT_ANNUITY"),
        _numeric_column(frame, "AMT_CREDIT"),
    )


def goods_credit_ratio(frame: pd.DataFrame) -> pd.Series:
    """`AMT_GOODS_PRICE / AMT_CREDIT`: proxy for the down payment."""
    return _safe_divide(
        _numeric_column(frame, "AMT_GOODS_PRICE"),
        _numeric_column(frame, "AMT_CREDIT"),
    )


def income_per_person(frame: pd.DataFrame) -> pd.Series:
    """`AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`: income per household member."""
    return _safe_divide(
        _numeric_column(frame, "AMT_INCOME_TOTAL"),
        _numeric_column(frame, "CNT_FAM_MEMBERS"),
    )


def employed_age_ratio(frame: pd.DataFrame) -> pd.Series:
    """Length of employment relative to age.

    Computed on `DAYS_EMPLOYED` after sentinel decoding; on the raw 365243 the
    ratio would come out negative and meaningless.
    """
    return _safe_divide(
        days_employed_cleaned(frame),
        _numeric_column(frame, "DAYS_BIRTH"),
    )


def bureau_inquiry_total(frame: pd.DataFrame) -> pd.Series:
    """Total number of credit bureau inquiries.

    Returns:
        Sum of the six columns. `min_count=1` makes an all-missing row yield
        NaN rather than 0.0, so "unknown" cannot pass for "zero inquiries".
    """
    columns = [_numeric_column(frame, name) for name in BUREAU_INQUIRY_COLUMNS]
    return pd.concat(columns, axis=1).sum(axis=1, min_count=1)


def ext_source_mean(frame: pd.DataFrame) -> pd.Series:
    """Mean of the three external scores, over the available values."""
    columns = [_numeric_column(frame, name) for name in EXT_SOURCE_COLUMNS]
    return pd.concat(columns, axis=1).mean(axis=1)


def ext_source_count(frame: pd.DataFrame) -> pd.Series:
    """How many of the three external scores are present.

    Absence is itself a signal here: the number of available scores says
    something about the client.
    """
    columns = [_numeric_column(frame, name) for name in EXT_SOURCE_COLUMNS]
    return pd.concat(columns, axis=1).notna().sum(axis=1).astype("float64")


RATIO_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "ANNUITY_INCOME_RATIO": annuity_income_ratio,
    "BUREAU_INQUIRY_TOTAL": bureau_inquiry_total,
    "CREDIT_INCOME_RATIO": credit_income_ratio,
    "CREDIT_TERM": credit_term,
    "EMPLOYED_AGE_RATIO": employed_age_ratio,
    "EXT_SOURCE_COUNT": ext_source_count,
    "EXT_SOURCE_MEAN": ext_source_mean,
    "GOODS_CREDIT_RATIO": goods_credit_ratio,
    "INCOME_PER_PERSON": income_per_person,
}
"""Feature name -> function. `FeatureEngineer` iterates over this mapping.

The keys must match `config.ENGINEERED_FEATURES`; a test enforces that.
"""
