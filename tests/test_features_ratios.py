"""Testy cech pochodnych — wartości liczone ręcznie plus przypadki brzegowe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.ratios import (
    RATIO_FUNCTIONS,
    annuity_income_ratio,
    bureau_inquiry_total,
    credit_income_ratio,
    credit_term,
    employed_age_ratio,
    ext_source_count,
    ext_source_mean,
    goods_credit_ratio,
    income_per_person,
)


@pytest.fixture
def row() -> pd.DataFrame:
    """Jeden wiersz z przykładu w `api/schemas.EXAMPLES`."""
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [202500.0],
            "AMT_CREDIT": [406597.5],
            "AMT_ANNUITY": [24700.5],
            "AMT_GOODS_PRICE": [351000.0],
            "CNT_FAM_MEMBERS": [2.0],
            "DAYS_BIRTH": [-9461.0],
            "DAYS_EMPLOYED": [-637.0],
        }
    )


# --- T1: wartości liczone ręcznie ------------------------------------------


def test_credit_income_ratio(row: pd.DataFrame) -> None:
    assert credit_income_ratio(row).iloc[0] == pytest.approx(406597.5 / 202500.0)


def test_annuity_income_ratio(row: pd.DataFrame) -> None:
    assert annuity_income_ratio(row).iloc[0] == pytest.approx(24700.5 / 202500.0)


def test_credit_term(row: pd.DataFrame) -> None:
    assert credit_term(row).iloc[0] == pytest.approx(24700.5 / 406597.5)


def test_goods_credit_ratio(row: pd.DataFrame) -> None:
    assert goods_credit_ratio(row).iloc[0] == pytest.approx(351000.0 / 406597.5)


def test_income_per_person(row: pd.DataFrame) -> None:
    assert income_per_person(row).iloc[0] == pytest.approx(202500.0 / 2.0)


def test_employed_age_ratio(row: pd.DataFrame) -> None:
    assert employed_age_ratio(row).iloc[0] == pytest.approx(-637.0 / -9461.0)


def test_bureau_inquiry_total_sums_the_six_columns() -> None:
    frame = pd.DataFrame(
        {
            "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0],
            "AMT_REQ_CREDIT_BUREAU_DAY": [1.0],
            "AMT_REQ_CREDIT_BUREAU_WEEK": [0.0],
            "AMT_REQ_CREDIT_BUREAU_MON": [2.0],
            "AMT_REQ_CREDIT_BUREAU_QRT": [0.0],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [3.0],
        }
    )
    assert bureau_inquiry_total(frame).iloc[0] == pytest.approx(6.0)


def test_bureau_inquiry_total_is_nan_when_all_six_are_missing() -> None:
    """Suma samych braków to brak, nie zero — inaczej "nie wiem" udaje "zero"."""
    frame = pd.DataFrame({"AMT_REQ_CREDIT_BUREAU_DAY": [np.nan]}, index=[0])
    assert np.isnan(bureau_inquiry_total(frame).iloc[0])


def test_ext_source_mean_ignores_missing() -> None:
    frame = pd.DataFrame(
        {"EXT_SOURCE_1": [0.4], "EXT_SOURCE_2": [np.nan], "EXT_SOURCE_3": [0.6]}
    )
    assert ext_source_mean(frame).iloc[0] == pytest.approx(0.5)


def test_ext_source_count_counts_present_values() -> None:
    frame = pd.DataFrame(
        {"EXT_SOURCE_1": [0.4], "EXT_SOURCE_2": [np.nan], "EXT_SOURCE_3": [0.6]}
    )
    assert ext_source_count(frame).iloc[0] == pytest.approx(2.0)


# --- T2: przypadki brzegowe ------------------------------------------------


@pytest.mark.parametrize(
    ("income", "credit"),
    [(0.0, 100.0), (np.nan, 100.0), (0.0, np.nan), (np.nan, np.nan)],
)
def test_zero_or_missing_denominator_gives_nan(income: float, credit: float) -> None:
    frame = pd.DataFrame({"AMT_INCOME_TOTAL": [income], "AMT_CREDIT": [credit]})
    assert np.isnan(credit_income_ratio(frame).iloc[0])


def test_zero_family_members_gives_nan() -> None:
    frame = pd.DataFrame({"AMT_INCOME_TOTAL": [1.0], "CNT_FAM_MEMBERS": [0.0]})
    assert np.isnan(income_per_person(frame).iloc[0])


def test_employed_age_ratio_handles_the_sentinel() -> None:
    """Iloraz liczy się na DAYS_EMPLOYED po dekodowaniu, nie na surowej wartości."""
    frame = pd.DataFrame({"DAYS_EMPLOYED": [365243.0], "DAYS_BIRTH": [-9461.0]})
    assert np.isnan(employed_age_ratio(frame).iloc[0])


def test_every_function_survives_a_completely_empty_frame() -> None:
    empty = pd.DataFrame(index=[0])
    for name, function in RATIO_FUNCTIONS.items():
        result = function(empty)
        assert len(result) == 1, name


# --- T3: reguła D5, jako własność ------------------------------------------


def test_no_engineered_column_is_ever_infinite() -> None:
    """Spec D5. Ramka jest zbudowana tak, żeby sprowokować dzielenie przez zero."""
    provocative = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [0.0, np.nan, 1e-300, 202500.0],
            "AMT_CREDIT": [1e308, 1.0, 1e308, 406597.5],
            "AMT_ANNUITY": [1e308, 1.0, 1.0, 24700.5],
            "AMT_GOODS_PRICE": [1.0, 1.0, 1.0, 351000.0],
            "CNT_FAM_MEMBERS": [0.0, 0.0, np.nan, 2.0],
            "DAYS_BIRTH": [0.0, -1.0, np.nan, -9461.0],
            "DAYS_EMPLOYED": [365243.0, -1.0, 0.0, -637.0],
        }
    )
    for name, function in RATIO_FUNCTIONS.items():
        values = function(provocative).to_numpy(dtype="float64")
        assert not np.isinf(values).any(), f"{name} wyprodukowało ±inf"
