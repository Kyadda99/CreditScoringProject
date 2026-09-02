"""Testy funkcji analitycznych EDA.

Wszystkie działają na ramkach syntetycznych — CI nie ma zbioru danych, a te
funkcje muszą być zielone także tam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda import (
    cardinality_report,
    correlation_screen,
    missingness_report,
    target_rate_by_bin,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full": [1.0, 2.0, 3.0, 4.0],
            "half": [1.0, np.nan, 3.0, np.nan],
            "empty": [np.nan, np.nan, np.nan, np.nan],
            "cat": ["A", "A", "B", None],
            "TARGET": [0, 1, 0, 1],
        }
    )


def test_missingness_counts_and_shares(frame: pd.DataFrame) -> None:
    report = missingness_report(frame).set_index("column")
    assert report.loc["full", "n_missing"] == 0
    assert report.loc["half", "n_missing"] == 2
    assert report.loc["half", "pct_missing"] == pytest.approx(50.0)
    assert report.loc["empty", "pct_missing"] == pytest.approx(100.0)


def test_missingness_sorted_worst_first(frame: pd.DataFrame) -> None:
    """Kolejność jest częścią kontraktu — raport czyta się od góry."""
    report = missingness_report(frame)
    assert report["pct_missing"].is_monotonic_decreasing


def test_cardinality_counts_distinct_ignoring_nan(frame: pd.DataFrame) -> None:
    report = cardinality_report(frame).set_index("column")
    assert report.loc["cat", "n_unique"] == 2
    assert report.loc["cat", "top_value"] == "A"
    assert report.loc["cat", "top_share"] == pytest.approx(2 / 3)


def test_cardinality_only_covers_non_numeric(frame: pd.DataFrame) -> None:
    """Faza 4 wymiaruje `nn.Embedding` z tej tabeli — liczby jej nie dotyczą."""
    report = cardinality_report(frame)
    assert set(report["column"]) == {"cat"}


def test_target_rate_by_bin_returns_one_row_per_bin(frame: pd.DataFrame) -> None:
    result = target_rate_by_bin(frame, "full", bins=2)
    assert len(result) == 2
    assert result["n"].sum() == 4
    assert set(result.columns) == {"bin", "n", "target_rate"}


def test_target_rate_by_bin_computes_the_rate() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "TARGET": [0, 0, 1, 1]})
    result = target_rate_by_bin(df, "x", bins=2)
    assert result["target_rate"].tolist() == pytest.approx([0.0, 1.0])


def test_correlation_screen_finds_redundant_pairs() -> None:
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "c": [3.0, 1.0, 2.0]}
    )
    pairs = correlation_screen(df, threshold=0.99)
    assert pairs[["left", "right"]].values.tolist() == [["a", "b"]]
    assert pairs["correlation"].iloc[0] == pytest.approx(1.0)


def test_correlation_screen_reports_each_pair_once() -> None:
    """Górny trójkąt, nie cała macierz — inaczej każda para jest podwójnie."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    assert len(correlation_screen(df, threshold=0.5)) == 1
