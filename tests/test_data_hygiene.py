"""Testy higieny wierszy — bez dostępu do prawdziwych danych.

Osobny plik, bo `test_data_contract.py` ma modułowy `pytestmark =
requires_data`. Te testy działają na ramkach syntetycznych i muszą być zielone
także w CI, gdzie zbioru nie ma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import drop_unusable_rows


def test_rows_without_a_target_are_dropped() -> None:
    frame = pd.DataFrame({"TARGET": [0.0, np.nan, 1.0], "CODE_GENDER": ["M", "F", "F"]})
    assert len(drop_unusable_rows(frame)) == 2


def test_xna_gender_rows_are_dropped() -> None:
    """Cztery wiersze na 307 tysięcy; kategoria "XNA" to brak, nie płeć."""
    frame = pd.DataFrame({"TARGET": [0.0, 1.0], "CODE_GENDER": ["M", "XNA"]})
    assert drop_unusable_rows(frame)["CODE_GENDER"].tolist() == ["M"]


def test_hygiene_is_a_no_op_on_clean_data() -> None:
    frame = pd.DataFrame({"TARGET": [0.0, 1.0], "CODE_GENDER": ["M", "F"]})
    pd.testing.assert_frame_equal(drop_unusable_rows(frame), frame)


def test_hygiene_tolerates_a_frame_without_those_columns() -> None:
    """Ramka żądania nie ma ani celu, ani (być może) płci."""
    frame = pd.DataFrame({"AMT_CREDIT": [1.0]})
    assert len(drop_unusable_rows(frame)) == 1
