"""Strażnik podziału train/test.

Podział liczymy z RANDOM_STATE, nie przechowujemy listy identyfikatorów.
`train_test_split` jest jednak funkcją ramki, którą dostanie — jeśli Faza 2
usunie choć jeden wiersz, podział cicho się przesunie, a porównanie modeli
z Faz 3 i 4 przestanie być porównaniem. Ten test zamienia cichy dryf w głośną
porażkę.

Gdy zmiana podziału jest ZAMIERZONA: uruchom test, przepisz wypisaną sumę
kontrolną do EXPECTED_TEST_SET_SHA256 i odnotuj powód w findings danej fazy.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from config import ID_COLUMN
from data import load_data
from train import split

pytestmark = pytest.mark.requires_data

EXPECTED_TEST_SET_SHA256 = (
    "fc6c4da2768918a908ce3cce54abc7ec66d9efc15d49e41231e7fb0941260bc2"
)
EXPECTED_TEST_ROWS = 61_503


def _checksum(ids: pd.Series) -> str:
    """SHA-256 po posortowanych identyfikatorach, złączonych przecinkami.

    Nie `hash()`: wbudowany hash Pythona jest solony per proces (PYTHONHASHSEED),
    więc dawałby inny wynik przy każdym uruchomieniu.
    """
    joined = ",".join(str(i) for i in sorted(ids.tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def test_test_split_has_not_drifted() -> None:
    df = load_data()
    _X_train, X_test, _y_train, _y_test = split(df)
    actual = _checksum(df.loc[X_test.index, ID_COLUMN])
    assert actual == EXPECTED_TEST_SET_SHA256, (
        f"Podział testowy się zmienił. Nowa suma: {actual}\n"
        "Jeśli to zamierzone, wpisz ją do EXPECTED_TEST_SET_SHA256 "
        "i opisz powód w findings tej fazy."
    )


def test_test_split_size_is_stable() -> None:
    df = load_data()
    _X_train, X_test, _y_train, _y_test = split(df)
    assert len(X_test) == EXPECTED_TEST_ROWS


def test_stratification_preserves_the_target_rate() -> None:
    df = load_data()
    _X_train, _X_test, y_train, y_test = split(df)
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.005
