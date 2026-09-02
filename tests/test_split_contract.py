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

# ZMIANA ZAMIERZONA (Faza 2). Higiena wierszy usuwa 4 rekordy
# `CODE_GENDER == "XNA"` (spec D4), więc `train_test_split` dostaje inną ramkę
# i podział się przesuwa: 61 503 -> 61 502 wiersze testowe. Ten test wykrył to
# dokładnie tak, jak zaprojektowano w Fazie 0. Skutek dla porównywalności jest
# pomijalny — 4 wiersze na 307 511 to 0.0013% zbioru — ale jest odnotowany
# w `docs/findings/02-eda.md`, bo baseline 0.7479 z Fazy 1 zmierzono na
# minimalnie innym zbiorze testowym.
EXPECTED_TEST_SET_SHA256 = (
    "d6fe6da52b849bf449b1eddfe70176c43ff9cc9a8624540884a24e0a5158d895"
)
EXPECTED_TEST_ROWS = 61_502


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
