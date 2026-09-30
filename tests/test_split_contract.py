"""Train/test split guard.

The split is computed from RANDOM_STATE rather than storing a list of ids.
`train_test_split` is however a function of the frame it receives: if a row is
removed, the split shifts silently and model comparisons stop being
comparisons. This test turns that silent drift into a loud failure.

When a split change is intentional: run the test, copy the printed checksum
into EXPECTED_TEST_SET_SHA256, and record why.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from config import ID_COLUMN
from data import load_data
from data_loader import DataLoader

pytestmark = pytest.mark.requires_data

# INTENTIONAL CHANGE. Row hygiene removes 4 records with
# `CODE_GENDER == "XNA"`, so `train_test_split` receives a different frame
# and the split shifts: 61,503 -> 61,502 test rows. This test caught it
# exactly as designed. The effect on comparability is negligible, 4 rows in
# 307,511 or 0.0013%, but the earlier 0.7479 baseline was measured on the
# previous split.
EXPECTED_TEST_SET_SHA256 = (
    "d6fe6da52b849bf449b1eddfe70176c43ff9cc9a8624540884a24e0a5158d895"
)
EXPECTED_TEST_ROWS = 61_502


def _checksum(ids: pd.Series) -> str:
    """SHA-256 over the sorted identifiers, joined by commas.

    Not `hash()`: Python's built-in hash is salted per process
    (PYTHONHASHSEED), so it would differ on every run.
    """
    joined = ",".join(str(i) for i in sorted(ids.tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def test_test_split_has_not_drifted() -> None:
    df = load_data()
    _X_train, X_test, _y_train, _y_test = DataLoader().split(df)
    actual = _checksum(df.loc[X_test.index, ID_COLUMN])
    assert actual == EXPECTED_TEST_SET_SHA256, (
        f"The test split changed. New checksum: {actual}\n"
        "If this is intentional, write it into EXPECTED_TEST_SET_SHA256 "
        "and record why."
    )


def test_test_split_size_is_stable() -> None:
    df = load_data()
    _X_train, X_test, _y_train, _y_test = DataLoader().split(df)
    assert len(X_test) == EXPECTED_TEST_ROWS


def test_stratification_preserves_the_target_rate() -> None:
    df = load_data()
    _X_train, _X_test, y_train, y_test = DataLoader().split(df)
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.005
