"""Data contract: `application_train.csv` is what we think it is.

These are contract tests, not statistics tests. They must fail loudly when the
file is swapped or truncated, or when a column changes name or type, because
every later stage derives from these columns.

The numbers below come from the dataset description, not from a measurement on
this machine. The first run with real data confirms or refutes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import ID_COLUMN, TARGET, split_feature_groups
from data import load_data

pytestmark = pytest.mark.requires_data

# The file has 307,511 rows; `load_data` drops 4 with `CODE_GENDER == "XNA"`
# as row hygiene. The contract describes what the rest of the system
# sees, so the count is after hygiene, not before.
EXPECTED_ROWS = 307_507
EXPECTED_COLUMNS = 122
TARGET_RATE_BAND = (0.05, 0.12)
"""Positive class share. A band, not a point: this detects a swapped file or
an inverted label, not a per-mille measurement."""

MANIFEST_PATH = Path(__file__).parent / "fixtures" / "column_manifest.json"
"""Full list of column names, recorded once from the real file and committed.
Without it a column rename would pass unnoticed, since the count would
still match."""

EXPECTED_DTYPE_KINDS: dict[str, str] = {
    ID_COLUMN: "i",
    TARGET: "i",
    "AMT_INCOME_TOTAL": "f",
    "AMT_CREDIT": "f",
    "DAYS_BIRTH": "i",
    "CNT_CHILDREN": "i",
}
"""dtype kind (`numpy.dtype.kind`) for the columns the rest of the project
rests on. Catches an int -> float swap that a column count cannot see.
`kind` is compared, not the exact dtype: width (int32 vs int64) depends on the
platform, not the contract."""

EXPECTED_GROUP_MEMBERSHIP: dict[str, str] = {
    "AMT_INCOME_TOTAL": "numeric",
    "DAYS_BIRTH": "numeric",
    "NAME_CONTRACT_TYPE": "categorical",
    "CODE_GENDER": "categorical",
    "FLAG_OWN_CAR": "categorical",  # holds "Y"/"N", not 0/1
    "FLAG_MOBIL": "binary",
    "FLAG_EMP_PHONE": "binary",
}
"""Specific columns in specific groups. An "every group is non-empty" test
would pass for any CSV with one number, one text column and one flag."""

LEAKAGE_PREFIXES = ("SK_ID_",)
"""Identifier columns. They carry no signal and the model can memorise them."""


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Raw data loaded once per module; the file is ~160 MB."""
    return load_data()


def test_shape_matches_contract(df: pd.DataFrame) -> None:
    """Detects a truncated file and a change in column count."""
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLUMNS)


def test_column_names_match_manifest(df: pd.DataFrame) -> None:
    """Detects a column rename, which the column count cannot see."""
    if not MANIFEST_PATH.exists():
        pytest.fail(
            f"Missing column manifest: {MANIFEST_PATH}\n"
            "Record it once from the real file and commit it:\n"
            "  CS_WRITE_MANIFEST=1 uv run pytest tests/test_data_contract.py"
        )
    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert list(df.columns) == expected


def test_dtypes_match_contract(df: pd.DataFrame) -> None:
    """Pins the column types preprocessing and the API schema rest on."""
    actual = {col: df[col].dtype.kind for col in EXPECTED_DTYPE_KINDS}
    assert actual == EXPECTED_DTYPE_KINDS


def test_coercion_introduced_no_nulls(df: pd.DataFrame) -> None:
    """`load_data` uses `errors="coerce"`; NaN here means junk in the CSV."""
    assert df[ID_COLUMN].notna().all()
    assert df[TARGET].notna().all()


def test_target_column_present_and_binary(df: pd.DataFrame) -> None:
    assert TARGET in df.columns
    assert set(df[TARGET].unique()) == {0, 1}


def test_target_rate_within_plausible_band(df: pd.DataFrame) -> None:
    """The dataset is heavily imbalanced, which drives the whole modelling stage."""
    rate = float(df[TARGET].mean())
    low, high = TARGET_RATE_BAND
    assert low < rate < high, f"positive class share {rate:.4f} outside the band"


def test_id_column_is_unique(df: pd.DataFrame) -> None:
    assert df[ID_COLUMN].is_unique


def test_no_identifier_survives_into_features(df: pd.DataFrame) -> None:
    """Identifiers must not reach any feature group."""
    features = split_feature_groups(df).all_features
    leaked = [c for c in features if c.startswith(LEAKAGE_PREFIXES)]
    assert not leaked, f"identifiers among the features: {leaked}"


def test_known_columns_land_in_expected_groups(df: pd.DataFrame) -> None:
    """Specific columns must land in specific groups."""
    groups = split_feature_groups(df)
    lookup = {
        **dict.fromkeys(groups.numeric, "numeric"),
        **dict.fromkeys(groups.categorical, "categorical"),
        **dict.fromkeys(groups.binary, "binary"),
    }
    actual = {col: lookup.get(col) for col in EXPECTED_GROUP_MEMBERSHIP}
    assert actual == EXPECTED_GROUP_MEMBERSHIP


def test_groups_cover_every_column_but_target_and_id(df: pd.DataFrame) -> None:
    """The groups plus target plus id must be exactly the file's columns."""
    groups = split_feature_groups(df)
    assert set(groups.all_features) | {TARGET, ID_COLUMN} == set(df.columns)


# --- Row hygiene ---
@pytest.mark.requires_data
def test_loader_applies_hygiene(df: pd.DataFrame) -> None:
    assert df["TARGET"].notna().all()
    assert "XNA" not in set(df["CODE_GENDER"].unique())
