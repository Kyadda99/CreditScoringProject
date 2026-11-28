"""Contract tests on the real Lending Club descriptions.

Skipped unless `uv run cs-nn-download-text` has run. The recorded numbers come
from the downloaded file and the cleaning rules, so a change in either shows
up here. When a change is intentional: copy the new values or the printed
checksum into the constants below, and record why.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from config import DESCRIPTIONS_PATH
from nn.text_data import load_descriptions, split_frame

RECORDED_CLASSES = {
    "debt_consolidation": 64351,
    "credit_card": 23291,
    "home_improvement": 7168,
    "other": 7116,
    "major_purchase": 3145,
    "small_business": 2799,
    "car": 1955,
    "wedding": 1252,
    "medical": 1016,
    "house": 839,
    "moving": 820,
    "vacation": 553,
}
RECORDED_SPLIT_SIZES = {"train": 80013, "val": 17146, "test": 17146}
EXPECTED_TEST_SET_SHA256 = (
    "48256c8fbcbae9110ecbcd7ed5e708d0c022c9f33b0f564b747bbbd00a470b05"
)

pytestmark = pytest.mark.requires_data(DESCRIPTIONS_PATH)


@pytest.fixture(scope="module")
def descriptions() -> pd.DataFrame:
    return load_descriptions(DESCRIPTIONS_PATH)


@pytest.fixture(scope="module")
def splits(descriptions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return split_frame(descriptions)


def test_descriptions_have_the_recorded_class_distribution(
    descriptions: pd.DataFrame,
) -> None:
    assert descriptions["label"].value_counts().to_dict() == RECORDED_CLASSES


def test_descriptions_have_no_nulls_empties_or_duplicates(
    descriptions: pd.DataFrame,
) -> None:
    assert descriptions.notna().all().all()
    assert (descriptions["text"].str.len() > 0).all()
    assert not descriptions["text"].duplicated().any()


def test_descriptions_carry_no_dated_prefix_or_markup(
    descriptions: pd.DataFrame,
) -> None:
    text = descriptions["text"]
    assert not text.str.contains(r"added on \d{1,2}/\d{1,2}/\d{2,4}").any()
    assert not text.str.contains("<br", regex=False).any()


def test_splits_have_the_recorded_sizes(splits: dict[str, pd.DataFrame]) -> None:
    assert {name: len(part) for name, part in splits.items()} == RECORDED_SPLIT_SIZES


def test_splits_share_no_text(splits: dict[str, pd.DataFrame]) -> None:
    train, val, test = (set(part["text"]) for part in splits.values())
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_splits_keep_every_class(splits: dict[str, pd.DataFrame]) -> None:
    for part in splits.values():
        assert set(part["label"]) == set(RECORDED_CLASSES)


def test_test_split_has_not_drifted(splits: dict[str, pd.DataFrame]) -> None:
    joined = "\n".join(sorted(splits["test"]["text"]))
    actual = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    assert actual == EXPECTED_TEST_SET_SHA256, (
        f"The text test split changed. New checksum: {actual}\n"
        "If this is intentional, write it into EXPECTED_TEST_SET_SHA256 "
        "and record why."
    )
