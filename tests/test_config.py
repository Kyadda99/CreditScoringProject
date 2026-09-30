"""Tests for splitting columns into feature groups, without real data."""

from __future__ import annotations

import pandas as pd
import pytest

from config import (
    DAYS_EMPLOYED_SENTINEL,
    DROPPED_COLUMNS,
    ENGINEERED_FEATURES,
    FEATURE_SOURCE_COLUMNS,
    ID_COLUMN,
    INFORMATIVE_MISSING,
    TARGET,
    FeatureGroups,
    split_feature_groups,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """A miniature frame mirroring the Home Credit column types.

    `FLAG_OWN_CAR` deliberately holds "Y"/"N", as it does in the real dataset,
    despite the `FLAG_` prefix. `FLAG_MOBIL` is the numeric flag.
    """
    return pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4],
            TARGET: [0, 1, 0, 0],
            "AMT_INCOME_TOTAL": [1000.0, 2000.0, 3000.0, 4000.0],
            "CNT_CHILDREN": [0, 1, 2, 3],
            "NAME_CONTRACT_TYPE": ["Cash", "Revolving", "Cash", "Cash"],
            "FLAG_OWN_CAR": ["Y", "N", "Y", "N"],
            "FLAG_MOBIL": [1, 1, 0, 1],
        }
    )


def test_target_and_id_are_not_features(frame: pd.DataFrame) -> None:
    groups = split_feature_groups(frame)
    assert TARGET not in groups.all_features
    assert ID_COLUMN not in groups.all_features


def test_columns_are_grouped_by_kind(frame: pd.DataFrame) -> None:
    groups = split_feature_groups(frame)
    assert groups.numeric == ("AMT_INCOME_TOTAL", "CNT_CHILDREN")
    assert groups.categorical == ("FLAG_OWN_CAR", "NAME_CONTRACT_TYPE")
    assert groups.binary == ("FLAG_MOBIL",)


def test_text_columns_never_land_in_numeric(frame: pd.DataFrame) -> None:
    """Regression: pandas 3.0 gives text a `str` dtype, not `object`.

    A `dtype == "object"` test would let a text column into the numeric group,
    which would only fail later, at the encoder.
    """
    groups = split_feature_groups(frame)
    assert "NAME_CONTRACT_TYPE" not in groups.numeric
    assert "FLAG_OWN_CAR" not in groups.numeric


def test_constant_and_empty_columns_are_not_flags() -> None:
    """Regression: `nunique() <= 2` treated empty and constant columns as flags."""
    df = pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3],
            TARGET: [0, 1, 0],
            "ALL_NAN": [float("nan")] * 3,
            "CONST_FIVE": [5.0, 5.0, 5.0],
        }
    )
    groups = split_feature_groups(df)
    assert groups.binary == ()
    assert groups.numeric == ("ALL_NAN", "CONST_FIVE")


def test_every_feature_lands_in_exactly_one_group(frame: pd.DataFrame) -> None:
    groups = split_feature_groups(frame)
    expected = set(frame.columns) - {TARGET, ID_COLUMN}
    assert set(groups.all_features) == expected
    assert len(groups.all_features) == len(expected), "column landed in two groups"


def test_groups_are_immutable() -> None:
    groups = FeatureGroups(numeric=("a",), categorical=(), binary=())
    with pytest.raises((AttributeError, TypeError)):
        groups.numeric = ("b",)  # type: ignore[misc]


def test_groups_round_trip_through_dict(frame: pd.DataFrame) -> None:
    """Groups must survive being stored with the model and rebuilt at serving."""
    groups = split_feature_groups(frame)
    assert FeatureGroups.from_dict(groups.to_dict()) == groups


def test_derivation_is_sample_dependent_so_groups_must_be_frozen(
    frame: pd.DataFrame,
) -> None:
    """Documents why the groups are stored with the model.

    On a one-row frame, exactly what `/score` receives, `CNT_CHILDREN == 0` looks
    like a 0/1 flag. Serving must therefore rebuild the groups with `from_dict()`
    rather than derive them from the request data.
    """
    train_groups = split_feature_groups(frame)
    request_groups = split_feature_groups(frame.head(1))

    assert "CNT_CHILDREN" in train_groups.numeric
    assert "CNT_CHILDREN" in request_groups.binary
    assert train_groups != request_groups


# --- EDA decisions ---


def test_dropped_columns_are_unique() -> None:
    assert len(DROPPED_COLUMNS) == len(set(DROPPED_COLUMNS))


def test_informative_missing_is_not_dropped() -> None:
    """A column cannot be both dropped and given a missingness indicator."""
    assert not set(INFORMATIVE_MISSING) & set(DROPPED_COLUMNS)


def test_feature_sources_are_not_dropped_from_the_input_contract() -> None:
    """A column feeding a derived feature must stay accepted by /score."""
    from api.schemas import FEATURE_GROUPS

    accepted = set(FEATURE_GROUPS.all_features)
    assert set(FEATURE_SOURCE_COLUMNS) <= accepted


def test_split_feature_groups_honours_dropped() -> None:
    frame = pd.DataFrame({"keep": [1.0, 2.0], "bin": [0.0, 1.0], "junk": [5.0, 6.0]})
    groups = split_feature_groups(frame, dropped=("junk",))
    assert "junk" not in groups.all_features
    assert groups.numeric == ("keep",)
    assert groups.binary == ("bin",)


def test_split_feature_groups_default_drops_nothing() -> None:
    """The default call drops nothing."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert set(split_feature_groups(frame).all_features) == {"a", "b"}


def test_sentinel_value_is_the_documented_one() -> None:
    assert DAYS_EMPLOYED_SENTINEL == 365243


def test_engineered_features_are_not_raw_columns() -> None:
    """Derived feature names must not collide with source columns."""
    assert not set(ENGINEERED_FEATURES) & set(FEATURE_SOURCE_COLUMNS)
