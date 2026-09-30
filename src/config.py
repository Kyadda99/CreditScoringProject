"""Project constants and the column contract.

The module imports without any data on disk. Feature groups are derived from
dtypes rather than hard-coded, so they cannot drift from the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
DATA_PATH: Path = RAW_DATA_DIR / "application_train.csv"

# --- Column contract -------------------------------------------------------

TARGET: str = "TARGET"
"""Target: 1 = the client had repayment difficulty, 0 = repaid on time."""

ID_COLUMN: str = "SK_ID_CURR"
"""Application id. Never a feature; unique per row."""

RANDOM_STATE: int = 42
"""One seed for the whole project: split, samplers, models."""

BINARY_VALUES: frozenset[float] = frozenset({0.0, 1.0})
"""Values a numeric column may hold to count as a 0/1 flag."""

# --- Cost assumption -------------------------------------------------------
#
# Assumed ratio, declared before any threshold is computed, not measured.
# A missed default forfeits unpaid principal; a false alarm forfeits the margin
# on one rejected application.

COST_FN: float = 10.0
"""Cost of a missed default (false negative)."""

COST_FP: float = 1.0
"""Cost of a false alarm (false positive)."""

# --- EDA decisions ---------------------------------------------------------
#
# Dropping a column needs no code: ColumnTransformer(remainder="drop") discards
# anything absent from the groups. These tuples exist so the choice is reviewable.

DAYS_EMPLOYED_SENTINEL: int = 365243
"""~1000 years of employment: how the dataset encodes "not employed".

Present in 18.01% of rows, where the default rate is 5.40% against 8.66%
elsewhere. The difference is real, so it is kept as `FLAG_NOT_EMPLOYED`.
"""

DROPPED_COLUMNS: tuple[str, ...] = (
    # 1. Housing block: _AVG / _MEDI / _MODE encode the same quantity three
    #    times (r = 0.96-0.998 within each triple). _MEDI stays. 28 columns.
    "APARTMENTS_AVG",
    "APARTMENTS_MODE",
    "BASEMENTAREA_AVG",
    "BASEMENTAREA_MODE",
    "COMMONAREA_AVG",
    "COMMONAREA_MODE",
    "ELEVATORS_AVG",
    "ELEVATORS_MODE",
    "ENTRANCES_AVG",
    "ENTRANCES_MODE",
    "FLOORSMAX_AVG",
    "FLOORSMAX_MODE",
    "FLOORSMIN_AVG",
    "FLOORSMIN_MODE",
    "LANDAREA_AVG",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_AVG",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_AVG",
    "NONLIVINGAREA_MODE",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_AVG",
    "YEARS_BUILD_MODE",
    # 2. Document flags whose dominant value covers >= 99% of rows.
    #    |r| with the target <= 0.012. 16 columns.
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_9",
    # 3. Sentinel duplicate. FLAG_EMP_PHONE == 0 for all 55,374 rows with
    #    DAYS_EMPLOYED == 365243 and only 12 others (r = 0.9998).
    "FLAG_EMP_PHONE",
    # 4. Near-identical pairs.
    "OBS_60_CNT_SOCIAL_CIRCLE",  # r = 0.9985 with OBS_30; OBS_30 stays
    "REGION_RATING_CLIENT",  # r = 0.9508 with _W_CITY, which has more signal
)
"""Columns excluded from the model."""

INFORMATIVE_MISSING: tuple[str, ...] = (
    "AMT_REQ_CREDIT_BUREAU_YEAR",  # missing = no credit bureau history
    "EXT_SOURCE_1",  # 56.4% missing, strongest feature family (r = -0.155)
    "EXT_SOURCE_3",  # 19.8% missing (r = -0.179)
    "OWN_CAR_AGE",  # not applicable: empty for 202,924 of 202,929 without a car
    "TOTALAREA_MODE",  # represents the housing block (48.3% missing)
)
"""Subset that gets an explicit missingness indicator.

Five columns, not all 67 with missing values. The housing block shares one
missingness pattern, so an indicator per `_MEDI` column would produce 14
collinear columns; `TOTALAREA_MODE` represents it.
"""

FEATURE_SOURCE_COLUMNS: tuple[str, ...] = (
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)
"""Raw columns that feed the derived features.

Input contract membership rule: `/score` accepts a column if it either becomes
a model feature or feeds a derived one. A column dropped from the model but
used in a ratio must still be accepted, or the ratio is always NaN.
"""

ENGINEERED_FEATURES: tuple[str, ...] = (
    "ANNUITY_INCOME_RATIO",
    "BUREAU_INQUIRY_TOTAL",
    "CREDIT_INCOME_RATIO",
    "CREDIT_TERM",
    "EMPLOYED_AGE_RATIO",
    "EXT_SOURCE_COUNT",
    "EXT_SOURCE_MEAN",
    "GOODS_CREDIT_RATIO",
    "INCOME_PER_PERSON",
)
"""Features computed by `FeatureEngineer`. This order sets column order."""


@dataclass(frozen=True)
class FeatureGroups:
    """Feature columns split into groups needing different preprocessing.

    The split drives `build_preprocessor()`: each group gets its own
    imputation, scaling and encoding pipeline.

    Attributes:
        numeric: Continuous and discrete numbers; median imputation, scaling.
        categorical: Non-numeric columns, including `FLAG_OWN_CAR`, which holds
            "Y"/"N" rather than 0/1; mode imputation, one-hot.
        binary: Numeric 0/1 flags; imputation, no scaling.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    binary: tuple[str, ...]

    @property
    def all_features(self) -> tuple[str, ...]:
        """All feature columns, in group order."""
        return self.numeric + self.categorical + self.binary

    def to_dict(self) -> dict[str, list[str]]:
        """Serialise the groups for storage beside the model."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "binary": list(self.binary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> FeatureGroups:
        """Rebuild the groups stored at training time.

        Args:
            payload: Dictionary from `to_dict()`.

        Returns:
            The same groups, in the same order.
        """
        return cls(
            numeric=tuple(payload["numeric"]),
            categorical=tuple(payload["categorical"]),
            binary=tuple(payload["binary"]),
        )


def split_feature_groups(
    df: pd.DataFrame, dropped: tuple[str, ...] = ()
) -> FeatureGroups:
    """Derive feature groups from the training frame.

    The result is a function of the sample passed in, not of a schema: a
    numeric column lands in `binary` based on the values actually present, so
    on a one-row frame `CNT_CHILDREN == 0` would look like a flag. Groups are
    therefore derived once, on the full training set, and stored with the model
    via `to_dict()`. Serving rebuilds them with `from_dict()` and never calls
    this function on request data, which would cause train/serve skew.

    The target and id columns are excluded; they are not features.

    Args:
        df: Full training frame loaded by `data.load_data`.
        dropped: Columns excluded by the EDA decisions
            (`config.DROPPED_COLUMNS`). Defaults to an empty tuple.

    Returns:
        Feature groups; each list sorted alphabetically so column order is
        stable across runs.
    """
    excluded = {TARGET, ID_COLUMN, *dropped}
    feature_cols = [c for c in df.columns if c not in excluded]

    numeric: list[str] = []
    categorical: list[str] = []
    binary: list[str] = []

    for col in feature_cols:
        series = df[col]
        # Branch on "is numeric", not dtype == "object": pandas 3.0 infers a
        # dedicated str dtype, so an object test would let text into numeric.
        if pd.api.types.is_bool_dtype(series):
            binary.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Test the values, not nunique() <= 2: counting uniques also
            # admits empty (0) and constant (1) columns as flags.
            values = set(series.dropna().unique())
            if values and values <= BINARY_VALUES:
                binary.append(col)
            else:
                numeric.append(col)
        else:
            categorical.append(col)

    return FeatureGroups(
        numeric=tuple(sorted(numeric)),
        categorical=tuple(sorted(categorical)),
        binary=tuple(sorted(binary)),
    )


# MLflow 3.x rejects the file backend ("./mlruns"): set_experiment raises
# unless MLFLOW_ALLOW_FILE_STORE is set. SQLite is used instead; the model
# registry and the @production alias require a database backend anyway.
# as_posix(): SQLAlchemy expects forward slashes, not Windows backslashes.
MLFLOW_TRACKING_URI: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
"""Shared by `train`, `registry` and `tune`.

Defined here rather than in `train.py` because `registry.py` needs it too and
importing from `train` would create a cycle.
"""
