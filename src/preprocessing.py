"""Preprocessor assembly: builds, never fits.

Each feature group gets its own pipeline. The returned `ColumnTransformer` is
unfitted; fitting happens inside a single `Pipeline` with the model so the
imputer cannot be learned on the test set.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import INFORMATIVE_MISSING, FeatureGroups


def build_preprocessor(
    groups: FeatureGroups,
    informative_missing: tuple[str, ...] = INFORMATIVE_MISSING,
) -> ColumnTransformer:
    """Assemble a preprocessor for the given column groups.

    Args:
        groups: Feature groups derived once on the training frame.
        informative_missing: Numeric columns whose missingness carries signal.
            They get the same pipeline plus `add_indicator=True`. Names outside
            `groups.numeric` are ignored, so a dropped column cannot return.

    Returns:
        An unfitted `ColumnTransformer`. Columns outside the groups are dropped
        (`remainder="drop"`), keeping the target and id out of the model.
    """
    informative = [c for c in groups.numeric if c in set(informative_missing)]
    plain_numeric = [c for c in groups.numeric if c not in set(informative)]

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # Same imputation plus a 0/1 column marking where the value was missing.
    # The indicator goes through the same StandardScaler as the value.
    numeric_informative = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": production will see categories
            # absent from training; without this the encoder raises and
            # /score returns 500.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    # 0/1 flags need imputation only; scaling would obscure regression
    # coefficients without adding anything.
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", numeric, plain_numeric),
        ("categorical", categorical, list(groups.categorical)),
        ("binary", binary, list(groups.binary)),
    ]
    # Added conditionally: ColumnTransformer with an empty column list is
    # unreliable in get_feature_names_out.
    if informative:
        transformers.insert(
            1, ("numeric_informative", numeric_informative, informative)
        )

    return ColumnTransformer(transformers, remainder="drop")
