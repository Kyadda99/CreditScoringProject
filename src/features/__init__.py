"""Feature engineering: pure functions plus the transformer that applies them.

The transformer is the first `Pipeline` step, before the `ColumnTransformer`,
so all derivation travels in one joblib artifact: `/score` accepts raw columns
and the container computes the features itself. There is no second call site
that could drift from training.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import ENGINEERED_FEATURES
from features.cleaning import days_employed_cleaned, not_employed_flag
from features.ratios import RATIO_FUNCTIONS

logger = logging.getLogger(__name__)

FLAG_NOT_EMPLOYED = "FLAG_NOT_EMPLOYED"
"""Flag added on every transform, including when no ratios are computed."""

_REQUIRED_AT_FIT: tuple[str, ...] = (
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
)
"""Columns without which the training frame is incomplete.

At `transform` a missing column is allowed: `/score` fields are optional and
the imputer handles them. At `fit` it is a pipeline fault and must be loud.
"""


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived features and decodes sentinels. No learned state.

    The step is deliberately stateless: anything learned from the data would
    have to be learned on the training fold alone, and `Pipeline` would have to
    police that. With no state there is nothing to police.

    Attributes:
        features: Names of the derived features to compute. An empty tuple
            means cleaning only.
    """

    def __init__(self, features: tuple[str, ...] = ENGINEERED_FEATURES) -> None:
        """Store the choice of derived features.

        Args:
            features: Names from `RATIO_FUNCTIONS` to compute.
        """
        # No normalisation or validation: sklearn.base.clone rebuilds the
        # object from get_params() and compares attributes against __init__.
        self.features = features

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        """Record the input columns and check the frame is complete.

        A training frame missing a source column is a pipeline fault, unlike
        `transform`, where an absent field is a normal request state.

        Args:
            X: Training frame.
            y: Ignored; required by the sklearn signature.

        Returns:
            `self`.

        Raises:
            ValueError: When the training frame lacks a source column.
        """
        missing = [c for c in _REQUIRED_AT_FIT if c not in X.columns]
        if missing:
            raise ValueError(
                "Training frame is missing source columns: " + ", ".join(missing)
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with the sentinel decoded and features added.

        Never raises on request data: a missing column yields a NaN feature for
        the fitted imputer, because `/score` fields are optional.

        Args:
            X: Frame of raw columns.

        Returns:
            A new frame: the input columns with `DAYS_EMPLOYED` decoded, then
            `FLAG_NOT_EMPLOYED`, then `self.features` in order.
        """
        frame = X.copy()
        if "DAYS_EMPLOYED" in frame.columns:
            frame["DAYS_EMPLOYED"] = days_employed_cleaned(X)
        frame[FLAG_NOT_EMPLOYED] = not_employed_flag(X)
        for name in self.features:
            frame[name] = RATIO_FUNCTIONS[name](X)
        return frame

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Column names `transform` produces.

        Args:
            input_features: Ignored; the names come from `fit`.

        Returns:
            An array of names in output order.
        """
        base = list(self.feature_names_in_)
        return np.asarray([*base, FLAG_NOT_EMPLOYED, *self.features], dtype=object)
