"""Data processing pipeline assembly.

`FeatureEngineer` is the first step, so feature derivation travels with the
artifact and `/score` accepts raw columns. A sampler, when present, sits after
the preprocessor because SMOTE needs a numeric matrix.
"""

from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import ENGINEERED_FEATURES, FeatureGroups
from features import FeatureEngineer
from preprocessing import build_preprocessor
from strategies import get_estimator_strategy


class Preprocessor:
    """Builds the transformation pipeline for fixed feature groups.

    Attributes:
        groups: Feature groups derived on the training frame after engineering.
        features: Names of the derived features to compute.
    """

    def __init__(
        self,
        groups: FeatureGroups,
        features: tuple[str, ...] = ENGINEERED_FEATURES,
    ) -> None:
        """Store the column contract; fit nothing.

        Args:
            groups: Model feature groups.
            features: Derived features; an empty tuple disables derivation.
        """
        self.groups = groups
        self.features = features

    def engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with derived features added.

        Args:
            frame: Raw columns.

        Returns:
            A new frame; the input is left untouched.
        """
        return FeatureEngineer(features=self.features).fit_transform(frame)

    def build_transformer(self) -> ColumnTransformer:
        """Return an unfitted `ColumnTransformer` for the feature groups."""
        return build_preprocessor(self.groups)

    def build_pipeline(
        self,
        estimator: BaseEstimator | None = None,
        sampler: BaseEstimator | None = None,
    ) -> Pipeline:
        """Assemble the full pipeline: features, preprocessing, sampler, model.

        Args:
            estimator: Estimator to plug in; `None` gives logistic regression.
            sampler: An `imblearn` sampler, or `None`.

        Returns:
            An unfitted pipeline: `sklearn` without a sampler, `imblearn` with.
        """
        if estimator is None:
            estimator = get_estimator_strategy("logistic_regression").build(
                class_weighted=False, scale_pos_weight=1.0
            )
        steps: list[tuple[str, BaseEstimator]] = [
            ("features", FeatureEngineer(features=self.features)),
            ("preprocessor", self.build_transformer()),
        ]
        if sampler is None:
            return Pipeline([*steps, ("model", estimator)])
        return ImbPipeline([*steps, ("sampler", sampler), ("model", estimator)])
