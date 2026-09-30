"""Data entry: loading, splitting and both column contracts.

Feature groups are derived from the training set only. Computed on the full
frame, the test set could influence which column counts as a binary flag.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    DROPPED_COLUMNS,
    ENGINEERED_FEATURES,
    FEATURE_SOURCE_COLUMNS,
    ID_COLUMN,
    RANDOM_STATE,
    TARGET,
    FeatureGroups,
    split_feature_groups,
)
from data import load_data
from features import FeatureEngineer
from settings import TrainingSettings

TEST_SIZE = 0.2


class DataLoader:
    """Supplies the training data and the column contracts.

    Attributes:
        path: Path to the data file.
        test_size: Test set fraction.
        random_state: Split seed.
    """

    def __init__(
        self,
        path: Path = DATA_PATH,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Store the loading configuration.

        Args:
            path: Path to the CSV file.
            test_size: Test set fraction.
            random_state: Split seed.
        """
        self.path = path
        self.test_size = test_size
        self.random_state = random_state

    @classmethod
    def from_settings(cls, settings: TrainingSettings) -> DataLoader:
        """Build a loader from the configuration.

        Args:
            settings: Training settings.

        Returns:
            A loader pointing at the configured dataset and split.
        """
        return cls(
            path=settings.data_path,
            test_size=settings.test_size,
        )

    def load(self) -> pd.DataFrame:
        """Load and clean the data.

        Returns:
            A frame including the target and id columns.
        """
        return load_data(self.path)

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the frame into train and test, stratified.

        Args:
            df: Full frame with the target and id columns.

        Returns:
            `(X_train, X_test, y_train, y_test)` without target or id.
        """
        y = df[TARGET]
        X = df.drop(columns=[TARGET, ID_COLUMN])
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

    def model_groups(
        self,
        X_train: pd.DataFrame,
        features: tuple[str, ...] = ENGINEERED_FEATURES,
    ) -> FeatureGroups:
        """Return the model feature groups, computed after engineering.

        Args:
            X_train: Raw training frame.
            features: Derived features.

        Returns:
            The feature groups that drive the `ColumnTransformer`.
        """
        engineered = FeatureEngineer(features=features).fit_transform(X_train)
        return split_feature_groups(engineered, dropped=DROPPED_COLUMNS)

    def input_contract(
        self,
        X_train: pd.DataFrame,
        dropped: tuple[str, ...] = DROPPED_COLUMNS,
    ) -> FeatureGroups:
        """Return the `/score` input contract: raw columns only.

        A column is accepted if it becomes a model feature or feeds a derived
        one. Without the second condition a ratio using a dropped column would
        always be NaN.

        Args:
            X_train: Raw training frame.
            dropped: Columns removed by the data analysis decisions.

        Returns:
            Groups of raw columns.
        """
        kept = set(X_train.columns) - set(dropped) | set(FEATURE_SOURCE_COLUMNS)
        accepted = [c for c in X_train.columns if c in kept]
        return split_feature_groups(X_train[accepted])
