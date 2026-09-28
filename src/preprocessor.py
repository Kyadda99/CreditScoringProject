"""Składanie potoku przetwarzania danych.

`FeatureEngineer` jest pierwszym krokiem, więc derywacja cech jedzie razem
z artefaktem, a `/score` przyjmuje surowe kolumny. Sampler, gdy jest, stoi za
preprocessorem: SMOTE potrzebuje macierzy numerycznej.
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
    """Buduje potok przekształceń dla ustalonych grup cech.

    Attributes:
        groups: Grupy cech wyprowadzone na ramce treningowej po inżynierii.
        features: Nazwy cech pochodnych do policzenia.
    """

    def __init__(
        self,
        groups: FeatureGroups,
        features: tuple[str, ...] = ENGINEERED_FEATURES,
    ) -> None:
        """Zapamiętuje kontrakt kolumn; niczego nie dopasowuje.

        Args:
            groups: Grupy cech modelu.
            features: Cechy pochodne; pusta krotka wyłącza derywację.
        """
        self.groups = groups
        self.features = features

    def engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Zwraca ramkę z doliczonymi cechami pochodnymi.

        Args:
            frame: Surowe kolumny.

        Returns:
            Nowa ramka; wejście pozostaje nietknięte.
        """
        return FeatureEngineer(features=self.features).fit_transform(frame)

    def build_transformer(self) -> ColumnTransformer:
        """Zwraca niedopasowany `ColumnTransformer` dla grup cech."""
        return build_preprocessor(self.groups)

    def build_pipeline(
        self,
        estimator: BaseEstimator | None = None,
        sampler: BaseEstimator | None = None,
    ) -> Pipeline:
        """Składa pełny potok: cechy, preprocessing, opcjonalny sampler, model.

        Args:
            estimator: Estymator do wpięcia; `None` daje regresję logistyczną.
            sampler: Sampler `imblearn` albo `None`.

        Returns:
            Niedopasowany potok — `sklearn` bez samplera, `imblearn` z samplerem.
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
