"""Wejście danych: wczytanie, podział i oba kontrakty kolumn.

Grupy cech wyprowadzamy wyłącznie ze zbioru treningowego. Policzone na pełnej
ramce pozwoliłyby zbiorowi testowemu wpłynąć na to, która kolumna uchodzi za
flagę binarną.
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
    """Dostarcza dane treningowe i kontrakty kolumn.

    Attributes:
        path: Ścieżka do pliku z danymi.
        test_size: Udział zbioru testowego.
        random_state: Ziarno podziału.
    """

    def __init__(
        self,
        path: Path = DATA_PATH,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Zapamiętuje konfigurację wczytywania.

        Args:
            path: Ścieżka do pliku CSV.
            test_size: Udział zbioru testowego.
            random_state: Ziarno podziału.
        """
        self.path = path
        self.test_size = test_size
        self.random_state = random_state

    @classmethod
    def from_settings(cls, settings: TrainingSettings) -> DataLoader:
        """Buduje loader z konfiguracji.

        Args:
            settings: Ustawienia treningu.

        Returns:
            Loader wskazujący na skonfigurowany zbiór i podział.
        """
        return cls(
            path=settings.data_path,
            test_size=settings.test_size,
        )

    def load(self) -> pd.DataFrame:
        """Wczytuje i czyści dane.

        Returns:
            Ramka z kolumną celu i identyfikatorem.
        """
        return load_data(self.path)

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Dzieli ramkę na train/test ze stratyfikacją.

        Args:
            df: Pełna ramka z celem i identyfikatorem.

        Returns:
            `(X_train, X_test, y_train, y_test)` bez celu i identyfikatora.
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
        """Zwraca grupy cech modelu, liczone po inżynierii cech.

        Args:
            X_train: Surowa ramka treningowa.
            features: Cechy pochodne.

        Returns:
            Grupy cech sterujące `ColumnTransformer`.
        """
        engineered = FeatureEngineer(features=features).fit_transform(X_train)
        return split_feature_groups(engineered, dropped=DROPPED_COLUMNS)

    def input_contract(
        self,
        X_train: pd.DataFrame,
        dropped: tuple[str, ...] = DROPPED_COLUMNS,
    ) -> FeatureGroups:
        """Zwraca kontrakt wejściowy `/score` — same surowe kolumny.

        Kolumna jest przyjmowana, jeśli zostaje cechą modelu **albo** karmi
        cechę pochodną. Bez drugiego warunku iloraz z usuniętej kolumny zawsze
        wychodziłby NaN.

        Args:
            X_train: Surowa ramka treningowa.
            dropped: Kolumny usunięte decyzją z analizy danych.

        Returns:
            Grupy surowych kolumn.
        """
        kept = set(X_train.columns) - set(dropped) | set(FEATURE_SOURCE_COLUMNS)
        accepted = [c for c in X_train.columns if c in kept]
        return split_feature_groups(X_train[accepted])
