"""Inżynieria cech: czyste funkcje plus transformer, który je składa.

Transformer jest **pierwszym krokiem** `Pipeline` (spec D1), przed
`ColumnTransformer`. Dzięki temu cała derywacja jedzie w jednym artefakcie
joblib: `/score` przyjmuje surowe kolumny, a kontener liczy cechy sam. Nie ma
drugiego miejsca wywołania, które mogłoby się rozjechać z treningiem — a to
właśnie ten błąd Faza 1 usuwała ustaleniem M4.
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
"""Nazwa flagi dodawanej zawsze — także w transzy bez ilorazów (spec D7)."""

_REQUIRED_AT_FIT: tuple[str, ...] = (
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
)
"""Kolumny, bez których ramka TRENINGOWA jest niekompletna.

Przy `transform` brak kolumny jest dozwolony — pola `/score` są opcjonalne
i imputer je obsłuży. Przy `fit` to usterka pipeline'u i musi być głośna.
"""


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Dokłada cechy pochodne i dekoduje sentinele. Bez stanu uczonego.

    Krok jest bezstanowy z rozmysłem: gdyby liczył cokolwiek na danych (średnie,
    progi), musiałby to robić wyłącznie na foldzie treningowym, a `Pipeline`
    i tak by tego pilnował. Tu nie ma czego pilnować — te same wejścia dają ten
    sam wynik w treningu i w serwowaniu.

    Attributes:
        features: Nazwy cech pochodnych do policzenia. Pusta krotka oznacza
            "tylko czyszczenie" — tak realizowana jest transza 1 ze spec D7.
    """

    def __init__(self, features: tuple[str, ...] = ENGINEERED_FEATURES) -> None:
        """Zapamiętuje wybór cech pochodnych.

        Args:
            features: Nazwy cech z `RATIO_FUNCTIONS` do policzenia.
        """
        # Bez normalizacji i bez walidacji: `sklearn.base.clone` odtwarza obiekt
        # z `get_params()` i porównuje atrybuty z tym, co dostał __init__.
        self.features = features

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        """Zapamiętuje kolumny wejściowe i sprawdza kompletność ramki.

        Ramka treningowa bez kolumny źródłowej to usterka pipeline'u — inaczej
        niż przy `transform`, gdzie brak pola jest normalnym stanem żądania.

        Args:
            X: Ramka treningowa.
            y: Ignorowane; sygnatura wymagana przez sklearn.

        Returns:
            `self`.

        Raises:
            ValueError: Gdy ramce treningowej brakuje kolumny źródłowej.
        """
        missing = [c for c in _REQUIRED_AT_FIT if c not in X.columns]
        if missing:
            raise ValueError(
                "Ramka treningowa nie ma kolumn źródłowych: " + ", ".join(missing)
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Zwraca kopię ramki z odkodowanym sentinelem i dołożonymi cechami.

        Nigdy nie rzuca na danych żądania: brak kolumny daje cechę NaN, którą
        dopasowany imputer obsłuży — pola `/score` są opcjonalne (Faza 1, D1).

        Args:
            X: Ramka surowych kolumn.

        Returns:
            Nowa ramka: kolumny wejściowe (z `DAYS_EMPLOYED` po dekodowaniu),
            `FLAG_NOT_EMPLOYED`, a potem cechy z `self.features` w podanej
            kolejności.
        """
        frame = X.copy()
        if "DAYS_EMPLOYED" in frame.columns:
            frame["DAYS_EMPLOYED"] = days_employed_cleaned(X)
        frame[FLAG_NOT_EMPLOYED] = not_employed_flag(X)
        for name in self.features:
            frame[name] = RATIO_FUNCTIONS[name](X)
        return frame

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Nazwy kolumn, jakie produkuje `transform`.

        Args:
            input_features: Ignorowane; nazwy pochodzą z `fit`.

        Returns:
            Tablica nazw w kolejności wyjściowej.
        """
        base = list(self.feature_names_in_)
        return np.asarray([*base, FLAG_NOT_EMPLOYED, *self.features], dtype=object)
