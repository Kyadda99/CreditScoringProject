"""Budowanie preprocessora — składanie, nigdy dopasowywanie.

Każda grupa cech z `config.FeatureGroups` dostaje własny pipeline, bo wymaga
innego traktowania. Zwracany `ColumnTransformer` jest **niedopasowany**:
dopasowanie należy do `train.py`, wewnątrz jednego `Pipeline` razem z modelem,
żeby nie dało się przypadkiem nauczyć imputera na zbiorze testowym.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FeatureGroups


def build_preprocessor(groups: FeatureGroups) -> ColumnTransformer:
    """Składa preprocessor dopasowany do podziału kolumn na grupy.

    Args:
        groups: Grupy cech wyprowadzone **raz**, na ramce treningowej.

    Returns:
        Niedopasowany `ColumnTransformer`. Kolumny spoza grup są odrzucane
        (`remainder="drop"`) — cel i identyfikator nie mogą wejść do modelu.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": w produkcji pojawi się wartość, której
            # nie było w treningu. Bez tego encoder rzuca wyjątkiem, a /score
            # zwraca 500 zamiast wyniku.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    # Flagi 0/1 wymagają tylko imputacji — skalowanie nic by nie wniosło,
    # a utrudniłoby interpretację współczynników regresji.
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        [
            ("numeric", numeric, list(groups.numeric)),
            ("categorical", categorical, list(groups.categorical)),
            ("binary", binary, list(groups.binary)),
        ],
        remainder="drop",
    )
