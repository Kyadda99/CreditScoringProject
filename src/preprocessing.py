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

from config import INFORMATIVE_MISSING, FeatureGroups


def build_preprocessor(
    groups: FeatureGroups,
    informative_missing: tuple[str, ...] = INFORMATIVE_MISSING,
) -> ColumnTransformer:
    """Składa preprocessor dopasowany do podziału kolumn na grupy.

    Args:
        groups: Grupy cech wyprowadzone **raz**, na ramce treningowej.
        informative_missing: Kolumny numeryczne, w których brak niesie sygnał
            (spec D3). Dostają ten sam pipeline plus `add_indicator=True`.
            Nazwy spoza `groups.numeric` są ignorowane — kolumna usunięta
            z modelu nie może wrócić bocznymi drzwiami.

    Returns:
        Niedopasowany `ColumnTransformer`. Kolumny spoza grup są odrzucane
        (`remainder="drop"`) — cel i identyfikator nie mogą wejść do modelu.
    """
    informative = [c for c in groups.numeric if c in set(informative_missing)]
    plain_numeric = [c for c in groups.numeric if c not in set(informative)]

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # Ta sama imputacja, plus kolumna 0/1 mówiąca "tu było pusto". Wskaźnik idzie
    # przez ten sam StandardScaler co wartość — po standaryzacji flaga 0/1 dalej
    # rozróżnia dwa stany, więc osobna ścieżka nic by nie wniosła poza złożonością.
    numeric_informative = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
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

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", numeric, plain_numeric),
        ("categorical", categorical, list(groups.categorical)),
        ("binary", binary, list(groups.binary)),
    ]
    # Gałąź dokładana warunkowo: `ColumnTransformer` z pustą listą kolumn bywa
    # kapryśny przy `get_feature_names_out`, a pusty transformer i tak niczego
    # nie liczy.
    if informative:
        transformers.insert(
            1, ("numeric_informative", numeric_informative, informative)
        )

    return ColumnTransformer(transformers, remainder="drop")
