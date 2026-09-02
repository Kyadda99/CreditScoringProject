"""Cechy pochodne — jedna czysta funkcja na cechę.

Każda bierze ramkę i zwraca serię o tym samym indeksie. Nic tu nie ma stanu,
więc każdą da się przetestować wartością policzoną ręcznie (spec, T1).

Reguła nadrzędna (spec D5): **żaden iloraz nie zwraca ±inf**. Faza 1 ustawiła
`allow_inf_nan=False` na modelu żądania, a `SimpleImputer` traktuje `inf` jak
zwykłą wartość i psuje średnią oraz wariancję całej kolumny w `StandardScaler`.
Zero i brak w mianowniku dają NaN — czyli dokładnie to, co dopasowany imputer
umie obsłużyć.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from features.cleaning import _numeric_column, days_employed_cleaned

BUREAU_INQUIRY_COLUMNS: tuple[str, ...] = (
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)

EXT_SOURCE_COLUMNS: tuple[str, ...] = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Dzieli tak, żeby wynik NIGDY nie był ±inf (spec D5).

    Zero w mianowniku zamieniamy na NaN *przed* dzieleniem, a nie po — po
    dzieleniu byłoby już za późno, bo `0 / 0` daje NaN, a `1 / 0` daje inf
    i trzeba by rozróżniać dwa przypadki. Dodatkowe czyszczenie wyniku łapie
    nadmiar zmiennoprzecinkowy przy skrajnych kwotach.

    Args:
        numerator: Licznik.
        denominator: Mianownik.

    Returns:
        Seria `float64`; NaN wszędzie tam, gdzie mianownik jest zerem lub
        brakiem, albo gdzie wynik przekroczył zakres float.
    """
    denom = denominator.astype("float64").mask(denominator == 0)
    result = numerator.astype("float64") / denom
    return result.replace([np.inf, -np.inf], np.nan)


def credit_income_ratio(frame: pd.DataFrame) -> pd.Series:
    """`AMT_CREDIT / AMT_INCOME_TOTAL` — dźwignia."""
    return _safe_divide(
        _numeric_column(frame, "AMT_CREDIT"),
        _numeric_column(frame, "AMT_INCOME_TOTAL"),
    )


def annuity_income_ratio(frame: pd.DataFrame) -> pd.Series:
    """`AMT_ANNUITY / AMT_INCOME_TOTAL` — obciążenie ratą (debt-to-income)."""
    return _safe_divide(
        _numeric_column(frame, "AMT_ANNUITY"),
        _numeric_column(frame, "AMT_INCOME_TOTAL"),
    )


def credit_term(frame: pd.DataFrame) -> pd.Series:
    """`AMT_ANNUITY / AMT_CREDIT` — odwrotność domyślnej długości kredytu."""
    return _safe_divide(
        _numeric_column(frame, "AMT_ANNUITY"),
        _numeric_column(frame, "AMT_CREDIT"),
    )


def goods_credit_ratio(frame: pd.DataFrame) -> pd.Series:
    """`AMT_GOODS_PRICE / AMT_CREDIT` — przybliżenie wkładu własnego."""
    return _safe_divide(
        _numeric_column(frame, "AMT_GOODS_PRICE"),
        _numeric_column(frame, "AMT_CREDIT"),
    )


def income_per_person(frame: pd.DataFrame) -> pd.Series:
    """`AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` — dochód na osobę w gospodarstwie."""
    return _safe_divide(
        _numeric_column(frame, "AMT_INCOME_TOTAL"),
        _numeric_column(frame, "CNT_FAM_MEMBERS"),
    )


def employed_age_ratio(frame: pd.DataFrame) -> pd.Series:
    """Staż pracy w relacji do wieku.

    Liczony na `DAYS_EMPLOYED` **po dekodowaniu sentinela** — na surowej
    wartości 365243 iloraz wyszedłby ujemny i bez sensu.
    """
    return _safe_divide(
        days_employed_cleaned(frame),
        _numeric_column(frame, "DAYS_BIRTH"),
    )


def bureau_inquiry_total(frame: pd.DataFrame) -> pd.Series:
    """Łączna liczba zapytań do biura kredytowego.

    To cecha "liczba zapytań kredytowych" wprost wymieniona w `CLAUDE.md`.

    Returns:
        Suma sześciu kolumn. `min_count=1` sprawia, że wiersz z samymi brakami
        daje NaN, a nie 0.0 — "nie wiem" nie może udawać "zero zapytań".
    """
    columns = [_numeric_column(frame, name) for name in BUREAU_INQUIRY_COLUMNS]
    return pd.concat(columns, axis=1).sum(axis=1, min_count=1)


def ext_source_mean(frame: pd.DataFrame) -> pd.Series:
    """Średnia z trzech zewnętrznych scoringów, po dostępnych wartościach."""
    columns = [_numeric_column(frame, name) for name in EXT_SOURCE_COLUMNS]
    return pd.concat(columns, axis=1).mean(axis=1)


def ext_source_count(frame: pd.DataFrame) -> pd.Series:
    """Ile z trzech scoringów w ogóle jest.

    Brak jest tu cechą, nie usterką (spec D3): liczba dostępnych scoringów sama
    w sobie mówi coś o kliencie.
    """
    columns = [_numeric_column(frame, name) for name in EXT_SOURCE_COLUMNS]
    return pd.concat(columns, axis=1).notna().sum(axis=1).astype("float64")


RATIO_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "ANNUITY_INCOME_RATIO": annuity_income_ratio,
    "BUREAU_INQUIRY_TOTAL": bureau_inquiry_total,
    "CREDIT_INCOME_RATIO": credit_income_ratio,
    "CREDIT_TERM": credit_term,
    "EMPLOYED_AGE_RATIO": employed_age_ratio,
    "EXT_SOURCE_COUNT": ext_source_count,
    "EXT_SOURCE_MEAN": ext_source_mean,
    "GOODS_CREDIT_RATIO": goods_credit_ratio,
    "INCOME_PER_PERSON": income_per_person,
}
"""Nazwa cechy -> funkcja. `FeatureEngineer` iteruje po tym słowniku.

Klucze muszą pokrywać się z `config.ENGINEERED_FEATURES` — pilnuje tego test
w `tests/test_features_engineer.py`.
"""
