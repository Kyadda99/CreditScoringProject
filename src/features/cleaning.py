"""Dekodowanie wartości sentinelowych.

`DAYS_EMPLOYED == 365243` to ~1000 lat stażu — tak zbiór koduje
"niezatrudniony". Zostawiona liczba wchodzi do `StandardScaler` jako wartość
oddalona o setki odchyleń i zdominowałaby wkład tej cechy w model liniowy.

Dekodowanie mieszka **w pipelinie**, nie w `data.py` (spec D4): klient `/score`
może przysłać 365243, a czyszczenie wykonywane tylko przy wczytywaniu danych
nie obroniłoby produkcji.

Wszystkie funkcje są czyste — biorą ramkę, zwracają nową serię, nie ruszają
wejścia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DAYS_EMPLOYED_SENTINEL

_DAYS_EMPLOYED = "DAYS_EMPLOYED"


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Zwraca kolumnę jako float albo serię NaN, gdy kolumny nie ma.

    Brak kolumny nie jest błędem: pola `/score` są opcjonalne (Faza 1, D1),
    a dopasowany imputer jest zaprojektowanym miejscem obsługi braków.

    Args:
        frame: Ramka wejściowa.
        name: Nazwa kolumny.

    Returns:
        Seria `float64` o indeksie ramki.
    """
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def days_employed_cleaned(frame: pd.DataFrame) -> pd.Series:
    """`DAYS_EMPLOYED` z wartością sentinelową zamienioną na NaN.

    Args:
        frame: Ramka mogąca zawierać `DAYS_EMPLOYED`.

    Returns:
        Seria `float64`; sentinel → NaN, reszta bez zmian.
    """
    series = _numeric_column(frame, _DAYS_EMPLOYED)
    return series.mask(series == DAYS_EMPLOYED_SENTINEL)


def not_employed_flag(frame: pd.DataFrame) -> pd.Series:
    """Jawna flaga 0/1: czy wiersz miał wartość sentinelową.

    Informacja "niezatrudniony" jest realna — udział defaultów w tej grupie to
    5.40% wobec 8.66% w pozostałych. Bez tej flagi znika razem z sentinelem.

    Args:
        frame: Ramka mogąca zawierać `DAYS_EMPLOYED`.

    Returns:
        Seria `float64` o wartościach 0.0/1.0. NaN daje 0.0 — brak wartości
        nie jest dowodem na brak zatrudnienia.
    """
    series = _numeric_column(frame, _DAYS_EMPLOYED)
    return (series == DAYS_EMPLOYED_SENTINEL).astype("float64")
