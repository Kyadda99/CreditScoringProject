"""Wczytywanie surowych danych i higiena wierszy.

Podział obowiązków po Fazie 2 (spec D4): tutaj mieszka **wyłącznie** to, co
nie ma sensu poza zbiorem treningowym — odrzucenie wierszy bez celu i wierszy
z `CODE_GENDER == "XNA"`.

Czyszczenie **kolumn** — sentinel `DAYS_EMPLOYED == 365243`, imputacja, cechy
pochodne — mieszka w `features/`, wewnątrz pipeline'u. Powód: klient `/score`
może przysłać sentinel, a kod wykonywany tylko przy wczytywaniu CSV nie
obroniłby produkcji.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATA_PATH, ID_COLUMN, TARGET

logger = logging.getLogger(__name__)


def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa wiersze, których nie da się użyć do uczenia.

    Wyłącznie higiena **wierszy**, i wyłącznie taka, która nie ma sensu poza
    zbiorem treningowym (spec D4). Wszystko, co klient `/score` mógłby przysłać
    źle — sentinele, braki, wartości spoza zakresu — mieszka w `FeatureEngineer`
    i w dopasowanym imputerze, bo tylko tam zadziała także w produkcji.

    Args:
        df: Ramka; brakujące kolumny są tolerowane, żeby dało się wywołać to
            także na ramce bez celu.

    Returns:
        Nowa ramka bez wierszy z brakującym celem i bez `CODE_GENDER == "XNA"`.
    """
    frame = df
    if TARGET in frame.columns:
        frame = frame[frame[TARGET].notna()]
    if "CODE_GENDER" in frame.columns:
        # "XNA" to nie trzecia kategoria płci, tylko brak zapisany tekstem.
        # Czterech wierszy na 307 tysięcy nie warto imputować — one-hot
        # zrobiłby z tego osobną kolumnę o czterech obserwacjach.
        frame = frame[frame["CODE_GENDER"] != "XNA"]
    if len(frame) == len(df):
        return df
    logger.info("Higiena wierszy: %d -> %d.", len(df), len(frame))
    return frame.reset_index(drop=True)


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Wczytuje `application_train.csv` do ramki danych.

    Args:
        path: Ścieżka do pliku CSV. Domyślnie `config.DATA_PATH`.

    Returns:
        Surowa ramka danych — bez czyszczenia, z typami nadanymi przez pandas.

    Raises:
        FileNotFoundError: Gdy plik nie istnieje; komunikat wskazuje skrypt
            pobierający, bo to najczęstsza przyczyna.
    """
    path = DATA_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"Brak pliku z danymi: {path}\n" "Uruchom: uv run cs-download"
        )

    logger.info("Wczytuję %s ...", path)
    df = pd.read_csv(path)

    # `pd.to_numeric(errors="coerce")` zamiast `astype`: astype po cichu zawija
    # przy zwezaniu (300 -> 44 dla int8), wiec uszkodzony plik przeszedlby przez
    # loader wygladajac poprawnie. Coerce zamienia smieci na NaN, co kontrakt
    # widzi i odrzuca. Zadnego zwezania typu - kontrakt ma ogladac to, co jest
    # w CSV, a nie to, co loader z tego zrobil.
    for column in (ID_COLUMN, TARGET):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = drop_unusable_rows(df)

    logger.info("Wczytano %d wierszy x %d kolumn.", len(df), df.shape[1])
    return df
