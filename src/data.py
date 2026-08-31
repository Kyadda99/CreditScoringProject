"""Wczytywanie surowych danych.

Faza 0 celowo **nie czyści** danych. Decyzje o imputacji, wartościach
sentinelowych (np. `DAYS_EMPLOYED == 365243`) i outlierach wymagają dowodów
z EDA — te powstają w Fazie 2. Tutaj tylko czytamy i rzutujemy typy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATA_PATH, ID_COLUMN, TARGET

logger = logging.getLogger(__name__)


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

    logger.info("Wczytano %d wierszy x %d kolumn.", len(df), df.shape[1])
    return df
