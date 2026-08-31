"""Stałe projektu i kontrakt kolumn.

Ten moduł jest importowalny **bez** obecności danych na dysku — CI nie ma
poświadczeń Kaggle i nie pobiera 160 MB. Dlatego podział na grupy cech jest
funkcją liczoną z ramki danych, a nie listą 122 nazw wpisaną ręcznie: nazwy
kolumn zmieniłyby się razem z danymi, a wyprowadzenie z dtypes nie może się
rozjechać.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- Ścieżki ---------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
DATA_PATH: Path = RAW_DATA_DIR / "application_train.csv"

# --- Kontrakt kolumn -------------------------------------------------------

TARGET: str = "TARGET"
"""Zmienna celu: 1 = klient miał trudności ze spłatą, 0 = spłacał terminowo."""

ID_COLUMN: str = "SK_ID_CURR"
"""Identyfikator wniosku. Nigdy nie jest cechą — unikalny na wiersz."""

RANDOM_STATE: int = 42
"""Jedno ziarno dla całego projektu (split, samplery, modele)."""

BINARY_VALUES: frozenset[float] = frozenset({0.0, 1.0})
"""Wartości, jakie może przyjmować kolumna liczbowa uznana za flagę 0/1."""


@dataclass(frozen=True)
class FeatureGroups:
    """Kolumny cech rozbite na grupy wymagające innego preprocessingu.

    Podział steruje `build_preprocessor()` w Fazie 1: każda grupa dostaje własny
    pipeline imputacji/skalowania/kodowania.

    Attributes:
        numeric: Ciągłe i dyskretne liczby (np. `AMT_INCOME_TOTAL`) —
            imputacja medianą, skalowanie.
        categorical: Kolumny nieliczbowe (np. `NAME_CONTRACT_TYPE`, a także
            `FLAG_OWN_CAR`, który w tym zbiorze trzyma `"Y"`/`"N"`, nie 0/1) —
            imputacja modą, one-hot.
        binary: Liczbowe flagi 0/1 (np. `FLAG_MOBIL`) — imputacja, bez
            skalowania.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    binary: tuple[str, ...]

    @property
    def all_features(self) -> tuple[str, ...]:
        """Wszystkie kolumny cech, w kolejności grup."""
        return self.numeric + self.categorical + self.binary

    def to_dict(self) -> dict[str, list[str]]:
        """Serializuje grupy do postaci nadającej się do zapisu obok modelu."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "binary": list(self.binary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> FeatureGroups:
        """Odtwarza grupy zapisane w czasie treningu.

        Args:
            payload: Słownik z `to_dict()`.

        Returns:
            Te same grupy, w tej samej kolejności.
        """
        return cls(
            numeric=tuple(payload["numeric"]),
            categorical=tuple(payload["categorical"]),
            binary=tuple(payload["binary"]),
        )


def split_feature_groups(df: pd.DataFrame) -> FeatureGroups:
    """Wyprowadza grupy cech z **ramki treningowej**.

    Ostrzeżenie:
        Wynik jest funkcją przekazanej próbki, nie schematu. Kolumna liczbowa
        trafia do `binary` na podstawie wartości, które faktycznie wystąpiły —
        na ramce jednowierszowej (scoring pojedynczego wniosku w Fazie 6)
        `CNT_CHILDREN == 0` wyglądałoby jak flaga. Dlatego grupy wyprowadzamy
        **raz**, na pełnym zbiorze treningowym, i zapisujemy przez `to_dict()`
        razem z artefaktem modelu. Serwowanie odtwarza je przez `from_dict()`
        i **nigdy** nie wywołuje tej funkcji na danych żądania — inaczej
        dostajemy train/serve skew.

    Kolumna celu i identyfikator są wyłączone — nie są cechami.

    Args:
        df: Pełna ramka treningowa wczytana przez `data.load_data`.

    Returns:
        Grupy cech; każda lista posortowana alfabetycznie dla stabilności
        kolejności kolumn między uruchomieniami.
    """
    feature_cols = [c for c in df.columns if c not in {TARGET, ID_COLUMN}]

    numeric: list[str] = []
    categorical: list[str] = []
    binary: list[str] = []

    for col in feature_cols:
        series = df[col]
        # Rozgałęziamy po "czy liczbowe", a nie po `dtype == "object"`: pandas 3.0
        # wnioskuje dla tekstu dedykowany dtype `str`, więc test na `object`
        # po cichu wpuściłby kolumny tekstowe do grupy numerycznej.
        if pd.api.types.is_bool_dtype(series):
            binary.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Test na wartości, nie na `nunique() <= 2`: przy liczeniu unikatów
            # kolumna pusta (0 unikatów) i stała (1 unikat) też przechodziły
            # jako flagi, choć jedna nie niesie nic, a druga może być dowolną
            # liczbą.
            values = set(series.dropna().unique())
            if values and values <= BINARY_VALUES:
                binary.append(col)
            else:
                numeric.append(col)
        else:
            categorical.append(col)

    return FeatureGroups(
        numeric=tuple(sorted(numeric)),
        categorical=tuple(sorted(categorical)),
        binary=tuple(sorted(binary)),
    )
