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

# --- Założenie kosztowe (Faza 3, spec D4) -----------------------------------
#
# Deklarowane ZANIM policzymy jakikolwiek próg. Przeoczony default forfeituje
# niespłacony kapitał; fałszywy alarm forfeituje marżę na jednym odrzuconym
# wniosku. 10:1 to obronione ZAŁOŻENIE, nie zmierzony fakt — i jako założenie
# jest raportowane w docs/findings/03-model-selection.md.

COST_FN: float = 10.0
"""Koszt przeoczonego defaultu (false negative)."""

COST_FP: float = 1.0
"""Koszt fałszywego alarmu (false positive)."""

# --- Decyzje z EDA (Faza 2) -------------------------------------------------
#
# Te krotki są *zapisem decyzji*, nie mechanizmem. Samo usunięcie kolumny nie
# wymaga kodu — `ColumnTransformer(remainder="drop")` odrzuca wszystko, czego
# nie ma w grupach. Stała istnieje po to, żeby dało się to skontrolować:
# `docs/findings/02-eda.md` uzasadnia każdy wpis, a testy pilnują spójności.

DAYS_EMPLOYED_SENTINEL: int = 365243
"""~1000 lat pracy. Tak zbiór koduje "niezatrudniony" — patrz spec D4.

Występuje w 18.01% wierszy, a udział defaultów w tej grupie wynosi 5.40%
wobec 8.66% w pozostałych — różnica jest realna, więc informacja zostaje
zachowana jawnie jako `FLAG_NOT_EMPLOYED`.
"""

DROPPED_COLUMNS: tuple[str, ...] = (
    # 1. Blok mieszkaniowy: `_AVG` / `_MEDI` / `_MODE` to trzy kodowania tego
    #    samego pojęcia (korelacje 0.96-0.998 wewnątrz każdej trójki). Decyzja
    #    blokowa: zostaje `_MEDI`, znikają `_AVG` i `_MODE`. 28 kolumn.
    "APARTMENTS_AVG",
    "APARTMENTS_MODE",
    "BASEMENTAREA_AVG",
    "BASEMENTAREA_MODE",
    "COMMONAREA_AVG",
    "COMMONAREA_MODE",
    "ELEVATORS_AVG",
    "ELEVATORS_MODE",
    "ENTRANCES_AVG",
    "ENTRANCES_MODE",
    "FLOORSMAX_AVG",
    "FLOORSMAX_MODE",
    "FLOORSMIN_AVG",
    "FLOORSMIN_MODE",
    "LANDAREA_AVG",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_AVG",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_AVG",
    "NONLIVINGAREA_MODE",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_AVG",
    "YEARS_BUILD_MODE",
    # 2. Flagi dokumentów, w których wartość dominująca pokrywa >= 99% wierszy.
    #    Reguła mechaniczna, nie wybieranie po uważaniu: |r| z celem <= 0.012,
    #    a po standaryzacji zostaje z nich szum. 16 kolumn.
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_9",
    # 3. Duplikat sentinela. `FLAG_EMP_PHONE == 0` dla wszystkich 55 374 wierszy
    #    z `DAYS_EMPLOYED == 365243` i tylko dla 12 innych — to niejawnie ta sama
    #    flaga co `FLAG_NOT_EMPLOYED`, tyle że nieudokumentowana (r = 0.9998).
    "FLAG_EMP_PHONE",
    # 4. Pary niemal identyczne (patrz screening korelacji w notebooku).
    "OBS_60_CNT_SOCIAL_CIRCLE",  # r = 0.9985 z OBS_30; zostaje OBS_30
    "REGION_RATING_CLIENT",  # r = 0.9508 z _W_CITY, które ma silniejszy sygnał
)
"""Kolumny wypadające z modelu. Każdy wpis ma uzasadnienie w `02-eda.md`."""

INFORMATIVE_MISSING: tuple[str, ...] = (
    "AMT_REQ_CREDIT_BUREAU_YEAR",  # brak = klient bez historii w biurze
    "EXT_SOURCE_1",  # 56.4% braków, najsilniejsza rodzina cech (r = -0.155)
    "EXT_SOURCE_3",  # 19.8% braków (r = -0.179)
    "OWN_CAR_AGE",  # not_applicable: puste dla 202 924 z 202 929 bez auta
    "TOTALAREA_MODE",  # reprezentant bloku mieszkaniowego (48.3% braków)
)
"""Podzbiór z jawnym wskaźnikiem braku (spec D3).

Celowo **pięć** kolumn, nie wszystkie 67 z brakami. Blok mieszkaniowy dzieli
jeden wzorzec braku, więc wskaźnik dla każdej z 14 kolumn `_MEDI` dałby 14
kolumn współliniowych; reprezentuje go `TOTALAREA_MODE`. Wybór jest decyzją,
nie efektem ubocznym — i to on odróżnia D3 od `add_indicator=True` wszędzie.
"""

FEATURE_SOURCE_COLUMNS: tuple[str, ...] = (
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)
"""Surowe kolumny karmiące cechy pochodne.

Reguła przynależności do kontraktu wejściowego (spec D2): kolumna jest
przyjmowana przez `/score`, jeśli **albo** zostaje cechą modelu, **albo**
karmi cechę pochodną. Kolumna usunięta z modelu, ale licząca się do ilorazu,
musi nadal być przyjmowana — inaczej iloraz zawsze wychodzi NaN.
"""

ENGINEERED_FEATURES: tuple[str, ...] = (
    "ANNUITY_INCOME_RATIO",
    "BUREAU_INQUIRY_TOTAL",
    "CREDIT_INCOME_RATIO",
    "CREDIT_TERM",
    "EMPLOYED_AGE_RATIO",
    "EXT_SOURCE_COUNT",
    "EXT_SOURCE_MEAN",
    "GOODS_CREDIT_RATIO",
    "INCOME_PER_PERSON",
)
"""Cechy liczone przez `FeatureEngineer`. Kolejność ustala kolejność kolumn."""


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


def split_feature_groups(
    df: pd.DataFrame, dropped: tuple[str, ...] = ()
) -> FeatureGroups:
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
        dropped: Kolumny wykluczone decyzją z EDA (`config.DROPPED_COLUMNS`).
            Domyślnie pusta krotka, więc wywołania z Fazy 1 działają bez zmian.

    Returns:
        Grupy cech; każda lista posortowana alfabetycznie dla stabilności
        kolejności kolumn między uruchomieniami.
    """
    excluded = {TARGET, ID_COLUMN, *dropped}
    feature_cols = [c for c in df.columns if c not in excluded]

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


# MLflow 3.x odrzuca backend plikowy ("./mlruns") — jest w trybie utrzymaniowym
# i `set_experiment` rzuca wyjątkiem, dopóki nie ustawi się MLFLOW_ALLOW_FILE_STORE.
# Zamiast wchodzić w wycofywany backend, bierzemy SQLite: rejestr modeli i alias
# @production i tak wymagają backendu bazodanowego. `mlflow.db` jest w .gitignore.
# as_posix(): SQLAlchemy oczekuje ukośników, a nie windowsowych backslashy.
MLFLOW_TRACKING_URI: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
"""Wspólne źródło prawdy dla `train`, `registry` i `tune`.

Mieszka tutaj, a nie w `train.py`, bo `registry.py` też go potrzebuje —
import z `train` zrobiłby cykl.
"""
