"""Kontrakt danych: `application_train.csv` jest tym, czym myślimy, że jest.

To nie są testy statystyk, tylko testy kontraktu — mają paść głośno, gdy plik
zostanie podmieniony, obcięty, gdy kolumna zmieni nazwę albo typ. Każdy
późniejszy etap (preprocessing, schemat API) wyprowadza się z tych kolumn, więc
cicha zmiana tutaj zepsułaby wszystko poniżej.

WARTOŚCI PROWIZORYCZNE: liczby poniżej pochodzą z opisu zbioru, nie z pomiaru
na tej maszynie. Pierwsze uruchomienie z danymi je potwierdza albo obala — i to
jest właśnie sens tego pliku. Rozbieżności zapisujemy w
`docs/findings/00-data-contract.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import ID_COLUMN, TARGET, split_feature_groups
from data import load_data

pytestmark = pytest.mark.requires_data

# Plik ma 307 511 wierszy; `load_data` odrzuca 4 z `CODE_GENDER == "XNA"`
# (higiena wierszy z Fazy 2, spec D4). Kontrakt opisuje to, co widzi reszta
# systemu, więc liczba jest po higienie — nie przed.
EXPECTED_ROWS = 307_507
EXPECTED_COLUMNS = 122
TARGET_RATE_BAND = (0.05, 0.12)
"""Odsetek klasy pozytywnej. Pasmo, nie punkt — chodzi o wykrycie podmiany
pliku lub odwrócenia etykiety, nie o pomiar z dokładnością do promila."""

MANIFEST_PATH = Path(__file__).parent / "fixtures" / "column_manifest.json"
"""Pełna lista nazw kolumn, spisana raz z prawdziwego pliku i zacommitowana.
Bez niej zmiana nazwy kolumny przeszłaby niezauważona — liczba kolumn by się
zgadzała."""

EXPECTED_DTYPE_KINDS: dict[str, str] = {
    ID_COLUMN: "i",
    TARGET: "i",
    "AMT_INCOME_TOTAL": "f",
    "AMT_CREDIT": "f",
    "DAYS_BIRTH": "i",
    "CNT_CHILDREN": "i",
}
"""Rodzaj dtype (`numpy.dtype.kind`) dla kolumn, na których stoi reszta
projektu. Pilnuje m.in. podmiany int -> float, której sama liczba kolumn nie
wykryje. Porównujemy `kind`, nie dokładny dtype: szerokość (int32 vs int64)
zależy od platformy, a nie od kontraktu."""

EXPECTED_GROUP_MEMBERSHIP: dict[str, str] = {
    "AMT_INCOME_TOTAL": "numeric",
    "DAYS_BIRTH": "numeric",
    "NAME_CONTRACT_TYPE": "categorical",
    "CODE_GENDER": "categorical",
    "FLAG_OWN_CAR": "categorical",  # trzyma "Y"/"N", nie 0/1
    "FLAG_MOBIL": "binary",
    "FLAG_EMP_PHONE": "binary",
}
"""Konkretne kolumny w konkretnych grupach. Test „każda grupa niepusta"
przechodził dla dowolnego CSV z jedną liczbą, jednym tekstem i jedną flagą."""

LEAKAGE_PREFIXES = ("SK_ID_",)
"""Kolumny identyfikacyjne. Nie niosą sygnału, a model potrafi je zapamiętać."""


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Surowe dane wczytane raz na moduł — plik ma ~160 MB."""
    return load_data()


def test_shape_matches_contract(df: pd.DataFrame) -> None:
    """Wykrywa obcięcie pliku i zmianę liczby kolumn."""
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLUMNS)


def test_column_names_match_manifest(df: pd.DataFrame) -> None:
    """Wykrywa zmianę nazwy kolumny, której liczba kolumn nie wykryje."""
    if not MANIFEST_PATH.exists():
        pytest.fail(
            f"Brak manifestu kolumn: {MANIFEST_PATH}\n"
            "Zapisz go raz z prawdziwego pliku i zacommituj:\n"
            "  CS_WRITE_MANIFEST=1 uv run pytest tests/test_data_contract.py"
        )
    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert list(df.columns) == expected


def test_dtypes_match_contract(df: pd.DataFrame) -> None:
    """Pilnuje typów kolumn, na których stoi preprocessing i schemat API."""
    actual = {col: df[col].dtype.kind for col in EXPECTED_DTYPE_KINDS}
    assert actual == EXPECTED_DTYPE_KINDS


def test_coercion_introduced_no_nulls(df: pd.DataFrame) -> None:
    """`load_data` używa `errors="coerce"`; NaN tutaj oznacza śmieci w CSV."""
    assert df[ID_COLUMN].notna().all()
    assert df[TARGET].notna().all()


def test_target_column_present_and_binary(df: pd.DataFrame) -> None:
    assert TARGET in df.columns
    assert set(df[TARGET].unique()) == {0, 1}


def test_target_rate_within_plausible_band(df: pd.DataFrame) -> None:
    """Zbiór jest silnie niezbalansowany — to determinuje cały Etap 3."""
    rate = float(df[TARGET].mean())
    low, high = TARGET_RATE_BAND
    assert low < rate < high, f"odsetek klasy pozytywnej {rate:.4f} poza pasmem"


def test_id_column_is_unique(df: pd.DataFrame) -> None:
    assert df[ID_COLUMN].is_unique


def test_no_identifier_survives_into_features(df: pd.DataFrame) -> None:
    """Identyfikatory nie mogą trafić do żadnej grupy cech."""
    features = split_feature_groups(df).all_features
    leaked = [c for c in features if c.startswith(LEAKAGE_PREFIXES)]
    assert not leaked, f"identyfikatory w cechach: {leaked}"


def test_known_columns_land_in_expected_groups(df: pd.DataFrame) -> None:
    """Konkretne kolumny muszą trafić do konkretnych grup."""
    groups = split_feature_groups(df)
    lookup = {
        **dict.fromkeys(groups.numeric, "numeric"),
        **dict.fromkeys(groups.categorical, "categorical"),
        **dict.fromkeys(groups.binary, "binary"),
    }
    actual = {col: lookup.get(col) for col in EXPECTED_GROUP_MEMBERSHIP}
    assert actual == EXPECTED_GROUP_MEMBERSHIP


def test_groups_cover_every_column_but_target_and_id(df: pd.DataFrame) -> None:
    """Suma grup + cel + id musi dać dokładnie kolumny pliku."""
    groups = split_feature_groups(df)
    assert set(groups.all_features) | {TARGET, ID_COLUMN} == set(df.columns)


# --- Faza 2: higiena wierszy (spec D4) -------------------------------------
@pytest.mark.requires_data
def test_loader_applies_hygiene(df: pd.DataFrame) -> None:
    assert df["TARGET"].notna().all()
    assert "XNA" not in set(df["CODE_GENDER"].unique())
