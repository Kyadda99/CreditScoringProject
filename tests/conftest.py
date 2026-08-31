"""Wspólna konfiguracja pytest.

Testy kontraktu danych wymagają ~160 MB CSV, którego CI nie ma (brak
poświadczeń Kaggle i brak sensu pobierania takiego pliku w workflow).
Marker `requires_data` pozwala pominąć je czysto — bez wyłączania całego pliku.
"""

from __future__ import annotations

import json
import os

import pytest

from config import DATA_PATH


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Pomija testy oznaczone `requires_data`, gdy zbioru nie ma na dysku."""
    if DATA_PATH.exists():
        return
    skip = pytest.mark.skip(reason=f"Brak {DATA_PATH}; uruchom `uv run cs-download`")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _write_column_manifest() -> None:
    """Spisuje manifest kolumn z prawdziwego pliku, gdy `CS_WRITE_MANIFEST=1`.

    Jednorazowa czynność po pierwszym pobraniu danych; wynik jest commitowany
    i od tego momentu każda zmiana nazwy kolumny wywala test kontraktu.
    """
    if os.environ.get("CS_WRITE_MANIFEST") != "1" or not DATA_PATH.exists():
        return

    from data import load_data
    from test_data_contract import MANIFEST_PATH

    columns = list(load_data().columns)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(columns, indent=2), encoding="utf-8")
    print(f"\nZapisano manifest {len(columns)} kolumn -> {MANIFEST_PATH}")
