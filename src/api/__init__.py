"""Warstwa HTTP: FastAPI, schematy żądań i konfiguracja serwisu.

Trzyma też **jedyną** definicję ścieżki do kontraktu cech. Wcześniej ta sama
ścieżka była wyprowadzana dwa razy — w `train.py` z `PROJECT_ROOT`, a w
`api/schemas.py` z `__file__` — i pokrywały się wyłącznie w katalogu
repozytorium. Uruchomienie `cs-train` z zainstalowanego pakietu (np. w
kontenerze) zapisywałoby kontrakt w `<venv>/lib/python3.12/src/api/`, czyli
w miejscu, którego nikt nie czyta, a API serwowałoby stary kontrakt bez
jednego słowa błędu.
"""

from __future__ import annotations

from pathlib import Path

FEATURE_SCHEMA_PATH: Path = Path(__file__).resolve().parent / "feature_schema.json"
"""Kontrakt cech: commitowany, generowany przez `train.write_feature_schema`
i czytany przez `api.schemas` przy imporcie. Wyprowadzany z położenia tego
pakietu, więc wskazuje to samo miejsce w repozytorium i po instalacji."""
