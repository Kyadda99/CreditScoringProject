"""Schematy żądań i odpowiedzi.

Model żądania jest **generowany** z `feature_schema.json`, a nie pisany ręcznie.
Powód: 120 pól przepisanych z palca rozjedzie się z kontraktem danych przy
pierwszej zmianie kolumny, i to po cichu. Generowanie z tego samego pliku, który
produkuje trening, sprawia, że rozjazd jest niemożliwy.

Koszt tej decyzji: brak statycznego typowania modelu żądania.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from api import FEATURE_SCHEMA_PATH
from config import FeatureGroups

EXAMPLES: dict[str, Any] = {
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "AMT_GOODS_PRICE": 351000.0,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "CNT_CHILDREN": 0,
    "CNT_FAM_MEMBERS": 1.0,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Single / not married",
}
"""Przykłady dla pól najbardziej sensownych w demie — żeby /docs dało się użyć.
Pozostałe pola nie mają przykładu; przy 120 polach byłby to szum."""


def load_feature_groups(path: Path = FEATURE_SCHEMA_PATH) -> FeatureGroups:
    """Wczytuje commitowany kontrakt cech.

    Args:
        path: Ścieżka do `feature_schema.json`.

    Returns:
        Grupy cech w takiej postaci, w jakiej zapisał je trening.

    Raises:
        FileNotFoundError: Gdy pliku nie ma — powstaje dopiero przy treningu.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Brak kontraktu cech: {path}\n"
            "Powstaje przy treningu — uruchom: uv run cs-train"
        )
    return FeatureGroups.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _example_for(name: str) -> list[Any] | None:
    """Zwraca listę z przykładem albo None, gdy pola nie ma w EXAMPLES."""
    return [EXAMPLES[name]] if name in EXAMPLES else None


def _build_request_model(groups: FeatureGroups) -> type[BaseModel]:
    """Buduje model żądania z grup cech.

    Kolumny numeryczne i binarne przyjmują liczbę, kategoryczne tekst. Wszystkie
    pola są opcjonalne — brak pola oznacza brak wartości, którą uzupełni
    dopasowany imputer.

    Args:
        groups: Grupy cech z commitowanego kontraktu.

    Returns:
        Klasa modelu pydantic o polach odpowiadających cechom.
    """
    fields: dict[str, Any] = {}
    # allow_inf_nan=False: bez tego `Infinity` w ciele żądania przechodzi walidację
    # i wywraca się dopiero w `_assert_all_finite` sklearn-a, co bare `except`
    # w /score zamieniał na 500. Nieskończoność w dochodzie to błąd klienta,
    # czyli 422, a nie awaria serwera.
    for name in groups.numeric:
        fields[name] = (
            float | None,
            Field(default=None, allow_inf_nan=False, examples=_example_for(name)),
        )
    # Flagi binarne przyjmują wyłącznie 0/1. Bez tego `{"FLAG_MOBIL": 7.3}`
    # zwracało 200 i realnie przesuwało wynik (0.0674 -> 0.0045), bo wartość
    # spoza dziedziny szła prosto do modelu.
    for name in groups.binary:
        fields[name] = (
            float | None,
            Field(
                default=None,
                ge=0.0,
                le=1.0,
                allow_inf_nan=False,
                examples=_example_for(name),
            ),
        )
    for name in groups.categorical:
        fields[name] = (
            str | None,
            Field(default=None, examples=_example_for(name)),
        )
    return create_model(
        "ScoreRequest",
        # extra="forbid": literówka w nazwie kolumny ma zwrócić 422, a nie
        # zostać po cichu zignorowana i dać wynik z samych imputacji.
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


FEATURE_GROUPS: FeatureGroups = load_feature_groups()
ScoreRequest: type[BaseModel] = _build_request_model(FEATURE_GROUPS)


class ScoreResponse(BaseModel):
    """Wynik scoringu jednego wniosku."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prawdopodobieństwo trudności ze spłatą.",
        examples=[0.0731],
    )
    decision: bool = Field(
        ...,
        description="True, gdy prawdopodobieństwo osiąga próg — czyli wniosek "
        "jest oznaczony jako ryzykowny.",
        examples=[False],
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Próg, który wyprodukował decyzję. Zwracany jawnie, żeby "
        "dało się zinterpretować wynik bez zaglądania do konfiguracji.",
        examples=[0.5],
    )


class HealthResponse(BaseModel):
    """Odpowiedź sondy zdrowia."""

    status: str = Field(..., examples=["ok"])
