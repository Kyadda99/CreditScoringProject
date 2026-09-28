"""Ocena pojedynczego wniosku na podstawie zapisanego artefaktu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from artifact import Bundle
from config import FeatureGroups

FALLBACK_THRESHOLD = 0.5
"""Próg stosowany, gdy nie ma go ani w konfiguracji, ani w artefakcie."""


class ThresholdSettings(Protocol):
    """Konfiguracja niosąca jawny próg albo jego brak.

    Protokół zamiast importu `api.config`: predyktor jest warstwą niżej niż
    serwis HTTP i nie może od niego zależeć.
    """

    score_threshold: float | None


@dataclass(frozen=True)
class ScoreResult:
    """Wynik oceny jednego wniosku.

    Attributes:
        probability: Prawdopodobieństwo trudności ze spłatą.
        decision: Czy prawdopodobieństwo osiągnęło próg.
        threshold: Próg, który wyprodukował decyzję.
    """

    probability: float
    decision: bool
    threshold: float


class Predictor:
    """Ocenia wnioski wytrenowanym potokiem.

    Attributes:
        bundle: Artefakt z potokiem i kontraktami kolumn.
        threshold: Próg decyzyjny zastosowany do prawdopodobieństwa.
    """

    def __init__(self, bundle: Bundle, threshold: float) -> None:
        """Przyjmuje gotowy artefakt i próg.

        Args:
            bundle: Wczytany artefakt.
            threshold: Próg decyzyjny.
        """
        self.bundle = bundle
        self.threshold = threshold

    @classmethod
    def from_bundle(cls, bundle: Bundle, threshold: float | None = None) -> Predictor:
        """Tworzy predyktor, biorąc próg z artefaktu, gdy nie podano innego.

        Args:
            bundle: Wczytany artefakt.
            threshold: Jawny próg; `None` oznacza próg zapisany w metadanych.

        Returns:
            Gotowy predyktor.
        """
        if threshold is None:
            threshold = float(bundle.metadata.get("threshold", FALLBACK_THRESHOLD))
        return cls(bundle, threshold)

    @classmethod
    def from_settings(cls, settings: ThresholdSettings, bundle: Bundle) -> Predictor:
        """Buduje predyktor; jawny próg z konfiguracji ma pierwszeństwo.

        Args:
            settings: Ustawienia serwisu.
            bundle: Wczytany artefakt.

        Returns:
            Predyktor z progiem z konfiguracji, a w jego braku z artefaktu,
            a w braku obu z wartości domyślnej.
        """
        return cls.from_bundle(bundle, settings.score_threshold)

    @property
    def input_groups(self) -> FeatureGroups:
        """Kontrakt wejściowy: surowe kolumny przyjmowane od klienta."""
        return self.bundle.input_groups

    def to_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Zamienia ciało żądania na jednowierszową ramkę surowych kolumn.

        Buduje wiersz z kontraktu **wejściowego**, nie z grup modelu: model
        konsumuje cechy pochodne, których klient nie przysyła — liczy je
        pierwszy krok potoku.

        Args:
            payload: Zwalidowane pola żądania, bez pustych.

        Returns:
            Ramka o jednym wierszu i kolumnach kontraktu wejściowego.
        """
        return self.build_frame(payload, self.input_groups)

    @staticmethod
    def build_frame(payload: dict[str, Any], groups: FeatureGroups) -> pd.DataFrame:
        """Buduje jednowierszową ramkę dla wskazanych grup kolumn.

        Kolejność kolumn bierzemy z kontraktu, a nie z żądania: kolejność
        kluczy w JSON-ie jest przypadkowa.

        Args:
            payload: Zwalidowane pola żądania, bez pustych.
            groups: Grupy kolumn, z których budujemy wiersz.

        Returns:
            Ramka o jednym wierszu i kolumnach `groups.all_features`.
        """
        row = {name: payload.get(name) for name in groups.all_features}
        frame = pd.DataFrame([row], columns=list(groups.all_features))

        # Bez rzutowania ramka ma dtype object, a wtedy SimpleImputer nie
        # rozpoznaje None jako braku i flagi binarne docierają do modelu jako NaN.
        numeric_like = [*groups.numeric, *groups.binary]
        if numeric_like:
            frame[numeric_like] = frame[numeric_like].astype("float64")
        categorical = list(groups.categorical)
        if categorical:
            frame[categorical] = (
                frame[categorical]
                .astype(object)
                .where(frame[categorical].notna(), np.nan)
            )
        return frame

    def score(self, payload: dict[str, Any]) -> ScoreResult:
        """Ocenia jeden wniosek.

        Args:
            payload: Zwalidowane pola żądania.

        Returns:
            Prawdopodobieństwo, decyzja i próg.

        Raises:
            ValueError: Gdy wartości cech są niepoprawne dla potoku.
        """
        frame = self.to_frame(payload)
        probability = float(self.bundle.pipeline.predict_proba(frame)[0, 1])
        return ScoreResult(
            probability=probability,
            decision=probability >= self.threshold,
            threshold=self.threshold,
        )
