"""Wymienne warianty estymatora i obsługi niezbalansowania.

Dwie osie zmienności zmierzone w siatce porównawczej: rodzina estymatora oraz
sposób radzenia sobie z rzadką klasą. Wagi klas ustawia się inaczej
w scikit-learn niż w XGBoost, więc wie o tym sama rodzina estymatora, a nie
warunek w kodzie wywołującym.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import RANDOM_STATE


class EstimatorStrategy(ABC):
    """Rodzina estymatora.

    Kontrakt: `build` zwraca **nowy, niedopasowany** estymator przy każdym
    wywołaniu i sam decyduje, jak wyrazić ważenie klas w swoim API.

    Attributes:
        name: Klucz w `ESTIMATOR_STRATEGIES`, używany w tabeli porównawczej.
    """

    name: str

    @abstractmethod
    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Tworzy estymator.

        Args:
            class_weighted: Czy rzadka klasa ma dostać wyższą wagę.
            scale_pos_weight: Stosunek negatywów do pozytywów; istotny tylko
                dla rodzin, które wyrażają wagi liczbą.

        Returns:
            Niedopasowany estymator.
        """


class ImbalanceStrategy(ABC):
    """Sposób radzenia sobie z niezbalansowaniem klas.

    Kontrakt: ramię albo prosi estymator o wagi (`uses_class_weights`), albo
    dokłada sampler (`make_sampler`), nigdy oba naraz — inaczej ten sam
    imbalans zostałby policzony dwa razy.

    Attributes:
        name: Klucz w `IMBALANCE_STRATEGIES`.
        uses_class_weights: Czy estymator ma dostać `class_weighted=True`.
    """

    name: str
    uses_class_weights: bool = False

    @abstractmethod
    def make_sampler(self) -> BaseEstimator | None:
        """Zwraca świeży sampler albo `None`, gdy ramię go nie używa."""


@dataclass(frozen=True)
class LogisticRegressionStrategy(EstimatorStrategy):
    """Regresja logistyczna — punkt odniesienia całego projektu."""

    name: str = "logistic_regression"

    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Tworzy regresję logistyczną. Patrz `EstimatorStrategy.build`."""
        # max_iter=1000: po one-hot mamy 211 kolumn, a lbfgs nie zbiega w 100.
        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced" if class_weighted else None,
        )


@dataclass(frozen=True)
class RandomForestStrategy(EstimatorStrategy):
    """Las losowy z ograniczoną głębokością."""

    name: str = "random_forest"

    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Tworzy las losowy. Patrz `EstimatorStrategy.build`."""
        # Ograniczenia głębokości i liścia: bez nich las na 246k wierszach
        # trenuje się długo, waży setki MB i przeucza na szumie rzadkiej klasy.
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced" if class_weighted else None,
        )


@dataclass(frozen=True)
class XGBoostStrategy(EstimatorStrategy):
    """Gradient boosting; wagi klas wyraża liczbą, nie etykietą."""

    name: str = "xgboost"

    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Tworzy XGBoosta. Patrz `EstimatorStrategy.build`."""
        # tree_method="hist": splity po histogramie, wielokrotnie szybsze na
        # 246k x 211. eval_metric="aucpr" spina metrykę wewnętrzną z tą,
        # którą wybieramy championa.
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight if class_weighted else 1.0,
        )


@dataclass(frozen=True)
class NoImbalanceHandling(ImbalanceStrategy):
    """Brak jakiejkolwiek korekty — model widzi rozkład, jaki jest."""

    name: str = "none"
    uses_class_weights: bool = False

    def make_sampler(self) -> BaseEstimator | None:
        """Nie używa samplera."""
        return None


@dataclass(frozen=True)
class ClassWeighting(ImbalanceStrategy):
    """Wyższa waga rzadkiej klasy, wyrażona w API estymatora."""

    name: str = "class_weight"
    uses_class_weights: bool = True

    def make_sampler(self) -> BaseEstimator | None:
        """Nie używa samplera."""
        return None


@dataclass(frozen=True)
class Resampling(ImbalanceStrategy):
    """Syntetyczne przykłady rzadkiej klasy (SMOTE)."""

    name: str = "resample"
    uses_class_weights: bool = False

    def make_sampler(self) -> BaseEstimator | None:
        """Zwraca świeży SMOTE."""
        # Bez n_jobs: parametr usunięty z SMOTE w imbalanced-learn 0.12.
        return SMOTE(random_state=RANDOM_STATE)


ESTIMATOR_STRATEGIES: dict[str, EstimatorStrategy] = {
    s.name: s
    for s in (
        LogisticRegressionStrategy(),
        RandomForestStrategy(),
        XGBoostStrategy(),
    )
}
"""Nazwa algorytmu -> strategia. Nowy algorytm dopisuje wpis, nie warunek."""

IMBALANCE_STRATEGIES: dict[str, ImbalanceStrategy] = {
    s.name: s for s in (NoImbalanceHandling(), ClassWeighting(), Resampling())
}
"""Nazwa ramienia -> strategia."""


def get_estimator_strategy(name: str) -> EstimatorStrategy:
    """Zwraca strategię estymatora po nazwie.

    Args:
        name: Klucz z `ESTIMATOR_STRATEGIES`.

    Returns:
        Strategia estymatora.

    Raises:
        ValueError: Gdy nazwa jest nieznana.
    """
    try:
        return ESTIMATOR_STRATEGIES[name]
    except KeyError:
        allowed = ", ".join(sorted(ESTIMATOR_STRATEGIES))
        raise ValueError(
            f"Nieznany estymator: {name!r}. Dozwolone: {allowed}."
        ) from None


def get_imbalance_strategy(name: str) -> ImbalanceStrategy:
    """Zwraca strategię niezbalansowania po nazwie.

    Args:
        name: Klucz z `IMBALANCE_STRATEGIES`.

    Returns:
        Strategia niezbalansowania.

    Raises:
        ValueError: Gdy nazwa jest nieznana.
    """
    try:
        return IMBALANCE_STRATEGIES[name]
    except KeyError:
        allowed = ", ".join(sorted(IMBALANCE_STRATEGIES))
        raise ValueError(
            f"Nieznane ramię niezbalansowania: {name!r}. Dozwolone: {allowed}."
        ) from None
