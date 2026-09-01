"""Nazwany rejestr modeli.

Faza 1 ma dokładnie jeden wpis — regresję logistyczną z domyślnymi
hiperparametrami. To celowe: jej ROC-AUC jest punktem odniesienia, wobec
którego mierzone są wszystkie późniejsze fazy. Faza 3 dokłada kolejne
algorytmy do tego samego słownika.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE


def get_models() -> dict[str, BaseEstimator]:
    """Zwraca modele bazowe, każdy niedopasowany.

    Returns:
        Słownik nazwa -> świeża instancja estymatora. Instancje są tworzone
        przy każdym wywołaniu, żeby dopasowanie jednego modelu nie wpływało
        na drugi.
    """
    return {
        # max_iter=1000 zamiast domyślnych 100: po one-hot mamy kilkaset kolumn
        # i lbfgs nie zbiega w 100 iteracjach — sklearn ostrzega, a model jest
        # cicho niedouczony.
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    }
