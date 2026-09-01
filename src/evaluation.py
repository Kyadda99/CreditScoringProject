"""Metryki klasyfikacji binarnej.

Zbiór jest silnie niezbalansowany (8.07% klasy pozytywnej), więc trafność sama
w sobie nic nie mówi — model przewidujący zawsze "nie zbankrutuje" osiąga 91.9%.
Metryką wiodącą jest ROC-AUC; metryki decyzyjne liczone są przy jawnym progu.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Liczy metryki dla prawdopodobieństw klasy pozytywnej.

    Args:
        y_true: Prawdziwe etykiety 0/1.
        y_proba: Prawdopodobieństwo klasy pozytywnej — `predict_proba(...)[:, 1]`,
            nie `predict()`.
        threshold: Próg decyzyjny. Faza 3 go stroi; tutaj jawnie 0.5.

    Returns:
        Słownik metryk. Wartości to zwykłe `float`, nie `np.float64` — MLflow
        i serializacja JSON nie przyjmują typów numpy bez konwersji.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: przy wysokim progu model może nie wskazać ani jednej
        # klasy pozytywnej, co bez tego daje ostrzeżenie i NaN.
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
