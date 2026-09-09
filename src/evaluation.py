"""Metryki klasyfikacji binarnej.

Zbiór jest silnie niezbalansowany (8.07% klasy pozytywnej), więc trafność sama
w sobie nic nie mówi — model przewidujący zawsze "nie zbankrutuje" osiąga 91.9%.
Metryką wiodącą jest ROC-AUC; metryki decyzyjne liczone są przy jawnym progu.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import COST_FN, COST_FP

DEFAULT_THRESHOLD = 0.5


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> dict[str, float]:
    """Liczy metryki dla prawdopodobieństw klasy pozytywnej.

    Zwracany słownik jest **jednym kontraktem metryk**: tym samym obiektem
    karmione są logowanie do MLflow, bramka jakości i wybór championa. Jedno
    źródło prawdy zamiast trzech lekko rozjeżdżających się definicji.

    Args:
        y_true: Prawdziwe etykiety 0/1.
        y_proba: Prawdopodobieństwo klasy pozytywnej — `predict_proba(...)[:, 1]`,
            nie `predict()`.
        threshold: Próg decyzyjny. Faza 3 wyznacza go poza zbiorem testowym
            (spec D4), a nie przyjmuje 0.5 z rozpędu.
        cost_fn: Koszt przeoczonego defaultu.
        cost_fp: Koszt fałszywego alarmu.

    Returns:
        Słownik metryk. Wartości to zwykłe `float`, nie `np.float64` — MLflow
        i serializacja JSON nie przyjmują typów numpy bez konwersji.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    # labels=[0, 1] jest konieczne: przy zdegenerowanej predykcji (same zera
    # albo same jedynki) confusion_matrix zwraca macierz 1x1, a ravel()
    # rozpada się wtedy na ValueError zamiast dać cztery liczby.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        # average_precision, nie auc(recall, precision): ta druga interpoluje
        # liniowo między punktami krzywej i zawyża wynik przy rzadkiej klasie.
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: przy wysokim progu model może nie wskazać ani jednej
        # klasy pozytywnej, co bez tego daje ostrzeżenie i NaN.
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "expected_cost": float(cost_fn * fn + cost_fp * fp),
        "threshold": float(threshold),
    }


def choose_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, float]:
    """Wybiera próg minimalizujący `cost_fn*FN + cost_fp*FP`.

    Przeszukuje wyłącznie progi **osiągalne** — czyli zaobserwowane wartości
    prawdopodobieństwa — w jednym przebiegu po posortowanej tablicy, zamiast
    pętli po siatce. Na 246k wierszach to różnica między sekundą a minutami.

    Wariant "nie oznaczaj nikogo" (próg powyżej maksimum) NIE jest rozważany:
    to status quo z Fazy 2, którego koszt raportujemy osobno dla porównania.

    Args:
        y_true: Prawdziwe etykiety 0/1.
        y_proba: Prawdopodobieństwa klasy pozytywnej. **Muszą pochodzić spoza
            zbioru testowego** (spec D4) — w praktyce z `cross_val_predict`
            na zbiorze treningowym. Dobranie progu na teście i raportowanie
            kosztu przy tym progu byłoby dopasowaniem do zbioru testowego
            i raportowaniem tego dopasowania.
        cost_fn: Koszt przeoczonego defaultu.
        cost_fp: Koszt fałszywego alarmu.

    Returns:
        `(próg, koszt przy tym progu)`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    # Malejąco: idąc w prawo obniżamy próg, więc kolejne wiersze wpadają do
    # klasy pozytywnej jeden po drugim. mergesort jest stabilny — remisy
    # zachowują kolejność, dzięki czemu maska `last_of_tie` niżej jest poprawna.
    order = np.argsort(-y_proba, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_proba[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = int(y_true.sum()) - tp
    cost = cost_fn * fn + cost_fp * fp

    # Przy remisach prawdopodobieństw próg nie potrafi przeciąć grupy w środku
    # — `>=` bierze całą grupę albo nic. Legalne są więc tylko pozycje na
    # KOŃCU serii jednakowych wartości; reszcie nadajemy nieskończony koszt,
    # żeby argmin nie wybrał progu, którego nie da się faktycznie zastosować.
    last_of_tie = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    cost = np.where(last_of_tie, cost, np.inf)

    best = int(np.argmin(cost))
    return float(p_sorted[best]), float(cost[best])
