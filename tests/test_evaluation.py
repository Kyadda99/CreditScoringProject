"""Testy metryk — na ręcznie policzonych przypadkach, nie na prawdziwych danych."""

from __future__ import annotations

import numpy as np

from evaluation import choose_threshold, evaluate

EXPECTED_KEYS = {
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "expected_cost",
    "threshold",
}
"""Kontrakt metryk (spec Fazy 3). Ten sam słownik karmi MLflow, bramkę
jakości i wybór championa, więc jego kształt jest pinowany testem."""


def test_returns_the_expected_metric_keys() -> None:
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
    assert set(metrics) == EXPECTED_KEYS


def test_perfect_separation_scores_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluate(y_true, y_proba)["roc_auc"] == 1.0


def test_inverted_predictions_score_zero() -> None:
    """Odwrócone etykiety dają 0.0 — łapie pomylony indeks predict_proba."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])
    assert evaluate(y_true, y_proba)["roc_auc"] == 0.0


def test_random_predictions_score_near_half() -> None:
    """Zasiane, nie losowe: bez seeda ten test jest z definicji niestabilny."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=5000)
    y_proba = rng.random(5000)
    assert abs(evaluate(y_true, y_proba)["roc_auc"] - 0.5) < 0.05


def test_threshold_changes_recall_but_not_roc_auc() -> None:
    """ROC-AUC nie zależy od progu; metryki decyzyjne owszem."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    low = evaluate(y_true, y_proba, threshold=0.5)
    high = evaluate(y_true, y_proba, threshold=0.95)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["recall"] == 1.0
    assert high["recall"] == 0.0


def test_threshold_is_echoed_back() -> None:
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.3)
    assert metrics["threshold"] == 0.3


def test_every_value_is_a_plain_float() -> None:
    """MLflow i JSON nie przyjmują np.float64 bez konwersji."""
    metrics = evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
    assert all(type(v) is float for v in metrics.values())


# --- Faza 3: PR-AUC, macierz pomyłek, koszt (spec D1, D4) -------------------

_Y_TRUE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
_Y_PROBA = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.2, 0.9, 0.95])
"""Dziesięć wierszy, trzy pozytywy. Przy progu 0.5 dodatnio typowane są
0.6, 0.7, 0.8 (negatywy) oraz 0.9, 0.95 (pozytywy)."""


def test_evaluate_reports_confusion_counts_that_sum_to_n() -> None:
    """tn+fp+fn+tp musi równać się liczbie wierszy — inaczej gubimy próbki."""
    m = evaluate(_Y_TRUE, _Y_PROBA, threshold=0.5)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 10.0
    assert m["tp"] == 2.0
    assert m["fp"] == 3.0
    assert m["fn"] == 1.0
    assert m["tn"] == 4.0


def test_evaluate_expected_cost_is_hand_computable() -> None:
    """10*FN + 1*FP — arytmetyka, nie magia."""
    m = evaluate(_Y_TRUE, _Y_PROBA, threshold=0.5)
    assert m["expected_cost"] == 10.0 * 1 + 1.0 * 3


def test_evaluate_expected_cost_follows_injected_costs() -> None:
    """Koszty są parametrem, a nie zaszytą stałą."""
    m = evaluate(_Y_TRUE, _Y_PROBA, threshold=0.5, cost_fn=3.0, cost_fp=2.0)
    assert m["expected_cost"] == 3.0 * 1 + 2.0 * 3


def test_evaluate_reports_pr_auc() -> None:
    """Idealne uszeregowanie daje PR-AUC 1.0."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluate(y_true, y_proba)["pr_auc"] == 1.0


def test_pr_auc_is_harsher_than_roc_auc_on_imbalance() -> None:
    """Powód wyboru PR-AUC na metrykę championa (spec D1).

    Przy rzadkiej klasie dodatniej ROC-AUC jest zawyżane przez łatwą
    większość; PR-AUC patrzy tylko na to, co dzieje się przy granicy.
    """
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.05, size=2000)
    y_proba = np.clip(rng.normal(0.05, 0.05, 2000) + 0.15 * y_true, 0.001, 0.999)
    m = evaluate(y_true, y_proba)
    assert m["pr_auc"] < m["roc_auc"]


def test_evaluate_survives_degenerate_predictions() -> None:
    """Przy wysokim progu model nie wskazuje ani jednego pozytywu.

    confusion_matrix bez labels=[0, 1] zwróciłby wtedy macierz 1x1, a .ravel()
    rzuciłby ValueError — to realna pułapka, nie hipotetyczna.
    """
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    m = evaluate(y_true, y_proba, threshold=0.99)
    assert m["tp"] == 0.0
    assert m["fn"] == 2.0
    assert m["recall"] == 0.0


# --- Faza 3: wybór progu decyzyjnego (spec D4) ------------------------------


def test_threshold_boundary_is_inclusive() -> None:
    """Prawdopodobieństwo równe progowi klasyfikuje POZYTYWNIE (`>=`, nie `>`).

    Gdyby próg był wyłączny, wiersz o prawdopodobieństwie dokładnie równym
    progowi wypadałby po innej stronie decyzji niż ta, którą zmierzyliśmy
    przy wyborze progu — i metryki opisywałyby inny model niż serwowany.
    """
    metrics = evaluate(np.array([0, 1]), np.array([0.3, 0.7]), threshold=0.7)
    assert metrics["tp"] == 1.0


def _separable(seed: int, n: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Etykiety i prawdopodobieństwa z realnym, ale niepełnym sygnałem."""
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, rate, size=n)
    y_proba = np.clip(rng.normal(rate, 0.08, n) + 0.25 * y_true, 0.001, 0.999)
    return y_true, y_proba


def test_choose_threshold_falls_when_missing_defaults_costs_more() -> None:
    """Im droższy FN, tym niżej trzeba postawić próg."""
    y_true, y_proba = _separable(42, 2000, 0.1)
    cheap_t, _ = choose_threshold(y_true, y_proba, cost_fn=2.0, cost_fp=1.0)
    dear_t, _ = choose_threshold(y_true, y_proba, cost_fn=50.0, cost_fp=1.0)
    assert dear_t < cheap_t


def test_choose_threshold_returns_the_cost_it_minimised() -> None:
    """Zwrócony koszt musi zgadzać się z tym, co evaluate liczy przy tym progu."""
    y_true, y_proba = _separable(7, 500, 0.2)
    threshold, cost = choose_threshold(y_true, y_proba)
    assert cost == evaluate(y_true, y_proba, threshold=threshold)["expected_cost"]


def test_choose_threshold_beats_the_default_half() -> None:
    """Cały sens fazy: 0.5 jest kosztowo gorsze niż próg dobrany do kosztów."""
    y_true, y_proba = _separable(1, 3000, 0.08)
    _, cost = choose_threshold(y_true, y_proba)
    assert cost <= evaluate(y_true, y_proba, threshold=0.5)["expected_cost"]


def test_choose_threshold_is_the_global_minimum() -> None:
    """Przemiatanie ma znaleźć minimum, a nie pierwsze lepsze miejsce."""
    y_true, y_proba = _separable(3, 400, 0.15)
    threshold, cost = choose_threshold(y_true, y_proba)
    for candidate in np.unique(y_proba):
        assert (
            cost
            <= evaluate(y_true, y_proba, threshold=float(candidate))["expected_cost"]
        )


def test_choose_threshold_handles_tied_probabilities() -> None:
    """Przy remisach `>=` bierze całą grupę albo nic — próg musi to szanować.

    Trzy wiersze mają dokładnie 0.4. Zwrócony próg musi dać koszt faktycznie
    osiągalny, a nie taki, który zakłada przecięcie remisu w środku.
    """
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.4, 0.4, 0.4, 0.9, 0.1, 0.8])
    threshold, cost = choose_threshold(y_true, y_proba)
    assert cost == evaluate(y_true, y_proba, threshold=threshold)["expected_cost"]
