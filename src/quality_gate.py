"""Twarde progi, które model musi przejść, zanim trafi na alias @production.

Bramka jest **wywoływana** z `train.py` i `tune.py` przed `promote()`.
W projekcie referencyjnym instruktora ten moduł istnieje, ale nie jest
importowany nigdzie — obie ścieżki promocji wołają `promote()` bezwarunkowo.
To najłatwiejszy błąd do odziedziczenia przez skopiowanie struktury i dlatego
jest tu opisany wprost.

Nieudana bramka **nie jest wyjątkiem**: to normalny wynik biegu treningowego.
Bieg kończy się, loguje komplet naruszeń i po prostu nie promuje.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_ROC_AUC = 0.75
"""Bez regresji względem Fazy 2 (0.7515).

Model szeregujący ryzyko gorzej niż baseline nie ma po co jechać na produkcję,
niezależnie od tego, jak dobrze wypada na pozostałych metrykach.
"""

MIN_RECALL = 0.50
"""Powód istnienia całej fazy: baseline łapał 1.19% defaultów.

Przy koszcie FN = 10 x FP model, który przepuszcza połowę defaultów, jest
kosztowo nie do obrony — a to i tak próg łagodny.
"""

MIN_PRECISION = 0.12
"""1.5x bazowa częstość 8.07%.

Model oznaczający wnioski losowo trafiłby w okolice 0.08, więc ten próg
wymaga realnego sygnału, a nie samej liczby alarmów. Bez niego kryterium
recall dałoby się spełnić, oznaczając wszystkich.
"""

_CRITERIA: tuple[tuple[str, float], ...] = (
    ("roc_auc", MIN_ROC_AUC),
    ("recall", MIN_RECALL),
    ("precision", MIN_PRECISION),
)
"""Kryteria sprawdzane po kolei, wszystkie, bez short-circuitu."""


def quality_gate(metrics: dict[str, float]) -> bool:
    """Sprawdza, czy metryki pozwalają na promocję modelu.

    Args:
        metrics: Słownik z `evaluate()`. Klucze spoza `_CRITERIA` są ignorowane
            — bramka dostaje pełny kontrakt metryk i wybiera z niego swoje.
            Brakujący klucz jest traktowany jak wartość dyskwalifikująca:
            niekompletny pomiar to nie jest zgoda na promocję.

    Returns:
        `True`, jeśli **wszystkie** kryteria są spełnione.
    """
    passed = True
    for key, minimum in _CRITERIA:
        # -inf, nie 0: brak klucza ma przegrać z każdym progiem, także ujemnym.
        value = metrics.get(key, float("-inf"))
        if value >= minimum:
            logger.info("PRZYJĘTE  %-10s %.4f >= %.4f", key, value, minimum)
        else:
            logger.warning("ODRZUCONE %-10s %.4f <  %.4f", key, value, minimum)
            passed = False

    logger.info("Bramka jakości: %s", "PRZESZŁA" if passed else "ODRZUCIŁA MODEL")
    return passed


CHAMPION_METRIC = "pr_auc"
"""Metryka rozstrzygająca, kto jest championem (spec D1)."""


def beats_incumbent(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    metric: str = CHAMPION_METRIC,
) -> bool:
    """Sprawdza, czy kandydat jest lepszy od modelu obecnie na @production.

    Progi bezwzględne z `quality_gate` nie wystarczają: model może je spełnić
    i **jednocześnie** być gorszy od tego, który już jest na produkcji. Dokładnie
    to zdarzyło się w tej fazie — strojony XGBoost przeszedł wszystkie trzy
    kryteria i wyparł nietrojony model o wyższym PR-AUC (0.2564 vs 0.2626).

    Projekt referencyjny instruktora ma ten sam brak: `promote()` nadpisuje
    alias bezwarunkowo, bez porównania z poprzednikiem.

    Porównanie jest **ostre** (`>`): przy remisie zostaje ten, który już jest.
    Wymiana modelu produkcyjnego kosztuje rebuild obrazu i weryfikację, więc
    musi mieć uzasadnienie, a nie tylko brak przeciwwskazań.

    Args:
        candidate: Metryki kandydata z `evaluate()`.
        incumbent: Metryki modelu spod aliasu, albo `None`, gdy aliasu nie ma
            jeszcze wcale — wtedy kandydat wygrywa walkowerem.
        metric: Klucz metryki rozstrzygającej.

    Returns:
        `True`, jeśli kandydata należy promować.
    """
    if incumbent is None:
        logger.info("Brak modelu na @production — promuję kandydata walkowerem.")
        return True

    new = candidate.get(metric, float("-inf"))
    old = incumbent.get(metric, float("-inf"))
    if new > old:
        logger.info("PRZYJĘTE  %s %.4f >  %.4f (obecny champion)", metric, new, old)
        return True

    logger.warning(
        "ODRZUCONE %s %.4f <= %.4f (obecny champion) — alias bez zmian",
        metric,
        new,
        old,
    )
    return False
