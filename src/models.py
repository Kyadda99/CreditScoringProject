"""Nazwany rejestr modeli i wariantów obsługi niezbalansowania.

Faza 1 miała jeden wpis — regresję logistyczną, punkt odniesienia, wobec
którego mierzone są wszystkie późniejsze fazy. Faza 3 rozszerza rejestr do
trzech algorytmów i parametryzuje go ramieniem `imbalance` (spec D2), dzięki
czemu siatka w `train.py` jest podwójną pętlą, a nie dziewięcioma kopiami
tego samego bloku.

Wagi klas i resampling są **alternatywami**, nie dodatkami: ramię `resample`
zostawia wagi nietknięte, bo inaczej ten sam imbalans byłby policzony dwa
razy i porównanie obu strategii przestałoby cokolwiek znaczyć.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import RANDOM_STATE

IMBALANCE_ARMS: tuple[str, ...] = ("none", "class_weight", "resample")
"""Trzy ramiona eksperymentu z niezbalansowaniem (spec D2)."""

SCALE_POS_WEIGHT: float = 11.39
"""Domyślny stosunek negatywów do pozytywów przy 8.07% klasy dodatniej.

Wartość domyślna jest udokumentowaną stałą, żeby testy nie potrzebowały
zbioru danych; `train.py` nadpisuje ją stosunkiem policzonym z `y_train`,
bo gdyby higiena wierszy kiedyś zmieniła bazową częstość, model ma iść
za danymi, a nie za komentarzem.
"""


def _check_arm(imbalance: str) -> None:
    """Odrzuca nieznane ramię.

    Args:
        imbalance: Nazwa ramienia do sprawdzenia.

    Raises:
        ValueError: Gdy nazwa nie jest jednym z `IMBALANCE_ARMS`.
    """
    if imbalance not in IMBALANCE_ARMS:
        raise ValueError(
            f"Nieznane ramię niezbalansowania: {imbalance!r}. "
            f"Dozwolone: {', '.join(IMBALANCE_ARMS)}."
        )


def get_models(
    imbalance: str = "none",
    scale_pos_weight: float = SCALE_POS_WEIGHT,
) -> dict[str, BaseEstimator]:
    """Zwraca modele bazowe dla wskazanego ramienia, każdy niedopasowany.

    Args:
        imbalance: Jedno z `IMBALANCE_ARMS`.
        scale_pos_weight: Stosunek klas dla XGBoost. Używany wyłącznie
            w ramieniu `class_weight`.

    Returns:
        Słownik nazwa -> świeża instancja estymatora. Instancje są tworzone
        przy każdym wywołaniu, żeby dopasowanie jednego modelu nie wpływało
        na drugi.

    Raises:
        ValueError: Gdy `imbalance` nie jest znanym ramieniem.
    """
    _check_arm(imbalance)
    weighted = imbalance == "class_weight"
    return {
        # max_iter=1000 zamiast domyślnych 100: po one-hot mamy 211 kolumn
        # i lbfgs nie zbiega w 100 iteracjach — sklearn ostrzega, a model
        # jest cicho niedouczony.
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced" if weighted else None,
        ),
        # max_depth i min_samples_leaf ograniczone świadomie: 246k wierszy
        # bez ograniczeń daje las, który trenuje się długo, waży setki MB
        # i przeucza się na szumie rzadkiej klasy.
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced" if weighted else None,
        ),
        # tree_method="hist": splity po histogramie, wielokrotnie szybsze na
        # 246k x 211 niż domyślny exact. eval_metric="aucpr" spina metrykę
        # wewnętrzną XGBoosta z tą, którą wybieramy championa (spec D1).
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight if weighted else 1.0,
        ),
    }


def get_sampler(imbalance: str) -> SMOTE | None:
    """Zwraca sampler dla ramienia `resample`, w pozostałych `None`.

    Sampler trafia do `imblearn.pipeline.Pipeline` **za** preprocessorem —
    SMOTE potrzebuje macierzy numerycznej, więc nie może stać przed
    one-hot encodingiem (spec D3).

    Args:
        imbalance: Jedno z `IMBALANCE_ARMS`.

    Returns:
        Świeża instancja `SMOTE` albo `None`.

    Raises:
        ValueError: Gdy `imbalance` nie jest znanym ramieniem.
    """
    _check_arm(imbalance)
    # Bez n_jobs: parametr został usunięty z SMOTE w imbalanced-learn 0.12.
    return SMOTE(random_state=RANDOM_STATE) if imbalance == "resample" else None
