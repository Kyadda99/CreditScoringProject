"""Trening i wybór championa.

Siatka algorytm x ramię niezbalansowania jedzie na jednym podziale, a próg
decyzyjny wyznaczamy dopiero dla championa, z predykcji out-of-fold na zbiorze
treningowym. Próg policzony na zbiorze testowym byłby dopasowaniem do testu.
"""

from __future__ import annotations

import logging
import time

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from config import COST_FN, COST_FP, RANDOM_STATE
from evaluation import DEFAULT_THRESHOLD, choose_threshold, evaluate
from preprocessor import Preprocessor
from settings import TrainingSettings
from strategies import (
    ESTIMATOR_STRATEGIES,
    IMBALANCE_STRATEGIES,
    get_estimator_strategy,
    get_imbalance_strategy,
)

logger = logging.getLogger(__name__)

ALL_MODELS: tuple[str, ...] = tuple(ESTIMATOR_STRATEGIES)
"""Algorytmy siatki — wprost z rejestru strategii, żeby nie było drugiej listy."""

ALL_ARMS: tuple[str, ...] = tuple(IMBALANCE_STRATEGIES)
"""Ramiona niezbalansowania — wprost z rejestru strategii."""

CV_SPLITS = 5
"""Foldy do wyznaczenia progu poza zbiorem testowym."""

TABLE_COLUMNS: tuple[str, ...] = (
    "model",
    "imbalance",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "threshold",
    "expected_cost",
)
"""Kolumny tabeli porównawczej, w kolejności raportowania."""


class ModelTrainer:
    """Prowadzi eksperyment: siatka, champion, próg.

    Attributes:
        preprocessor: Składacz potoku dla ustalonych grup cech.
        models: Algorytmy do przetrenowania.
        arms: Ramiona niezbalansowania.
        cv_splits: Liczba foldów przy wyznaczaniu progu.
        track: Czy logować biegi do MLflow.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        models: tuple[str, ...] = ALL_MODELS,
        arms: tuple[str, ...] = ALL_ARMS,
        cv_splits: int = CV_SPLITS,
        track: bool = True,
    ) -> None:
        """Zapamiętuje zakres eksperymentu.

        Args:
            preprocessor: Składacz potoku.
            models: Nazwy algorytmów.
            arms: Nazwy ramion niezbalansowania.
            cv_splits: Liczba foldów do predykcji out-of-fold.
            track: `False` wyłącza MLflow — używane w testach.
        """
        self.preprocessor = preprocessor
        self.models = models
        self.arms = arms
        self.cv_splits = cv_splits
        self.track = track

    @classmethod
    def from_settings(
        cls, settings: TrainingSettings, preprocessor: Preprocessor
    ) -> ModelTrainer:
        """Buduje trenera z konfiguracji.

        Args:
            settings: Ustawienia treningu.
            preprocessor: Składacz potoku dla grup cech.

        Returns:
            Trener o zakresie wziętym z konfiguracji.
        """
        return cls(
            preprocessor,
            models=settings.models,
            arms=settings.arms,
            cv_splits=settings.cv_splits,
            track=settings.track,
        )

    @staticmethod
    def positive_ratio(y: pd.Series) -> float:
        """Zwraca stosunek negatywów do pozytywów.

        Liczony z faktycznych etykiet, a nie ze stałej: gdyby higiena wierszy
        kiedyś zmieniła bazową częstość, model ma iść za danymi.

        Args:
            y: Etykiety 0/1.

        Returns:
            `neg / pos`.
        """
        positives = int(y.sum())
        return float((len(y) - positives) / positives)

    def _pipeline_for(self, name: str, arm: str, ratio: float) -> Pipeline:
        """Składa potok dla jednej komórki siatki.

        Args:
            name: Nazwa algorytmu.
            arm: Ramię niezbalansowania.
            ratio: Stosunek klas dla estymatorów ważących liczbą.

        Returns:
            Niedopasowany potok.
        """
        imbalance = get_imbalance_strategy(arm)
        estimator = get_estimator_strategy(name).build(
            class_weighted=imbalance.uses_class_weights,
            scale_pos_weight=ratio,
        )
        return self.preprocessor.build_pipeline(estimator, imbalance.make_sampler())

    def run_params(self, name: str, arm: str, ratio: float) -> dict[str, object]:
        """Buduje parametry biegu opisujące jedną komórkę siatki.

        Wagę bierzemy z tej samej strategii co model, a nie z porównania nazwy
        ramienia — inaczej zapis biegu mógłby opisywać inny model niż ten,
        który faktycznie powstał.

        Args:
            name: Nazwa algorytmu.
            arm: Ramię niezbalansowania.
            ratio: Stosunek negatywów do pozytywów.

        Returns:
            Słownik parametrów do zapisania przy biegu.
        """
        imbalance = get_imbalance_strategy(arm)
        return {
            "model": name,
            "imbalance": arm,
            "scale_pos_weight": ratio if imbalance.uses_class_weights else 1.0,
            "random_state": RANDOM_STATE,
            "n_features": len(self.preprocessor.groups.all_features),
            "n_engineered": len(self.preprocessor.features),
        }

    def run_grid(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Trenuje siatkę algorytm x ramię na jednym podziale.

        Każda komórka to jeden bieg MLflow. Metryki liczone są przy progu 0.5 —
        strojenie progu dotyczy dopiero championa, bo próg dobrany osobno dla
        każdej komórki uczyniłby kolumny tabeli nieporównywalnymi.

        Args:
            X_train: Cechy treningowe.
            y_train: Etykiety treningowe.
            X_test: Cechy testowe.
            y_test: Etykiety testowe.

        Returns:
            `{(nazwa_modelu, ramię): metryki}`.
        """
        ratio = self.positive_ratio(y_train)
        logger.info("scale_pos_weight z danych: %.2f", ratio)

        results: dict[tuple[str, str], dict[str, float]] = {}
        for arm in self.arms:
            for name in self.models:
                pipeline = self._pipeline_for(name, arm, ratio)

                logger.info("Trenuję %s / %s ...", name, arm)
                started = time.perf_counter()
                pipeline.fit(X_train, y_train)
                elapsed = time.perf_counter() - started

                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics = evaluate(
                    y_test.to_numpy(), y_proba, threshold=DEFAULT_THRESHOLD
                )
                metrics["fit_seconds"] = float(elapsed)
                results[(name, arm)] = metrics

                if self.track:
                    with mlflow.start_run(run_name=f"{name}__{arm}"):
                        mlflow.log_params(self.run_params(name, arm, ratio))
                        mlflow.log_metrics(metrics)

                logger.info(
                    "  %-20s %-13s PR-AUC=%.4f ROC-AUC=%.4f recall=%.4f (%.1fs)",
                    name,
                    arm,
                    metrics["pr_auc"],
                    metrics["roc_auc"],
                    metrics["recall"],
                    elapsed,
                )
        return results

    @staticmethod
    def comparison_table(
        results: dict[tuple[str, str], dict[str, float]],
    ) -> pd.DataFrame:
        """Zamienia wynik `run_grid` w tabelę porównawczą, malejąco po PR-AUC.

        Args:
            results: Wynik `run_grid`.

        Returns:
            Ramka z jedną linią na komórkę siatki.
        """
        rows = [
            {"model": model, "imbalance": arm, **{c: m[c] for c in TABLE_COLUMNS[2:]}}
            for (model, arm), m in results.items()
        ]
        return (
            pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
            .sort_values("pr_auc", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def select_champion(
        results: dict[tuple[str, str], dict[str, float]],
    ) -> tuple[str, str]:
        """Wybiera komórkę siatki o najwyższym PR-AUC.

        PR-AUC, nie ROC-AUC: przy 8.07% klasy dodatniej ROC-AUC jest zawyżane
        przez łatwą większość, a champion ma być dobry przy granicy decyzyjnej,
        bo to tam zapada decyzja kredytowa.

        Args:
            results: Wynik `run_grid`.

        Returns:
            `(nazwa_modelu, ramię)`.
        """
        model, arm = max(results, key=lambda key: results[key]["pr_auc"])
        logger.info(
            "Champion: %s / %s (PR-AUC %.4f)",
            model,
            arm,
            results[(model, arm)]["pr_auc"],
        )
        return model, arm

    def fit_champion(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        arm: str,
    ) -> tuple[Pipeline, float, dict[str, float]]:
        """Dopasowuje championa i wyznacza jego próg decyzyjny.

        Kolejność jest istotna:

        1. `cross_val_predict` na zbiorze **treningowym** daje prawdopodobieństwa
           out-of-fold — każdy wiersz oceniony przez model, który go nie widział.
        2. `choose_threshold` minimalizuje na nich koszt `10*FN + 1*FP`.
        3. Próg jedzie **niezmieniony** na zbiór testowy.

        Args:
            X_train: Cechy treningowe.
            y_train: Etykiety treningowe.
            X_test: Cechy testowe.
            y_test: Etykiety testowe.
            model_name: Nazwa algorytmu.
            arm: Ramię niezbalansowania.

        Returns:
            `(dopasowany pipeline, próg, metryki na teście przy tym progu)`.
        """
        ratio = self.positive_ratio(y_train)

        logger.info("Wyznaczam próg z predykcji OOF (%d foldów) ...", self.cv_splits)
        oof_proba = cross_val_predict(
            self._pipeline_for(model_name, arm, ratio),
            X_train,
            y_train,
            cv=StratifiedKFold(
                n_splits=self.cv_splits, shuffle=True, random_state=RANDOM_STATE
            ),
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold, oof_cost = choose_threshold(y_train.to_numpy(), oof_proba)
        logger.info(
            "Próg z OOF: %.4f (koszt %.0f). Analityczne optimum dla %.0f:1 to %.4f "
            "— różnica jest sygnałem kalibracji, nie błędem.",
            threshold,
            oof_cost,
            COST_FN / COST_FP,
            1.0 / (1.0 + COST_FN / COST_FP),
        )

        # Świeży potok: ten z cross_val_predict został dopasowany per fold,
        # a championa uczymy na całym zbiorze treningowym.
        pipeline = self._pipeline_for(model_name, arm, ratio)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), y_proba, threshold=threshold)
        return pipeline, threshold, metrics
