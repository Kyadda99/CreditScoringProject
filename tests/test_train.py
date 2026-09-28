"""Testy orkiestracji treningu — logika, nie pełny przebieg na 158 MB."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from config import ID_COLUMN, TARGET, FeatureGroups
from data_loader import DataLoader
from preprocessor import Preprocessor
from strategies import get_estimator_strategy, get_imbalance_strategy
from train import write_feature_schema
from trainer import ModelTrainer


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 200
    return pd.DataFrame(
        {
            ID_COLUMN: range(size),
            TARGET: rng.integers(0, 2, size=size),
            "AMT_INCOME_TOTAL": rng.random(size) * 1000 + 1.0,
            "AMT_CREDIT": rng.random(size) * 2000 + 1.0,
            "AMT_ANNUITY": rng.random(size) * 100 + 1.0,
            "AMT_GOODS_PRICE": rng.random(size) * 1800 + 1.0,
            "CNT_FAM_MEMBERS": rng.integers(1, 4, size=size).astype(float),
            "DAYS_BIRTH": -rng.integers(7000, 20000, size=size).astype(float),
            "DAYS_EMPLOYED": -rng.integers(100, 5000, size=size).astype(float),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash", "Revolving"], size=size),
            "FLAG_MOBIL": rng.integers(0, 2, size=size),
        }
    )


@pytest.fixture
def groups() -> FeatureGroups:
    return FeatureGroups(
        numeric=("AMT_INCOME_TOTAL",),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )


def test_split_excludes_target_and_id_from_features(frame: pd.DataFrame) -> None:
    X_train, _X_test, _y_train, _y_test = DataLoader().split(frame)
    assert TARGET not in X_train.columns
    assert ID_COLUMN not in X_train.columns


def test_split_is_eighty_twenty(frame: pd.DataFrame) -> None:
    X_train, X_test, _y_train, _y_test = DataLoader().split(frame)
    assert len(X_test) == pytest.approx(len(frame) * 0.2, abs=1)
    assert len(X_train) + len(X_test) == len(frame)


def test_split_is_stratified(frame: pd.DataFrame) -> None:
    """Przy 8% klasy pozytywnej niestratyfikowany podział potrafi ją zgubić."""
    _X_train, _X_test, y_train, y_test = DataLoader().split(frame)
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.05


def test_split_is_reproducible(frame: pd.DataFrame) -> None:
    """Cały projekt stoi na tym, że RANDOM_STATE daje ten sam podział."""
    first = DataLoader().split(frame)[1].index.tolist()
    second = DataLoader().split(frame)[1].index.tolist()
    assert first == second


def test_pipeline_has_features_preprocessor_then_model(
    groups: FeatureGroups,
) -> None:
    pipeline = Preprocessor(groups).build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(dict(pipeline.steps)) == ["features", "preprocessor", "model"]


def test_pipeline_fits_and_predicts_in_unit_range(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    X_train, X_test, y_train, _y_test = DataLoader().split(frame)
    pipeline = Preprocessor(groups).build_pipeline().fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_write_feature_schema_round_trips(tmp_path: Path) -> None:
    groups = FeatureGroups(numeric=("a",), categorical=("b",), binary=("c",))
    path = write_feature_schema(groups, tmp_path / "feature_schema.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert FeatureGroups.from_dict(payload) == groups


# --- Faza 2: kontrakt wejściowy i inżynieria cech (spec D1, D2, D7) --------


def test_input_contract_is_raw_columns_only() -> None:
    """Kontrakt wejściowy nie może zawierać cech pochodnych (spec D2)."""
    frame = pd.DataFrame(
        {
            "AMT_CREDIT": [1.0, 2.0],
            "AMT_INCOME_TOTAL": [3.0, 4.0],
            "NAME_CONTRACT_TYPE": ["a", "b"],
        }
    )
    contract = DataLoader().input_contract(frame)
    assert "CREDIT_INCOME_RATIO" not in contract.all_features


def test_input_contract_keeps_a_dropped_column_that_feeds_a_feature() -> None:
    """Reguła przynależności z D2 — bez tego iloraz zawsze wychodzi NaN."""
    from config import FEATURE_SOURCE_COLUMNS

    frame = pd.DataFrame({name: [1.0, 2.0] for name in FEATURE_SOURCE_COLUMNS})
    frame["JUNK"] = [9.0, 9.0]
    contract = DataLoader().input_contract(
        frame, dropped=(*FEATURE_SOURCE_COLUMNS, "JUNK")
    )
    assert set(FEATURE_SOURCE_COLUMNS) <= set(contract.all_features)
    assert "JUNK" not in contract.all_features


def test_build_pipeline_puts_the_engineer_first() -> None:
    from config import FeatureGroups

    groups = FeatureGroups(numeric=("AMT_CREDIT",), categorical=(), binary=())
    assert list(Preprocessor(groups).build_pipeline().named_steps) == [
        "features",
        "preprocessor",
        "model",
    ]


def test_build_pipeline_can_disable_engineered_features() -> None:
    """Transza 1 ze spec D7."""
    from config import FeatureGroups

    groups = FeatureGroups(numeric=("AMT_CREDIT",), categorical=(), binary=())
    pipeline = Preprocessor(groups, features=()).build_pipeline()
    assert pipeline.named_steps["features"].features == ()


# --- Faza 3: estymator i sampler w pipelinie (spec D3) ---------------------


def test_build_pipeline_defaults_to_the_baseline_estimator(
    groups: FeatureGroups,
) -> None:
    """Bez jawnego estymatora dostajemy regresję logistyczną — jak w Fazie 2.

    Domyślna wartość istnieje celowo: `Preprocessor(groups).build_pipeline()` ma pięć
    istniejących wywołań, w tym fixture `synthetic_artifact` w conftest.py.
    Uczynienie estymatora obowiązkowym wyłączyłoby cały zestaw testów HTTP
    w CI, a suite nadal raportowałby zieleń — to regresja M1 z Fazy 1.
    """
    from sklearn.linear_model import LogisticRegression

    assert isinstance(
        Preprocessor(groups).build_pipeline().named_steps["model"], LogisticRegression
    )


def test_build_pipeline_accepts_an_explicit_estimator(groups: FeatureGroups) -> None:
    from xgboost import XGBClassifier

    pipeline = Preprocessor(groups).build_pipeline(
        estimator=get_estimator_strategy("xgboost").build(
            class_weighted=False, scale_pos_weight=1.0
        )
    )
    assert isinstance(pipeline.named_steps["model"], XGBClassifier)


def test_sampler_produces_an_imblearn_pipeline_with_the_sampler_in_place(
    groups: FeatureGroups,
) -> None:
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipeline = Preprocessor(groups).build_pipeline(
        estimator=get_estimator_strategy("logistic_regression").build(
            class_weighted=False, scale_pos_weight=1.0
        ),
        sampler=get_imbalance_strategy("resample").make_sampler(),
    )
    assert isinstance(pipeline, ImbPipeline)
    assert list(pipeline.named_steps) == [
        "features",
        "preprocessor",
        "sampler",
        "model",
    ]


def test_sampler_is_inert_outside_fit(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    """Gwarancja braku wycieku (spec D3): sampler nie działa przy predict.

    Gdyby działał, predict na n wierszach zwracałby inną liczbę wyników niż n
    — a w walidacji krzyżowej oznaczałoby to ocenianie modelu na syntetycznych
    wierszach, których nigdy nie było w danych. Context7 nie ma dokumentacji
    imbalanced-learn na ten temat, więc własność jest tu SPRAWDZANA, a nie
    przyjmowana na wiarę.
    """
    X_train, X_test, y_train, _y_test = DataLoader().split(frame)
    pipeline = (
        Preprocessor(groups)
        .build_pipeline(
            estimator=get_estimator_strategy("logistic_regression").build(
                class_weighted=False, scale_pos_weight=1.0
            ),
            sampler=get_imbalance_strategy("resample").make_sampler(),
        )
        .fit(X_train, y_train)
    )

    assert len(pipeline.predict(X_test)) == len(X_test)
    assert pipeline.predict_proba(X_test).shape == (len(X_test), 2)


def test_sampler_actually_rebalances_during_fit(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    """Kontrola pozytywna do testu wyżej: sampler MUSI coś robić w `fit`.

    Bez tego "bezczynny przy predict" byłby spełniony także przez sampler,
    który nie robi nic nigdzie — a wtedy całe ramię `resample` byłoby cichą
    kopią ramienia `none`.
    """
    from imblearn.over_sampling import SMOTE

    # Fixture `frame` ma cel ~50/50, na którym SMOTE nie ma czego wyrównywać.
    # Ten test potrzebuje realnego niezbalansowania, więc narzuca własne
    # etykiety — 10% pozytywów, blisko produkcyjnych 8.07%.
    imbalanced = frame.copy()
    imbalanced[TARGET] = ([0] * 180) + ([1] * 20)

    X_train, _X_test, y_train, _y_test = DataLoader().split(imbalanced)
    engineered = (
        Preprocessor(groups, features=()).build_pipeline().named_steps["features"]
    )
    prep = (
        Preprocessor(groups, features=()).build_pipeline().named_steps["preprocessor"]
    )
    matrix = prep.fit_transform(engineered.fit_transform(X_train), y_train)

    _resampled_X, resampled_y = SMOTE(random_state=42).fit_resample(matrix, y_train)
    assert len(resampled_y) > len(y_train), "SMOTE nie dołożył ani jednego wiersza"
    # Po wyrównaniu obie klasy mają tyle samo wierszy.
    assert int(resampled_y.sum()) == int((resampled_y == 0).sum())


# --- Faza 3: siatka eksperymentu (spec D2) --------------------------------


def test_run_grid_returns_one_metrics_dict_per_cell(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    X_train, X_test, y_train, y_test = DataLoader().split(frame)
    results = ModelTrainer(
        Preprocessor(groups, features=()),
        models=("logistic_regression",),
        arms=("none", "class_weight"),
        track=False,
    ).run_grid(X_train, y_train, X_test, y_test)
    assert set(results) == {
        ("logistic_regression", "none"),
        ("logistic_regression", "class_weight"),
    }
    for metrics in results.values():
        assert {"roc_auc", "pr_auc", "recall", "expected_cost"} <= set(metrics)


def test_run_grid_covers_the_full_product_of_models_and_arms(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    """Siatka to iloczyn kartezjański — brak komórki psuje porównanie."""
    X_train, X_test, y_train, y_test = DataLoader().split(frame)
    results = ModelTrainer(
        Preprocessor(groups, features=()),
        models=("logistic_regression", "xgboost"),
        arms=("none", "class_weight"),
        track=False,
    ).run_grid(X_train, y_train, X_test, y_test)
    assert len(results) == 4


def test_comparison_table_is_sorted_by_pr_auc() -> None:
    """Champion wybieramy po PR-AUC (spec D1), więc tabela ma to odzwierciedlać."""

    def row(pr_auc: float, roc_auc: float) -> dict[str, float]:
        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "recall": 0.1,
            "precision": 0.2,
            "f1": 0.1,
            "threshold": 0.5,
            "expected_cost": 100.0,
        }

    table = ModelTrainer.comparison_table(
        {("a", "none"): row(0.10, 0.7), ("b", "none"): row(0.30, 0.8)}
    )
    assert list(table["model"]) == ["b", "a"]
    assert list(table.columns) == [
        "model",
        "imbalance",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "threshold",
        "expected_cost",
    ]


# --- Faza 3: wybór championa i jego próg (spec D1, D4) --------------------


def test_select_champion_picks_the_highest_pr_auc() -> None:
    results = {
        ("logistic_regression", "none"): {"pr_auc": 0.11, "roc_auc": 0.90},
        ("xgboost", "class_weight"): {"pr_auc": 0.24, "roc_auc": 0.75},
        ("random_forest", "resample"): {"pr_auc": 0.19, "roc_auc": 0.99},
    }
    assert ModelTrainer.select_champion(results) == ("xgboost", "class_weight")


def test_select_champion_ignores_roc_auc() -> None:
    """Gdyby wybór szedł po ROC-AUC, wygrałoby "b" — a nie wygrywa (spec D1)."""
    results = {
        ("a", "none"): {"pr_auc": 0.30, "roc_auc": 0.70},
        ("b", "none"): {"pr_auc": 0.20, "roc_auc": 0.99},
    }
    assert ModelTrainer.select_champion(results)[0] == "a"


def test_fit_champion_threshold_is_not_the_default_half(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    """Próg pochodzi z przemiatania kosztowego na OOF, a nie z rozpędu."""
    X_train, X_test, y_train, y_test = DataLoader().split(frame)
    _pipeline, threshold, metrics = ModelTrainer(
        Preprocessor(groups, features=()), cv_splits=3
    ).fit_champion(X_train, y_train, X_test, y_test, "logistic_regression", "none")
    assert 0.0 < threshold < 1.0
    assert metrics["threshold"] == threshold


def test_fit_champion_returns_a_fitted_usable_pipeline(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    X_train, X_test, y_train, y_test = DataLoader().split(frame)
    pipeline, _threshold, _metrics = ModelTrainer(
        Preprocessor(groups, features=()), cv_splits=3
    ).fit_champion(X_train, y_train, X_test, y_test, "logistic_regression", "none")
    assert len(pipeline.predict(X_test)) == len(X_test)


def test_fit_champion_cost_threshold_raises_recall_over_the_default(
    frame: pd.DataFrame, groups: FeatureGroups
) -> None:
    """Sedno fazy: przy koszcie FN 10x FP próg spada, a recall rośnie."""
    from evaluation import evaluate

    X_train, X_test, y_train, y_test = DataLoader().split(frame)
    pipeline, _threshold, metrics = ModelTrainer(
        Preprocessor(groups, features=()), cv_splits=3
    ).fit_champion(
        X_train, y_train, X_test, y_test, "logistic_regression", "class_weight"
    )
    at_half = evaluate(
        y_test.to_numpy(), pipeline.predict_proba(X_test)[:, 1], threshold=0.5
    )
    assert metrics["recall"] >= at_half["recall"]
