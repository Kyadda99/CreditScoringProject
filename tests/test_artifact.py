"""Testy zapisu i odczytu artefaktu modelu."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from artifact import Bundle, load_bundle, save_bundle
from config import FeatureGroups
from preprocessing import build_preprocessor


@pytest.fixture
def groups() -> FeatureGroups:
    return FeatureGroups(
        numeric=("AMT_INCOME_TOTAL",),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [1000.0, 2000.0, 3000.0, 4000.0],
            "NAME_CONTRACT_TYPE": ["Cash", "Revolving", "Cash", "Cash"],
            "FLAG_MOBIL": [1, 1, 0, 1],
        }
    )


@pytest.fixture
def fitted_pipeline(groups: FeatureGroups, frame: pd.DataFrame) -> Pipeline:
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(groups)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return pipeline.fit(frame, np.array([0, 1, 0, 1]))


def test_round_trip_preserves_predictions(
    tmp_path: Path,
    fitted_pipeline: Pipeline,
    groups: FeatureGroups,
    frame: pd.DataFrame,
) -> None:
    path = save_bundle(
        fitted_pipeline, groups, groups, {"roc_auc": 0.5}, tmp_path / "m.joblib"
    )
    loaded = load_bundle(path)
    np.testing.assert_allclose(
        loaded.pipeline.predict_proba(frame), fitted_pipeline.predict_proba(frame)
    )


def test_round_trip_preserves_groups(
    tmp_path: Path, fitted_pipeline: Pipeline, groups: FeatureGroups
) -> None:
    """Sedno pakietu: grupy nie mogą rozjechać się z pipeline'em."""
    path = save_bundle(fitted_pipeline, groups, groups, {}, tmp_path / "m.joblib")
    assert load_bundle(path).groups == groups


def test_round_trip_preserves_metadata(
    tmp_path: Path, fitted_pipeline: Pipeline, groups: FeatureGroups
) -> None:
    metadata = {"roc_auc": 0.74, "rows_train": 246008}
    path = save_bundle(fitted_pipeline, groups, groups, metadata, tmp_path / "m.joblib")
    assert load_bundle(path).metadata == metadata


def test_save_creates_missing_parent_directories(
    tmp_path: Path, fitted_pipeline: Pipeline, groups: FeatureGroups
) -> None:
    path = save_bundle(
        fitted_pipeline, groups, groups, {}, tmp_path / "a" / "b" / "m.joblib"
    )
    assert path.exists()


def test_load_missing_file_raises_with_a_useful_message(tmp_path: Path) -> None:
    """Komunikat ma wskazywać, co uruchomić — to najczęstsza przyczyna."""
    with pytest.raises(FileNotFoundError, match="cs-train"):
        load_bundle(tmp_path / "nie-ma-mnie.joblib")


def test_bundle_is_immutable(
    tmp_path: Path, fitted_pipeline: Pipeline, groups: FeatureGroups
) -> None:
    path = save_bundle(fitted_pipeline, groups, groups, {}, tmp_path / "m.joblib")
    bundle = load_bundle(path)
    assert isinstance(bundle, Bundle)
    with pytest.raises((AttributeError, TypeError)):
        bundle.pipeline = None  # type: ignore[misc]


# --- Faza 2: dwa kontrakty w jednym artefakcie (spec D2) -------------------


def test_bundle_round_trips_both_contracts(tmp_path: Path) -> None:
    from sklearn.dummy import DummyClassifier

    model = FeatureGroups(numeric=("A", "RATIO"), categorical=(), binary=())
    inputs = FeatureGroups(numeric=("A", "B"), categorical=(), binary=())
    pipeline = Pipeline([("model", DummyClassifier())]).fit([[0.0], [1.0]], [0, 1])

    path = save_bundle(pipeline, model, inputs, {"roc_auc": 0.5}, tmp_path / "b.joblib")
    restored = load_bundle(path)

    assert restored.groups == model
    assert restored.input_groups == inputs


def test_the_two_contracts_are_allowed_to_differ(tmp_path: Path) -> None:
    """Sedno D2: model konsumuje cechy, których klient nie może przysłać."""
    from sklearn.dummy import DummyClassifier

    model = FeatureGroups(numeric=("RATIO",), categorical=(), binary=())
    inputs = FeatureGroups(numeric=("AMT_CREDIT",), categorical=(), binary=())
    pipeline = Pipeline([("model", DummyClassifier())]).fit([[0.0], [1.0]], [0, 1])

    path = save_bundle(pipeline, model, inputs, {}, tmp_path / "b.joblib")
    assert load_bundle(path).input_groups.numeric == ("AMT_CREDIT",)


def test_a_phase_1_artifact_fails_loudly(tmp_path: Path) -> None:
    """Stary artefakt bez kontraktu wejściowego musi wywalić się od razu."""
    import joblib

    path = tmp_path / "old.joblib"
    joblib.dump(
        {
            "pipeline": None,
            "feature_groups": {"numeric": [], "categorical": [], "binary": []},
            "metadata": {},
        },
        path,
    )
    with pytest.raises(KeyError, match="cs-train"):
        load_bundle(path)
