"""Tests for the MLflow model registry wrapper.

Everything runs on a temporary SQLite database in `tmp_path`. The registry
needs a database backend, but not the dataset, so these tests run in CI.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import registry as registry_module
from registry import (
    BUNDLE_ARTIFACT_PATH,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    export_champion,
    list_models,
    load_production,
    promote,
    register,
)

_X = np.array([[0.0], [1.0], [2.0], [3.0]])
_Y = np.array([0, 0, 1, 1])


@pytest.fixture()
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """An MLflow backend isolated to a single test."""
    uri = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setattr(registry_module, "MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-registry")
    return uri


def _logged_run() -> str:
    """Log a trivial sklearn model and return its run_id."""
    model = LogisticRegression().fit(_X, _Y)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name="model")
        return run.info.run_id


def _logged_run_with_bundle(tmp_path: Path) -> str:
    """A run holding both the sklearn model and the Bundle artifact."""
    from artifact import save_bundle
    from config import FeatureGroups

    model = LogisticRegression().fit(_X, _Y)
    groups = FeatureGroups(numeric=("a",), categorical=(), binary=())

    local = tmp_path / "pipeline.joblib"
    save_bundle(model, groups, groups, {"threshold": 0.09, "pr_auc": 0.42}, local)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name="model")
        mlflow.log_artifact(str(local), artifact_path=BUNDLE_ARTIFACT_PATH)
        return run.info.run_id


def test_register_creates_a_version(tracking: str) -> None:
    mv = register(_logged_run())
    assert mv.name == REGISTERED_MODEL_NAME
    assert int(mv.version) >= 1


def test_promote_then_load_returns_a_usable_model(tracking: str) -> None:
    """Full cycle: register, set the alias, load by alias, predict."""
    promote(register(_logged_run()))
    loaded = load_production()
    assert loaded.predict(np.array([[3.0]]))[0] in (0, 1)


def test_promote_moves_the_alias_to_the_newer_version(tracking: str) -> None:
    """The @production alias points at exactly one model, the last promoted."""
    from mlflow import MlflowClient

    promote(register(_logged_run()))
    second = register(_logged_run())
    promote(second)

    mv = MlflowClient(tracking_uri=tracking).get_model_version_by_alias(
        REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
    )
    assert mv.version == second.version


def test_load_production_without_an_alias_fails_loudly(tracking: str) -> None:
    """A missing champion is an error, not a silent `None`."""
    with pytest.raises(Exception):  # noqa: B017
        load_production()


def test_list_models_logs_the_alias(
    tracking: str, caplog: pytest.LogCaptureFixture
) -> None:
    promote(register(_logged_run()))
    caplog.set_level("INFO")
    list_models()
    assert any("@production" in r.getMessage() for r in caplog.records)


# --- Exporting the champion to the container ---


def test_export_champion_writes_a_loadable_bundle(
    tracking: str, tmp_path: Path
) -> None:
    from artifact import load_bundle

    promote(register(_logged_run_with_bundle(tmp_path)))
    dest = tmp_path / "exported.joblib"
    export_champion(dest=dest)

    assert load_bundle(dest).metadata["threshold"] == 0.09


def test_export_champion_stamps_registry_provenance(
    tracking: str, tmp_path: Path
) -> None:
    """The exported joblib must record which registry version it came from."""
    from artifact import load_bundle

    mv = register(_logged_run_with_bundle(tmp_path))
    promote(mv)
    dest = tmp_path / "exported.joblib"
    export_champion(dest=dest)

    metadata = load_bundle(dest).metadata
    assert metadata["registry_version"] == mv.version
    assert metadata["registry_alias"] == PRODUCTION_ALIAS
    assert metadata["registry_run_id"] == mv.run_id
    assert metadata["registry_model"] == REGISTERED_MODEL_NAME


def test_export_champion_preserves_both_column_contracts(
    tracking: str, tmp_path: Path
) -> None:
    """The export must not lose input_groups, or the container cannot start."""
    from artifact import load_bundle

    promote(register(_logged_run_with_bundle(tmp_path)))
    dest = tmp_path / "exported.joblib"
    export_champion(dest=dest)

    bundle = load_bundle(dest)
    assert bundle.input_groups.numeric == ("a",)
    assert bundle.groups.numeric == ("a",)


def test_export_champion_follows_the_alias_not_the_latest_version(
    tracking: str, tmp_path: Path
) -> None:
    """A newer, unpromoted version must not displace the champion.

    This is the point of the quality gate: a model that did not pass stays in
    MLflow as a run and has no right to reach the container.
    """
    from artifact import load_bundle

    promoted = register(_logged_run_with_bundle(tmp_path))
    promote(promoted)
    register(_logged_run_with_bundle(tmp_path))  # newer version, not promoted

    dest = tmp_path / "exported.joblib"
    export_champion(dest=dest)
    assert load_bundle(dest).metadata["registry_version"] == promoted.version


def test_log_model_handles_our_custom_pipeline_types(tracking: str) -> None:
    """A pipeline with FeatureEngineer and XGBoost must log to MLflow.

    Regression: MLflow 3.15 serializes sklearn through skops by default, which
    refuses classes outside its trusted list and fails on this pipeline:

        UntrustedTypesFoundException: ['features.FeatureEngineer',
        'numpy.dtype', 'xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']

    The quality gate passed and nothing was registered; the failure appeared only
    on real data, after a full training run. This reproduces it on 240 rows.
    """
    import numpy as np
    import pandas as pd

    from config import ID_COLUMN, TARGET, FeatureGroups
    from preprocessor import Preprocessor
    from strategies import get_estimator_strategy

    rng = np.random.default_rng(42)
    n = 240
    target = pd.Series(rng.binomial(1, 0.25, n), name=TARGET)
    frame = pd.DataFrame(
        {
            ID_COLUMN: range(n),
            "AMT_INCOME_TOTAL": rng.random(n) * 1000 + 1.0,
            "AMT_CREDIT": rng.random(n) * 2000 + 1.0,
            "AMT_ANNUITY": rng.random(n) * 100 + 1.0,
            "AMT_GOODS_PRICE": rng.random(n) * 1800 + 1.0,
            "CNT_FAM_MEMBERS": rng.integers(1, 4, size=n).astype(float),
            "DAYS_BIRTH": -rng.integers(7000, 20000, size=n).astype(float),
            "DAYS_EMPLOYED": -rng.integers(100, 5000, size=n).astype(float),
        }
    ).drop(columns=[ID_COLUMN])
    groups = FeatureGroups(
        numeric=("AMT_INCOME_TOTAL", "AMT_CREDIT"), categorical=(), binary=()
    )

    # A pipeline with this project's transformer and XGBoost: exactly the
    # composition the real run failed on.
    pipeline = (
        Preprocessor(groups)
        .build_pipeline(
            estimator=get_estimator_strategy("xgboost").build(
                class_weighted=False, scale_pos_weight=1.0
            )
        )
        .fit(frame, target)
    )

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            pipeline,
            name=registry_module.MODEL_ARTIFACT_PATH,
            serialization_format=registry_module.SERIALIZATION_FORMAT,
        )
        run_id = run.info.run_id

    promote(register(run_id))
    loaded = load_production()
    assert len(loaded.predict(frame)) == n


def test_production_metrics_is_none_before_any_promotion(tracking: str) -> None:
    """The first champion has nothing to compare against; that is not an error."""
    from registry import production_metrics

    assert production_metrics() is None


def test_production_metrics_reads_the_promoted_runs_metrics(tracking: str) -> None:
    from registry import production_metrics

    model = LogisticRegression().fit(_X, _Y)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name="model")
        mlflow.log_metrics({"pr_auc": 0.2626, "roc_auc": 0.7684})
        run_id = run.info.run_id
    promote(register(run_id))

    metrics = production_metrics()
    assert metrics is not None
    assert metrics["pr_auc"] == 0.2626


def test_production_metrics_follows_the_alias(tracking: str) -> None:
    """After the alias moves, the comparison uses the new champion."""
    from registry import production_metrics

    versions = []
    for pr_auc in (0.20, 0.30):
        model = LogisticRegression().fit(_X, _Y)
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, name="model")
            mlflow.log_metrics({"pr_auc": pr_auc})
            versions.append(register(run.info.run_id))

    promote(versions[0])
    assert production_metrics()["pr_auc"] == 0.20
    promote(versions[1])
    assert production_metrics()["pr_auc"] == 0.30
