"""from_settings factories and behaviour without optional configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import Settings
from artifact import load_bundle
from config import RANDOM_STATE
from data_loader import DataLoader
from predictor import Predictor
from settings import TrainingSettings


def test_training_settings_have_defaults_for_every_field() -> None:
    settings = TrainingSettings()
    assert settings.test_size == pytest.approx(0.2)
    assert "xgboost" in settings.models
    assert settings.track is True


def test_training_settings_read_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CS_CV_SPLITS", "3")
    monkeypatch.setenv("CS_DATA_PATH", "/tmp/other.csv")
    settings = TrainingSettings()
    assert settings.cv_splits == 3
    assert settings.data_path == Path("/tmp/other.csv")


def test_seed_is_not_configurable_so_it_cannot_half_apply() -> None:
    """The seed is single and project-wide.

    Making it configurable for the split alone would lie in the metadata: the
    estimators, the sampler and cross-validation all take it from `config`.
    """
    assert not hasattr(TrainingSettings(), "random_state")
    assert DataLoader().random_state == RANDOM_STATE


def test_data_loader_from_settings_uses_them() -> None:
    settings = TrainingSettings(test_size=0.3)
    loader = DataLoader.from_settings(settings)
    assert loader.test_size == pytest.approx(0.3)
    assert loader.path == settings.data_path


def test_threshold_precedence_and_degradation(synthetic_artifact: Path) -> None:
    bundle = load_bundle(synthetic_artifact)

    explicit = Predictor.from_settings(
        Settings(model_path=synthetic_artifact, score_threshold=0.1), bundle
    )
    assert explicit.threshold == pytest.approx(0.1)

    from_artifact = Predictor.from_settings(
        Settings(model_path=synthetic_artifact, score_threshold=None), bundle
    )
    assert from_artifact.threshold == pytest.approx(bundle.metadata["threshold"])


def test_missing_threshold_in_metadata_falls_back_to_default(
    synthetic_artifact: Path,
) -> None:
    bundle = load_bundle(synthetic_artifact)
    stripped = type(bundle)(
        pipeline=bundle.pipeline,
        groups=bundle.groups,
        input_groups=bundle.input_groups,
        metadata={},
    )
    predictor = Predictor.from_settings(
        Settings(model_path=synthetic_artifact, score_threshold=None), stripped
    )
    assert predictor.threshold == pytest.approx(0.5)
