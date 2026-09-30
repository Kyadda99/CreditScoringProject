"""Service configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import Settings


def test_defaults_are_usable_without_any_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service must start without .env; every field has a usable default."""
    monkeypatch.delenv("CS_SCORE_THRESHOLD", raising=False)
    monkeypatch.delenv("CS_MODEL_PATH", raising=False)
    settings = Settings(_env_file=None)
    # None, not 0.5: no override means take the threshold from the artifact, so
    # the service applies the rule the stored metrics were measured at.
    assert settings.score_threshold is None
    assert isinstance(settings.model_path, Path)


def test_env_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CS_SCORE_THRESHOLD", "0.31")
    assert Settings(_env_file=None).score_threshold == 0.31


def test_threshold_outside_zero_one_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A threshold outside [0, 1] is always a mistake; fail rather than score."""
    monkeypatch.setenv("CS_SCORE_THRESHOLD", "1.5")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment holds hundreds of variables; unknown ones must not fail."""
    monkeypatch.setenv("CS_SOMETHING_UNRELATED", "x")
    assert Settings(_env_file=None).score_threshold is None
