"""Testy konfiguracji serwisu."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import Settings


def test_defaults_are_usable_without_any_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serwis musi wstać bez .env — każde pole ma sensowną wartość domyślną."""
    monkeypatch.delenv("CS_SCORE_THRESHOLD", raising=False)
    monkeypatch.delenv("CS_MODEL_PATH", raising=False)
    settings = Settings(_env_file=None)
    # None, nie 0.5: brak nadpisania znaczy "weź próg z artefaktu", żeby serwis
    # stosował tę samą regułę decyzyjną, przy której policzono zapisane metryki.
    assert settings.score_threshold is None
    assert isinstance(settings.model_path, Path)


def test_env_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CS_SCORE_THRESHOLD", "0.31")
    assert Settings(_env_file=None).score_threshold == 0.31


def test_threshold_outside_zero_one_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Próg poza [0, 1] to zawsze pomyłka — lepiej nie wstać niż cicho scorować."""
    monkeypatch.setenv("CS_SCORE_THRESHOLD", "1.5")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Środowisko zawiera setki zmiennych; nieznane nie mogą wywalać startu."""
    monkeypatch.setenv("CS_SOMETHING_UNRELATED", "x")
    assert Settings(_env_file=None).score_threshold is None
