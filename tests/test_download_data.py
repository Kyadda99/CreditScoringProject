"""The shared kagglehub wrapper: routing, cache location, credential hint."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import download_data
from download_data import kaggle_download


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    seen: list[tuple[str, str, str]] = []
    fake = SimpleNamespace(
        dataset_download=lambda handle, path: seen.append(("dataset", handle, path))
        or "dataset.csv",
        competition_download=lambda handle, path: seen.append(
            ("competition", handle, path)
        )
        or "competition.csv",
    )
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    monkeypatch.setattr(download_data, "load_dotenv", lambda *a, **k: None)
    # The code under test writes os.environ; a copy keeps that out of later tests.
    environ = {k: v for k, v in os.environ.items() if k != "KAGGLEHUB_CACHE"}
    monkeypatch.setattr(os, "environ", environ)
    return seen


def test_dataset_and_competition_are_routed(calls: list[tuple[str, str, str]]) -> None:
    assert kaggle_download("owner/set", "a.csv") == Path("dataset.csv")
    assert kaggle_download("comp", "b.csv", competition=True) == Path("competition.csv")
    assert calls == [
        ("dataset", "owner/set", "a.csv"),
        ("competition", "comp", "b.csv"),
    ]


def test_cache_dir_becomes_the_default_cache(
    calls: list[tuple[str, str, str]], tmp_path: Path
) -> None:
    kaggle_download("owner/set", "a.csv", cache_dir=tmp_path)
    assert os.environ["KAGGLEHUB_CACHE"] == str(tmp_path)


def test_a_cache_set_in_dotenv_wins(
    calls: list[tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        download_data,
        "load_dotenv",
        lambda *a, **k: monkeypatch.setenv("KAGGLEHUB_CACHE", "from-dotenv"),
    )
    kaggle_download("owner/set", "a.csv", cache_dir=tmp_path)
    assert os.environ["KAGGLEHUB_CACHE"] == "from-dotenv"


def test_no_cache_dir_leaves_the_cache_alone(
    calls: list[tuple[str, str, str]],
) -> None:
    kaggle_download("owner/set", "a.csv")
    assert "KAGGLEHUB_CACHE" not in os.environ


def test_failure_exits_with_every_credential_option(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _fail(handle: str, path: str) -> str:
        raise RuntimeError("403")

    monkeypatch.setitem(
        sys.modules, "kagglehub", SimpleNamespace(dataset_download=_fail)
    )
    monkeypatch.setattr(download_data, "load_dotenv", lambda *a, **k: None)
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        kaggle_download("owner/set", "a.csv", hint="Extra advice.")
    text = caplog.text
    for option in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "kaggle.json",
        "Extra advice.",
    ):
        assert option in text
