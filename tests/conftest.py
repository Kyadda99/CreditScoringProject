"""Shared pytest configuration.

The data contract tests need downloaded datasets, which CI does not have.
`@pytest.mark.requires_data(*paths)` skips a test when any named path is
missing; with no arguments it means the Home Credit CSV.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from api.cache import ScoreCache, model_version
from api.config import get_settings
from artifact import Bundle
from config import DATA_PATH, DESCRIPTIONS_PATH, DOCUMENTS_DIR

DOWNLOAD_COMMANDS: dict[Path, str] = {
    DATA_PATH: "cs-download",
    DESCRIPTIONS_PATH: "cs-nn-download-text",
    DOCUMENTS_DIR: "cs-nn-download-documents",
}


def missing_data(item: pytest.Item) -> list[Path]:
    """Return the `requires_data` paths of `item` that are not on disk."""
    marker = item.get_closest_marker("requires_data")
    if marker is None:
        return []
    paths = [Path(p) for p in marker.args] or [DATA_PATH]
    return [path for path in paths if not path.exists()]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked `requires_data` when a dataset they name is absent."""
    for item in items:
        missing = missing_data(item)
        if missing:
            commands = sorted(
                {DOWNLOAD_COMMANDS.get(p, "the download") for p in missing}
            )
            reason = f"Missing {', '.join(map(str, missing))}; run " + ", ".join(
                f"`uv run {c}`" for c in commands
            )
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session", autouse=True)
def _write_column_manifest() -> None:
    """Write the column manifest from the real file when CS_WRITE_MANIFEST=1.

    Run once after the first download; the result is committed, and from then on
    any column rename fails the contract test.
    """
    if os.environ.get("CS_WRITE_MANIFEST") != "1" or not DATA_PATH.exists():
        return

    from data import load_data
    from test_data_contract import MANIFEST_PATH

    columns = list(load_data().columns)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(columns, indent=2), encoding="utf-8")
    print(f"\nWrote a manifest of {len(columns)} columns -> {MANIFEST_PATH}")


@pytest.fixture(scope="session")
def synthetic_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny but real model artifact, without the 158 MB CSV.

    These HTTP tests once hung on the `requires_data` marker, so CI skipped them
    all and a regression in /score went green. The CI service stack serves the
    same artifact, so the two cannot drift.
    """
    from synthetic_artifact import build_synthetic_artifact

    return build_synthetic_artifact(
        tmp_path_factory.mktemp("artifact") / "pipeline.joblib"
    )


@pytest.fixture
def client(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Client that runs the lifespan; without `with` the model never loads.

    Points at the synthetic artifact from conftest.py, not a model trained on real
    data, so the whole HTTP set runs in CI.
    """
    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    # get_settings is lru_cached: without clearing it, settings from one test
    # would leak into the next.
    get_settings.cache_clear()
    with TestClient(app_module.app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def cached_client(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Client whose cache is backed by fakeredis rather than a real server."""
    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    get_settings.cache_clear()

    def fake_from_settings(settings: object, bundle: Bundle) -> ScoreCache:
        return ScoreCache(
            client=fakeredis.FakeAsyncRedis(),
            version=model_version(bundle),
            ttl_seconds=300,
        )

    monkeypatch.setattr(ScoreCache, "from_settings", fake_from_settings)
    with TestClient(app_module.app) as test_client:
        yield test_client
    get_settings.cache_clear()
