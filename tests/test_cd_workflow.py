"""Contract tests for the CD workflow that ships to Cloud Run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CD = PROJECT_ROOT / ".github" / "workflows" / "cd.yml"
CI = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return _load(CD)


@pytest.fixture(scope="module")
def job(workflow: dict[str, Any]) -> dict[str, Any]:
    return workflow["jobs"]["deploy"]


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML reads the bare key `on` as the boolean True.
    return workflow[True]


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [s for s in job["steps"] if s.get("name") == name]
    assert matches, f"no step named {name!r}"
    return matches[0]


def _index(job: dict[str, Any], name: str) -> int:
    return [s.get("name") for s in job["steps"]].index(name)


def test_cd_runs_only_after_a_green_ci_on_main(
    workflow: dict[str, Any], job: dict[str, Any]
) -> None:
    assert _triggers(workflow)["workflow_run"] == {
        "workflows": [_load(CI)["name"]],
        "types": ["completed"],
        "branches": ["main"],
    }
    assert "github.event.workflow_run.conclusion == 'success'" in job["if"]


def test_cd_ignores_ci_runs_from_pull_requests_and_forks(job: dict[str, Any]) -> None:
    """The branch filter matches a fork's own `main` too, so it is not enough."""
    assert "github.event.workflow_run.event == 'push'" in job["if"]
    assert (
        "github.event.workflow_run.head_repository.full_name == github.repository"
        in job["if"]
    )


def test_cd_checks_out_the_commit_ci_tested(job: dict[str, Any]) -> None:
    assert "github.event.workflow_run.head_sha" in job["env"]["SHA"]
    assert _step(job, "Checkout")["with"]["ref"] == "${{ env.SHA }}"


def test_cd_authenticates_with_wif_only(
    workflow: dict[str, Any], job: dict[str, Any]
) -> None:
    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}
    auth = _step(job, "Authenticate to Google Cloud")
    assert auth["uses"].startswith("google-github-actions/auth@")
    assert set(auth["with"]) == {"workload_identity_provider", "service_account"}
    assert all(v.startswith("${{ vars.") for v in auth["with"].values())
    text = CD.read_text(encoding="utf-8")
    assert "secrets." not in text
    assert "credentials_json" not in text


def test_cd_deploys_never_overlap(workflow: dict[str, Any]) -> None:
    concurrency = workflow["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert "github.ref" not in concurrency["group"]
    assert "run_id" not in concurrency["group"]


def test_image_is_tagged_with_the_commit_sha_only(job: dict[str, Any]) -> None:
    build = _step(job, "Build and push the image")
    assert build["with"]["push"] is True
    tags = str(build["with"]["tags"]).strip()
    assert "\n" not in tags and "," not in tags
    assert tags.endswith(":${{ env.SHA }}")
    assert "latest" not in tags


def test_cd_reuses_the_ci_layer_cache(job: dict[str, Any]) -> None:
    ci_build = _load(CI)["jobs"]["build"]["steps"]
    ci_cache_to = next(s["with"]["cache-to"] for s in ci_build if "with" in s)
    ci_scope = next(p for p in ci_cache_to.split(",") if p.startswith("scope="))
    cache_from = _step(job, "Build and push the image")["with"]["cache-from"]
    assert ci_scope in cache_from.split(",")


def test_traffic_moves_only_after_the_tagged_revision_passes_smoke(
    job: dict[str, Any],
) -> None:
    order = [
        "Build and push the image",
        "Deploy a tagged revision",
        "Smoke-test the tagged revision",
        "Move traffic to the revision",
        "Smoke-test the public URL",
    ]
    indices = [_index(job, name) for name in order]
    assert indices == sorted(indices)
    assert "deploy/cloudrun/deploy.sh" in _step(job, "Deploy a tagged revision")["run"]
    assert "--to-revisions" in _step(job, "Move traffic to the revision")["run"]
    for name in ("Move traffic to the revision", "Smoke-test the public URL"):
        assert "if" not in _step(job, name), f"{name} must not override failure()"


def test_smoke_waits_long_enough_for_a_cold_start(job: dict[str, Any]) -> None:
    assert int(job["env"]["SMOKE_TIMEOUT"]) >= 180


def test_failed_smoke_can_be_forced_for_rollback_proof(
    workflow: dict[str, Any], job: dict[str, Any]
) -> None:
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert inputs["force_smoke_failure"]["type"] == "boolean"
    assert inputs["force_smoke_failure"]["default"] is False
    assert "force_smoke_failure" in str(_step(job, "Smoke-test the tagged revision"))


def test_traffic_split_is_always_recorded(job: dict[str, Any]) -> None:
    step = _step(job, "Traffic split")
    assert step["if"].startswith("always()")
    assert _index(job, "Traffic split") == len(job["steps"]) - 1


def test_no_service_account_key_material_in_repo() -> None:
    this_file = Path(__file__).resolve()
    for root in (".github", "deploy", "src", "tests"):
        for path in (PROJECT_ROOT / root).rglob("*"):
            if not path.is_file() or path.resolve() == this_file:
                continue
            if path.suffix in {".pyc", ".joblib"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for needle in ("private_key", "credentials_json"):
                assert needle not in text, f"{needle} in {path}"


def test_cd_refuses_a_commit_that_is_no_longer_the_tip_of_main(
    job: dict[str, Any],
) -> None:
    """A re-run of an older green CI run must not roll production back."""
    guard = _step(job, "Refuse a commit that is not the tip of main")
    assert _index(job, "Checkout") < _index(job, guard["name"])
    assert _index(job, guard["name"]) < _index(job, "Authenticate to Google Cloud")
    assert "origin main" in guard["run"]
    assert "$SHA" in guard["run"]
    assert "exit 1" in guard["run"]
    assert "if" not in guard


def test_traffic_split_needs_a_successful_login(job: dict[str, Any]) -> None:
    auth = _step(job, "Authenticate to Google Cloud")
    split = _step(job, "Traffic split")
    assert split["if"] == f"always() && steps.{auth['id']}.outcome == 'success'"
