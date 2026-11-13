"""Contract tests for the CI job that deploys onto a kind cluster."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from load.thresholds import Thresholds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
LOAD_ENV = ("CS_LOAD_MAX_FAIL_RATIO", "CS_LOAD_MAX_P95_MS", "CS_LOAD_MIN_REQUESTS")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kind_job() -> dict[str, Any]:
    return _load(WORKFLOW)["jobs"]["kind"]


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [s for s in job["steps"] if s.get("name") == name]
    assert matches, f"no step named {name!r}"
    return matches[0]


def _minor(version: str) -> int:
    match = re.search(r"v1\.(\d+)", version)
    assert match, f"no v1.N version in {version!r}"
    return int(match.group(1))


def test_kind_job_waits_for_build_and_unit_tests(kind_job: dict[str, Any]) -> None:
    assert set(kind_job["needs"]) == {"lint-and-test", "build"}


def test_ci_kubectl_is_within_skew_of_the_node_image(
    kind_job: dict[str, Any],
) -> None:
    cluster = _load(PROJECT_ROOT / "deploy" / "kind" / "cluster.yaml")
    node_image = cluster["nodes"][0]["image"]
    kubectl = kind_job["env"]["KUBECTL_VERSION"]
    assert abs(_minor(kubectl) - _minor(node_image)) <= 1


def test_kind_job_gates_load_with_explicit_thresholds(
    kind_job: dict[str, Any],
) -> None:
    env = kind_job["env"]
    assert set(LOAD_ENV) <= env.keys()
    Thresholds.from_env({name: str(env[name]) for name in LOAD_ENV})


def test_kind_job_keeps_evidence_of_a_red_run(kind_job: dict[str, Any]) -> None:
    assert _step(kind_job, "Cluster diagnostics")["if"] == "failure()"
    assert _step(kind_job, "Upload Locust results")["if"] == "always()"


def test_kind_job_loads_the_tag_the_dev_overlay_deploys(
    kind_job: dict[str, Any],
) -> None:
    overlay = _load(
        PROJECT_ROOT / "deploy" / "k8s" / "overlays" / "dev" / "kustomization.yaml"
    )
    tag = overlay["images"][0]["newTag"]
    tags = _step(kind_job, "Load the serving image")["with"]["tags"]
    assert tags == f"credit-scoring/api:{tag}"


def test_executed_deploy_scripts_are_executable_in_git() -> None:
    """A checkout on Linux takes the mode from the index, not the working tree."""
    listing = subprocess.run(
        ["git", "ls-files", "-s", "--", "deploy/*.sh"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    modes = {line.split("\t")[1]: line.split()[0] for line in listing}
    sourced = {"deploy/kubectl.sh"}
    for path, mode in modes.items():
        if path not in sourced:
            assert mode == "100755", f"{path} is {mode}, not executable"
    assert "deploy/k8s/smoke.sh" in modes, "smoke.sh is not tracked"
