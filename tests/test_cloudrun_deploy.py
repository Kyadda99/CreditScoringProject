"""Behaviour of deploy/cloudrun/deploy.sh against a stub gcloud."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "deploy" / "cloudrun" / "deploy.sh"
MODEL_VERSION = PROJECT_ROOT / "deploy" / "model-version"
IMAGE = "europe-central2-docker.pkg.dev/p/credit-scoring/api:abc1234"
TAG = "sha-abc1234"
REVISION = "credit-scoring-api-00002-abc"
TAG_URL = "https://sha-abc1234---credit-scoring-api-xyz.a.run.app"
ENV = {
    "CS_GCP_REGION": "europe-central2",
    "CS_RUN_SERVICE": "credit-scoring-api",
    "CS_RUNTIME_SA": "cs-api-runtime@p.iam.gserviceaccount.com",
    "CS_MODEL_BUCKET": "p-cs-models",
}

STUB = f"""#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG"
case "$*" in
  "run services describe"*"metadata.name"*)
    case "$STUB_DESCRIBE" in
      exists) echo credit-scoring-api ;;
      missing) echo "ERROR: (gcloud.run.services.describe) \
Cannot find service [credit-scoring-api]" >&2; exit 1 ;;
      *) echo "ERROR: PERMISSION_DENIED: run.services.get denied" >&2; exit 1 ;;
    esac ;;
  "run services describe"*"latestCreatedRevisionName"*)
    echo credit-scoring-api-00003-out-of-band ;;
  "run services describe"*"--filter"*)
    echo "ERROR: (gcloud.run.services.describe) unrecognized arguments: \
--filter" >&2; exit 2 ;;
  "run services describe"*"status.traffic.tag"*)
    printf '\\tcredit-scoring-api-00001-aaa\\thttps://credit-scoring-api-xyz.a.run.app\\n'
    printf 'manual\\tcredit-scoring-api-00001-aaa\\thttps://manual---x.a.run.app\\n'
    printf '{TAG}\\t{REVISION}\\t{TAG_URL}\\n' ;;
  "run deploy"*) echo "Deploying... noise" ;;
esac
"""


def _bash() -> str:
    if sys.platform == "win32":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        if git_bash.exists():
            return str(git_bash)
        pytest.skip("Git Bash not found")
    found = shutil.which("bash")
    if found is None:
        pytest.skip("bash not found")
    return found


def _run(
    tmp_path: Path,
    describe: str = "exists",
    version_text: str = "1\n",
    unset: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "gcloud"
    stub.write_bytes(STUB.encode())
    stub.chmod(0o755)
    version_file = tmp_path / "model-version"
    version_file.write_bytes(version_text.encode())
    env = {
        **os.environ,
        **ENV,
        "STUB_LOG": (tmp_path / "calls.log").as_posix(),
        "STUB_DESCRIBE": describe,
        "CS_MODEL_VERSION_FILE": version_file.as_posix(),
    }
    for name in unset:
        env.pop(name)
    env["STUB_BIN"] = str(bin_dir)
    # Git Bash needs the Windows path converted before it goes on PATH.
    command = (
        'PATH="$(cygpath -u "$STUB_BIN" 2>/dev/null || echo "$STUB_BIN"):$PATH"; '
        'exec bash "$0" "$@"'
    )
    return subprocess.run(
        [_bash(), "-c", command, SCRIPT.as_posix(), IMAGE, TAG],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _deploy_call(tmp_path: Path) -> str:
    calls = [c for c in _calls(tmp_path) if c.startswith("run deploy")]
    assert len(calls) == 1, _calls(tmp_path)
    return calls[0]


def test_existing_service_gets_a_no_traffic_tagged_revision(tmp_path: Path) -> None:
    result = _run(tmp_path, describe="exists")
    assert result.returncode == 0, result.stderr
    call = _deploy_call(tmp_path)
    assert "--no-traffic" in call
    assert f"--tag {TAG}" in call
    assert f"--image {IMAGE}" in call
    assert "--allow-unauthenticated" not in call


def test_missing_service_is_created_public_with_traffic(tmp_path: Path) -> None:
    result = _run(tmp_path, describe="missing")
    assert result.returncode == 0, result.stderr
    call = _deploy_call(tmp_path)
    assert "--allow-unauthenticated" in call
    assert "--no-traffic" not in call


def test_describe_error_other_than_not_found_aborts(tmp_path: Path) -> None:
    result = _run(tmp_path, describe="denied")
    assert result.returncode != 0
    assert not any(c.startswith("run deploy") for c in _calls(tmp_path))


def test_model_path_comes_from_model_version_file(tmp_path: Path) -> None:
    result = _run(tmp_path, version_text="7")
    assert result.returncode == 0, result.stderr
    call = _deploy_call(tmp_path)
    assert "CS_MODEL_PATH=/models/7/pipeline.joblib" in call
    assert f"bucket={ENV['CS_MODEL_BUCKET']}" in call
    assert "readonly=true" in call
    assert "mount-path=/models" in call


@pytest.mark.parametrize("text", ["", "v1", "1.0", "0", "01"])
def test_malformed_model_version_aborts(tmp_path: Path, text: str) -> None:
    result = _run(tmp_path, version_text=text)
    assert result.returncode != 0
    assert _calls(tmp_path) == []


@pytest.mark.parametrize("text", ["3\n", "3\r\n", " 3 "])
def test_model_version_tolerates_trailing_newline_and_crlf(
    tmp_path: Path, text: str
) -> None:
    result = _run(tmp_path, version_text=text)
    assert result.returncode == 0, result.stderr
    assert "CS_MODEL_PATH=/models/3/pipeline.joblib" in _deploy_call(tmp_path)


def test_stdout_carries_only_revision_and_url(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"revision={REVISION}", f"url={TAG_URL}"]


def test_revision_comes_from_the_tagged_traffic_entry(tmp_path: Path) -> None:
    """Another deploy can become the latest revision between deploy and lookup."""
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"revision={REVISION}" in result.stdout.splitlines()
    assert not any("latestCreatedRevisionName" in c for c in _calls(tmp_path))


def test_missing_required_env_aborts_with_usage(tmp_path: Path) -> None:
    result = _run(tmp_path, unset=("CS_MODEL_BUCKET",))
    assert result.returncode == 64
    assert "CS_MODEL_BUCKET" in result.stderr
    assert _calls(tmp_path) == []


def test_committed_model_version_is_a_registry_version() -> None:
    assert re.fullmatch(r"[1-9][0-9]*", MODEL_VERSION.read_text().strip())
