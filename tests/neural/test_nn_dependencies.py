"""torch stays out of the scoring service's dependency set."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
TORCH = re.compile(r"^(torch|torchvision)\b", re.IGNORECASE)


def test_torch_is_not_a_project_dependency() -> None:
    deps = PYPROJECT["project"]["dependencies"]
    assert [d for d in deps if TORCH.match(d)] == []


def test_torch_lives_in_the_nn_group() -> None:
    nn_group = PYPROJECT["dependency-groups"]["nn"]
    assert {TORCH.match(d).group(1) for d in nn_group if TORCH.match(d)} == {
        "torch",
        "torchvision",
    }


# Settings that change what `uv export` resolves, or that uv refuses together
# with --frozen (CI sets UV_LOCKED). --frozen stays, so the lock is never
# rewritten by a test.
_UV_ENV_OVERRIDES = frozenset(
    {"UV_LOCKED", "UV_FROZEN", "UV_NO_GROUP", "UV_NO_DEFAULT_GROUPS", "UV_GROUP"}
)
_WHEEL = re.compile(
    r"^(torch|torchvision)==([\d.]+)(?:\+\w+)? ; sys_platform == '(linux|win32)'"
)


def _export(*args: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k not in _UV_ENV_OVERRIDES}
    result = subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_scoring_export_has_no_torch() -> None:
    lines = _export("--no-default-groups")
    assert [line for line in lines if TORCH.match(line)] == []


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_linux_resolves_the_cpu_wheel() -> None:
    lines = _export("--no-default-groups", "--group", "nn")
    linux = [
        line
        for line in lines
        if TORCH.match(line) and "sys_platform == 'linux'" in line
    ]
    assert len(linux) == 2
    assert all("+cpu" in line for line in linux)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_windows_and_linux_get_the_same_releases() -> None:
    versions: dict[str, dict[str, str]] = {"linux": {}, "win32": {}}
    for line in _export("--no-default-groups", "--group", "nn"):
        match = _WHEEL.match(line)
        if match:
            name, version, platform = match.groups()
            versions[platform][name] = version
    assert set(versions["linux"]) == {"torch", "torchvision"}
    assert versions["linux"] == versions["win32"]
