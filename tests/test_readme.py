"""Keeps the README runnable: its example request, links and commands."""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from api.schemas import ScoreRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
SMOKE = PROJECT_ROOT / "deploy" / "k8s" / "smoke.sh"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

PAYLOAD = re.compile(r"-d '(\{.*?\})'", re.DOTALL)
LINK = re.compile(r"\]\(([^)#\s]+)")


def _payload_after(text: str, marker: str) -> dict[str, Any]:
    """Returns the first `-d '{...}'` JSON body that follows `marker`."""
    start = text.index(marker)
    match = PAYLOAD.search(text, start)
    assert match is not None, f"no -d '{{...}}' payload after {marker!r}"
    return json.loads(match.group(1))


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.splitlines())


def test_readme_score_example_is_a_valid_score_request() -> None:
    ScoreRequest.model_validate(_payload_after(_readme(), "/score"))


def test_readme_score_example_matches_the_smoke_payload() -> None:
    readme = _payload_after(_readme(), "/score")
    smoke = _payload_after(SMOKE.read_text(encoding="utf-8"), "/score")
    assert readme == smoke


def test_readme_links_point_at_tracked_files() -> None:
    tracked = _tracked_files()
    targets = [t for t in LINK.findall(_readme()) if "://" not in t]
    missing = [
        t
        for t in targets
        if not any(
            path == t.rstrip("/") or path.startswith(t.rstrip("/") + "/")
            for path in tracked
        )
    ]
    assert targets, "README links to no repository files"
    assert missing == []


def test_readme_names_every_project_script() -> None:
    scripts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    readme = _readme()
    assert [name for name in scripts["scripts"] if name not in readme] == []
