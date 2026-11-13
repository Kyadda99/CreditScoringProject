"""The built wheel ships every top-level module under src/."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_every_top_level_module_is_listed_in_py_modules() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    listed = set(pyproject["tool"]["setuptools"]["py-modules"])
    on_disk = {path.stem for path in (PROJECT_ROOT / "src").glob("*.py")}

    assert on_disk - listed == set()
