"""The `requires_data` marker skips on exactly the paths a test names."""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.mark.structures import Mark

from config import DATA_PATH
from conftest import missing_data


class _Item:
    def __init__(self, mark: Mark | None) -> None:
        self._mark = mark

    def get_closest_marker(self, name: str) -> Mark | None:
        return self._mark if name == "requires_data" else None


def test_unmarked_test_needs_nothing() -> None:
    assert missing_data(_Item(None)) == []


def test_bare_marker_means_the_home_credit_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: False)
    assert missing_data(_Item(pytest.mark.requires_data.mark)) == [DATA_PATH]


def test_marker_arguments_replace_the_default(tmp_path: Path) -> None:
    present = tmp_path / "present.csv"
    present.write_text("x", encoding="utf-8")
    absent = tmp_path / "absent"
    mark = pytest.mark.requires_data(present, absent).mark
    assert missing_data(_Item(mark)) == [absent]
