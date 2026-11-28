"""verify_env.py reports the neural-network data without failing on its absence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_env", PROJECT_ROOT / "verify_env.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_nn_data_is_reported_with_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _load()._report_nn_data(tmp_path / "descriptions.csv", tmp_path / "documents")
    out = capsys.readouterr().out
    assert "cs-nn-download-text" in out
    assert "cs-nn-download-documents" in out


def test_present_nn_data_is_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = tmp_path / "descriptions.csv"
    text.write_text("desc,purpose\n", encoding="utf-8")
    image = tmp_path / "documents" / "train" / "letter" / "a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    _load()._report_nn_data(text, tmp_path / "documents")
    out = capsys.readouterr().out
    assert "descriptions.csv" in out
    assert "1 images" in out


def test_main_runs_without_the_nn_group(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    for name in ("PIL", "torch", "torchvision", "nn.documents_data", "nn.device"):
        monkeypatch.setitem(sys.modules, name, None)
    real_version = module._version
    monkeypatch.setattr(
        module,
        "_version",
        lambda dist: None if dist in {"torch", "torchvision"} else real_version(dist),
    )
    module.main()
    assert "Neural-network data" in capsys.readouterr().out
