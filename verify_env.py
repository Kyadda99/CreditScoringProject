"""Quick environment check.

Reports library versions, GPU availability and whether the dataset is present.
Exits non-zero when something required is missing, so it can be used in a
script or in CI. The neural-network data is reported but never fails the check.

Usage:
    uv run python verify_env.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path

# Optional libraries are reported as information, not as an error.
REQUIRED: tuple[str, ...] = ("pandas", "numpy")
OPTIONAL: tuple[str, ...] = (
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "lightgbm",
    "torch",
    "torchvision",
    "mlflow",
    "optuna",
    "fastapi",
)


def _version(dist: str) -> str | None:
    """Return the installed distribution version, or None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _report_packages() -> bool:
    """Print library versions. Return False when a required one is missing."""
    ok = True
    print("Libraries")
    for dist in REQUIRED:
        version = _version(dist)
        if version is None:
            print(f"  [MISS] {dist:<20} required - run `uv sync`")
            ok = False
        else:
            print(f"  [ OK ] {dist:<20} {version}")
    for dist in OPTIONAL:
        version = _version(dist)
        status = (
            f"[ OK ] {dist:<20} {version}"
            if version
            else f"[  - ] {dist:<20} not needed yet"
        )
        print(f"  {status}")
    return ok


def _report_gpu() -> None:
    """Print GPU status. A missing torch is not an error."""
    print("\nGPU")
    if _version("torch") is None:
        print("  [  - ] torch not installed - GPU check skipped")
        return
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        print(f"  [ OK ] CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("  [ OK ] MPS (Apple Silicon)")
    else:
        print("  [ !! ] No accelerator - neural network training runs on CPU")


def _report_data() -> bool:
    """Print dataset status. Return False when the file is absent."""
    from config import DATA_PATH

    print("\nData")
    if not DATA_PATH.exists():
        print(f"  [MISS] {DATA_PATH}")
        print("         run: uv run cs-download")
        return False
    size_mb = DATA_PATH.stat().st_size / 1024**2
    print(f"  [ OK ] {DATA_PATH.name} ({size_mb:.1f} MB)")
    return True


def _report_nn_data(descriptions: Path, documents: Path) -> None:
    """Print whether the neural-network datasets are on disk."""
    print("\nNeural-network data")
    if descriptions.exists():
        size_mb = descriptions.stat().st_size / 1024**2
        print(f"  [ OK ] {descriptions.name} ({size_mb:.1f} MB)")
    else:
        print(f"  [  - ] {descriptions} - run: uv run cs-nn-download-text")
    images = sum(1 for _ in documents.rglob("*.png")) if documents.exists() else 0
    if images:
        print(f"  [ OK ] {documents.name}: {images} images")
    else:
        print(f"  [  - ] {documents} - run: uv run cs-nn-download-documents")


def main() -> int:
    """Run every check. Return the process exit code."""
    host = f"{platform.system()} {platform.machine()}"
    print(f"Python {platform.python_version()} on {host}\n")
    packages_ok = _report_packages()
    _report_gpu()
    data_ok = _report_data()
    from config import DESCRIPTIONS_PATH, DOCUMENTS_DIR

    _report_nn_data(DESCRIPTIONS_PATH, DOCUMENTS_DIR)

    if packages_ok and data_ok:
        print("\nEnvironment ready.")
        return 0
    print("\nEnvironment incomplete - see [MISS] above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
