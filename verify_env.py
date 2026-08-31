"""Szybki przegląd środowiska — uruchamiany na początku sesji.

Raportuje wersje bibliotek, dostępność GPU i obecność zbioru danych.
Kończy się kodem != 0, gdy czegoś krytycznego brakuje, żeby dało się go wpiąć
w skrypt czy CI.

Usage:
    uv run python verify_env.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys

# Biblioteki dochodzą fazami: pandas w Fazie 0, scikit-learn w Fazie 1,
# torch w Fazie 4. Brak tych późniejszych to informacja, nie błąd.
REQUIRED: tuple[str, ...] = ("pandas", "numpy")
OPTIONAL: tuple[str, ...] = (
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "lightgbm",
    "torch",
    "mlflow",
    "optuna",
    "fastapi",
)


def _version(dist: str) -> str | None:
    """Zwraca wersję zainstalowanej dystrybucji albo None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _report_packages() -> bool:
    """Wypisuje wersje bibliotek. Zwraca False, gdy brakuje wymaganej."""
    ok = True
    print("Biblioteki")
    for dist in REQUIRED:
        version = _version(dist)
        if version is None:
            print(f"  [BRAK] {dist:<20} wymagane — uruchom `uv sync`")
            ok = False
        else:
            print(f"  [ OK ] {dist:<20} {version}")
    for dist in OPTIONAL:
        version = _version(dist)
        status = (
            f"[ OK ] {dist:<20} {version}"
            if version
            else f"[  - ] {dist:<20} jeszcze niepotrzebne"
        )
        print(f"  {status}")
    return ok


def _report_gpu() -> None:
    """Wypisuje status GPU. Brak torcha nie jest błędem przed Fazą 4."""
    print("\nGPU")
    if _version("torch") is None:
        print("  [  - ] torch niezainstalowany — sprawdzenie GPU odłożone do Fazy 4")
        return
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        print(f"  [ OK ] CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("  [ OK ] MPS (Apple Silicon)")
    else:
        print("  [ !! ] Brak akceleratora — trening NN pójdzie na CPU")


def _report_data() -> bool:
    """Wypisuje status zbioru danych. Zwraca False, gdy pliku nie ma."""
    from config import DATA_PATH

    print("\nDane")
    if not DATA_PATH.exists():
        print(f"  [BRAK] {DATA_PATH}")
        print("         uruchom: uv run python scripts/download_data.py")
        return False
    size_mb = DATA_PATH.stat().st_size / 1024**2
    print(f"  [ OK ] {DATA_PATH.name} ({size_mb:.1f} MB)")
    return True


def main() -> int:
    """Uruchamia wszystkie kontrole. Zwraca kod wyjścia procesu."""
    host = f"{platform.system()} {platform.machine()}"
    print(f"Python {platform.python_version()} na {host}\n")
    packages_ok = _report_packages()
    _report_gpu()
    data_ok = _report_data()

    if packages_ok and data_ok:
        print("\nŚrodowisko gotowe.")
        return 0
    print("\nŚrodowisko niekompletne — zobacz [BRAK] powyżej.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
