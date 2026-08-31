"""Pobiera `application_train.csv` z konkursu Home Credit Default Risk.

Uruchamiane raz na maszynę; plik trafia do `data/raw/` (gitignored). Skrypt jest
idempotentny — jeśli plik już istnieje, nic nie pobiera.

Wymaga poświadczeń Kaggle. Konkurs wymaga **wcześniejszej akceptacji regulaminu
w przeglądarce**, w przeciwnym razie API zwróci 403 mimo poprawnych kluczy:
https://www.kaggle.com/competitions/home-credit-default-risk/rules

Usage:
    uv run cs-download
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMPETITION = "home-credit-default-risk"
FILENAME = "application_train.csv"
DEST_DIR = RAW_DATA_DIR
DEST = DEST_DIR / FILENAME


def download() -> Path:
    """Pobiera plik konkursowy do `data/raw/`, jeśli jeszcze go tam nie ma.

    Returns:
        Ścieżka do pobranego pliku CSV.

    Raises:
        SystemExit: gdy Kaggle odrzuci żądanie (brak kluczy lub nieakceptowany
            regulamin konkursu).
    """
    if DEST.exists():
        size_mb = DEST.stat().st_size / 1024**2
        logger.info(
            "Plik już istnieje: %s (%.1f MB) — pomijam pobieranie.", DEST, size_mb
        )
        return DEST

    import kagglehub

    logger.info("Pobieram %s z konkursu %s ...", FILENAME, COMPETITION)
    try:
        cached = kagglehub.competition_download(COMPETITION, path=FILENAME)
    except Exception as exc:
        logger.error("Pobieranie nie powiodło się: %s", exc)
        logger.error(
            "Sprawdź: (1) ~/.kaggle/kaggle.json lub zmienne "
            "KAGGLE_USERNAME/KAGGLE_KEY, (2) czy regulamin konkursu został "
            "zaakceptowany w przeglądarce."
        )
        raise SystemExit(1) from exc

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, DEST)
    size_mb = DEST.stat().st_size / 1024**2
    logger.info("Zapisano %s (%.1f MB).", DEST, size_mb)
    return DEST


def main() -> None:
    """Punkt wejścia CLI."""
    path = download()
    print(f"OK: {path}")


if __name__ == "__main__":
    sys.exit(main())
