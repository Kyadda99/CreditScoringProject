"""Pobiera `application_train.csv` z konkursu Home Credit Default Risk.

Uruchamiane raz na maszynę; plik trafia do `data/raw/` (gitignored). Skrypt jest
idempotentny — jeśli plik już istnieje, nic nie pobiera.

Wymaga poświadczeń Kaggle. Podaje się je na jeden z dwóch sposobów:
skopiować `.env.example` do `.env` i wypełnić `KAGGLE_USERNAME`/`KAGGLE_KEY`,
albo zostawić `~/.kaggle/kaggle.json` z witryny Kaggle. Zmienne z `.env` mają
pierwszeństwo — `kagglehub` sprawdza środowisko przed plikiem.

Konkurs wymaga dodatkowo **wcześniejszej akceptacji regulaminu w przeglądarce**,
w przeciwnym razie API zwróci 403 mimo poprawnych kluczy:
https://www.kaggle.com/competitions/home-credit-default-risk/rules

Usage:
    uv run cs-download
"""

from __future__ import annotations

import logging
import shutil
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from config import PROJECT_ROOT, RAW_DATA_DIR

# `kagglehub` czyta KAGGLE_USERNAME/KAGGLE_KEY ze środowiska procesu i nie wie
# nic o plikach `.env`. Bez tej linii poprawnie wypełniony `.env` byłby po cichu
# ignorowany, a skrypt padłby na "User is not authenticated" — czyli dokładnie
# tym samym błędem, co przy braku poświadczeń. Ścieżka jest liczona od
# PROJECT_ROOT, a nie od cwd, żeby `uv run cs-download` działało z podkatalogu.
# `override=False`: zmienne ustawione naprawdę w powłoce wygrywają, więc `.env`
# jest wygodą lokalną, a nie nadpisaniem środowiska CI.
load_dotenv(PROJECT_ROOT / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMPETITION = "home-credit-default-risk"
FILENAME = "application_train.csv"
DEST_DIR = RAW_DATA_DIR
DEST = DEST_DIR / FILENAME


def _materialise(cached: Path, dest: Path) -> None:
    """Kładzie prawdziwy CSV pod `dest`, rozpakowując go w razie potrzeby.

    `kagglehub.competition_download` dla pojedynczego pliku z konkursu zwraca
    **archiwum ZIP**, a nie CSV — mimo że ścieżka kończy się na `.csv`. Zwykłe
    skopiowanie dawało 36 MB archiwum pod nazwą `application_train.csv`;
    `pd.read_csv` wywracał się na nim daleko od przyczyny.

    Args:
        cached: Ścieżka zwrócona przez kagglehub — ZIP albo goły CSV.
        dest: Docelowa ścieżka CSV.

    Raises:
        SystemExit: Gdy archiwum nie zawiera oczekiwanego pliku.
    """
    if not zipfile.is_zipfile(cached):
        shutil.copy2(cached, dest)
        return

    with zipfile.ZipFile(cached) as archive:
        if FILENAME not in archive.namelist():
            logger.error(
                "Archiwum %s nie zawiera %s. Zawartość: %s",
                cached,
                FILENAME,
                archive.namelist(),
            )
            raise SystemExit(1)
        logger.info("Rozpakowuję %s z archiwum ...", FILENAME)
        # Strumieniowo, nie `archive.read()`: rozpakowany plik ma ~158 MB
        # i nie ma powodu trzymać go w całości w pamięci.
        with archive.open(FILENAME) as source, dest.open("wb") as target:
            shutil.copyfileobj(source, target)


def download() -> Path:
    """Pobiera plik konkursowy do `data/raw/`, jeśli jeszcze go tam nie ma.

    Returns:
        Ścieżka do pobranego pliku CSV.

    Raises:
        SystemExit: gdy Kaggle odrzuci żądanie (brak kluczy lub nieakceptowany
            regulamin konkursu).
    """
    # `zipfile.is_zipfile` zamiast samego `exists()`: wcześniejsza wersja tego
    # skryptu zapisywała archiwum ZIP pod nazwą `.csv`, a `exists()` uznawało je
    # za gotowe dane i pomijało pobieranie już na zawsze. Uszkodzony plik
    # przeżywał każde kolejne uruchomienie.
    if DEST.exists() and not zipfile.is_zipfile(DEST):
        size_mb = DEST.stat().st_size / 1024**2
        logger.info(
            "Plik już istnieje: %s (%.1f MB) — pomijam pobieranie.", DEST, size_mb
        )
        return DEST

    if DEST.exists():
        logger.warning("%s to archiwum ZIP, nie CSV — rozpakowuję ponownie.", DEST)

    import kagglehub

    logger.info("Pobieram %s z konkursu %s ...", FILENAME, COMPETITION)
    try:
        cached = kagglehub.competition_download(COMPETITION, path=FILENAME)
    except Exception as exc:
        logger.error("Pobieranie nie powiodło się: %s", exc)
        logger.error(
            "Sprawdź: (1) poświadczenia — KAGGLE_API_TOKEN albo para "
            "KAGGLE_USERNAME+KAGGLE_KEY w .env, albo ~/.kaggle/kaggle.json "
            "(szczegóły w .env.example), (2) czy regulamin konkursu został "
            "zaakceptowany w przeglądarce — bez tego API zwraca 403 mimo "
            "poprawnych kluczy."
        )
        raise SystemExit(1) from exc

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    _materialise(Path(cached), DEST)
    size_mb = DEST.stat().st_size / 1024**2
    logger.info("Zapisano %s (%.1f MB).", DEST, size_mb)
    return DEST


def main() -> None:
    """Punkt wejścia CLI."""
    path = download()
    print(f"OK: {path}")


if __name__ == "__main__":
    sys.exit(main())
