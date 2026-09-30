"""Download `application_train.csv` from the Home Credit Default Risk competition.

Run once per machine; the file lands in `data/raw/` (gitignored). The script is
idempotent: if the file already exists it downloads nothing.

Requires Kaggle credentials, either in `.env` or in `~/.kaggle/kaggle.json`.
Environment variables win: kagglehub checks the environment before the file.

The competition rules must also be accepted in a browser first, or the API
returns 403 even with valid keys:
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

# kagglehub reads KAGGLE_USERNAME/KAGGLE_KEY from the process environment
# and knows nothing about .env files. The path is resolved from PROJECT_ROOT,
# not cwd, so cs-download works from a subdirectory. override=False keeps real
# shell variables winning, so .env is a local convenience, not a CI override.
load_dotenv(PROJECT_ROOT / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMPETITION = "home-credit-default-risk"
FILENAME = "application_train.csv"
DEST_DIR = RAW_DATA_DIR
DEST = DEST_DIR / FILENAME


def _materialise(cached: Path, dest: Path) -> None:
    """Place the real CSV at `dest`, unpacking it when needed.

    For a single competition file `kagglehub.competition_download` returns a
    ZIP archive, not a CSV, even though the path ends in `.csv`. Copying it
    directly produced a 36 MB archive named `application_train.csv`, on which
    `pd.read_csv` failed far from the cause.

    Args:
        cached: Path returned by kagglehub: a ZIP or a bare CSV.
        dest: Destination CSV path.

    Raises:
        SystemExit: When the archive does not contain the expected file.
    """
    if not zipfile.is_zipfile(cached):
        shutil.copy2(cached, dest)
        return

    with zipfile.ZipFile(cached) as archive:
        if FILENAME not in archive.namelist():
            logger.error(
                "Archive %s does not contain %s. Contents: %s",
                cached,
                FILENAME,
                archive.namelist(),
            )
            raise SystemExit(1)
        logger.info("Extracting %s from the archive ...", FILENAME)
        # Streamed, not archive.read(): the unpacked file is ~158 MB.
        with archive.open(FILENAME) as source, dest.open("wb") as target:
            shutil.copyfileobj(source, target)


def download() -> Path:
    """Download the competition file to `data/raw/` if it is not there yet.

    Returns:
        Path to the downloaded CSV.

    Raises:
        SystemExit: When Kaggle rejects the request, from missing keys or
            unaccepted competition rules.
    """
    # is_zipfile rather than exists() alone: an earlier version of this script
    # saved a ZIP under a .csv name, and exists() then treated the corrupt file
    # as ready data and skipped the download on every later run.
    if DEST.exists() and not zipfile.is_zipfile(DEST):
        size_mb = DEST.stat().st_size / 1024**2
        logger.info(
            "File already exists: %s (%.1f MB); skipping download.", DEST, size_mb
        )
        return DEST

    if DEST.exists():
        logger.warning("%s is a ZIP archive, not a CSV; extracting again.", DEST)

    import kagglehub

    logger.info("Downloading %s from competition %s ...", FILENAME, COMPETITION)
    try:
        cached = kagglehub.competition_download(COMPETITION, path=FILENAME)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        logger.error(
            "Check: (1) credentials - KAGGLE_API_TOKEN, or the pair "
            "KAGGLE_USERNAME+KAGGLE_KEY, or ~/.kaggle/kaggle.json; "
            "(2) whether the competition rules were accepted in a browser; "
            "see .env.example. "
            "without which the API returns 403 even with valid keys."
        )
        raise SystemExit(1) from exc

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    _materialise(Path(cached), DEST)
    size_mb = DEST.stat().st_size / 1024**2
    logger.info("Saved %s (%.1f MB).", DEST, size_mb)
    return DEST


def main() -> None:
    """CLI entry point."""
    path = download()
    print(f"OK: {path}")


if __name__ == "__main__":
    sys.exit(main())
