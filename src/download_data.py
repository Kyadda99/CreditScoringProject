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
import os
import shutil
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from config import PROJECT_ROOT, RAW_DATA_DIR

logger = logging.getLogger(__name__)

COMPETITION = "home-credit-default-risk"
FILENAME = "application_train.csv"
DEST_DIR = RAW_DATA_DIR
DEST = DEST_DIR / FILENAME

CREDENTIALS_HINT = (
    "Check credentials: KAGGLE_API_TOKEN, or the pair "
    "KAGGLE_USERNAME+KAGGLE_KEY, or ~/.kaggle/kaggle.json; see .env.example."
)
RULES_HINT = (
    "Also check that the competition rules were accepted in a browser, "
    "without which the API returns 403 even with valid keys."
)


def kaggle_download(
    handle: str,
    path: str,
    *,
    competition: bool = False,
    cache_dir: Path | None = None,
    hint: str = "",
) -> Path:
    """Download one file through kagglehub with the project's setup.

    kagglehub reads credentials and its cache location from the process
    environment and knows nothing about .env files, so `.env` is loaded first.
    The path is resolved from PROJECT_ROOT, not cwd, so the scripts work from
    a subdirectory. override=False keeps real shell variables winning, so .env
    is a local convenience, not a CI override.

    Args:
        handle: Dataset `owner/name`, or the competition name.
        path: File inside the dataset or competition.
        competition: Download from a competition instead of a dataset.
        cache_dir: Default kagglehub cache; a `KAGGLEHUB_CACHE` already set in
            the environment or `.env` wins. None keeps kagglehub's default.
        hint: Extra advice logged when the download fails.

    Returns:
        The path kagglehub returns.

    Raises:
        SystemExit: When Kaggle rejects the request.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if cache_dir is not None:
        os.environ.setdefault("KAGGLEHUB_CACHE", str(cache_dir))
    import kagglehub

    fetch = (
        kagglehub.competition_download if competition else kagglehub.dataset_download
    )
    try:
        return Path(fetch(handle, path=path))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        logger.error(" ".join(filter(None, (CREDENTIALS_HINT, hint))))
        raise SystemExit(1) from exc


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

    logger.info("Downloading %s from competition %s ...", FILENAME, COMPETITION)
    cached = kaggle_download(COMPETITION, FILENAME, competition=True, hint=RULES_HINT)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    _materialise(cached, DEST)
    size_mb = DEST.stat().st_size / 1024**2
    logger.info("Saved %s (%.1f MB).", DEST, size_mb)
    return DEST


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = download()
    print(f"OK: {path}")


if __name__ == "__main__":
    sys.exit(main())
