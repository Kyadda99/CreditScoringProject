"""Raw data loading and row hygiene.

This module holds only what makes no sense outside the training set: dropping
rows without a target and rows with `CODE_GENDER == "XNA"`.

Column cleaning (the `DAYS_EMPLOYED` sentinel, imputation, derived features)
lives in `features/`, inside the pipeline, because a `/score` client can send a
sentinel and code that runs only when reading the CSV would not catch it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATA_PATH, ID_COLUMN, TARGET

logger = logging.getLogger(__name__)


def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be used for training.

    Row hygiene only, and only the kind that makes no sense outside the
    training set. Anything a `/score` client could send wrong lives in
    `FeatureEngineer` and the fitted imputer, where it also applies in
    production.

    Args:
        df: Frame; missing columns are tolerated so this can also run on a
            frame without the target.

    Returns:
        A new frame without rows missing the target or with
        `CODE_GENDER == "XNA"`.
    """
    frame = df
    if TARGET in frame.columns:
        frame = frame[frame[TARGET].notna()]
    if "CODE_GENDER" in frame.columns:
        # "XNA" is a missing value written as text, not a third category.
        # Four rows in 307k; one-hot would give them their own column.
        frame = frame[frame["CODE_GENDER"] != "XNA"]
    if len(frame) == len(df):
        return df
    logger.info("Row hygiene: %d -> %d.", len(df), len(frame))
    return frame.reset_index(drop=True)


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load `application_train.csv` into a data frame.

    Args:
        path: Path to the CSV file. Defaults to `config.DATA_PATH`.

    Returns:
        The raw frame, uncleaned, with dtypes as pandas inferred them.

    Raises:
        FileNotFoundError: When the file is absent; the message points at the
            download script, the usual cause.
    """
    path = DATA_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n" "Run: uv run cs-download"
        )

    logger.info("Loading %s ...", path)
    df = pd.read_csv(path)

    # to_numeric(errors="coerce") rather than astype: astype wraps silently
    # when narrowing (300 -> 44 for int8), so a corrupt file would pass the
    # loader looking correct. Coerce turns junk into NaN, which the contract
    # rejects. No narrowing: the contract must see what the CSV holds.
    for column in (ID_COLUMN, TARGET):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = drop_unusable_rows(df)

    logger.info("Loaded %d rows x %d columns.", len(df), df.shape[1])
    return df
