"""Lending Club loan descriptions: download, cleaning and splits.

The text model predicts the loan `purpose` the borrower selected from the free
text they wrote (`desc`). Only `accepted_2007_to_2018Q4.csv.gz` is used, and
only those two columns; `desc` is populated mainly for loans issued 2007-2014.

Usage:
    uv run cs-nn-download-text
"""

from __future__ import annotations

import html
import logging
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DESCRIPTIONS_PATH, RANDOM_STATE
from download_data import kaggle_download

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "wordsforthewise/lending-club"
KAGGLE_FILE = "accepted_2007_to_2018Q4.csv.gz"

TEXT_COLUMN = "desc"
LABEL_COLUMN = "purpose"
OTHER_LABEL = "other"

# Classes below this size after deduplication are merged into `other`.
# At a 15 % test share it leaves at least 75 test texts per class, enough for a
# per-class F1 that is not dominated by a handful of examples.
MIN_CLASS_COUNT = 500

VAL_SHARE = 0.15
TEST_SHARE = 0.15

# The platform prepends this to every text a borrower adds, and a long text
# can carry several. Older listings carry the member id instead of
# "Borrower", and a few borrowers typed the phrase with other punctuation.
# The date says when the loan was listed and the id identifies the member;
# the purpose mix shifts with both, so neither may reach the model.
_BORROWER_PREFIX = re.compile(
    r"(?:\b(?:borrower|\d+)\s+)?added on \d{1,2}/\d{1,2}/\d{2,4}\s*[>.,:-]?", re.I
)
# A tag starts with a letter and holds no other bracket; a bare `<` such as
# "<1 year" is text.
_HTML_TAG = re.compile(r"</?[a-z][^<>]*>", re.I)
_WHITESPACE = re.compile(r"\s+")


def clean_description(raw: str) -> str:
    """Normalise one borrower description.

    Removes the "Borrower added on MM/DD/YY >" prefixes (also with a member id
    in place of "Borrower") and HTML tags, decodes HTML entities, lowercases
    and collapses whitespace. Training and serving both call this function, so
    the model always sees text cleaned the same way.

    Args:
        raw: The `desc` value as published.

    Returns:
        The cleaned text; empty when nothing but markup was present.
    """
    # Entities are decoded first: some prefixes are written "&gt;" not ">".
    text = html.unescape(raw)
    text = _BORROWER_PREFIX.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip().lower()


def load_descriptions(
    path: Path = DESCRIPTIONS_PATH, *, min_class_count: int = MIN_CLASS_COUNT
) -> pd.DataFrame:
    """Load, clean and deduplicate the descriptions.

    Deduplication runs on the cleaned text and before any split, so the same
    text cannot land in both train and test. A text seen with more than one
    label is dropped entirely rather than resolved arbitrarily.

    Args:
        path: CSV with at least the `desc` and `purpose` columns.
        min_class_count: Classes smaller than this are relabelled `other`.

    Returns:
        A frame with columns `text` and `label`, in file order, index reset.
    """
    raw = pd.read_csv(path, usecols=[TEXT_COLUMN, LABEL_COLUMN], dtype=str)
    raw = raw.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    frame = pd.DataFrame(
        {
            "text": raw[TEXT_COLUMN].map(clean_description),
            "label": raw[LABEL_COLUMN].str.strip(),
        }
    )
    frame = frame[frame["text"] != ""]

    labels_per_text = frame.groupby("text")["label"].transform("nunique")
    frame = frame[labels_per_text == 1].drop_duplicates(subset="text")

    counts = frame["label"].value_counts()
    rare = counts[counts < min_class_count].index
    frame.loc[frame["label"].isin(rare), "label"] = OTHER_LABEL
    return frame.reset_index(drop=True)


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split stratified on `label` into 70 % train, 15 % val and 15 % test.

    Args:
        df: Output of `load_descriptions`.

    Returns:
        Frames keyed `train`, `val` and `test`, each with a fresh index.
    """
    holdout = VAL_SHARE + TEST_SHARE
    train, rest = train_test_split(
        df, test_size=holdout, stratify=df["label"], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        rest,
        test_size=TEST_SHARE / holdout,
        stratify=rest["label"],
        random_state=RANDOM_STATE,
    )
    return {
        name: part.reset_index(drop=True)
        for name, part in (("train", train), ("val", val), ("test", test))
    }


def download(dest: Path = DESCRIPTIONS_PATH) -> Path:
    """Download the accepted-loans file and keep only rows with a description.

    The 1.6 GB CSV is reduced to the two columns the model uses, so later loads
    take seconds. Text is stored raw; cleaning stays in `load_descriptions`.

    Args:
        dest: Where to write the reduced CSV.

    Returns:
        `dest`.

    Raises:
        SystemExit: When Kaggle rejects the request.
    """
    if dest.exists():
        logger.info("Already present: %s; skipping download.", dest)
        return dest

    logger.info("Downloading %s from %s ...", KAGGLE_FILE, KAGGLE_DATASET)
    # kagglehub caches under the user profile by default, and C: is nearly
    # full; the cache lives next to the data instead.
    cached = kaggle_download(
        KAGGLE_DATASET, KAGGLE_FILE, cache_dir=dest.parent / "kagglehub"
    )

    frame = pd.read_csv(cached, usecols=[TEXT_COLUMN, LABEL_COLUMN], dtype=str)
    frame = frame.dropna(subset=[TEXT_COLUMN])
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    frame.to_csv(partial, index=False)
    shutil.move(partial, dest)
    logger.info("Saved %d descriptions to %s.", len(frame), dest)
    return dest


def main() -> None:
    """CLI entry point: download, then report the class and split sizes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = download()
    frame = load_descriptions(path)
    print(f"{len(frame)} descriptions after cleaning and deduplication")
    print(frame["label"].value_counts().to_string())
    for name, part in split_frame(frame).items():
        print(f"{name}: {len(part)}")


if __name__ == "__main__":
    sys.exit(main())
