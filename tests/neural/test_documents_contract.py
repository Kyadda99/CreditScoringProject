"""Contract tests on the real preprocessed RVL-CDIP images.

Skipped unless `uv run cs-nn-download-documents` has run. The counts are
literals, not read from the module under test, so a change to the subsample
fails here. When a change is intentional: copy the new values or the printed
checksum into the constants below, and record why.
"""

from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from config import DOCUMENTS_DIR
from nn.documents_data import DOCUMENT_CLASSES, PREPROCESS_SIZE, count_documents

RECORDED_PER_CLASS = {"train": 385, "val": 99, "test": 100}
EXPECTED_TEST_SET_SHA256 = (
    "009bfdcc0be6e5f67f027b543db7d5d613df6e832847916854b1c370d1ae1c61"
)

pytestmark = pytest.mark.requires_data(DOCUMENTS_DIR)


def test_documents_have_the_recorded_counts() -> None:
    expected = {
        split: dict.fromkeys(DOCUMENT_CLASSES, count)
        for split, count in RECORDED_PER_CLASS.items()
    }
    assert count_documents(DOCUMENTS_DIR) == expected


def test_documents_are_grayscale_png_at_the_preprocess_size() -> None:
    for path in DOCUMENTS_DIR.rglob("*.png"):
        with Image.open(path) as image:
            assert (image.format, image.mode) == ("PNG", "L"), path
            assert min(image.size) == PREPROCESS_SIZE, path


def test_documents_hold_no_duplicate_within_or_across_splits() -> None:
    digests = [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DOCUMENTS_DIR.rglob("*.png")
    ]
    assert len(digests) == len(set(digests))


def test_test_split_has_not_drifted() -> None:
    root = DOCUMENTS_DIR / "test"
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.png"))
    actual = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    assert actual == EXPECTED_TEST_SET_SHA256, (
        f"The document test split changed. New checksum: {actual}\n"
        "If this is intentional, write it into EXPECTED_TEST_SET_SHA256 "
        "and record why."
    )
