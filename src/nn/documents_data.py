"""RVL-CDIP document images: download, subsample and preprocess to PNG.

The source is a parquet mirror on the Hugging Face Hub that keeps the original
grayscale TIFFs and the 16 official labels, 400 train / 100 validation /
100 test images per class.

Usage:
    uv run cs-nn-download-documents
"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import DOCUMENTS_DIR, RANDOM_STATE, RVL_CDIP_DIR

logger = logging.getLogger(__name__)

HF_REPOSITORY = "jinhybr/rvl_cdip_400_train_val_test"
# Pinned so a later push to the mirror cannot change the data silently.
HF_REVISION = "13462a42e3375e80a9bd46a64c58e1bcaba77874"
PARQUET_FILES: dict[str, int] = {
    "train-00000-of-00002-fdc52038be08b4c4.parquet": 373_855_204,
    "train-00001-of-00002-f58eaf6bb83fd3ad.parquet": 361_181_648,
    "validation-00000-of-00001-9d1404b06c7fb129.parquet": 180_034_202,
    "test-00000-of-00001-f6660aa778291464.parquet": 186_404_543,
}


# Label ids 0-15 of RVL-CDIP in order; this order is also the model output order.
DOCUMENT_CLASSES: tuple[str, ...] = (
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific_report",
    "scientific_publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
)

# Class name -> output index. torchvision's ImageFolder numbers the class
# folders alphabetically instead, which disagrees for 14 of the 16 classes, so
# training and serving must both take indices from here.
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(DOCUMENT_CLASSES)}

# Output split name -> split name in the source files.
SOURCE_SPLITS: dict[str, str] = {"train": "train", "val": "validation", "test": "test"}

# The largest balanced counts left after duplicate removal: advertisement
# loses 15 train pages and form loses one validation page.
PER_CLASS: dict[str, int] = {"train": 385, "val": 99, "test": 100}

# Short side after preprocessing. Training crops 224 from it, and 256 keeps
# the PNG tree near 1 GB instead of the 1000 px originals.
PREPROCESS_SIZE = 256

# Seconds without data before a download gives up instead of hanging.
DOWNLOAD_TIMEOUT = 60


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    """Convert to grayscale and scale so the shorter side equals `size`."""
    image = image.convert("L")
    width, height = image.size
    scale = size / min(width, height)
    target = (round(width * scale), round(height * scale))
    return image.resize(target, Image.Resampling.LANCZOS)


def _split_files(source: Path, source_split: str) -> list[Path]:
    """Return the parquet files of one source split in a stable order."""
    files = sorted(source.glob(f"{source_split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {source_split}-*.parquet in {source}")
    return files


def _pixel_hash(data: bytes) -> str:
    """Hash the decoded grayscale pixels, so re-encoded copies still match."""
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L")
        return hashlib.sha256(f"{gray.size}".encode() + gray.tobytes()).hexdigest()


def _excluded_rows(source: Path) -> dict[str, set[int]]:
    """Find duplicate images across and within the source splits.

    The mirror holds a few pages many times over, sometimes in two splits and
    sometimes under two labels. The first copy in priority order test,
    validation, train is kept, so the evaluation splits stay intact and train
    never holds a test image. Every copy of a page seen with two labels is
    excluded.

    Returns:
        Row ids to skip, keyed by source split name.
    """
    import pyarrow.parquet as pq

    first: dict[str, tuple[str, int]] = {}
    labels: dict[str, set[int]] = defaultdict(set)
    rows_by_hash: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for source_split in ("test", "validation", "train"):
        row = 0
        for path in _split_files(source, source_split):
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=256, columns=["image", "label"]
            ):
                cells = batch.column("image").to_pylist()
                for cell, label in zip(
                    cells, batch.column("label").to_pylist(), strict=True
                ):
                    digest = _pixel_hash(cell["bytes"])
                    first.setdefault(digest, (source_split, row))
                    labels[digest].add(label)
                    rows_by_hash[digest].append((source_split, row))
                    row += 1

    excluded: dict[str, set[int]] = defaultdict(set)
    for digest, rows in rows_by_hash.items():
        for source_split, row in rows:
            if len(labels[digest]) > 1 or first[digest] != (source_split, row):
                excluded[source_split].add(row)
    return dict(excluded)


def _select_rows(
    labels: np.ndarray, count: int, seed: int, split: str
) -> dict[int, list[int]]:
    """Pick `count` row ids per class, reproducibly for a given seed.

    Rows labelled -1 are excluded duplicates and never chosen.
    """
    rng = np.random.default_rng(seed)
    chosen: dict[int, list[int]] = {}
    for label_id, name in enumerate(DOCUMENT_CLASSES):
        rows = np.flatnonzero(labels == label_id)
        if len(rows) < count:
            raise ValueError(
                f"{split}: class {name} has {len(rows)} images, {count} requested"
            )
        chosen[label_id] = sorted(rng.choice(rows, size=count, replace=False))
    return chosen


@dataclass(frozen=True)
class _SplitPlan:
    """The rows chosen for one split, ready to be written."""

    files: list[Path]
    file_sizes: list[int]
    wanted: dict[int, int]


def _plan_split(
    source: Path,
    source_split: str,
    count: int,
    seed: int,
    excluded: set[int],
    split: str,
) -> _SplitPlan:
    """Choose one split's rows without writing anything."""
    import pyarrow.parquet as pq

    files = _split_files(source, source_split)
    per_file_labels = [
        pq.read_table(path, columns=["label"]).column("label").to_numpy()
        for path in files
    ]
    labels = np.concatenate(per_file_labels)
    labels[list(excluded)] = -1
    chosen = _select_rows(labels, count, seed, split)
    return _SplitPlan(
        files=files,
        file_sizes=[len(file_labels) for file_labels in per_file_labels],
        wanted={row: label for label, rows in chosen.items() for row in rows},
    )


def _write_split(plan: _SplitPlan, out_dir: Path, source_split: str) -> None:
    """Write the planned images as PNGs under `out_dir/<class>/`."""
    import pyarrow.parquet as pq

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for name in DOCUMENT_CLASSES:
        (out_dir / name).mkdir(parents=True)

    offset = 0
    for path, size in zip(plan.files, plan.file_sizes, strict=True):
        local = [i for i in range(size) if i + offset in plan.wanted]
        if local:
            images = pq.read_table(path, columns=["image"]).column("image").take(local)
            for i, cell in zip(local, images.to_pylist(), strict=True):
                row = i + offset
                with Image.open(io.BytesIO(cell["bytes"])) as image:
                    resized = _resize_short_side(image, PREPROCESS_SIZE)
                label = DOCUMENT_CLASSES[plan.wanted[row]]
                resized.save(out_dir / label / f"{source_split}_{row:06d}.png")
        offset += size


def prepare_documents(
    dest: Path,
    per_class: dict[str, int],
    *,
    source: Path = RVL_CDIP_DIR,
    seed: int = RANDOM_STATE,
) -> Path:
    """Subsample the mirror and write `dest/{train,val,test}/<class>/*.png`.

    Each output split is drawn from the matching source split after duplicate
    pages are removed (see `_excluded_rows`), so no image is in two splits.
    Every split is planned before any is written, so a count that cannot be met
    leaves the existing tree untouched. Existing split folders are then
    replaced, which keeps a rerun with smaller counts from leaving stale images
    behind.

    Args:
        dest: Root of the processed tree.
        per_class: Images per class for each of `train`, `val` and `test`.
        source: Folder holding the mirror's parquet files.
        seed: Seed for the per-class selection.

    Returns:
        `dest`.

    Raises:
        ValueError: When a split holds fewer images of a class than requested.
    """
    logger.info("Hashing source images to find duplicates ...")
    excluded = _excluded_rows(source)
    logger.info(
        "Excluded duplicates: %s",
        {split: len(rows) for split, rows in sorted(excluded.items())},
    )
    plans = {
        split: _plan_split(
            source,
            SOURCE_SPLITS[split],
            count,
            seed,
            excluded.get(SOURCE_SPLITS[split], set()),
            split,
        )
        for split, count in per_class.items()
    }
    for split, plan in plans.items():
        logger.info("Writing %s: %d images per class ...", split, per_class[split])
        _write_split(plan, dest / split, SOURCE_SPLITS[split])
    return dest


def download(dest: Path = RVL_CDIP_DIR) -> Path:
    """Download the mirror's parquet files at the pinned revision.

    A file already present with the expected size is not fetched again.

    Args:
        dest: Folder for the parquet files.

    Returns:
        `dest`.

    Raises:
        ValueError: When `dest` holds parquet files of another revision, which
            `prepare_documents` would otherwise read as extra shards.
        OSError: When a downloaded file does not have the expected size.
    """
    dest.mkdir(parents=True, exist_ok=True)
    unknown = sorted(
        p.name for p in dest.glob("*.parquet") if p.name not in PARQUET_FILES
    )
    if unknown:
        raise ValueError(
            f"Remove parquet files not in this revision from {dest}: {unknown}"
        )
    base = f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/{HF_REVISION}"
    for name, size in PARQUET_FILES.items():
        target = dest / name
        if target.exists() and target.stat().st_size == size:
            logger.info("Already present: %s", name)
            continue
        logger.info("Downloading %s (%.0f MB) ...", name, size / 1024**2)
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(
            f"{base}/data/{name}", timeout=DOWNLOAD_TIMEOUT
        ) as response:
            with partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024**2)
        if partial.stat().st_size != size:
            raise OSError(
                f"{name}: expected {size} bytes, got {partial.stat().st_size}"
            )
        partial.replace(target)
    return dest


def count_documents(root: Path = DOCUMENTS_DIR) -> dict[str, dict[str, int]]:
    """Count PNGs per split and class under a processed tree."""
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for split in SOURCE_SPLITS:
        for name in DOCUMENT_CLASSES:
            counts[split][name] = len(list((root / split / name).glob("*.png")))
    return dict(counts)


def main() -> None:
    """CLI entry point: download, preprocess, then report the counts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_documents(DOCUMENTS_DIR, PER_CLASS, source=download())
    for split, per_label in count_documents().items():
        print(f"{split}: {sum(per_label.values())} images, {per_label}")


if __name__ == "__main__":
    sys.exit(main())
