"""Subsampling and preprocessing of the RVL-CDIP document images."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageDraw

from nn import documents_data
from nn.documents_data import (
    CLASS_TO_INDEX,
    DOCUMENT_CLASSES,
    PREPROCESS_SIZE,
    prepare_documents,
)

AVAILABLE = {"train": 4, "validation": 2, "test": 2}
PER_CLASS = {"train": 3, "val": 1, "test": 1}
SPLIT_OFFSET = {"train": 0, "validation": 10_000, "test": 20_000}


def _image_bytes(index: int) -> bytes:
    """Returns a TIFF whose pixels encode `index`, so distinct ids never collide.

    Size and colour mode also follow from `index`, so a copy is pixel-identical.
    """
    size = (300, 400) if index % 2 else (400, 300)
    image = Image.new("RGB" if index % 3 == 0 else "L", size, color="white")
    draw = ImageDraw.Draw(image)
    for bit in range(16):
        if index >> bit & 1:
            draw.rectangle((bit * 10, 0, bit * 10 + 9, 9), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF")
    return buffer.getvalue()


def _image_id(hf_split: str, label: int, i: int) -> int:
    return SPLIT_OFFSET[hf_split] + label * 100 + i


def _write_split(
    source: Path,
    hf_split: str,
    per_class: int,
    shards: int,
    copies: dict[tuple[int, int], int] | None = None,
) -> None:
    """Writes one split; `copies` maps (label, i) to the image id it duplicates."""
    copies = copies or {}
    rows = [
        (label, i) for i in range(per_class) for label in range(len(DOCUMENT_CLASSES))
    ]
    for shard in range(shards):
        part = rows[shard::shards]
        images = [
            {
                "bytes": _image_bytes(
                    copies.get((label, i), _image_id(hf_split, label, i))
                ),
                "path": None,
            }
            for label, i in part
        ]
        table = pa.table(
            {
                "image": pa.array(
                    images,
                    type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                ),
                "label": pa.array([label for label, _ in part], type=pa.int64()),
            }
        )
        pq.write_table(
            table, source / f"{hf_split}-{shard:05d}-of-{shards:05d}.parquet"
        )


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("rvl_cdip")
    for hf_split, count in AVAILABLE.items():
        _write_split(root, hf_split, count, shards=2 if hf_split == "train" else 1)
    return root


@pytest.fixture(scope="module")
def duplicated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Source whose train split repeats test, train and (relabelled) val images."""
    root = tmp_path_factory.mktemp("rvl_cdip_dup")
    train_copies = {
        (0, 1): _image_id("test", 0, 0),
        (2, 1): _image_id("train", 2, 0),
        (3, 1): _image_id("validation", 4, 0),
    }
    _write_split(root, "train", AVAILABLE["train"], shards=2, copies=train_copies)
    _write_split(root, "validation", AVAILABLE["validation"], shards=1)
    _write_split(root, "test", AVAILABLE["test"], shards=1)
    return root


def _hashes(dest: Path, split: str) -> list[str]:
    return [
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((dest / split).rglob("*.png"))
    ]


def _tree(dest: Path) -> dict[str, dict[str, list[str]]]:
    return {
        split: {
            label: sorted(p.name for p in (dest / split / label).glob("*.png"))
            for label in DOCUMENT_CLASSES
        }
        for split in PER_CLASS
    }


def test_classes_are_the_sixteen_rvl_cdip_labels_in_label_order() -> None:
    assert len(DOCUMENT_CLASSES) == 16
    assert DOCUMENT_CLASSES[0] == "letter"
    assert DOCUMENT_CLASSES[11] == "invoice"
    assert DOCUMENT_CLASSES[15] == "memo"
    assert all(" " not in name for name in DOCUMENT_CLASSES)


def test_prepare_writes_the_requested_count_per_split_and_class(
    source: Path, tmp_path: Path
) -> None:
    prepare_documents(tmp_path, PER_CLASS, source=source)
    for split, per_label in _tree(tmp_path).items():
        for files in per_label.values():
            assert len(files) == PER_CLASS[split]


def test_prepare_returns_the_destination(source: Path, tmp_path: Path) -> None:
    assert prepare_documents(tmp_path, PER_CLASS, source=source) == tmp_path


def test_prepare_writes_grayscale_png_with_the_short_side_resized(
    source: Path, tmp_path: Path
) -> None:
    prepare_documents(tmp_path, PER_CLASS, source=source)
    sizes = set()
    for path in tmp_path.rglob("*.png"):
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.mode == "L"
            assert min(image.size) == PREPROCESS_SIZE
            sizes.add(image.size)
    assert sizes == {(256, 341), (341, 256)}


def test_prepare_is_deterministic(source: Path, tmp_path: Path) -> None:
    first = prepare_documents(tmp_path / "a", PER_CLASS, source=source)
    second = prepare_documents(tmp_path / "b", PER_CLASS, source=source)
    assert _tree(first) == _tree(second)
    for path in first.rglob("*.png"):
        twin = second / path.relative_to(first)
        assert path.read_bytes() == twin.read_bytes()


def test_prepare_selection_depends_on_the_seed(source: Path, tmp_path: Path) -> None:
    trees = [
        _tree(
            prepare_documents(tmp_path / str(seed), PER_CLASS, source=source, seed=seed)
        )
        for seed in range(5)
    ]
    assert len({str(tree["train"]) for tree in trees}) > 1


def test_prepare_rejects_a_count_above_what_the_split_holds(
    source: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="train"):
        prepare_documents(tmp_path, {"train": 5, "val": 1, "test": 1}, source=source)


def test_prepare_removes_stale_files(source: Path, tmp_path: Path) -> None:
    stale = tmp_path / "train" / "letter" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    prepare_documents(tmp_path, PER_CLASS, source=source)
    assert not stale.exists()


def test_duplicates_are_kept_only_in_the_highest_priority_split(
    duplicated: Path, tmp_path: Path
) -> None:
    prepare_documents(tmp_path, PER_CLASS, source=duplicated)
    train, val, test = (_hashes(tmp_path, split) for split in ("train", "val", "test"))
    assert len(set(train)) == len(train)
    assert set(train).isdisjoint(test)
    assert set(train).isdisjoint(val)
    assert set(val).isdisjoint(test)


def test_duplicates_reduce_what_a_class_can_supply(
    duplicated: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="letter"):
        prepare_documents(
            tmp_path, {"train": 4, "val": 1, "test": 1}, source=duplicated
        )


def test_an_image_with_conflicting_labels_is_dropped_everywhere(
    duplicated: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="val: class advertisement"):
        prepare_documents(
            tmp_path, {"train": 3, "val": 2, "test": 1}, source=duplicated
        )


def test_download_skips_files_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(documents_data, "PARQUET_FILES", {"a.parquet": 3})
    (tmp_path / "a.parquet").write_bytes(b"abc")
    monkeypatch.setattr(documents_data.urllib.request, "urlopen", _no_network)
    assert documents_data.download(tmp_path) == tmp_path


def test_excluded_rows_are_counted_per_split_across_shards(duplicated: Path) -> None:
    # Train rows 8 and 9 sit in the first shard, row 41 in the second: a row
    # counter that restarted per file would name different rows.
    assert documents_data._excluded_rows(duplicated) == {
        "train": {8, 9, 41},
        "validation": {4},
    }


def _decoded_label(path: Path) -> int:
    """Reads back the id `_image_bytes` drew into the image, then its label."""
    with Image.open(path) as image:
        scale = PREPROCESS_SIZE / 300
        index = sum(
            1 << bit
            for bit in range(16)
            if image.getpixel((int((bit * 10 + 4.5) * scale), int(4.5 * scale))) < 128
        )
    return index % 10_000 // 100


def test_every_image_lands_in_its_true_class_folder(
    duplicated: Path, tmp_path: Path
) -> None:
    prepare_documents(tmp_path, PER_CLASS, source=duplicated)
    for path in tmp_path.rglob("*.png"):
        assert DOCUMENT_CLASSES[_decoded_label(path)] == path.parent.name, path


def test_prepare_checks_every_split_before_writing_any(
    source: Path, tmp_path: Path
) -> None:
    stale = tmp_path / "train" / "letter" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    with pytest.raises(ValueError, match="val"):
        prepare_documents(tmp_path, {"train": 3, "val": 5, "test": 1}, source=source)
    assert stale.exists()


def test_class_to_index_follows_the_label_order() -> None:
    assert list(CLASS_TO_INDEX) == list(DOCUMENT_CLASSES)
    assert list(CLASS_TO_INDEX.values()) == list(range(16))
    assert sorted(DOCUMENT_CLASSES) != list(DOCUMENT_CLASSES)


def test_download_rejects_parquet_files_from_another_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents_data, "PARQUET_FILES", {"a.parquet": 3})
    (tmp_path / "a.parquet").write_bytes(b"abc")
    (tmp_path / "train-old.parquet").write_bytes(b"x")
    with pytest.raises(ValueError, match="train-old.parquet"):
        documents_data.download(tmp_path)


def test_download_sets_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def _urlopen(url: str, **kwargs: object) -> io.BytesIO:
        seen.update(kwargs)
        return io.BytesIO(b"abc")

    monkeypatch.setattr(documents_data, "PARQUET_FILES", {"a.parquet": 3})
    monkeypatch.setattr(documents_data.urllib.request, "urlopen", _urlopen)
    documents_data.download(tmp_path)
    assert seen.get("timeout")
