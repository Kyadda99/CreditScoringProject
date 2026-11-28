"""Cleaning, deduplication and splitting of the Lending Club descriptions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nn.text_data import (
    MIN_CLASS_COUNT,
    OTHER_LABEL,
    clean_description,
    load_descriptions,
    split_frame,
)


def _write_csv(path: Path, rows: list[tuple[str | None, str]]) -> Path:
    frame = pd.DataFrame(rows, columns=["desc", "purpose"])
    frame["loan_amnt"] = 1000
    frame.to_csv(path, index=False)
    return path


def test_clean_strips_the_borrower_prefix() -> None:
    raw = "  Borrower added on 08/18/12 > Pay off my cards."
    assert clean_description(raw) == "pay off my cards."


def test_clean_strips_every_prefix_in_one_text() -> None:
    raw = (
        "  Borrower added on 03/12/10 > Upgrade our home,<br/>"
        " Borrower added on 03/15/10 > thank you all."
    )
    assert clean_description(raw) == "upgrade our home, thank you all."


def test_clean_strips_a_prefix_whose_marker_is_an_entity() -> None:
    raw = "Thanks. Borrower added on 10/17/11 &gt; Renovate the kitchen."
    assert clean_description(raw) == "thanks. renovate the kitchen."


@pytest.mark.parametrize("separator", [".", " -", ",", ""])
def test_clean_strips_a_typed_prefix_with_any_separator(separator: str) -> None:
    raw = f"Borrower added on 11/26/10{separator} loan for a car"
    assert clean_description(raw) == "loan for a car"


@pytest.mark.parametrize(
    "raw",
    [
        "591060 added on 12/15/09 > Financing for a car",
        "  Borrower added on 12/15/09 > 591060 added on 12/16/09 > Financing for a car",
        "added on 06/09/12 > Financing for a car",
    ],
)
def test_clean_strips_the_member_id_prefix(raw: str) -> None:
    assert clean_description(raw) == "financing for a car"


def test_clean_keeps_a_tag_like_run_that_contains_another_bracket() -> None:
    raw = "Repairs <sears)< taken apart too. <br/>"
    assert clean_description(raw) == "repairs <sears)< taken apart too."


def test_clean_removes_br_tags_in_every_spelling() -> None:
    raw = "first<br>second<br/>third<BR />fourth"
    assert clean_description(raw) == "first second third fourth"


def test_clean_keeps_a_literal_less_than_sign() -> None:
    raw = "Employed <1 year.<br/>"
    assert clean_description(raw) == "employed <1 year."


def test_clean_unescapes_html_entities() -> None:
    raw = "Tom &amp; Jerry&rsquo;s &quot;loan&quot;"
    assert clean_description(raw) == 'tom & jerry’s "loan"'


def test_clean_collapses_whitespace() -> None:
    assert clean_description("  a \t b\n\n c  ") == "a b c"


def test_clean_returns_empty_for_a_prefix_only_text() -> None:
    assert clean_description("  Borrower added on 01/04/13 > <br>") == ""


def test_load_returns_text_and_label_columns(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "lc.csv", [("Pay off cards.", "credit_card")])
    frame = load_descriptions(path, min_class_count=1)
    assert list(frame.columns) == ["text", "label"]
    assert frame.iloc[0].to_dict() == {"text": "pay off cards.", "label": "credit_card"}


def test_load_drops_missing_and_empty_after_cleaning(tmp_path: Path) -> None:
    rows = [
        (None, "car"),
        ("  Borrower added on 01/04/13 > <br>", "car"),
        ("Buy a car.", "car"),
    ]
    frame = load_descriptions(_write_csv(tmp_path / "lc.csv", rows), min_class_count=1)
    assert frame["text"].tolist() == ["buy a car."]


def test_load_deduplicates_on_the_cleaned_text(tmp_path: Path) -> None:
    rows = [
        ("  Borrower added on 01/04/13 > Buy a car.<br>", "car"),
        ("  Borrower added on 02/05/14 > Buy a car.", "car"),
        ("BUY A CAR.", "car"),
    ]
    frame = load_descriptions(_write_csv(tmp_path / "lc.csv", rows), min_class_count=1)
    assert len(frame) == 1


def test_load_drops_texts_that_carry_conflicting_labels(tmp_path: Path) -> None:
    rows = [
        ("Consolidate.", "debt_consolidation"),
        ("Consolidate.", "credit_card"),
        ("Buy a car.", "car"),
    ]
    frame = load_descriptions(_write_csv(tmp_path / "lc.csv", rows), min_class_count=1)
    assert frame["text"].tolist() == ["buy a car."]


def test_load_merges_rare_classes_into_other(tmp_path: Path) -> None:
    rows = [(f"car text {i}", "car") for i in range(3)]
    rows += [(f"boat text {i}", "vacation") for i in range(2)]
    rows += [("misc", OTHER_LABEL)]
    frame = load_descriptions(_write_csv(tmp_path / "lc.csv", rows), min_class_count=3)
    assert frame["label"].value_counts().to_dict() == {"car": 3, OTHER_LABEL: 3}


def test_min_class_count_leaves_enough_test_rows() -> None:
    assert MIN_CLASS_COUNT * 0.15 >= 75


def _balanced_frame(per_class: int = 40) -> pd.DataFrame:
    labels = ["car", "credit_card", "wedding"]
    return pd.DataFrame(
        {
            "text": [f"{label} {i}" for label in labels for i in range(per_class)],
            "label": [label for label in labels for _ in range(per_class)],
        }
    )


def test_split_returns_train_val_test_70_15_15() -> None:
    splits = split_frame(_balanced_frame())
    assert set(splits) == {"train", "val", "test"}
    assert {name: len(part) for name, part in splits.items()} == {
        "train": 84,
        "val": 18,
        "test": 18,
    }


def test_split_is_stratified() -> None:
    sizes = {"car": 200, "credit_card": 60, "wedding": 20}
    frame = pd.DataFrame(
        {
            "text": [f"{label} {i}" for label, n in sizes.items() for i in range(n)],
            "label": [label for label, n in sizes.items() for _ in range(n)],
        }
    )
    for part in split_frame(frame).values():
        shares = part["label"].value_counts(normalize=True)
        for label, n in sizes.items():
            assert shares[label] == pytest.approx(n / len(frame), abs=0.02)


def test_split_shares_no_text_between_parts() -> None:
    splits = split_frame(_balanced_frame())
    texts = [set(part["text"]) for part in splits.values()]
    assert texts[0].isdisjoint(texts[1])
    assert texts[0].isdisjoint(texts[2])
    assert texts[1].isdisjoint(texts[2])


def test_split_is_deterministic() -> None:
    first = split_frame(_balanced_frame())
    second = split_frame(_balanced_frame())
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_split_resets_the_index() -> None:
    for part in split_frame(_balanced_frame()).values():
        assert part.index.tolist() == list(range(len(part)))


@pytest.mark.parametrize("raw", ["", "   ", "<br>"])
def test_clean_of_blank_input_is_empty(raw: str) -> None:
    assert clean_description(raw) == ""
