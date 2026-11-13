"""Synthetic model artifact tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api.schemas import load_feature_groups
from artifact import load_bundle
from synthetic_artifact import build_synthetic_artifact, main


def test_bundle_carries_the_committed_input_contract(tmp_path: Path) -> None:
    path = build_synthetic_artifact(tmp_path / "pipeline.joblib")

    assert load_bundle(path).input_groups == load_feature_groups()


def test_bundle_scores_a_sparse_application(tmp_path: Path) -> None:
    bundle = load_bundle(build_synthetic_artifact(tmp_path / "pipeline.joblib"))
    row = pd.DataFrame(
        [{"AMT_CREDIT": 100000.0}], columns=list(bundle.input_groups.all_features)
    )

    proba = bundle.pipeline.predict_proba(row)

    assert proba.shape == (1, 2)


def test_same_seed_gives_the_same_scores(tmp_path: Path) -> None:
    first = load_bundle(build_synthetic_artifact(tmp_path / "a.joblib", seed=7))
    second = load_bundle(build_synthetic_artifact(tmp_path / "b.joblib", seed=7))
    row = pd.DataFrame(
        [{"AMT_CREDIT": 5000.0}], columns=list(first.input_groups.all_features)
    )

    assert (
        first.pipeline.predict_proba(row) == second.pipeline.predict_proba(row)
    ).all()


def test_cli_writes_to_the_given_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "pipeline.joblib"

    main(["--output", str(target)])

    assert load_bundle(target).metadata["threshold"] == 0.5


def test_metadata_marks_the_artifact_as_synthetic(tmp_path: Path) -> None:
    path = build_synthetic_artifact(tmp_path / "pipeline.joblib")

    assert load_bundle(path).metadata["synthetic"] is True


def test_cli_refuses_to_run_without_an_output_path() -> None:
    with pytest.raises(SystemExit):
        main([])
