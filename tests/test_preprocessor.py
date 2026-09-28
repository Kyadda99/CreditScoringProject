"""Kontrakt Preprocessora: składa, nigdy nie dopasowuje."""

from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from config import DROPPED_COLUMNS, FeatureGroups, split_feature_groups
from preprocessor import Preprocessor
from strategies import get_imbalance_strategy
from test_characterisation import synthetic_frame


def _groups(frame: pd.DataFrame) -> FeatureGroups:
    """Zwraca grupy cech modelu dla ramki po inżynierii cech."""
    engineered = Preprocessor(split_feature_groups(frame)).engineer(frame)
    return split_feature_groups(engineered, dropped=DROPPED_COLUMNS)


def test_build_pipeline_matches_the_documented_step_names() -> None:
    frame, _ = synthetic_frame()
    pipeline = Preprocessor(_groups(frame)).build_pipeline()
    assert [name for name, _ in pipeline.steps] == [
        "features",
        "preprocessor",
        "model",
    ]
    assert isinstance(pipeline, Pipeline)


def test_sampler_produces_an_imblearn_pipeline() -> None:
    frame, _ = synthetic_frame()
    sampler = get_imbalance_strategy("resample").make_sampler()
    pipeline = Preprocessor(_groups(frame)).build_pipeline(sampler=sampler)
    assert isinstance(pipeline, ImbPipeline)
    assert [name for name, _ in pipeline.steps][2] == "sampler"


def test_returned_pipeline_is_unfitted() -> None:
    frame, _ = synthetic_frame()
    pipeline = Preprocessor(_groups(frame)).build_pipeline()
    # clone() przechodzi tylko na niedopasowanym estymatorze o czystych parametrach.
    assert clone(pipeline) is not pipeline


def test_feature_engineer_import_path_is_stable() -> None:
    from features import FeatureEngineer

    assert FeatureEngineer.__module__ == "features"
    frame, _ = synthetic_frame()
    step = dict(Preprocessor(_groups(frame)).build_pipeline().steps)["features"]
    assert type(step).__module__ == "features"


def test_engineer_adds_the_derived_columns_without_mutating_input() -> None:
    frame, _ = synthetic_frame()
    before = frame.copy()
    engineered = Preprocessor(split_feature_groups(frame)).engineer(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert "CREDIT_INCOME_RATIO" in engineered.columns
