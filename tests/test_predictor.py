"""Kontrakt Predictora: wiersz żądania, prawdopodobieństwo, decyzja."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from artifact import load_bundle
from predictor import Predictor, ScoreResult


@pytest.fixture
def predictor(synthetic_artifact: Path) -> Predictor:
    return Predictor.from_bundle(load_bundle(synthetic_artifact))


def test_missing_fields_become_nan_not_none(predictor: Predictor) -> None:
    frame = predictor.to_frame({})
    numeric_like = [
        *predictor.input_groups.numeric,
        *predictor.input_groups.binary,
    ]
    assert frame[numeric_like].dtypes.eq("float64").all()
    assert frame[numeric_like].isna().all().all()


def test_partial_payload_has_no_nan_after_pipeline(predictor: Predictor) -> None:
    frame = predictor.to_frame({"AMT_INCOME_TOTAL": 202500.0})
    transformed = predictor.bundle.pipeline[:-1].transform(frame)
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()


def test_column_order_follows_the_contract_not_the_payload(
    predictor: Predictor,
) -> None:
    payload = dict(
        reversed(list({n: 1.0 for n in predictor.input_groups.numeric}.items()))
    )
    frame = predictor.to_frame(payload)
    assert list(frame.columns) == list(predictor.input_groups.all_features)


def test_score_returns_probability_decision_and_threshold(predictor: Predictor) -> None:
    result = predictor.score({"AMT_INCOME_TOTAL": 202500.0})
    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.probability <= 1.0
    assert result.decision == (result.probability >= result.threshold)


def test_threshold_boundary_classifies_as_positive(synthetic_artifact: Path) -> None:
    bundle = load_bundle(synthetic_artifact)
    predictor = Predictor.from_bundle(bundle, threshold=0.0)
    assert predictor.score({}).decision is True


def test_from_bundle_takes_the_threshold_from_metadata(
    synthetic_artifact: Path,
) -> None:
    bundle = load_bundle(synthetic_artifact)
    assert Predictor.from_bundle(bundle).threshold == pytest.approx(0.5)


def test_predictor_does_not_depend_on_the_api_package() -> None:
    """Zależności idą w jedną stronę: api zna Predictora, nie odwrotnie."""
    import predictor as predictor_module

    source = Path(predictor_module.__file__).read_text(encoding="utf-8")
    assert "from api" not in source
    assert "import api" not in source
