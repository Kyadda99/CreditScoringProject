"""Tests for the generated request schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from api.schemas import ScoreRequest, ScoreResponse, load_feature_groups
from artifact import load_bundle


def test_empty_payload_is_valid() -> None:
    """Every field is optional; the imputer fills the rest."""
    assert ScoreRequest().model_dump(exclude_none=True) == {}


def test_partial_payload_is_valid() -> None:
    request = ScoreRequest(AMT_INCOME_TOTAL=150000.0)
    assert request.model_dump(exclude_none=True) == {"AMT_INCOME_TOTAL": 150000.0}


def test_unknown_field_is_rejected() -> None:
    """extra="forbid": a misspelled column name must be an error, not silence."""
    with pytest.raises(ValidationError):
        ScoreRequest(NIE_MA_TAKIEJ_KOLUMNY=1.0)


def test_target_and_id_are_not_accepted() -> None:
    """The target and id are not features; sending them is a mistake."""
    with pytest.raises(ValidationError):
        ScoreRequest(TARGET=1)
    with pytest.raises(ValidationError):
        ScoreRequest(SK_ID_CURR=100002)


def test_wrong_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(AMT_INCOME_TOTAL="not a number")


def test_categorical_field_accepts_a_string() -> None:
    request = ScoreRequest(NAME_CONTRACT_TYPE="Cash loans")
    assert request.NAME_CONTRACT_TYPE == "Cash loans"


@pytest.mark.requires_data
def test_schema_matches_the_trained_bundle() -> None:
    """The committed contract and the artifact must not drift apart.

    The comparison uses the input contract; the model groups contain derived
    features that `feature_schema.json` by definition does not know.
    """
    assert load_feature_groups() == load_bundle().input_groups


def test_response_rejects_a_probability_outside_zero_one() -> None:
    with pytest.raises(ValidationError):
        ScoreResponse(probability=1.4, decision=True, threshold=0.5)


def test_schema_matches_the_bundle_input_contract(synthetic_artifact: Path) -> None:
    """The committed contract and the artifact must not drift apart."""
    from api.schemas import FEATURE_GROUPS
    from artifact import load_bundle

    assert load_bundle(synthetic_artifact).input_groups == FEATURE_GROUPS
