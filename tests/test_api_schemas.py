"""Testy generowanego schematu żądania."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from api.schemas import ScoreRequest, ScoreResponse, load_feature_groups
from artifact import load_bundle


def test_empty_payload_is_valid() -> None:
    """Każde pole jest opcjonalne — imputer uzupełni resztę."""
    assert ScoreRequest().model_dump(exclude_none=True) == {}


def test_partial_payload_is_valid() -> None:
    request = ScoreRequest(AMT_INCOME_TOTAL=150000.0)
    assert request.model_dump(exclude_none=True) == {"AMT_INCOME_TOTAL": 150000.0}


def test_unknown_field_is_rejected() -> None:
    """extra='forbid' — literówka w nazwie kolumny ma być błędem, nie ciszą."""
    with pytest.raises(ValidationError):
        ScoreRequest(NIE_MA_TAKIEJ_KOLUMNY=1.0)


def test_target_and_id_are_not_accepted() -> None:
    """Cel i identyfikator nie są cechami; przysłanie ich to pomyłka."""
    with pytest.raises(ValidationError):
        ScoreRequest(TARGET=1)
    with pytest.raises(ValidationError):
        ScoreRequest(SK_ID_CURR=100002)


def test_wrong_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(AMT_INCOME_TOTAL="nie liczba")


def test_categorical_field_accepts_a_string() -> None:
    request = ScoreRequest(NAME_CONTRACT_TYPE="Cash loans")
    assert request.NAME_CONTRACT_TYPE == "Cash loans"


@pytest.mark.requires_data
def test_schema_matches_the_trained_bundle() -> None:
    """Sedno D5: commitowany kontrakt i artefakt nie mogą się rozjechać.

    Od Fazy 2 porównujemy kontrakt **wejściowy** (spec D2) — grupy modelu
    zawierają cechy pochodne, których `feature_schema.json` z definicji nie zna.
    """
    assert load_feature_groups() == load_bundle().input_groups


def test_response_rejects_a_probability_outside_zero_one() -> None:
    with pytest.raises(ValidationError):
        ScoreResponse(probability=1.4, decision=True, threshold=0.5)


def test_schema_matches_the_bundle_input_contract(synthetic_artifact: Path) -> None:
    """T5 — kontrakt commitowany i artefakt nie mogą się rozjechać."""
    from api.schemas import FEATURE_GROUPS
    from artifact import load_bundle

    assert load_bundle(synthetic_artifact).input_groups == FEATURE_GROUPS
