"""Request and response schemas.

The request model is generated from `feature_schema.json`, the same file
training produces, rather than hand-written. The cost is no static typing on
the request model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from api import FEATURE_SCHEMA_PATH
from config import FeatureGroups

EXAMPLES: dict[str, Any] = {
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "AMT_GOODS_PRICE": 351000.0,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "CNT_CHILDREN": 0,
    "CNT_FAM_MEMBERS": 1.0,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Single / not married",
}
"""Examples for the fields most useful in /docs."""


def load_feature_groups(path: Path = FEATURE_SCHEMA_PATH) -> FeatureGroups:
    """Load the committed feature contract.

    Args:
        path: Path to `feature_schema.json`.

    Returns:
        Feature groups exactly as training wrote them.

    Raises:
        FileNotFoundError: When the file is absent; training creates it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature contract: {path}\n"
            "Training creates it - run: uv run cs-train"
        )
    return FeatureGroups.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _example_for(name: str) -> list[Any] | None:
    """Return a one-item example list, or None when the field has none."""
    return [EXAMPLES[name]] if name in EXAMPLES else None


def _build_request_model(groups: FeatureGroups) -> type[BaseModel]:
    """Build the request model from feature groups.

    Numeric and binary columns take a number, categorical ones text. Every
    field is optional; an absent field means a missing value for the imputer.

    Args:
        groups: Feature groups from the committed contract.

    Returns:
        A pydantic model class with one field per feature.
    """
    fields: dict[str, Any] = {}
    # allow_inf_nan=False: otherwise Infinity passes validation and only fails
    # in sklearn's _assert_all_finite, which /score turned into a 500.
    for name in groups.numeric:
        fields[name] = (
            float | None,
            Field(default=None, allow_inf_nan=False, examples=_example_for(name)),
        )
    # Binary flags accept only 0/1. Without this an out-of-domain value such as
    # {"FLAG_MOBIL": 7.3} returned 200 and moved the score (0.0674 -> 0.0045).
    for name in groups.binary:
        fields[name] = (
            float | None,
            Field(
                default=None,
                ge=0.0,
                le=1.0,
                allow_inf_nan=False,
                examples=_example_for(name),
            ),
        )
    for name in groups.categorical:
        fields[name] = (
            str | None,
            Field(default=None, examples=_example_for(name)),
        )
    return create_model(
        "ScoreRequest",
        # extra="forbid": a misspelled column returns 422 instead of being
        # ignored and scored from imputations alone.
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


FEATURE_GROUPS: FeatureGroups = load_feature_groups()
ScoreRequest: type[BaseModel] = _build_request_model(FEATURE_GROUPS)


class ScoreResponse(BaseModel):
    """Scoring result for a single application."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of difficulty repaying.",
        examples=[0.0731],
    )
    decision: bool = Field(
        ...,
        description="True when the probability reaches the threshold, marking "
        "the application as risky.",
        examples=[False],
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold that produced the decision. Returned explicitly "
        "so the result can be interpreted without reading the configuration.",
        examples=[0.5],
    )


class HealthResponse(BaseModel):
    """Health probe response."""

    status: str = Field(..., examples=["ok"])
