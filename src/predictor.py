"""Scoring a single application from a saved artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from artifact import Bundle
from config import FeatureGroups

FALLBACK_THRESHOLD = 0.5
"""Threshold used when neither the configuration nor the artifact has one."""


class ThresholdSettings(Protocol):
    """Configuration carrying an explicit threshold, or none.

    A protocol rather than an import of `api.config`: the predictor sits below
    the HTTP service and must not depend on it.
    """

    score_threshold: float | None


@dataclass(frozen=True)
class ScoreResult:
    """Result of scoring one application.

    Attributes:
        probability: Probability of repayment difficulty.
        decision: Whether the probability reached the threshold.
        threshold: Threshold that produced the decision.
    """

    probability: float
    decision: bool
    threshold: float


class Predictor:
    """Scores applications with a trained pipeline.

    Attributes:
        bundle: Artifact holding the pipeline and both column contracts.
        threshold: Decision threshold applied to the probability.
    """

    def __init__(self, bundle: Bundle, threshold: float) -> None:
        """Take a loaded artifact and a threshold.

        Args:
            bundle: Loaded artifact.
            threshold: Decision threshold.
        """
        self.bundle = bundle
        self.threshold = threshold

    @classmethod
    def from_bundle(cls, bundle: Bundle, threshold: float | None = None) -> Predictor:
        """Build a predictor, taking the threshold from the artifact if absent.

        Args:
            bundle: Loaded artifact.
            threshold: Explicit threshold; `None` uses the stored metadata.

        Returns:
            A ready predictor.
        """
        if threshold is None:
            threshold = float(bundle.metadata.get("threshold", FALLBACK_THRESHOLD))
        return cls(bundle, threshold)

    @classmethod
    def from_settings(cls, settings: ThresholdSettings, bundle: Bundle) -> Predictor:
        """Build a predictor; an explicit configured threshold wins.

        Args:
            settings: Service settings.
            bundle: Loaded artifact.

        Returns:
            A predictor using the configured threshold, else the artifact's,
            else the fallback.
        """
        return cls.from_bundle(bundle, settings.score_threshold)

    @property
    def input_groups(self) -> FeatureGroups:
        """Input contract: the raw columns accepted from the client."""
        return self.bundle.input_groups

    def to_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Turn the request body into a one-row frame of raw columns.

        Builds the row from the input contract, not the model groups: the model
        consumes derived features the client does not send, and the first
        pipeline step computes them.

        Args:
            payload: Validated request fields, with empties removed.

        Returns:
            A one-row frame with the input contract's columns.
        """
        return self.build_frame(payload, self.input_groups)

    @staticmethod
    def build_frame(payload: dict[str, Any], groups: FeatureGroups) -> pd.DataFrame:
        """Build a one-row frame for the given column groups.

        Column order comes from the contract, not the request: JSON key order
        is arbitrary.

        Args:
            payload: Validated request fields, with empties removed.
            groups: Column groups the row is built from.

        Returns:
            A one-row frame with the columns of `groups.all_features`.
        """
        row = {name: payload.get(name) for name in groups.all_features}
        frame = pd.DataFrame([row], columns=list(groups.all_features))

        # Without the cast the frame is object dtype, SimpleImputer does not
        # recognise None as missing, and binary flags reach the model as NaN.
        numeric_like = [*groups.numeric, *groups.binary]
        if numeric_like:
            frame[numeric_like] = frame[numeric_like].astype("float64")
        categorical = list(groups.categorical)
        if categorical:
            frame[categorical] = (
                frame[categorical]
                .astype(object)
                .where(frame[categorical].notna(), np.nan)
            )
        return frame

    def score(self, payload: dict[str, Any]) -> ScoreResult:
        """Score one application.

        Args:
            payload: Validated request fields.

        Returns:
            The probability, the decision and the threshold.

        Raises:
            ValueError: When the feature values are invalid for the pipeline.
        """
        frame = self.to_frame(payload)
        probability = float(self.bundle.pipeline.predict_proba(frame)[0, 1])
        return ScoreResult(
            probability=probability,
            decision=probability >= self.threshold,
            threshold=self.threshold,
        )
