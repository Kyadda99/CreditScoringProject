"""Model artifact save and load.

The pipeline and the feature groups travel in one file. `split_feature_groups`
depends on the sample, so serving must rebuild exactly the groups the model was
trained on; two separate files could be updated independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from config import PROJECT_ROOT, FeatureGroups

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH: Path = PROJECT_ROOT / "models" / "pipeline.joblib"

_PIPELINE_KEY = "pipeline"
_GROUPS_KEY = "feature_groups"
_METADATA_KEY = "metadata"
_INPUT_GROUPS_KEY = "input_feature_groups"


@dataclass(frozen=True)
class Bundle:
    """A trained model together with everything needed to use it.

    Attributes:
        pipeline: Fitted `Pipeline`: preprocessing and estimator together.
        groups: Feature groups derived on the training set after feature
            engineering. These drive the `ColumnTransformer`.
        input_groups: Raw columns `/score` accepts. Different from `groups`
            because the model consumes derived features the client cannot
            send. `to_frame` builds its row from this contract, and `lifespan`
            compares the committed `feature_schema.json` against it.
        metadata: Run context: metrics, versions, counts.
    """

    pipeline: Pipeline
    groups: FeatureGroups
    input_groups: FeatureGroups
    metadata: dict[str, Any]


def save_bundle(
    pipeline: Pipeline,
    groups: FeatureGroups,
    input_groups: FeatureGroups,
    metadata: dict[str, Any],
    path: Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Write the model, both column contracts and metadata to one file.

    Args:
        pipeline: Fitted pipeline.
        groups: Model feature groups, after feature engineering.
        input_groups: Input contract: raw columns the API accepts.
        metadata: Any joblib-serialisable dictionary.
        path: Destination path; parent directories are created.

    Returns:
        Path of the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            _PIPELINE_KEY: pipeline,
            _GROUPS_KEY: groups.to_dict(),
            _INPUT_GROUPS_KEY: input_groups.to_dict(),
            _METADATA_KEY: metadata,
        },
        path,
    )
    logger.info("Saved artifact: %s (%.1f MB)", path, path.stat().st_size / 1024**2)
    return path


def load_bundle(path: Path = DEFAULT_ARTIFACT_PATH) -> Bundle:
    """Load an artifact written by `save_bundle`.

    Args:
        path: Path to the `.joblib` file.

    Returns:
        The restored `Bundle`.

    Raises:
        FileNotFoundError: When the file is absent; the message points at
            `cs-train`, the usual cause.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}\nRun: uv run cs-train")
    payload = joblib.load(path)
    # An older artifact carries only one column contract. Serving from it
    # would build the request row from the model groups, whose derived
    # features the client never sends, yielding silent NaNs.
    if _INPUT_GROUPS_KEY not in payload:
        raise KeyError(
            f"Artifact {path} has no input contract. "
            "Retrain the model: uv run cs-train"
        )
    metadata = payload[_METADATA_KEY]

    # The artifact is built outside the image and copied in, so nothing
    # guarantees version agreement. A warning, not an exception: a mismatch
    # usually works, but it is the first thing to check on odd results.
    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "Artifact trained on scikit-learn %s, loaded on %s. "
            "sklearn pickles are not guaranteed across versions.",
            trained_with,
            sklearn.__version__,
        )

    return Bundle(
        pipeline=payload[_PIPELINE_KEY],
        groups=FeatureGroups.from_dict(payload[_GROUPS_KEY]),
        input_groups=FeatureGroups.from_dict(payload[_INPUT_GROUPS_KEY]),
        metadata=metadata,
    )
