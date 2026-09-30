"""Training orchestration: data -> split -> pipeline -> metrics -> artifact.

This module composes pieces from the other modules and writes the result. The
logic lives where it can be tested without the dataset on disk.

Usage:
    uv run cs-train
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import sklearn
from sklearn.pipeline import Pipeline

from api import FEATURE_SCHEMA_PATH as _API_FEATURE_SCHEMA_PATH
from artifact import DEFAULT_ARTIFACT_PATH, save_bundle
from config import (
    COST_FN,
    COST_FP,
    DROPPED_COLUMNS,
    INFORMATIVE_MISSING,
    FeatureGroups,
)
from data_loader import DataLoader
from quality_gate import beats_incumbent, quality_gate
from registry import (
    BUNDLE_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    SERIALIZATION_FORMAT,
    export_champion,
    production_metrics,
    promote,
    register,
)
from settings import TrainingSettings, get_training_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRID_EXPERIMENT_NAME = "credit-scoring-phase3"
"""A separate experiment so grid runs do not mix with the baseline."""
# Imported, not derived again: deriving it from PROJECT_ROOT drifted from the
# API everywhere outside the repository directory.
FEATURE_SCHEMA_PATH: Path = _API_FEATURE_SCHEMA_PATH


def write_feature_schema(
    groups: FeatureGroups, path: Path = FEATURE_SCHEMA_PATH
) -> Path:
    """Write the input contract as a committed file for the API schema.

    The API needs the field names and types to build the request model, but
    the artifact is gitignored and absent from CI and a fresh clone. This file
    is what `api/schemas.py` generates `ScoreRequest` from.

    It takes the input contract (raw columns), not the model groups. The model
    groups contain derived features the client cannot send; writing those here
    would produce a `/score` that accepts `CREDIT_INCOME_RATIO`.

    Args:
        groups: Input contract from `input_contract()`.
        path: Destination path.

    Returns:
        Path of the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(groups.to_dict(), indent=2), encoding="utf-8")
    logger.info("Wrote feature contract: %s", path)
    return path


def champion_metadata(
    metrics: dict[str, float],
    model_name: str,
    arm: str,
    engineered: bool,
    loader: DataLoader,
    rows_train: int,
    rows_test: int,
) -> dict[str, object]:
    """Build the artifact metadata describing this particular run.

    The split comes from `loader`, not from a constant: the metadata must
    describe what actually happened, since it reaches the registry and the
    artifact.

    Args:
        metrics: Champion metrics on the test set.
        model_name: Algorithm name.
        arm: Imbalance arm.
        engineered: Whether derived features were computed.
        loader: Loader that performed the split.
        rows_train: Number of training rows.
        rows_test: Number of test rows.

    Returns:
        The metadata dictionary.
    """
    return {
        **metrics,
        "model": model_name,
        "imbalance": arm,
        "engineered": engineered,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "n_dropped": len(DROPPED_COLUMNS),
        "n_informative_missing": len(INFORMATIVE_MISSING),
        "test_size": loader.test_size,
        "random_state": loader.random_state,
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "rows_train": rows_train,
        "rows_test": rows_test,
    }


def build_parser(settings: TrainingSettings) -> argparse.ArgumentParser:
    """Build the CLI parser, taking its defaults from the configuration.

    Args:
        settings: Training settings.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Train the grid of scoring models.")
    parser.add_argument(
        "--engineered",
        action=argparse.BooleanOptionalAction,
        default=settings.engineered,
        help="Derived features; --no-engineered measures cleaning alone.",
    )
    parser.add_argument(
        "--models",
        default=",".join(settings.models),
        help="Comma-separated list of algorithms.",
    )
    parser.add_argument(
        "--imbalance",
        default=",".join(settings.arms),
        help="Comma-separated list of imbalance arms.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Stop at the comparison table; do not touch the registry.",
    )
    return parser


def promote_champion(
    pipeline: Pipeline,
    threshold: float,
    metrics: dict[str, float],
    model_name: str,
    arm: str,
    groups: FeatureGroups,
    inputs: FeatureGroups,
    engineered: bool,
    loader: DataLoader,
    rows_train: int,
    rows_test: int,
) -> bool:
    """Run the champion through both gates and promote it if it passes.

    Args:
        pipeline: Fitted champion pipeline.
        threshold: Decision threshold.
        metrics: Metrics on the test set.
        model_name: Algorithm name.
        arm: Imbalance arm.
        groups: Model feature groups.
        inputs: Input contract.
        engineered: Whether derived features were computed.
        loader: Loader that performed the split.
        rows_train: Number of training rows.
        rows_test: Number of test rows.

    Returns:
        `True` when the model was promoted.
    """
    # Gate before promotion: absolute thresholds answer "good enough?".
    if not quality_gate(metrics):
        logger.warning(
            "Champion did not clear the gate - not promoting. The run stays "
            "in MLflow, but the @production alias is unchanged."
        )
        return False

    # Second gate: a model can meet the thresholds and still be worse than
    # the one already in production.
    if not beats_incumbent(metrics, production_metrics()):
        logger.warning(
            "Candidate does not beat the current champion - not promoting. "
            "The run stays in MLflow, the @production alias is unchanged."
        )
        return False

    metadata = champion_metadata(
        metrics=metrics,
        model_name=model_name,
        arm=arm,
        engineered=engineered,
        loader=loader,
        rows_train=rows_train,
        rows_test=rows_test,
    )

    with mlflow.start_run(run_name=f"champion__{model_name}__{arm}") as run:
        mlflow.log_params(
            {"model": model_name, "imbalance": arm, "threshold": threshold}
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_PATH,
            serialization_format=SERIALIZATION_FORMAT,
        )
        # The bundle is logged as an artifact of this run so the export can
        # find it later by alias -> version -> run_id.
        local = save_bundle(pipeline, groups, inputs, metadata, DEFAULT_ARTIFACT_PATH)
        mlflow.log_artifact(str(local), artifact_path=BUNDLE_ARTIFACT_PATH)
        run_id = run.info.run_id

    promote(register(run_id))
    export_champion()
    write_feature_schema(inputs)
    logger.info("Champion promoted and exported.")
    return True


def cli() -> None:
    """Entry point for `uv run cs-train`; runs the Prefect flow."""
    # Local import: `flows` imports this module, so a top-level import would
    # close a cycle.
    from flows import training_pipeline

    settings = get_training_settings()
    args = build_parser(settings).parse_args()

    training_pipeline(
        register_champion=not args.no_register,
        settings=settings.model_copy(
            update={
                "engineered": args.engineered,
                "models": tuple(args.models.split(",")),
                "arms": tuple(args.imbalance.split(",")),
            }
        ),
    )


if __name__ == "__main__":
    cli()
