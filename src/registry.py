"""Thin wrapper over the MLflow Model Registry.

Aliases, not the retired stages. The registry holds one model named
`credit-scoring` whose versions are successive champions; the `@production`
alias points at exactly one of them.

The registry is a training-side concern: the container does not use MLflow and
does not reach the network at startup. The champion enters the image through
`export_champion` as a plain joblib file, with its provenance recorded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from artifact import DEFAULT_ARTIFACT_PATH, load_bundle, save_bundle
from config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "credit-scoring"
"""One registry name; versions are successive champion generations."""

PRODUCTION_ALIAS = "production"
"""Alias marking the model that ships to the container."""

MODEL_ARTIFACT_PATH = "model"
"""sklearn artifact path inside the run: `runs:/<id>/model`.

Must match the `name=` passed to `mlflow.sklearn.log_model`, or `register`
finds nothing to register.
"""

BUNDLE_ARTIFACT_PATH = "bundle"
"""Artifact directory holding the `Bundle` inside the run."""

BUNDLE_FILENAME = "pipeline.joblib"
"""Artifact file name inside the `bundle/` directory."""

SERIALIZATION_FORMAT = "cloudpickle"
"""Serialization format for the sklearn model in MLflow.

MLflow 3.15 defaults to skops, which refuses to serialize classes outside its
trusted list and fails on this pipeline:

    UntrustedTypesFoundException: Untrusted types found in the file:
    ['features.FeatureEngineer', 'numpy.dtype', 'xgboost.core.Booster',
     'xgboost.sklearn.XGBClassifier']

Listing `skops_trusted_types` was the alternative, but that list would need
updating on every change to the pipeline's composition.

The security argument for skops does not apply here: this project produces and
loads only its own artifacts, and the container loads joblib regardless.
"""


def _client() -> MlflowClient:
    """Return a client pointing at the shared SQLite backend.

    Built on every call rather than as a module singleton: tests swap the URI,
    and a singleton would hold the one from before the swap.

    Returns:
        An MLflow client.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def register(run_id: str, name: str = REGISTERED_MODEL_NAME) -> ModelVersion:
    """Register the model logged in the given run.

    Args:
        run_id: Run in which `mlflow.sklearn.log_model(..., name="model")`
            wrote the pipeline.
        name: Registry name.

    Returns:
        The created model version.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", name=name
    )
    logger.info("Registered %s v%s (run %s)", name, mv.version, run_id[:8])
    return mv


def promote(mv: ModelVersion, alias: str = PRODUCTION_ALIAS) -> None:
    """Point the alias at the given version.

    The alias is moved, not duplicated: after this call `@production` points at
    exactly one version.

    Args:
        mv: Version returned by `register`.
        alias: Alias name.
    """
    _client().set_registered_model_alias(name=mv.name, alias=alias, version=mv.version)
    logger.info("%s v%s -> @%s", mv.name, mv.version, alias)


def load_production(
    name: str = REGISTERED_MODEL_NAME, alias: str = PRODUCTION_ALIAS
) -> Pipeline:
    """Load the model behind the alias.

    Used on the training side, for example to compare against a new candidate.
    The API never calls it: the container loads joblib, not the registry.

    Args:
        name: Registry name.
        alias: Alias.

    Returns:
        The fitted pipeline.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def list_models(name: str | None = None) -> None:
    """Log the registered models, their versions and aliases.

    Args:
        name: Restrict to one name; `None` shows all.
    """
    client = _client()
    names = [name] if name else [m.name for m in client.search_registered_models()]
    for model_name in names:
        registered = client.get_registered_model(model_name)
        # Aliases live on the model (alias -> version), not on the version,
        # so the map is inverted to print them on the matching row.
        by_version: dict[str, list[str]] = {}
        for alias, version in registered.aliases.items():
            by_version.setdefault(version, []).append(f"@{alias}")
        for mv in sorted(
            client.search_model_versions(f"name='{model_name}'"),
            key=lambda v: int(v.version),
        ):
            logger.info(
                "%-20s v%-3s %-14s run %s",
                model_name,
                mv.version,
                ", ".join(by_version.get(mv.version, [])) or "-",
                (mv.run_id or "")[:8],
            )


def export_champion(
    name: str = REGISTERED_MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
    dest: Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Pull the champion artifact from the registry and write it as joblib.

    This is the seam between the registry and the container. The alias points
    at a version, the version knows its run, and the run holds exactly the
    `Bundle` that version describes, so the file in `models/` has documented
    provenance.

    The version number exists only after registration, so it could not be in
    the `Bundle` logged earlier; it is added here and the file rewritten.

    Args:
        name: Registry name.
        alias: Champion alias.
        dest: Destination joblib path.

    Returns:
        Path of the written artifact.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mv = _client().get_model_version_by_alias(name, alias)
    logger.info("Champion: %s v%s (run %s)", name, mv.version, (mv.run_id or "")[:8])

    local = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path=f"{BUNDLE_ARTIFACT_PATH}/{BUNDLE_FILENAME}"
    )
    bundle = load_bundle(Path(local))
    metadata = {
        **bundle.metadata,
        "registry_model": name,
        "registry_version": mv.version,
        "registry_alias": alias,
        "registry_run_id": mv.run_id,
    }
    path = save_bundle(
        bundle.pipeline, bundle.groups, bundle.input_groups, metadata, dest
    )
    logger.info("Exported champion to %s", path)
    return path


def cli() -> None:
    """Entry point for `uv run cs-export`."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Export the champion from the registry to models/pipeline.joblib."
    )
    parser.add_argument("--alias", default=PRODUCTION_ALIAS)
    parser.add_argument("--name", default=REGISTERED_MODEL_NAME)
    args = parser.parse_args()

    list_models(args.name)
    export_champion(name=args.name, alias=args.alias)


def production_metrics(
    name: str = REGISTERED_MODEL_NAME, alias: str = PRODUCTION_ALIAS
) -> dict[str, float] | None:
    """Return the metrics of the model currently behind the alias.

    Needed so `beats_incumbent` can compare a candidate against the current
    champion instead of promoting unconditionally.

    Args:
        name: Registry name.
        alias: Champion alias.

    Returns:
        Metrics of the run the version came from, or `None` when the alias,
        or the model itself, does not exist yet.
    """
    client = _client()
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        logger.info("No @%s alias for %s; this is the first champion.", alias, name)
        return None
    return dict(client.get_run(mv.run_id).data.metrics)
