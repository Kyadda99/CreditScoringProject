"""HTTP layer: FastAPI, request schemas and service configuration.

Holds the single definition of the feature contract path.
"""

from __future__ import annotations

from pathlib import Path

FEATURE_SCHEMA_PATH: Path = Path(__file__).resolve().parent / "feature_schema.json"
"""Committed feature contract, written by `train.write_feature_schema` and read
by `api.schemas` at import. Derived from this package's location so it resolves
to the same file in the repository and once installed."""
