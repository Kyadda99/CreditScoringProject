"""Structured events on stdout.

Written with `print` rather than `logging` so each line is exactly one JSON
object with no prefix, which is what log collectors parse.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime


def _emit(event: str, **fields: object) -> None:
    """Write one event as a single JSON line on stdout.

    Args:
        event: Event name.
        **fields: Event payload.
    """
    record = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        "severity": "INFO",
        **fields,
    }
    print(json.dumps(record), flush=True)


def emit_request(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Emit the per-request event.

    Args:
        request_id: Value of the X-Request-ID header.
        method: HTTP method.
        path: Request path.
        status_code: Response status.
        duration_ms: Wall-clock handling time.
    """
    _emit(
        "request",
        request_id=request_id,
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
    )


def emit_prediction(
    request_id: str,
    probability: float,
    decision: bool,
    threshold: float,
    cache: str,
) -> None:
    """Emit the per-prediction event.

    Args:
        request_id: Value of the X-Request-ID header.
        probability: Predicted probability.
        decision: Decision derived from the threshold.
        threshold: Threshold applied.
        cache: HIT, MISS or BYPASS.
    """
    _emit(
        "prediction",
        request_id=request_id,
        probability=probability,
        decision=decision,
        threshold=threshold,
        cache=cache,
    )


def emit_model_loaded(
    model_path: str, registry_version: int | None, load_ms: float
) -> None:
    """Emit the startup event, once per process after the model is accepted.

    Args:
        model_path: Artifact path the model was loaded from.
        registry_version: Registry version stored in the artifact; `None` for
            an artifact that did not come from the registry.
        load_ms: Time spent loading the artifact.
    """
    _emit(
        "model_loaded",
        model_path=model_path,
        registry_version=registry_version,
        load_ms=load_ms,
    )
