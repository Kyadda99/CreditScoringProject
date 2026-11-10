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
