"""Pass/fail limits for a headless load run.

Kept free of Locust imports so the unit tests run without the `load` group.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadSummary:
    """The aggregate figures of one load run.

    `exceptions` counts errors raised in the locustfile itself, which record
    no request and so never show in the failure ratio.
    """

    requests: int
    fail_ratio: float
    p95_ms: float
    exceptions: int = 0


@dataclass(frozen=True)
class Thresholds:
    """Limits a load run must stay within.

    The minimum request count catches a run that never reached the service,
    whose failure ratio and p95 would otherwise read as zero.
    """

    max_fail_ratio: float = 0.01
    max_p95_ms: float = 5000.0
    min_requests: int = 100

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Thresholds:
        """Read `CS_LOAD_*` overrides; an unset variable keeps its default.

        Raises:
            ValueError: a value does not parse or is out of range, so a typo
                cannot silently loosen the gate.
        """
        defaults = cls()
        max_fail_ratio = _read(
            environ, "CS_LOAD_MAX_FAIL_RATIO", float, defaults.max_fail_ratio
        )
        if max_fail_ratio > 1:
            raise ValueError(
                f"CS_LOAD_MAX_FAIL_RATIO must be between 0 and 1, got {max_fail_ratio}"
            )
        return cls(
            max_fail_ratio=max_fail_ratio,
            max_p95_ms=_read(environ, "CS_LOAD_MAX_P95_MS", float, defaults.max_p95_ms),
            min_requests=_read(
                environ, "CS_LOAD_MIN_REQUESTS", int, defaults.min_requests
            ),
        )


def _read(
    environ: Mapping[str, str],
    name: str,
    parse: type[int] | type[float],
    default: int | float,
) -> int | float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = parse(raw)
    except ValueError:
        raise ValueError(f"{name} is not a valid number: {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {raw!r}")
    return value


def violations(summary: LoadSummary, thresholds: Thresholds) -> list[str]:
    """Return one line per breached limit; an empty list means the run passed."""
    found = []
    if summary.requests < thresholds.min_requests:
        found.append(
            f"only {summary.requests} requests completed, "
            f"minimum {thresholds.min_requests}"
        )
    if summary.fail_ratio > thresholds.max_fail_ratio:
        found.append(
            f"fail ratio {summary.fail_ratio:.2%} above {thresholds.max_fail_ratio:.2%}"
        )
    if summary.p95_ms > thresholds.max_p95_ms:
        found.append(
            f"p95 {summary.p95_ms:.0f} ms above {thresholds.max_p95_ms:.0f} ms"
        )
    if summary.exceptions:
        found.append(f"{summary.exceptions} exceptions raised in the locustfile")
    return found
