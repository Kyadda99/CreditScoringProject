"""The pass/fail limits a headless load run is judged against."""

from __future__ import annotations

import pytest

from load.thresholds import LoadSummary, Thresholds, violations


def test_a_healthy_run_passes() -> None:
    assert violations(LoadSummary(500, 0.0, 800.0), Thresholds()) == []


def test_fail_ratio_above_the_limit_fails() -> None:
    found = violations(LoadSummary(500, 0.02, 800.0), Thresholds())
    assert len(found) == 1
    assert "fail ratio" in found[0]


def test_fail_ratio_at_the_limit_passes() -> None:
    assert violations(LoadSummary(500, 0.01, 800.0), Thresholds()) == []


def test_slow_p95_fails() -> None:
    found = violations(LoadSummary(500, 0.0, 5001.0), Thresholds())
    assert len(found) == 1
    assert "p95" in found[0]


def test_too_few_requests_fail_the_gate() -> None:
    found = violations(LoadSummary(0, 0.0, 0.0), Thresholds())
    assert len(found) == 1
    assert "requests" in found[0]


def test_every_breach_is_reported() -> None:
    assert len(violations(LoadSummary(10, 0.5, 9000.0), Thresholds())) == 3


def test_locustfile_exceptions_fail_the_gate() -> None:
    found = violations(LoadSummary(500, 0.0, 800.0, exceptions=2), Thresholds())
    assert len(found) == 1
    assert "exception" in found[0]


def test_from_env_reads_overrides() -> None:
    environ = {
        "CS_LOAD_MAX_FAIL_RATIO": "0.05",
        "CS_LOAD_MAX_P95_MS": "8000",
        "CS_LOAD_MIN_REQUESTS": "20",
    }
    assert Thresholds.from_env(environ) == Thresholds(0.05, 8000.0, 20)


def test_from_env_defaults_when_unset() -> None:
    assert Thresholds.from_env({}) == Thresholds()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CS_LOAD_MAX_FAIL_RATIO", "1%"),
        ("CS_LOAD_MAX_FAIL_RATIO", "1.5"),
        ("CS_LOAD_MAX_P95_MS", "-1"),
        ("CS_LOAD_MIN_REQUESTS", "ten"),
        ("CS_LOAD_MAX_FAIL_RATIO", "nan"),
        ("CS_LOAD_MAX_P95_MS", "nan"),
        ("CS_LOAD_MAX_P95_MS", "inf"),
    ],
)
def test_malformed_threshold_raises(name: str, value: str) -> None:
    with pytest.raises(ValueError, match=name):
        Thresholds.from_env({name: value})
