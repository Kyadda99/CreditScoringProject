from __future__ import annotations

import json

import pytest

from api.events import emit_prediction, emit_request


def test_request_event_is_one_json_line(capsys: pytest.CaptureFixture[str]) -> None:
    emit_request(
        request_id="abc",
        method="POST",
        path="/score",
        status_code=200,
        duration_ms=12.5,
    )
    line = capsys.readouterr().out.strip()
    event = json.loads(line)
    assert event["event"] == "request"
    assert event["request_id"] == "abc"
    assert event["status_code"] == 200
    assert event["duration_ms"] == 12.5
    assert event["timestamp"].endswith("+00:00")


def test_prediction_event_carries_the_cache_outcome(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_prediction(
        request_id="abc",
        probability=0.25,
        decision=False,
        threshold=0.5,
        cache="HIT",
    )
    event = json.loads(capsys.readouterr().out.strip())
    assert event["event"] == "prediction"
    assert event["cache"] == "HIT"
