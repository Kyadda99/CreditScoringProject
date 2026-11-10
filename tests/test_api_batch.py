from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

PAYLOAD = {"AMT_CREDIT": 100000.0}


def test_batch_scores_every_application(client: TestClient) -> None:
    response = client.post(
        "/score/batch", json={"applications": [PAYLOAD, PAYLOAD, PAYLOAD]}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3


def test_batch_preserves_input_order(client: TestClient) -> None:
    applications = [{"AMT_CREDIT": value} for value in (1000.0, 500000.0, 20000.0)]
    response = client.post("/score/batch", json={"applications": applications})
    singles = [
        client.post("/score", json=one).json()["probability"] for one in applications
    ]
    assert [r["probability"] for r in response.json()["results"]] == singles


def test_empty_batch_is_rejected(client: TestClient) -> None:
    assert client.post("/score/batch", json={"applications": []}).status_code == 422


def test_oversized_batch_is_rejected(client: TestClient) -> None:
    response = client.post("/score/batch", json={"applications": [PAYLOAD] * 101})
    assert response.status_code == 422


def test_batch_emits_one_prediction_event_per_application(
    client: TestClient, capfd: pytest.CaptureFixture[str]
) -> None:
    capfd.readouterr()
    client.post("/score/batch", json={"applications": [PAYLOAD, PAYLOAD, PAYLOAD]})
    events = [
        json.loads(line)
        for line in capfd.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert sum(e["event"] == "prediction" for e in events) == 3
