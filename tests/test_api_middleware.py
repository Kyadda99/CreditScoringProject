from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/health")
    uuid.UUID(response.headers["X-Request-ID"])


def test_an_inbound_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "caller-supplied"})
    assert response.headers["X-Request-ID"] == "caller-supplied"


def test_response_carries_a_processing_time(client: TestClient) -> None:
    response = client.get("/health")
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0
