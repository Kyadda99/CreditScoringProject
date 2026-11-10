"""Integration tests against a running compose stack.

Start it first:

    docker compose --profile serve up -d
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("CS_INTEGRATION_BASE_URL", "http://localhost:8080")
PAYLOAD = {"AMT_CREDIT": 100000.0}
TIMEOUT = 15.0


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client


def test_health(client: httpx.Client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_score_against_the_real_redis(client: httpx.Client) -> None:
    unique = dict(PAYLOAD, AMT_CREDIT=float(uuid.uuid4().int % 1_000_000))
    first = client.post("/score", json=unique)
    second = client.post("/score", json=unique)
    assert first.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert first.json() == second.json()


def test_request_headers_are_present(client: httpx.Client) -> None:
    response = client.post("/score", json=PAYLOAD)
    uuid.UUID(response.headers["X-Request-ID"])
    assert float(response.headers["X-Process-Time-Ms"]) > 0


def test_batch_against_the_stack(client: httpx.Client) -> None:
    response = client.post("/score/batch", json={"applications": [PAYLOAD] * 5})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 5


def test_bad_input_is_rejected_not_crashed(client: httpx.Client) -> None:
    assert client.post("/score", json={"NOT_A_COLUMN": 1}).status_code == 422
