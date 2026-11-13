"""Integration tests against a running compose stack.

Start it first:

    docker compose --profile serve up -d
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import NoReturn

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("CS_INTEGRATION_BASE_URL", "http://localhost:8080")
PAYLOAD = {"AMT_CREDIT": 100000.0}
TIMEOUT = 15.0


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


def stack_required(value: str | None) -> bool:
    """Parse CS_INTEGRATION_REQUIRE_STACK.

    An unrecognised value raises: a typo that quietly meant "optional" would
    let CI pass by skipping.
    """
    normalised = (value or "").strip().lower()
    if normalised in _TRUE:
        return True
    if normalised in _FALSE:
        return False
    raise ValueError(
        f"CS_INTEGRATION_REQUIRE_STACK={value!r}: expected 1/0, true/false"
    )


REQUIRE_STACK = stack_required(os.environ.get("CS_INTEGRATION_REQUIRE_STACK"))


def unreachable_stack(base_url: str, exc: Exception, *, required: bool) -> NoReturn:
    """Skip the suite, or fail it when a stack was promised.

    Skipping suits a local `uv run pytest` with the stack down: there is no
    answer to "does the real stack behave?", not a broken service. The CI job
    that starts the stack sets CS_INTEGRATION_REQUIRE_STACK=1, so a stack that
    died before the tests ran cannot pass as a skip.
    """
    message = f"no stack reachable at {base_url}: {exc.__class__.__name__}"
    if required:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        try:
            client.get("/health")
        except httpx.TransportError as exc:
            unreachable_stack(BASE_URL, exc, required=REQUIRE_STACK)
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
