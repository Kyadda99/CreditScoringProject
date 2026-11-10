from __future__ import annotations

import fakeredis
import pytest
import redis.asyncio as redis

from api.cache import ScoreCache, cache_key
from predictor import ScoreResult

RESULT = ScoreResult(probability=0.25, decision=False, threshold=0.5)


def test_key_changes_with_the_payload() -> None:
    a = cache_key("v1", {"AMT_CREDIT": 1.0})
    b = cache_key("v1", {"AMT_CREDIT": 2.0})
    assert a != b


def test_key_changes_with_the_model_version() -> None:
    a = cache_key("v1", {"AMT_CREDIT": 1.0})
    b = cache_key("v2", {"AMT_CREDIT": 1.0})
    assert a != b


def test_key_is_stable_across_key_order() -> None:
    a = cache_key("v1", {"A": 1.0, "B": "x"})
    b = cache_key("v1", {"B": "x", "A": 1.0})
    assert a == b


def test_key_handles_values_pydantic_allows_through() -> None:
    payload = {"NAME_INCOME_TYPE": "Wyzsze wyksztalcenie", "AMT_CREDIT": 1.5}
    assert cache_key("v1", payload) == cache_key("v1", dict(payload))


def test_key_handles_none_values() -> None:
    assert cache_key("v1", {"A": None}) != cache_key("v1", {"A": 0.0})


class _DeadRedis:
    async def get(self, key: str) -> bytes | None:
        raise redis.ConnectionError("connection refused")

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise redis.ConnectionError("connection refused")


@pytest.fixture
def cache() -> ScoreCache:
    return ScoreCache(
        client=fakeredis.FakeAsyncRedis(),
        version="v1",
        ttl_seconds=300,
    )


async def test_miss_then_hit(cache: ScoreCache) -> None:
    payload = {"AMT_CREDIT": 1.0}
    assert await cache.get(payload) == (None, "MISS")
    await cache.set(payload, RESULT)
    assert await cache.get(payload) == (RESULT, "HIT")


async def test_a_new_model_version_does_not_serve_the_old_answer() -> None:
    server = fakeredis.FakeServer()
    payload = {"AMT_CREDIT": 1.0}
    old = ScoreCache(
        client=fakeredis.FakeAsyncRedis(server=server), version="v1", ttl_seconds=300
    )
    await old.set(payload, RESULT)
    new = ScoreCache(
        client=fakeredis.FakeAsyncRedis(server=server), version="v2", ttl_seconds=300
    )
    assert await new.get(payload) == (None, "MISS")


async def test_set_applies_the_ttl(cache: ScoreCache) -> None:
    payload = {"AMT_CREDIT": 1.0}
    await cache.set(payload, RESULT)
    assert 0 < await cache.client.ttl(cache_key("v1", payload)) <= 300


async def test_disabled_cache_always_bypasses() -> None:
    disabled = ScoreCache.disabled()
    assert disabled.enabled is False
    assert await disabled.get({"AMT_CREDIT": 1.0}) == (None, "BYPASS")
    await disabled.set({"AMT_CREDIT": 1.0}, RESULT)


async def test_unreachable_redis_bypasses_instead_of_raising() -> None:
    dead = ScoreCache(client=_DeadRedis(), version="v1", ttl_seconds=300)
    assert await dead.get({"AMT_CREDIT": 1.0}) == (None, "BYPASS")
    await dead.set({"AMT_CREDIT": 1.0}, RESULT)
