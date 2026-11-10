"""Measures the cost of blocking the event loop with CPU-bound scoring.

Two arms, one difference: an `async def` endpoint that calls `predict_proba`
inline, against one that hands it to the thread pool. Run with:

    uv run python scripts/measure_async.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

import httpx
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

CONCURRENCY = 20
REQUESTS = 100


def build_app(predictor: Any, *, blocking: bool) -> FastAPI:  # noqa: ANN401
    """Build a one-endpoint app in either arm.

    Args:
        predictor: Loaded predictor.
        blocking: `True` calls the model inline on the event loop.

    Returns:
        The application.
    """
    app = FastAPI()

    @app.post("/score")
    async def score(body: dict[str, Any]) -> dict[str, float]:
        if blocking:
            result = predictor.score(body)
        else:
            result = await run_in_threadpool(predictor.score, body)
        return {"probability": result.probability}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def measure(app: FastAPI, payload: dict[str, Any]) -> tuple[float, float, float]:
    """Drive the app concurrently and return p50, p95 and elapsed seconds.

    Args:
        app: Application under test.
        payload: Request body.

    Returns:
        p50 latency in ms, p95 latency in ms, total wall-clock seconds.
    """
    transport = httpx.ASGITransport(app=app)
    latencies: list[float] = []
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def one() -> None:
            async with semaphore:
                start = time.perf_counter()
                await client.post("/score", json=payload)
                latencies.append((time.perf_counter() - start) * 1000)

        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(REQUESTS)))
        elapsed = time.perf_counter() - started

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    return p50, p95, elapsed


async def measure_health_under_load(
    app: FastAPI, payload: dict[str, Any]
) -> tuple[float, float]:
    """Time an unrelated endpoint while scoring load runs.

    This is the failure mode the thread pool exists to prevent: a blocking
    call on the event loop stalls every other connection, including probes.

    Args:
        app: Application under test.
        payload: Scoring request body.

    Returns:
        p50 and p95 latency in ms of `/health` during the load.
    """
    transport = httpx.ASGITransport(app=app)
    health: list[float] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def load() -> None:
            await asyncio.gather(
                *(client.post("/score", json=payload) for _ in range(CONCURRENCY))
            )

        async def probe() -> None:
            for _ in range(20):
                start = time.perf_counter()
                await client.get("/health")
                health.append((time.perf_counter() - start) * 1000)
                await asyncio.sleep(0.01)

        await asyncio.gather(load(), probe())

    health.sort()
    return statistics.median(health), health[int(len(health) * 0.95) - 1]


async def main() -> None:
    """Run both arms and print the comparison."""
    from api.config import get_settings
    from artifact import load_bundle
    from predictor import Predictor

    settings = get_settings()
    bundle = load_bundle(settings.model_path)
    predictor = Predictor.from_settings(settings, bundle)
    payload = {"AMT_CREDIT": 100000.0}

    arms = (("async def, blocking", True), ("run_in_threadpool", False))

    print(f"Scoring throughput: {REQUESTS} requests, {CONCURRENCY} concurrent\n")
    print(f"{'arm':<28}{'p50 ms':>10}{'p95 ms':>10}{'total s':>10}")
    for label, blocking in arms:
        p50, p95, elapsed = await measure(
            build_app(predictor, blocking=blocking), payload
        )
        print(f"{label:<28}{p50:>10.1f}{p95:>10.1f}{elapsed:>10.2f}")

    print(f"\n/health latency while {CONCURRENCY} scores are in flight\n")
    print(f"{'arm':<28}{'p50 ms':>10}{'p95 ms':>10}")
    for label, blocking in arms:
        p50, p95 = await measure_health_under_load(
            build_app(predictor, blocking=blocking), payload
        )
        print(f"{label:<28}{p50:>10.1f}{p95:>10.1f}")


if __name__ == "__main__":
    asyncio.run(main())
