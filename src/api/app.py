"""Scoring service: /health and /score.

The model is loaded once in `lifespan` and held on `app.state`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from api.cache import ScoreCache
from api.config import get_settings
from api.events import emit_prediction
from api.middleware import RequestContextMiddleware
from api.schemas import (
    FEATURE_GROUPS,
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from artifact import load_bundle
from predictor import Predictor, ScoreResult

# uvicorn configures only its own loggers; root stays unhandled at WARNING.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and settings once at process start.

    A missing artifact is fatal and stops the container from starting.

    Args:
        app: Application whose state receives the model and settings.

    Yields:
        Nothing; control returns to FastAPI for the life of the process.
    """
    settings = get_settings()
    logger.info("Loading model: %s", settings.model_path)
    bundle = load_bundle(settings.model_path)

    # Fail closed. Compares the input contract (raw columns), not the model's
    # feature groups: the two differ because the model consumes derived
    # features the client never sends. Without this check a mismatch is silent.
    if bundle.input_groups != FEATURE_GROUPS:
        raise RuntimeError(
            "Input contract does not match the model artifact.\n"
            f"  feature_schema.json: {len(FEATURE_GROUPS.all_features)} columns\n"
            f"  {settings.model_path}: "
            f"{len(bundle.input_groups.all_features)} columns\n"
            "Retrain the model and commit the generated feature_schema.json: "
            "uv run cs-train"
        )

    # The artifact threshold is the default; CS_SCORE_THRESHOLD overrides it.
    predictor = Predictor.from_settings(settings, bundle)
    threshold = predictor.threshold
    source = (
        "CS_SCORE_THRESHOLD" if settings.score_threshold is not None else "artifact"
    )

    cache = ScoreCache.from_settings(settings, bundle)

    app.state.bundle = bundle
    app.state.predictor = predictor
    app.state.settings = settings
    app.state.cache = cache
    logger.info(
        "Model ready. Training ROC-AUC: %s, threshold: %.2f (source: %s), "
        "input columns: %d, model features: %d, sklearn: %s",
        bundle.metadata.get("roc_auc"),
        threshold,
        source,
        len(bundle.input_groups.all_features),
        len(bundle.groups.all_features),
        bundle.metadata.get("sklearn_version"),
    )
    yield
    await cache.aclose()
    logger.info("Shutting down.")


app = FastAPI(
    title="Credit Scoring API",
    description="Probability of difficulty repaying a loan.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 without echoing the raw input back.

    The default FastAPI handler puts the original value in an `input` field.
    For `Infinity` the JSON encoder fails on it and the client gets a 500
    instead of a 422.

    Args:
        request: Request that failed validation.
        exc: Validation exception carrying the list of errors.

    Returns:
        A 422 response with the location, message and type of each error.
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                for error in exc.errors()
            ]
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health probe. Returns 200 only when startup succeeded.

    Returns:
        A constant `ok` status.
    """
    return HealthResponse(status="ok")


async def _score_off_the_loop(
    predictor: Predictor, body: dict[str, Any]
) -> ScoreResult:
    """Run the CPU-bound scoring call without blocking the event loop.

    `predict_proba` takes about 24 ms, long enough to stall every other
    connection on the loop if called inline from a coroutine.

    Args:
        predictor: Loaded predictor.
        body: Validated request fields with empties removed.

    Returns:
        The scoring result.

    Raises:
        HTTPException: 422 for invalid feature values, 500 otherwise.
    """
    try:
        return await run_in_threadpool(predictor.score, body)
    except ValueError as exc:
        # A sklearn ValueError is almost always bad input, not a server fault.
        logger.warning("Rejected input: %s", exc)
        raise HTTPException(status_code=422, detail="Invalid feature values.") from None
    except Exception:
        # Exception text is withheld from the response: it can leak column
        # names and model details. The full trace goes to the logs.
        logger.exception("Scoring failed.")
        raise HTTPException(status_code=500, detail="Scoring failed.") from None


@app.post("/score", response_model=ScoreResponse)
async def score(
    payload: ScoreRequest,  # type: ignore[valid-type]
    request: Request,
    response: Response,
) -> ScoreResponse:
    """Score a single loan application.

    Args:
        payload: Application features; every field is optional.
        request: Request, used to reach `app.state`.
        response: Response, used to set the cache header.

    Returns:
        The probability, the decision, and the threshold that produced it.

    Raises:
        HTTPException: 422 for invalid feature values, 500 when scoring fails.
    """
    predictor = request.app.state.predictor
    cache = request.app.state.cache
    body = payload.model_dump(exclude_none=True)

    result, status = await cache.get(body)
    if result is None:
        result = await _score_off_the_loop(predictor, body)
        if status == "MISS":
            await cache.set(body, result)

    response.headers["X-Cache"] = status
    emit_prediction(
        request_id=request.state.request_id,
        probability=result.probability,
        decision=result.decision,
        threshold=result.threshold,
        cache=status,
    )
    return ScoreResponse(
        probability=result.probability,
        decision=result.decision,
        threshold=result.threshold,
    )


def main() -> None:
    """Run the service locally."""
    import os

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",  # noqa: S104 - container must listen beyond localhost
        port=int(os.environ.get("PORT", "8080")),
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(
    payload: BatchScoreRequest, request: Request
) -> BatchScoreResponse:
    """Score many applications concurrently.

    Concurrency is bounded by a semaphore: the thread pool is shared with
    every other request, and an unbounded gather would starve it.

    Args:
        payload: The batch.
        request: Request, used to reach `app.state`.

    Returns:
        One result per application, in the order received.

    Raises:
        HTTPException: 422 for invalid feature values, 500 when scoring fails.
    """
    predictor = request.app.state.predictor
    cache = request.app.state.cache
    settings = request.app.state.settings
    semaphore = asyncio.Semaphore(settings.batch_concurrency)

    async def one(application: BaseModel) -> ScoreResponse:
        body = application.model_dump(exclude_none=True)
        async with semaphore:
            result, status = await cache.get(body)
            if result is None:
                result = await _score_off_the_loop(predictor, body)
                if status == "MISS":
                    await cache.set(body, result)
        emit_prediction(
            request_id=request.state.request_id,
            probability=result.probability,
            decision=result.decision,
            threshold=result.threshold,
            cache=status,
        )
        return ScoreResponse(
            probability=result.probability,
            decision=result.decision,
            threshold=result.threshold,
        )

    results = await asyncio.gather(*(one(a) for a in payload.applications))
    return BatchScoreResponse(results=list(results))
