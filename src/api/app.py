"""Serwis scoringowy: /health i /score.

Model ładowany jest **raz**, w `lifespan`, i trzymany na `app.state`. Ładowanie
przy każdym żądaniu byłoby o rzędy wielkości wolniejsze i nie ma żadnego powodu,
żeby to robić — pipeline jest niezmienny.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.schemas import (
    FEATURE_GROUPS,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from artifact import load_bundle
from config import FeatureGroups

# uvicorn konfiguruje wyłącznie własne loggery ("uvicorn", "uvicorn.access")
# i zostawia root bez handlera na poziomie WARNING — przez co każdy
# `logger.info(...)` z tego modułu przepadał w ciszy. W kontenerze widać było
# tylko logi uvicorna, więc kryterium Fazy 1 ("udowodnij, że model ładuje się
# raz") było niesprawdzalne, a w Fazie 6 nic nie trafiłoby do Cloud Logging.
# app.py jest punktem wejścia aplikacji, nie biblioteką, więc konfiguracja
# logowania należy tutaj.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


def to_frame(payload: dict[str, Any], groups: FeatureGroups) -> pd.DataFrame:
    """Zamienia ciało żądania na jednowierszową ramkę w kolejności z treningu.

    Brakujące pola stają się NaN — dokładnie tym, co widział imputer w czasie
    treningu. Kolejność kolumn bierzemy z `groups`, a nie z żądania: kolejność
    kluczy w JSON-ie jest przypadkowa, a `ColumnTransformer` wymaga tej samej
    kolejności, na której był dopasowany.

    Args:
        payload: Zwalidowane ciało żądania, bez pól pustych.
        groups: Grupy cech odtworzone z artefaktu.

    Returns:
        Ramka o jednym wierszu i dokładnie kolumnach `groups.all_features`.
    """
    row = {name: payload.get(name) for name in groups.all_features}
    frame = pd.DataFrame([row], columns=list(groups.all_features))

    # Bez tego kroku ramka ma dtype `object`, bo brakujące pola to `None`.
    # `SimpleImputer` szuka `np.nan`, a w tablicy `object` `None != np.nan`, więc
    # braków w ogóle NIE rozpoznaje: przepuszcza `None` dalej, a rzutowanie na
    # float robi z nich NaN — i LogisticRegression wywala się na "Input X
    # contains NaN". Kolumny numeryczne ratowała konwersja object->float w
    # sklearn, kategoryczne handle_unknown="ignore"; flagi binarne nie miały
    # żadnej z tych furtek i to one wywracały scoring.
    numeric_like = [*groups.numeric, *groups.binary]
    if numeric_like:
        frame[numeric_like] = frame[numeric_like].astype("float64")
    categorical = list(groups.categorical)
    if categorical:
        frame[categorical] = (
            frame[categorical].astype(object).where(frame[categorical].notna(), np.nan)
        )
    return frame


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ładuje model i ustawienia raz, przy starcie procesu.

    Brak artefaktu jest błędem krytycznym — wyjątek tutaj zatrzymuje start
    kontenera. To celowe: serwis, który odpowiada na /health, a wywraca się na
    każdym /score, jest gorszy niż serwis, który w ogóle nie wstał.

    Args:
        app: Aplikacja, na której stanie lądują model i ustawienia.

    Yields:
        Nic — sterowanie wraca do FastAPI na czas życia procesu.
    """
    settings = get_settings()
    logger.info("Ładuję model: %s", settings.model_path)
    bundle = load_bundle(settings.model_path)

    # Fail closed. Kontrakt cech i artefakt wchodzą do obrazu z DWÓCH różnych
    # źródeł: schemat przez koło z gita, artefakt przez `COPY models/…` z dysku
    # hosta. Bez tej kontroli rozjazd nie daje żadnego błędu — pole obecne
    # w schemacie, a nieznane modelowi, jest po cichu gubione, a cecha, której
    # model potrzebuje, ale schemat jej nie zna, wraca jako 422, choć wewnątrz
    # i tak byłaby imputowana. Serwis scorowałby dalej, tyle że źle.
    if bundle.groups != FEATURE_GROUPS:
        raise RuntimeError(
            "Kontrakt cech nie zgadza się z artefaktem modelu.\n"
            f"  feature_schema.json: {len(FEATURE_GROUPS.all_features)} cech\n"
            f"  {settings.model_path}: {len(bundle.groups.all_features)} cech\n"
            "Przetrenuj model i zacommituj wygenerowany feature_schema.json: "
            "uv run cs-train"
        )

    # Próg z artefaktu jest domyślny, a nie zaszyty w kodzie: metryki zapisane
    # przy treningu opisują decyzję podjętą PRZY TYM progu. Gdy Faza 3 go
    # dostroi, serwis pójdzie za artefaktem, zamiast cicho stosować inną regułę
    # niż ta, którą zmierzono. `CS_SCORE_THRESHOLD` nadal ma pierwszeństwo.
    threshold = settings.score_threshold
    source = "CS_SCORE_THRESHOLD"
    if threshold is None:
        threshold = float(bundle.metadata.get("threshold", 0.5))
        source = "artefakt"

    app.state.bundle = bundle
    app.state.threshold = threshold
    app.state.settings = settings
    logger.info(
        "Model gotowy. ROC-AUC z treningu: %s, próg: %.2f (źródło: %s), "
        "cech: %d, sklearn: %s",
        bundle.metadata.get("roc_auc"),
        threshold,
        source,
        len(bundle.groups.all_features),
        bundle.metadata.get("sklearn_version"),
    )
    yield
    logger.info("Zamykam serwis.")


app = FastAPI(
    title="Credit Scoring API",
    description="Prawdopodobieństwo trudności ze spłatą kredytu.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Zwraca 422 bez odsyłania surowego wejścia z powrotem.

    Domyślny handler FastAPI wkłada do treści błędu pole `input` z oryginalną
    wartością. Dla `Infinity` enkoder JSON się na tym wywraca i klient dostaje
    500 zamiast 422 — czyli wygląda to na awarię serwera, choć jest to błędne
    żądanie. Odsyłanie wejścia z powrotem jest zresztą samo w sobie zbędne.

    Args:
        request: Żądanie, które nie przeszło walidacji.
        exc: Wyjątek walidacji z listą błędów.

    Returns:
        Odpowiedź 422 z lokalizacją, komunikatem i typem każdego błędu.
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
    """Sonda zdrowia. Zwraca 200 wyłącznie, gdy start się powiódł.

    Returns:
        Stały status `ok`.
    """
    return HealthResponse(status="ok")


# Celowo `def`, nie `async def`. `predict_proba` to praca CPU-bound trwająca
# ~24 ms; na pętli zdarzeń żądania ustawiałyby się w kolejce jedno za drugim
# (~42 rps na workera). FastAPI uruchamia endpointy synchroniczne w puli
# wątków, więc zwykłe `def` jest tu poprawną odpowiedzią — i tego wymaga
# CLAUDE.md ("keep CPU-bound work off the event loop").
@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:  # type: ignore[valid-type]
    """Ocenia jeden wniosek kredytowy.

    Args:
        payload: Cechy wniosku; każde pole opcjonalne.
        request: Żądanie — po to, żeby sięgnąć po `app.state`.

    Returns:
        Prawdopodobieństwo, decyzja i próg, który ją wyprodukował.

    Raises:
        HTTPException: 500, gdy scoring się nie powiedzie.
    """
    bundle = request.app.state.bundle
    threshold = request.app.state.threshold

    frame = to_frame(payload.model_dump(exclude_none=True), bundle.groups)
    try:
        probability = float(bundle.pipeline.predict_proba(frame)[0, 1])
    except ValueError as exc:
        # ValueError ze sklearn-a to niemal zawsze złe wejście, nie awaria
        # serwera — 422, nie 500. Wcześniej łapał to bare `except` i każde
        # takie żądanie wyglądało jak błąd po naszej stronie.
        logger.warning("Odrzucone wejście: %s", exc)
        raise HTTPException(status_code=422, detail="Invalid feature values.") from None
    except Exception:
        # Treść wyjątku nie trafia do odpowiedzi — mogłaby ujawnić nazwy kolumn
        # i szczegóły modelu. Pełny ślad idzie do logów.
        logger.exception("Scoring nie powiódł się.")
        raise HTTPException(status_code=500, detail="Scoring failed.") from None

    return ScoreResponse(
        probability=probability,
        decision=probability >= threshold,
        threshold=threshold,
    )


def main() -> None:
    """Uruchamia serwis lokalnie: `uv run cs-serve`.

    Istnieje, bo bez tego jedyną wspieraną drogą uruchomienia był kontener,
    a `CLAUDE.md` wymaga, żeby polecenia startowe szły przez
    `[project.scripts]`.
    """
    import os

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",  # noqa: S104 — kontener musi nasłuchiwać poza localhostem
        port=int(os.environ.get("PORT", "8080")),
    )
