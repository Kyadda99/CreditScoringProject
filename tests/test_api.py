"""Testy warstwy HTTP."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from api.config import get_settings
from config import FeatureGroups

# UWAGA: NIE ma tu `pytestmark` na cały moduł. Wcześniej był i pomijał w CI
# wszystkie 10 testów HTTP, bo `requires_data` zależy od 158 MB CSV — czyli
# kryteria ukończenia Fazy 1 (/health, /score, 422, "model ładuje się raz")
# pilnowały testy, które nigdy się nie wykonywały, a regresja w /score wchodziła
# na zielono. Teraz wszystkie działają na syntetycznym artefakcie zbudowanym
# w `conftest.py` z **commitowanego** kontraktu cech, więc nie potrzebują danych.


@pytest.fixture
def client(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Klient uruchamiający lifespan — bez `with` model nigdy się nie ładuje.

    Wskazuje na syntetyczny artefakt z `conftest.py`, nie na model wytrenowany
    na prawdziwych danych. Dzięki temu cały zestaw HTTP działa w CI, gdzie nie
    ma ani 158 MB CSV, ani wytrenowanego modelu — a to właśnie te testy
    pilnują kryteriów ukończenia Fazy 1.
    """
    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    # get_settings jest lru_cache'owane: bez czyszczenia ustawienia z jednego
    # testu wyciekłyby do kolejnych.
    get_settings.cache_clear()
    with TestClient(app_module.app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_returns_a_probability_decision_and_threshold(
    client: TestClient,
) -> None:
    response = client.post("/score", json={"AMT_INCOME_TOTAL": 202500.0})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["decision"], bool)
    assert body["decision"] is (body["probability"] >= body["threshold"])


def test_score_accepts_an_empty_payload(client: TestClient) -> None:
    """Wszystkie pola są opcjonalne — imputer uzupełnia całość."""
    assert client.post("/score", json={}).status_code == 200


def test_score_rejects_an_unknown_field(client: TestClient) -> None:
    assert client.post("/score", json={"NIE_MA_TAKIEJ": 1}).status_code == 422


def test_score_rejects_a_wrong_type(client: TestClient) -> None:
    response = client.post("/score", json={"AMT_INCOME_TOTAL": "nie liczba"})
    assert response.status_code == 422


def test_model_is_loaded_once_not_per_request(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kryterium ukończenia Fazy 1 — mierzone, nie zakładane z lektury kodu."""
    calls: list[int] = []
    original = app_module.load_bundle

    def counting_load(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "load_bundle", counting_load)
    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    get_settings.cache_clear()

    with TestClient(app_module.app) as client:
        for _ in range(5):
            assert client.post("/score", json={}).status_code == 200

    assert len(calls) == 1, f"model wczytany {len(calls)} razy zamiast raz"


def test_to_frame_produces_training_column_order() -> None:
    """Kolejność kolumn musi być ta z treningu — ColumnTransformer jej pilnuje."""
    groups = FeatureGroups(
        numeric=("AMT_INCOME_TOTAL",),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )
    frame = app_module.to_frame({"AMT_INCOME_TOTAL": 1000.0}, groups)
    assert list(frame.columns) == list(groups.all_features)
    assert len(frame) == 1
    assert pd.isna(frame.loc[0, "NAME_CONTRACT_TYPE"])


def test_to_frame_gives_numeric_and_binary_columns_a_float_dtype() -> None:
    """Regresja: `None` w kolumnie `object` nie jest brakiem dla SimpleImputer.

    Imputer szuka `np.nan`, a w tablicy `object` `None != np.nan`, więc braków
    nie rozpoznaje. Skutkiem było 32 kolumny binarne wychodzące z preprocessingu
    jako NaN i `/score` zwracające 500 na każdym niepełnym żądaniu.
    """
    groups = FeatureGroups(
        numeric=("AMT_INCOME_TOTAL",),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )
    frame = app_module.to_frame({}, groups)
    assert frame["AMT_INCOME_TOTAL"].dtype == "float64"
    assert frame["FLAG_MOBIL"].dtype == "float64"
    assert frame["FLAG_MOBIL"].isna().all(), "brak musi być NaN, nie None"


def test_empty_payload_leaves_no_nan_after_preprocessing(
    synthetic_artifact: Path,
) -> None:
    """Regresja na poziomie pipeline'u, nie tylko dtype'ów.

    Sprawdza to, co faktycznie wywracało scoring: że po preprocessingu pustego
    żądania nie zostaje ani jedna wartość nieskończona lub NaN.
    """
    import numpy as np

    from artifact import load_bundle

    bundle = load_bundle(synthetic_artifact)
    frame = app_module.to_frame({}, bundle.groups)
    transformed = bundle.pipeline.named_steps["preprocessor"].transform(frame)
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_lifespan_logs_that_the_model_was_loaded(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kryterium Fazy 1 wymaga DOWODU z logu, więc komunikat musi powstawać.

    Ten test sprawdza, że rekord logu w ogóle powstaje. Tego, czy log
    **wychodzi** z procesu, pod pytestem sprawdzić się nie da — plugin logujący
    pytesta i tak przestawia root logger, więc każdy taki test mierzyłby
    środowisko testowe, a nie aplikację. Za wyjście logu odpowiada
    `logging.basicConfig` w `api.app`; dowód pochodzi z uruchomionego kontenera
    i jest zapisany w `docs/findings/01-skeleton.md`.
    """
    import logging

    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    get_settings.cache_clear()
    app_logger = logging.getLogger("api.app")
    handler = Collector(level=logging.INFO)
    app_logger.addHandler(handler)
    previous_level = app_logger.level
    app_logger.setLevel(logging.INFO)
    try:
        with TestClient(app_module.app):
            pass
    finally:
        app_logger.removeHandler(handler)
        app_logger.setLevel(previous_level)

    assert any("Ładuję model" in r.getMessage() for r in records)


def test_lifespan_fails_closed_when_schema_and_artifact_disagree(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kontrakt cech i artefakt wchodzą do obrazu z dwóch różnych źródeł.

    Schemat jedzie kołem z gita, artefakt przez `COPY models/…` z dysku hosta.
    Bez tej kontroli rozjazd nie dawał żadnego błędu: pole obecne w schemacie,
    a nieznane modelowi, było po cichu gubione, a cecha potrzebna modelowi,
    ale nieznana schematowi, wracała jako 422. Serwis scorował dalej — źle.
    """
    from config import FeatureGroups

    monkeypatch.setattr(
        app_module,
        "FEATURE_GROUPS",
        FeatureGroups(numeric=("COS_INNEGO",), categorical=(), binary=()),
    )
    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="Kontrakt wejściowy nie zgadza"):
        with TestClient(app_module.app):
            pass
    get_settings.cache_clear()


def test_score_rejects_infinity_with_422_not_500(client: TestClient) -> None:
    """Regresja: `Infinity` docierało do sklearn-a i wracało jako 500.

    Nieskończoność w dochodzie to błąd klienta, nie awaria serwera.
    """
    response = client.post(
        "/score",
        content=b'{"AMT_INCOME_TOTAL": Infinity}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_score_rejects_a_binary_flag_outside_zero_one(client: TestClient) -> None:
    """Regresja: `{"FLAG_MOBIL": 7.3}` zwracało 200.

    Wartość spoza dziedziny szła prosto do modelu i realnie przesuwała wynik —
    z 0.0674 na 0.0045.
    """
    assert client.post("/score", json={"FLAG_MOBIL": 7.3}).status_code == 422
    assert client.post("/score", json={"FLAG_MOBIL": 1.0}).status_code == 200


def test_threshold_defaults_to_the_value_stored_in_the_artifact(
    client: TestClient, synthetic_artifact: Path
) -> None:
    """Metryki z treningu opisują decyzję przy progu z artefaktu.

    Gdyby serwis trzymał własną stałą, Faza 3 dostroiłaby próg, a serwis nadal
    stosowałby inną regułę niż ta, którą zmierzono — bez żadnego sygnału.
    """
    from artifact import load_bundle

    stored = float(load_bundle(synthetic_artifact).metadata["threshold"])
    assert client.post("/score", json={}).json()["threshold"] == stored


# --- Faza 2: kontrakt wejściowy vs kontrakt modelu (spec D2) ---------------


def test_lifespan_starts_when_the_two_contracts_differ(client: TestClient) -> None:
    """Bez repointu strażnika każdy kontener po Fazie 2 odmawia startu."""
    assert client.get("/health").status_code == 200
    bundle = app_module.app.state.bundle
    assert bundle.groups != bundle.input_groups


def test_score_still_works_with_engineered_features(client: TestClient) -> None:
    response = client.post("/score", json={"AMT_CREDIT": 400000.0})
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_engineered_feature_names_are_rejected_as_input() -> None:
    """Klient nie może przysłać cechy pochodnej — liczy ją kontener (D2)."""
    from api.schemas import ScoreRequest

    assert "CREDIT_INCOME_RATIO" not in ScoreRequest.model_fields


def test_to_frame_builds_the_row_from_the_input_contract() -> None:
    inputs = FeatureGroups(numeric=("AMT_CREDIT",), categorical=("X",), binary=())
    frame = app_module.to_frame({"AMT_CREDIT": 1.0}, inputs)
    assert list(frame.columns) == ["AMT_CREDIT", "X"]


def test_sentinel_is_decoded_inside_the_served_pipeline(client: TestClient) -> None:
    """Spec D4: klient może przysłać 365243 i pipeline serwisu to rozpoznaje.

    Sprawdzamy krok `features` załadowanego artefaktu, a nie samo
    prawdopodobieństwo: syntetyczny model uczy się na 40 losowych wierszach
    i saturuje wynik, więc na `/score` obie odpowiedzi byłyby identyczne
    niezależnie od tego, czy dekodowanie działa.
    """
    bundle = app_module.app.state.bundle
    engineer = bundle.pipeline.named_steps["features"]
    inputs = bundle.input_groups

    base = {"AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 406597.5}
    employed = engineer.transform(
        app_module.to_frame({**base, "DAYS_EMPLOYED": -637.0}, inputs)
    )
    idle = engineer.transform(
        app_module.to_frame({**base, "DAYS_EMPLOYED": 365243.0}, inputs)
    )

    assert employed["FLAG_NOT_EMPLOYED"].iloc[0] == 0.0
    assert idle["FLAG_NOT_EMPLOYED"].iloc[0] == 1.0
    assert pd.isna(idle["DAYS_EMPLOYED"].iloc[0])


def test_score_accepts_the_sentinel_without_error(client: TestClient) -> None:
    """Wartość sentinelowa nie może wywrócić scoringu na 500."""
    payload = {"AMT_INCOME_TOTAL": 202500.0, "DAYS_EMPLOYED": 365243.0}
    assert client.post("/score", json=payload).status_code == 200
