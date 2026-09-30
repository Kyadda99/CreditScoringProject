"""HTTP layer tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from api.config import get_settings
from config import FeatureGroups
from predictor import Predictor

# No module-level pytestmark here. It used to exist and made CI skip all 10
# HTTP tests, because `requires_data` needs the 158 MB CSV, so a regression
# in /score went green. They now run against the synthetic artifact built in
# conftest.py from the committed feature contract, so they need no data.


@pytest.fixture
def client(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Client that runs the lifespan; without `with` the model never loads.

    Points at the synthetic artifact from conftest.py, not a model trained on real
    data, so the whole HTTP set runs in CI.
    """
    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    # get_settings is lru_cached: without clearing it, settings from one test
    # would leak into the next.
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
    """Every field is optional; the imputer fills the rest."""
    assert client.post("/score", json={}).status_code == 200


def test_score_rejects_an_unknown_field(client: TestClient) -> None:
    assert client.post("/score", json={"NO_SUCH_COLUMN": 1}).status_code == 422


def test_score_rejects_a_wrong_type(client: TestClient) -> None:
    response = client.post("/score", json={"AMT_INCOME_TOTAL": "not a number"})
    assert response.status_code == 422


def test_model_is_loaded_once_not_per_request(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model loads once per process, measured rather than assumed."""
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

    assert len(calls) == 1, f"model loaded {len(calls)} times instead of once"


def test_to_frame_produces_training_column_order() -> None:
    """Column order must match training; ColumnTransformer depends on it."""
    groups = FeatureGroups(
        numeric=("AMT_INCOME_TOTAL",),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )
    frame = Predictor.build_frame({"AMT_INCOME_TOTAL": 1000.0}, groups)
    assert list(frame.columns) == list(groups.all_features)
    assert len(frame) == 1
    assert pd.isna(frame.loc[0, "NAME_CONTRACT_TYPE"])


def test_to_frame_gives_numeric_and_binary_columns_a_float_dtype() -> None:
    """Regression: `None` in an object column is not missing to SimpleImputer.

    The imputer looks for np.nan, and in an object array None != np.nan, so it
    finds no missing values. The result was 32 binary columns leaving
    preprocessing as NaN and /score returning 500 on every partial request.
    """
    groups = FeatureGroups(
        numeric=("AMT_INCOME_TOTAL",),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )
    frame = Predictor.build_frame({}, groups)
    assert frame["AMT_INCOME_TOTAL"].dtype == "float64"
    assert frame["FLAG_MOBIL"].dtype == "float64"
    assert frame["FLAG_MOBIL"].isna().all(), "missing must be NaN, not None"


def test_empty_payload_leaves_no_nan_after_preprocessing(
    synthetic_artifact: Path,
) -> None:
    """Regression at pipeline level, not just dtypes.

    Checks what actually broke scoring: after preprocessing an empty request, not
    one infinite or NaN value remains.
    """
    import numpy as np

    from artifact import load_bundle

    bundle = load_bundle(synthetic_artifact)
    frame = Predictor.build_frame({}, bundle.groups)
    transformed = bundle.pipeline.named_steps["preprocessor"].transform(frame)
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_lifespan_logs_that_the_model_was_loaded(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log record must be created when the model loads.

    Whether the log leaves the process cannot be tested under pytest: its logging
    plugin reassigns the root logger, so such a test would measure the test
    environment. Emission is handled by `logging.basicConfig` in `api.app`.
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

    assert any("Loading model" in r.getMessage() for r in records)


def test_lifespan_fails_closed_when_schema_and_artifact_disagree(
    synthetic_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The feature contract and the artifact enter the image from two sources.

    The schema arrives in the wheel from git, the artifact by COPY from the host
    disk. Without this check a mismatch raised nothing: a field present in the
    schema but unknown to the model was silently dropped, and a feature the model
    needed but the schema did not know came back as 422. The service kept scoring,
    incorrectly.
    """
    from config import FeatureGroups

    monkeypatch.setattr(
        app_module,
        "FEATURE_GROUPS",
        FeatureGroups(numeric=("SOMETHING_ELSE",), categorical=(), binary=()),
    )
    monkeypatch.setenv("CS_MODEL_PATH", str(synthetic_artifact))
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="Input contract does not match"):
        with TestClient(app_module.app):
            pass
    get_settings.cache_clear()


def test_score_rejects_infinity_with_422_not_500(client: TestClient) -> None:
    """Regression: Infinity reached sklearn and came back as a 500.

    An infinite income is a client error, not a server failure.
    """
    response = client.post(
        "/score",
        content=b'{"AMT_INCOME_TOTAL": Infinity}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_score_rejects_a_binary_flag_outside_zero_one(client: TestClient) -> None:
    """Regression: {"FLAG_MOBIL": 7.3} returned 200.

    An out-of-domain value went straight to the model and moved the score from
    0.0674 to 0.0045.
    """
    assert client.post("/score", json={"FLAG_MOBIL": 7.3}).status_code == 422
    assert client.post("/score", json={"FLAG_MOBIL": 1.0}).status_code == 200


def test_threshold_defaults_to_the_value_stored_in_the_artifact(
    client: TestClient, synthetic_artifact: Path
) -> None:
    """Training metrics describe the decision at the artifact threshold.

    If the service held its own constant, tuning the threshold would leave the
    service applying a different rule from the measured one, with no signal.
    """
    from artifact import load_bundle

    stored = float(load_bundle(synthetic_artifact).metadata["threshold"])
    assert client.post("/score", json={}).json()["threshold"] == stored


# --- Input contract vs model contract ---


def test_lifespan_starts_when_the_two_contracts_differ(client: TestClient) -> None:
    """Without repointing the guard, every container refuses to start."""
    assert client.get("/health").status_code == 200
    bundle = app_module.app.state.bundle
    assert bundle.groups != bundle.input_groups


def test_score_still_works_with_engineered_features(client: TestClient) -> None:
    response = client.post("/score", json={"AMT_CREDIT": 400000.0})
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_engineered_feature_names_are_rejected_as_input() -> None:
    """A client cannot send a derived feature; the container computes it."""
    from api.schemas import ScoreRequest

    assert "CREDIT_INCOME_RATIO" not in ScoreRequest.model_fields


def test_to_frame_builds_the_row_from_the_input_contract() -> None:
    inputs = FeatureGroups(numeric=("AMT_CREDIT",), categorical=("X",), binary=())
    frame = Predictor.build_frame({"AMT_CREDIT": 1.0}, inputs)
    assert list(frame.columns) == ["AMT_CREDIT", "X"]


def test_sentinel_is_decoded_inside_the_served_pipeline(client: TestClient) -> None:
    """A client may send 365243 and the served pipeline decodes it.

    Checks the `features` step of the loaded artifact rather than the probability:
    the synthetic model trains on 40 random rows and saturates, so both /score
    responses would be identical whether or not decoding works.
    """
    bundle = app_module.app.state.bundle
    engineer = bundle.pipeline.named_steps["features"]
    inputs = bundle.input_groups

    base = {"AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 406597.5}
    employed = engineer.transform(
        Predictor.build_frame({**base, "DAYS_EMPLOYED": -637.0}, inputs)
    )
    idle = engineer.transform(
        Predictor.build_frame({**base, "DAYS_EMPLOYED": 365243.0}, inputs)
    )

    assert employed["FLAG_NOT_EMPLOYED"].iloc[0] == 0.0
    assert idle["FLAG_NOT_EMPLOYED"].iloc[0] == 1.0
    assert pd.isna(idle["DAYS_EMPLOYED"].iloc[0])


def test_score_accepts_the_sentinel_without_error(client: TestClient) -> None:
    """The sentinel value must not turn scoring into a 500."""
    payload = {"AMT_INCOME_TOTAL": 202500.0, "DAYS_EMPLOYED": 365243.0}
    assert client.post("/score", json=payload).status_code == 200
