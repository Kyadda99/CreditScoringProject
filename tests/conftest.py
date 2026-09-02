"""Wspólna konfiguracja pytest.

Testy kontraktu danych wymagają ~160 MB CSV, którego CI nie ma (brak
poświadczeń Kaggle i brak sensu pobierania takiego pliku w workflow).
Marker `requires_data` pozwala pominąć je czysto — bez wyłączania całego pliku.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import sklearn

from config import DATA_PATH


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Pomija testy oznaczone `requires_data`, gdy zbioru nie ma na dysku."""
    if DATA_PATH.exists():
        return
    skip = pytest.mark.skip(reason=f"Brak {DATA_PATH}; uruchom `uv run cs-download`")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _write_column_manifest() -> None:
    """Spisuje manifest kolumn z prawdziwego pliku, gdy `CS_WRITE_MANIFEST=1`.

    Jednorazowa czynność po pierwszym pobraniu danych; wynik jest commitowany
    i od tego momentu każda zmiana nazwy kolumny wywala test kontraktu.
    """
    if os.environ.get("CS_WRITE_MANIFEST") != "1" or not DATA_PATH.exists():
        return

    from data import load_data
    from test_data_contract import MANIFEST_PATH

    columns = list(load_data().columns)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(columns, indent=2), encoding="utf-8")
    print(f"\nZapisano manifest {len(columns)} kolumn -> {MANIFEST_PATH}")


@pytest.fixture(scope="session")
def synthetic_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Buduje maleńki, ale PRAWDZIWY artefakt modelu — bez 158 MB CSV.

    Powód: testy HTTP pilnują kryteriów ukończenia Fazy 1 (`/health`, `/score`
    zwracające prawdopodobieństwo, 422 na błędnym ciele). Gdy wisiały na
    markerze `requires_data`, CI pomijało je wszystkie i regresja w `/score`
    wchodziła na zielono. Kolumny bierzemy z **commitowanego**
    `feature_schema.json`, więc pipeline ma dokładnie ten kontrakt co produkcja
    — tylko nauczony na kilkunastu zmyślonych wierszach.
    """
    import numpy as np
    import pandas as pd

    from api.schemas import load_feature_groups
    from artifact import save_bundle
    from config import DROPPED_COLUMNS, split_feature_groups
    from features import FeatureEngineer
    from train import build_pipeline

    inputs = load_feature_groups()  # kontrakt WEJŚCIOWY — surowe kolumny
    rng = np.random.default_rng(0)
    rows = 40
    data: dict[str, object] = {}
    for name in inputs.numeric:
        # Dodatnie i z dala od zera: mianowniki ilorazów nie mogą być zerem,
        # bo cała kolumna wyszłaby NaN i imputer nie miałby czego się nauczyć.
        data[name] = rng.uniform(1.0, 100.0, size=rows)
    for name in inputs.binary:
        data[name] = rng.integers(0, 2, size=rows).astype(float)
    for name in inputs.categorical:
        data[name] = rng.choice(["A", "B"], size=rows)
    frame = pd.DataFrame(data, columns=list(inputs.all_features))
    target = rng.integers(0, 2, size=rows)

    # Grupy MODELU wyprowadzamy z ramki po inżynierii — tak samo jak `train.main`.
    engineered = FeatureEngineer().fit_transform(frame)
    groups = split_feature_groups(engineered, dropped=DROPPED_COLUMNS)

    pipeline = build_pipeline(groups).fit(frame, target)
    path = tmp_path_factory.mktemp("artifact") / "pipeline.joblib"
    return save_bundle(
        pipeline,
        groups,
        inputs,
        {"threshold": 0.5, "roc_auc": 0.5, "sklearn_version": sklearn.__version__},
        path,
    )
