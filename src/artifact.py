"""Zapis i odczyt artefaktu modelu.

Pipeline i grupy cech jadą w **jednym** pliku. To nie jest wygoda, tylko
zabezpieczenie: `split_feature_groups` zależy od próbki, więc serwowanie musi
odtworzyć dokładnie te grupy, na których model był uczony. Dwa osobne pliki
dałoby się zaktualizować niezależnie — i dokładnie ten błąd tu eliminujemy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from config import PROJECT_ROOT, FeatureGroups

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH: Path = PROJECT_ROOT / "models" / "pipeline.joblib"

_PIPELINE_KEY = "pipeline"
_GROUPS_KEY = "feature_groups"
_METADATA_KEY = "metadata"


@dataclass(frozen=True)
class Bundle:
    """Wytrenowany model wraz ze wszystkim, czego potrzeba, żeby go użyć.

    Attributes:
        pipeline: Dopasowany `Pipeline` — preprocessing i estymator razem.
        groups: Grupy cech wyprowadzone na zbiorze treningowym. Kolejność
            `groups.all_features` wyznacza kolejność kolumn przy scoringu.
        metadata: Kontekst uruchomienia — metryki, wersje, liczności.
    """

    pipeline: Pipeline
    groups: FeatureGroups
    metadata: dict[str, Any]


def save_bundle(
    pipeline: Pipeline,
    groups: FeatureGroups,
    metadata: dict[str, Any],
    path: Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Zapisuje model, grupy i metadane do jednego pliku.

    Args:
        pipeline: Dopasowany pipeline.
        groups: Grupy cech użyte przy treningu.
        metadata: Dowolny słownik serializowalny przez joblib.
        path: Ścieżka docelowa; katalogi nadrzędne są tworzone.

    Returns:
        Ścieżka zapisanego pliku.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            _PIPELINE_KEY: pipeline,
            _GROUPS_KEY: groups.to_dict(),
            _METADATA_KEY: metadata,
        },
        path,
    )
    logger.info("Zapisano artefakt: %s (%.1f MB)", path, path.stat().st_size / 1024**2)
    return path


def load_bundle(path: Path = DEFAULT_ARTIFACT_PATH) -> Bundle:
    """Wczytuje artefakt zapisany przez `save_bundle`.

    Args:
        path: Ścieżka do pliku `.joblib`.

    Returns:
        Odtworzony `Bundle`.

    Raises:
        FileNotFoundError: Gdy pliku nie ma; komunikat wskazuje `cs-train`,
            bo to najczęstsza przyczyna.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Brak artefaktu modelu: {path}\nUruchom: uv run cs-train"
        )
    payload = joblib.load(path)
    metadata = payload[_METADATA_KEY]

    # `sklearn_version` było zapisywane i nigdy nieczytane. Artefakt powstaje
    # POZA obrazem i jest do niego kopiowany, więc nic nie gwarantuje zgodności
    # wersji — a joblib nie obiecuje przenośności piklów między wersjami
    # scikit-learn. Ostrzeżenie, nie wyjątek: rozbieżność zwykle działa, ale
    # gdy zacznie dawać dziwne wyniki, to jest pierwsze miejsce do sprawdzenia.
    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "Artefakt wytrenowany na scikit-learn %s, wczytywany na %s. "
            "Pikle sklearn nie są gwarantowane między wersjami.",
            trained_with,
            sklearn.__version__,
        )

    return Bundle(
        pipeline=payload[_PIPELINE_KEY],
        groups=FeatureGroups.from_dict(payload[_GROUPS_KEY]),
        metadata=metadata,
    )
