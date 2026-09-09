"""Cienkie opakowanie na MLflow Model Registry.

Aliasy, nie wycofane "stages" (`CLAUDE.md`, MLflow 3.x). Rejestr trzyma
**jeden** model o nazwie `credit-scoring`, którego wersje są kolejnymi
championami — alias `@production` wskazuje dokładnie jeden z nich (spec D6).

Projekt referencyjny instruktora rejestruje osobną nazwę per algorytm, co
dałoby tutaj trzy konkurujące aliasy `@production` znaczące trzy różne rzeczy.

**Rejestr jest sprawą treningową** (spec D7): kontener nie zna MLflow i nie
sięga po sieć przy starcie. Champion trafia do obrazu przez `export_champion`
jako zwykły joblib, ale z zapisanym pochodzeniem — wersją rejestru i biegiem,
z którego wyszedł.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from artifact import DEFAULT_ARTIFACT_PATH, load_bundle, save_bundle
from config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "credit-scoring"
"""Jedna nazwa w rejestrze; wersje to kolejne generacje championa."""

PRODUCTION_ALIAS = "production"
"""Alias wskazujący model, który ma pojechać do kontenera."""

MODEL_ARTIFACT_PATH = "model"
"""Ścieżka artefaktu sklearn wewnątrz biegu — `runs:/<id>/model`.

Musi zgadzać się z `name=` przekazanym do `mlflow.sklearn.log_model`,
inaczej `register` nie znajdzie czego rejestrować.
"""

BUNDLE_ARTIFACT_PATH = "bundle"
"""Katalog artefaktu z `Bundle` wewnątrz biegu (spec D7)."""

BUNDLE_FILENAME = "pipeline.joblib"
"""Nazwa pliku artefaktu wewnątrz katalogu `bundle/`."""

SERIALIZATION_FORMAT = "cloudpickle"
"""Format zapisu modelu sklearn w MLflow.

MLflow 3.15 domyślnie używa **skops**, który odmawia serializacji klas spoza
listy zaufanych i wywala się na naszym pipelinie:

    UntrustedTypesFoundException: Untrusted types found in the file:
    ['features.FeatureEngineer', 'numpy.dtype', 'xgboost.core.Booster',
     'xgboost.sklearn.XGBClassifier']

Alternatywą było wyliczenie `skops_trusted_types`, ale ta lista musiałaby być
aktualizowana przy każdej zmianie składu pipeline'u — cichy błąd czekający na
Fazę 4, która dokłada model PyTorch.

Argument bezpieczeństwa za skops tutaj nie obowiązuje: produkujemy i wczytujemy
wyłącznie własne artefakty, a kontener i tak ładuje `joblib` (spec D7). Trzymanie
kopii rejestrowej w skops, gdy wdrażana kopia jest picklem, byłoby teatrem.
"""


def _client() -> MlflowClient:
    """Zwraca klienta wskazującego współdzielony backend SQLite.

    Tworzony przy każdym wywołaniu, a nie jako singleton modułu: testy
    podmieniają URI, a singleton zapamiętałby ten sprzed podmiany.

    Returns:
        Klient MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def register(run_id: str, name: str = REGISTERED_MODEL_NAME) -> ModelVersion:
    """Rejestruje model zalogowany w danym biegu.

    Args:
        run_id: Bieg, w którym `mlflow.sklearn.log_model(..., name="model")`
            zapisał pipeline.
        name: Nazwa w rejestrze.

    Returns:
        Utworzona wersja modelu.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", name=name
    )
    logger.info("Zarejestrowano %s v%s (bieg %s)", name, mv.version, run_id[:8])
    return mv


def promote(mv: ModelVersion, alias: str = PRODUCTION_ALIAS) -> None:
    """Przypina alias do wskazanej wersji.

    Alias jest przenoszony, a nie dublowany — po tym wywołaniu `@production`
    wskazuje dokładnie jedną wersję.

    Args:
        mv: Wersja zwrócona przez `register`.
        alias: Nazwa aliasu.
    """
    _client().set_registered_model_alias(name=mv.name, alias=alias, version=mv.version)
    logger.info("%s v%s -> @%s", mv.name, mv.version, alias)


def load_production(
    name: str = REGISTERED_MODEL_NAME, alias: str = PRODUCTION_ALIAS
) -> Pipeline:
    """Wczytuje model spod aliasu.

    Używane po stronie **treningu** (np. do porównania z nowym kandydatem).
    API tego nie woła — kontener ładuje joblib, nie rejestr (spec D7).

    Args:
        name: Nazwa w rejestrze.
        alias: Alias.

    Returns:
        Dopasowany pipeline.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def list_models(name: str | None = None) -> None:
    """Wypisuje do logu zarejestrowane modele, ich wersje i aliasy.

    Args:
        name: Ogranicza do jednej nazwy; `None` pokazuje wszystkie.
    """
    client = _client()
    names = [name] if name else [m.name for m in client.search_registered_models()]
    for model_name in names:
        registered = client.get_registered_model(model_name)
        # Aliasy mieszkają na MODELU (alias -> wersja), nie na wersji, więc
        # odwracamy mapę, żeby wypisać je przy odpowiednim wierszu.
        by_version: dict[str, list[str]] = {}
        for alias, version in registered.aliases.items():
            by_version.setdefault(version, []).append(f"@{alias}")
        for mv in sorted(
            client.search_model_versions(f"name='{model_name}'"),
            key=lambda v: int(v.version),
        ):
            logger.info(
                "%-20s v%-3s %-14s bieg %s",
                model_name,
                mv.version,
                ", ".join(by_version.get(mv.version, [])) or "-",
                (mv.run_id or "")[:8],
            )


def export_champion(
    name: str = REGISTERED_MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
    dest: Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Wyciąga z rejestru artefakt championa i zapisuje go jako joblib.

    To jest spoina między rejestrem a kontenerem (spec D7). Alias wskazuje
    wersję, wersja zna swój bieg, a bieg zawiera **dokładnie ten** `Bundle`,
    który ta wersja opisuje. Dzięki temu plik w `models/` ma udokumentowane
    pochodzenie, zamiast być "tym, co akurat leżało na dysku".

    Numer wersji powstaje dopiero PRZY rejestracji, więc nie mógł znaleźć się
    w `Bundle` logowanym wcześniej — dopisujemy go tutaj i zapisujemy ponownie.

    Args:
        name: Nazwa w rejestrze.
        alias: Alias championa.
        dest: Docelowa ścieżka joblib.

    Returns:
        Ścieżka zapisanego artefaktu.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mv = _client().get_model_version_by_alias(name, alias)
    logger.info("Champion: %s v%s (bieg %s)", name, mv.version, (mv.run_id or "")[:8])

    local = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path=f"{BUNDLE_ARTIFACT_PATH}/{BUNDLE_FILENAME}"
    )
    bundle = load_bundle(Path(local))
    metadata = {
        **bundle.metadata,
        "registry_model": name,
        "registry_version": mv.version,
        "registry_alias": alias,
        "registry_run_id": mv.run_id,
    }
    path = save_bundle(
        bundle.pipeline, bundle.groups, bundle.input_groups, metadata, dest
    )
    logger.info("Wyeksportowano championa do %s", path)
    return path


def cli() -> None:
    """Wejście `uv run cs-export`."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Eksportuje championa z rejestru do models/pipeline.joblib."
    )
    parser.add_argument("--alias", default=PRODUCTION_ALIAS)
    parser.add_argument("--name", default=REGISTERED_MODEL_NAME)
    args = parser.parse_args()

    list_models(args.name)
    export_champion(name=args.name, alias=args.alias)


def production_metrics(
    name: str = REGISTERED_MODEL_NAME, alias: str = PRODUCTION_ALIAS
) -> dict[str, float] | None:
    """Zwraca metryki modelu stojącego obecnie pod aliasem.

    Potrzebne, żeby `beats_incumbent` mogło porównać kandydata z obecnym
    championem, zamiast promować bezwarunkowo.

    Args:
        name: Nazwa w rejestrze.
        alias: Alias championa.

    Returns:
        Metryki biegu, z którego pochodzi wskazana wersja, albo `None`,
        gdy aliasu (lub całego modelu) jeszcze nie ma.
    """
    client = _client()
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        logger.info("Brak aliasu @%s dla %s — to pierwszy champion.", alias, name)
        return None
    return dict(client.get_run(mv.run_id).data.metrics)
