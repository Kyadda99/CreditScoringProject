"""Wejście CLI treningu: konfiguracja i metadane artefaktu."""

from __future__ import annotations

import pytest

from settings import TrainingSettings


def test_cli_defaults_come_from_training_settings() -> None:
    """Zmienne CS_* mają realnie sterować treningiem, a nie być ozdobą."""
    from train import build_parser

    settings = TrainingSettings(models=("xgboost",), arms=("none",))
    args = build_parser(settings).parse_args([])
    assert args.models == "xgboost"
    assert args.imbalance == "none"


def test_champion_metadata_records_the_split_that_was_actually_used() -> None:
    """Metadane opisują przebieg, więc muszą brać wartości z loadera."""
    from data_loader import DataLoader
    from train import champion_metadata

    loader = DataLoader(test_size=0.25, random_state=7)
    metadata = champion_metadata(
        metrics={"roc_auc": 0.5},
        model_name="xgboost",
        arm="none",
        engineered=True,
        loader=loader,
        rows_train=10,
        rows_test=5,
    )
    assert metadata["test_size"] == pytest.approx(0.25)
    assert metadata["random_state"] == 7
    assert metadata["rows_train"] == 10
    assert metadata["roc_auc"] == pytest.approx(0.5)


def test_engineered_flag_defaults_to_the_setting() -> None:
    """`--no-engineered` ma działać, a CS_ENGINEERED ma nie być nadpisywane."""
    from train import build_parser

    off = build_parser(TrainingSettings(engineered=False))
    assert off.parse_args([]).engineered is False
    assert off.parse_args(["--engineered"]).engineered is True

    on = build_parser(TrainingSettings(engineered=True))
    assert on.parse_args([]).engineered is True
    assert on.parse_args(["--no-engineered"]).engineered is False
