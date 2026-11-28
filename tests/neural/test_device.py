"""Device selection and seeding."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from nn import device as device_module
from nn.device import SEED, seed_everything, select_device


def _availability(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_cuda_wins_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _availability(monkeypatch, cuda=True, mps=True)
    assert select_device() == torch.device("cuda")


def test_mps_is_used_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _availability(monkeypatch, cuda=False, mps=True)
    assert select_device() == torch.device("mps")


def test_cpu_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _availability(monkeypatch, cuda=False, mps=False)
    assert select_device() == torch.device("cpu")


def test_seed_is_the_project_seed() -> None:
    assert SEED == 42


def test_seed_everything_makes_every_generator_repeat() -> None:
    def draw() -> tuple[float, float, list[float]]:
        return random.random(), float(np.random.rand()), torch.rand(3).tolist()

    seed_everything(7)
    first = draw()
    seed_everything(7)
    assert draw() == first


def test_seed_everything_seeds_every_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    monkeypatch.setattr(device_module.torch.cuda, "manual_seed_all", seen.append)
    seed_everything(3)
    assert seen
    assert set(seen) == {3}
