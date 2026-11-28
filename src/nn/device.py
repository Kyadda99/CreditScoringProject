"""Device selection and seeding for the neural networks."""

from __future__ import annotations

import random

import numpy as np
import torch

from config import RANDOM_STATE

SEED: int = RANDOM_STATE


def select_device() -> torch.device:
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and torch, including every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
