"""Small helpers shared by the PyTorch modules."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device(preferred: str | None = None) -> torch.device:
    """cuda > mps (Apple silicon) > cpu, unless a device is given."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
