"""Small PyTorch helpers for the trainers."""

from __future__ import annotations

import torch


def worker_init(_worker_id: int) -> None:
    """Give each data-loading worker its own augmentation stream; otherwise
    every worker starts from a copy of the same random state."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng.seed(torch.initial_seed())


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
