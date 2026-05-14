"""Global seeding: makes a training run reproducible.

We seed Python's RNG, NumPy, and PyTorch (CPU and CUDA), and force cuDNN into
deterministic mode. The env's RNG is seeded separately at reset time, and the
action-space RNG is seeded in `common.envs.make_env`.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Trades a small amount of speed for reproducible kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
