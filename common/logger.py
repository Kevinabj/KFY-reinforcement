"""Append-only CSV logger.

One file per (algo, env, seed). Columns:
    step              env-step counter at episode end
    episode           1-indexed episode counter
    episode_return    sum of rewards in the episode
    episode_length    number of env steps in the episode
    wall_clock_s      seconds since the run started
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path


class Logger:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.f)
        self.writer.writerow(["step", "episode", "episode_return", "episode_length", "wall_clock_s"])
        self.start = time.time()
        self.episode = 0

    def log_episode(self, step: int, ep_return: float, ep_length: int) -> None:
        self.episode += 1
        self.writer.writerow(
            [step, self.episode, float(ep_return), int(ep_length), round(time.time() - self.start, 3)]
        )
        if self.episode % 10 == 0:
            self.f.flush()

    def close(self) -> None:
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
