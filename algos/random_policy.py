"""Random-policy placeholder.

Exists only to validate the full training pipeline end-to-end before any real
algorithm is in place. Same `train(env, cfg, seed, total_steps, log_dir)`
interface every real algorithm will expose.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from common.logger import Logger


def train(env: gym.Env, cfg: dict, seed: int, total_steps: int, log_dir: str) -> None:
    log_path = Path(log_dir) / "random" / env.spec.id / f"seed{seed}.csv"
    obs, _ = env.reset(seed=seed)
    ep_return, ep_length = 0.0, 0
    with Logger(log_path) as logger:
        for step in range(1, total_steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            if terminated or truncated:
                logger.log_episode(step, ep_return, ep_length)
                obs, _ = env.reset()
                ep_return, ep_length = 0.0, 0
