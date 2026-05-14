"""Gymnasium environment factory.

`make_env(env_id, seed)` returns an environment ready for training:
  - episode statistics tracked in `info["episode"]` on terminal steps
  - continuous action spaces rescaled to [-1, 1] for cleaner actor outputs
  - action-space RNG seeded so `env.action_space.sample()` is reproducible

The first `env.reset(seed=seed)` call must still be made by the trainer.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RescaleAction


CONTINUOUS_ENVS = {"Pendulum-v1", "MountainCarContinuous-v0"}
DISCRETE_ENVS = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"}


def is_discrete(env_id: str) -> bool:
    return env_id in DISCRETE_ENVS


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)
    if isinstance(env.action_space, gym.spaces.Box):
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env.action_space.seed(seed)
    return env


def env_info(env: gym.Env) -> dict:
    """Returns shapes / sizes the algorithms need to instantiate networks."""
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = int(obs_space.shape[0])
    if isinstance(act_space, gym.spaces.Discrete):
        return {
            "obs_dim": obs_dim,
            "discrete": True,
            "n_actions": int(act_space.n),
            "act_dim": 1,
            "act_low": None,
            "act_high": None,
        }
    if isinstance(act_space, gym.spaces.Box):
        return {
            "obs_dim": obs_dim,
            "discrete": False,
            "n_actions": None,
            "act_dim": int(act_space.shape[0]),
            "act_low": float(act_space.low.min()),
            "act_high": float(act_space.high.max()),
        }
    raise ValueError(f"Unsupported action space: {act_space}")
