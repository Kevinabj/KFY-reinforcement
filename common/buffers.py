"""Experience buffers.

Two classes:
  - ReplayBuffer: circular off-policy buffer used by DQN, SAC, TD3.
  - RolloutBuffer: fixed-size on-policy buffer with GAE-lambda advantages
    used by PPO.

Both keep storage in numpy and only move data to the device (GPU) on sample.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


def _as_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        discrete: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discrete = discrete
        self.device = device

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        if discrete:
            self.actions = np.zeros(self.capacity, dtype=np.int64)
        else:
            self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)
        if self.discrete:
            self.actions[self.idx] = int(action)
        else:
            self.actions[self.idx] = action
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        ids = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": _as_tensor(self.obs[ids], self.device, torch.float32),
            "next_obs": _as_tensor(self.next_obs[ids], self.device, torch.float32),
            "rewards": _as_tensor(self.rewards[ids], self.device, torch.float32),
            "dones": _as_tensor(self.dones[ids], self.device, torch.float32),
        }
        if self.discrete:
            batch["actions"] = _as_tensor(self.actions[ids], self.device, torch.long)
        else:
            batch["actions"] = _as_tensor(self.actions[ids], self.device, torch.float32)
        return batch

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """On-policy rollout for PPO with GAE-lambda advantages."""

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        act_dim: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
        discrete: bool = False,
    ) -> None:
        self.n_steps = int(n_steps)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.discrete = discrete

        self.obs = np.zeros((self.n_steps, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        if discrete:
            self.actions = np.zeros(self.n_steps, dtype=np.int64)
        else:
            self.actions = np.zeros((self.n_steps, act_dim), dtype=np.float32)

        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action,
        log_prob: float,
        reward: float,
        done: float,
        value: float,
    ) -> None:
        self.obs[self.ptr] = obs
        if self.discrete:
            self.actions[self.ptr] = int(action)
        else:
            self.actions[self.ptr] = action
        self.log_probs[self.ptr] = float(log_prob)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = float(value)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(self, last_value: float, last_done: float) -> None:
        """Computes GAE-lambda advantages and returns in-place.

        Convention: `dones[t]` is the terminated flag of the transition stored
        at index t (i.e., did step t end the episode). For the bootstrap into
        the value of the state AFTER step t we mask out `dones[t]`, because if
        step t terminated then the obs stored at index t+1 is a fresh episode
        start, not the actual next state of the same trajectory.

        Caller passes `last_value = V(obs_after_last_step)` and
        `last_done = terminated_at_last_step` so the boundary case (t = n_steps-1)
        gets the right mask.
        """
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                non_terminal = 1.0 - last_done
            else:
                next_value = self.values[t + 1]
                non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get(self, batch_size: int) -> Iterator[dict]:
        """Yields mini-batches of the full rollout in random order."""
        assert self.is_full(), "Rollout buffer must be full before sampling."
        ids = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            mb = ids[start : start + batch_size]
            batch = {
                "obs": _as_tensor(self.obs[mb], self.device, torch.float32),
                "old_log_probs": _as_tensor(self.log_probs[mb], self.device, torch.float32),
                "advantages": _as_tensor(self.advantages[mb], self.device, torch.float32),
                "returns": _as_tensor(self.returns[mb], self.device, torch.float32),
                "old_values": _as_tensor(self.values[mb], self.device, torch.float32),
            }
            if self.discrete:
                batch["actions"] = _as_tensor(self.actions[mb], self.device, torch.long)
            else:
                batch["actions"] = _as_tensor(self.actions[mb], self.device, torch.float32)
            yield batch

    def reset(self) -> None:
        self.ptr = 0
