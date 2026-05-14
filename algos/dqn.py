"""DQN (Mnih et al. 2013/2015) with target network and Huber loss.

Implementation notes (paste these into the diary if they bite us in practice):
  - We bootstrap with `terminated` only, not `terminated or truncated`. A
    truncation (time-limit) does not mean the state has no future value, so
    masking it out of the Bellman target biases Q downward on long-horizon
    envs like Acrobot.
  - Huber (smooth-L1) loss is materially more stable than MSE early in
    training, when Q values are noisy and outliers are common.
  - The target network is hard-synced (copy) every `target_sync` env steps.
    A polyak (soft) update is an alternative; we chose hard sync because it
    is what the original DQN paper used and it keeps the ablation cleaner.
  - epsilon decays linearly from `eps_start` to `eps_end` over the first
    `eps_decay_fraction * total_steps`, then stays at `eps_end`.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from common.buffers import ReplayBuffer
from common.envs import env_info
from common.logger import Logger
from common.nets import MLPQNet, count_parameters
from common.seeding import get_device


DEFAULTS = {
    "gamma": 0.99,
    "lr": 1e-3,
    "buffer_size": 50_000,
    "batch_size": 64,
    "hidden": (64, 64),
    "learning_starts": 1_000,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_sync": 1_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_fraction": 0.10,
    "max_grad_norm": 10.0,
}


def _epsilon(step: int, total_steps: int, cfg: dict) -> float:
    decay_end = max(1, int(cfg["eps_decay_fraction"] * total_steps))
    frac = min(1.0, step / decay_end)
    return cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])


def train(env: gym.Env, cfg: dict, seed: int, total_steps: int, log_dir: str) -> None:
    cfg = {**DEFAULTS, **(cfg or {})}
    device = get_device()

    info = env_info(env)
    assert info["discrete"], "DQN requires a discrete action space."
    obs_dim = info["obs_dim"]
    n_actions = info["n_actions"]

    qnet = MLPQNet(obs_dim, n_actions, hidden=tuple(cfg["hidden"])).to(device)
    target_qnet = MLPQNet(obs_dim, n_actions, hidden=tuple(cfg["hidden"])).to(device)
    target_qnet.load_state_dict(qnet.state_dict())
    for p in target_qnet.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(qnet.parameters(), lr=cfg["lr"])

    buffer = ReplayBuffer(
        capacity=cfg["buffer_size"],
        obs_dim=obs_dim,
        act_dim=1,
        device=device,
        discrete=True,
    )

    log_path = Path(log_dir) / "dqn" / env.spec.id / f"seed{seed}.csv"

    obs, _ = env.reset(seed=seed)
    ep_return, ep_length = 0.0, 0
    with Logger(log_path) as logger:
        for step in range(1, total_steps + 1):
            eps = _epsilon(step, total_steps, cfg)

            # action selection: epsilon-greedy
            if step < cfg["learning_starts"] or np.random.random() < eps:
                action = int(env.action_space.sample())
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(qnet(obs_t).argmax(dim=-1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1

            # Bootstrap only on real termination; truncation should not mask future value.
            buffer.add(obs, action, float(reward), next_obs, done=float(terminated))

            obs = next_obs

            if terminated or truncated:
                logger.log_episode(step, ep_return, ep_length)
                obs, _ = env.reset()
                ep_return, ep_length = 0.0, 0

            # gradient updates
            if step >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
                for _ in range(cfg["gradient_steps"]):
                    batch = buffer.sample(cfg["batch_size"])
                    with torch.no_grad():
                        next_q_max = target_qnet(batch["next_obs"]).max(dim=-1).values
                        target = batch["rewards"] + cfg["gamma"] * (1.0 - batch["dones"]) * next_q_max
                    q_sa = qnet(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
                    loss = F.smooth_l1_loss(q_sa, target)
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(qnet.parameters(), cfg["max_grad_norm"])
                    optim.step()

            if step % cfg["target_sync"] == 0:
                target_qnet.load_state_dict(qnet.state_dict())

    # Save final policy for parameter counting / reuse.
    out = Path(log_dir) / "dqn" / env.spec.id / f"seed{seed}.pt"
    torch.save({"qnet": qnet.state_dict(), "n_params": count_parameters(qnet)}, out)
