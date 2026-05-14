"""TD3 (Fujimoto et al. 2018).

Three key innovations over DDPG, each independently important and ablation-worthy:
  1. Twin Q-networks with min taken in the bootstrap target.
  2. Target policy smoothing: noise added to the target action, then clipped,
     so the critic is regressed against a noisy local average of values.
  3. Delayed policy updates: the actor is updated every `policy_delay` critic
     updates, letting the critic stabilise before the actor chases it.

Implementation notes for the diary:
  - Exploration noise is independent of the target-smoothing noise. Common
     mistake to flag: re-using one for the other.
  - We clip the smoothed target action to the action range so target Q is
     evaluated at valid actions; without this, training can drift on envs
     with narrow action bounds.
  - Bootstrap uses terminated only (same convention as DQN/SAC).
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
from common.nets import DeterministicActor, MLPCritic, count_parameters
from common.seeding import get_device


DEFAULTS = {
    "gamma": 0.99,
    "tau": 0.005,
    "lr_actor": 3.0e-4,
    "lr_critic": 3.0e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "hidden": [256, 256],
    "learning_starts": 1_000,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_delay": 2,
    "exploration_noise": 0.1,
    "target_noise": 0.2,
    "target_noise_clip": 0.5,
}


def _polyak_update(online: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)


def train(env: gym.Env, cfg: dict, seed: int, total_steps: int, log_dir: str) -> None:
    cfg = {**DEFAULTS, **(cfg or {})}
    device = get_device()
    info = env_info(env)
    assert not info["discrete"], "TD3 requires a continuous action space."
    obs_dim = info["obs_dim"]
    act_dim = info["act_dim"]

    actor = DeterministicActor(obs_dim, act_dim, hidden=tuple(cfg["hidden"])).to(device)
    actor_target = DeterministicActor(obs_dim, act_dim, hidden=tuple(cfg["hidden"])).to(device)
    actor_target.load_state_dict(actor.state_dict())
    for p in actor_target.parameters():
        p.requires_grad = False

    q1 = MLPCritic(obs_dim, act_dim, hidden=tuple(cfg["hidden"])).to(device)
    q2 = MLPCritic(obs_dim, act_dim, hidden=tuple(cfg["hidden"])).to(device)
    q1_target = MLPCritic(obs_dim, act_dim, hidden=tuple(cfg["hidden"])).to(device)
    q2_target = MLPCritic(obs_dim, act_dim, hidden=tuple(cfg["hidden"])).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    for p in q1_target.parameters():
        p.requires_grad = False
    for p in q2_target.parameters():
        p.requires_grad = False

    optim_actor = torch.optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    optim_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg["lr_critic"])

    buffer = ReplayBuffer(
        capacity=cfg["buffer_size"],
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        discrete=False,
    )

    log_path = Path(log_dir) / "td3" / env.spec.id / f"seed{seed}.csv"
    update_count = 0

    obs, _ = env.reset(seed=seed)
    ep_return, ep_length = 0.0, 0
    with Logger(log_path) as logger:
        for step in range(1, total_steps + 1):
            if step < cfg["learning_starts"]:
                action_np = env.action_space.sample().astype(np.float32)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a = actor(obs_t).squeeze(0).cpu().numpy()
                noise = np.random.randn(act_dim).astype(np.float32) * cfg["exploration_noise"]
                action_np = np.clip(a + noise, -1.0, 1.0).astype(np.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_return += float(reward)
            ep_length += 1
            buffer.add(obs, action_np, float(reward), next_obs, done=float(terminated))

            obs = next_obs
            if terminated or truncated:
                logger.log_episode(step, ep_return, ep_length)
                obs, _ = env.reset()
                ep_return, ep_length = 0.0, 0

            # gradient updates
            if step >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
                for _ in range(cfg["gradient_steps"]):
                    batch = buffer.sample(cfg["batch_size"])

                    # critic update
                    with torch.no_grad():
                        next_a_mean = actor_target(batch["next_obs"])
                        noise = (torch.randn_like(next_a_mean) * cfg["target_noise"]).clamp(
                            -cfg["target_noise_clip"], cfg["target_noise_clip"]
                        )
                        next_a = (next_a_mean + noise).clamp(-1.0, 1.0)
                        next_q1 = q1_target(batch["next_obs"], next_a)
                        next_q2 = q2_target(batch["next_obs"], next_a)
                        next_q = torch.min(next_q1, next_q2)
                        target = batch["rewards"] + cfg["gamma"] * (1.0 - batch["dones"]) * next_q

                    q1_pred = q1(batch["obs"], batch["actions"])
                    q2_pred = q2(batch["obs"], batch["actions"])
                    critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

                    optim_q.zero_grad()
                    critic_loss.backward()
                    optim_q.step()

                    update_count += 1

                    # delayed policy + target updates
                    if update_count % cfg["policy_delay"] == 0:
                        new_a = actor(batch["obs"])
                        actor_loss = -q1(batch["obs"], new_a).mean()
                        optim_actor.zero_grad()
                        actor_loss.backward()
                        optim_actor.step()

                        _polyak_update(q1, q1_target, cfg["tau"])
                        _polyak_update(q2, q2_target, cfg["tau"])
                        _polyak_update(actor, actor_target, cfg["tau"])

    out = Path(log_dir) / "td3" / env.spec.id / f"seed{seed}.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "n_params_actor": count_parameters(actor),
            "n_params_critic_each": count_parameters(q1),
        },
        out,
    )
