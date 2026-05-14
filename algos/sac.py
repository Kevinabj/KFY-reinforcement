"""SAC (Haarnoja et al. 2018) with automatic temperature tuning.

Implementation notes for the diary:
  - Twin Q-networks Q1, Q2 with target copies updated by Polyak averaging
    (tau = 0.005). The policy loss uses min(Q1, Q2) to reduce overestimation,
    same as TD3.
  - Squashed-Gaussian policy: u ~ N(mu, sigma), a = tanh(u). The log-prob
    carries the tanh change-of-variables correction; without it, the
    entropy term is wrong and training silently fails. This was the
    canonical SAC bug to flag in the diary.
  - Automatic temperature tuning: we optimise log_alpha so the expected
    log-prob matches target_entropy = -|A|. The log parameterisation keeps
    alpha positive without a hard constraint.
  - Reparameterised sampling (`rsample`) for the policy loss so gradients
    flow through the action; non-reparameterised would block them.
  - Bootstrap mask uses terminated only; truncation does not zero the
    future value (same convention as DQN).
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
from common.nets import MLPCritic, SquashedGaussianActor, count_parameters
from common.seeding import get_device


DEFAULTS = {
    "gamma": 0.99,
    "tau": 0.005,
    "lr_actor": 3.0e-4,
    "lr_critic": 3.0e-4,
    "lr_alpha": 3.0e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "hidden": [256, 256],
    "learning_starts": 1_000,
    "train_freq": 1,
    "gradient_steps": 1,
    "autotune_alpha": True,
    "init_log_alpha": 0.0,
    "target_entropy_scale": 1.0,   # target entropy = -act_dim * scale
}


def _polyak_update(online: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)


def train(env: gym.Env, cfg: dict, seed: int, total_steps: int, log_dir: str) -> None:
    cfg = {**DEFAULTS, **(cfg or {})}
    device = get_device()
    info = env_info(env)
    assert not info["discrete"], "SAC requires a continuous action space."
    obs_dim = info["obs_dim"]
    act_dim = info["act_dim"]

    actor = SquashedGaussianActor(obs_dim, act_dim, hidden=tuple(cfg["hidden"])).to(device)
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

    target_entropy = -float(act_dim) * cfg["target_entropy_scale"]
    log_alpha = torch.tensor(float(cfg["init_log_alpha"]), device=device, requires_grad=cfg["autotune_alpha"])
    if cfg["autotune_alpha"]:
        optim_alpha = torch.optim.Adam([log_alpha], lr=cfg["lr_alpha"])

    buffer = ReplayBuffer(
        capacity=cfg["buffer_size"],
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        discrete=False,
    )

    log_path = Path(log_dir) / "sac" / env.spec.id / f"seed{seed}.csv"

    obs, _ = env.reset(seed=seed)
    ep_return, ep_length = 0.0, 0
    with Logger(log_path) as logger:
        for step in range(1, total_steps + 1):
            # action selection
            if step < cfg["learning_starts"]:
                action_np = env.action_space.sample().astype(np.float32)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a, _ = actor.sample(obs_t)
                action_np = a.squeeze(0).cpu().numpy().astype(np.float32)

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
                    alpha = log_alpha.exp().detach()

                    # critic update
                    with torch.no_grad():
                        next_a, next_log_pi = actor.sample(batch["next_obs"])
                        next_q1 = q1_target(batch["next_obs"], next_a)
                        next_q2 = q2_target(batch["next_obs"], next_a)
                        next_q = torch.min(next_q1, next_q2) - alpha * next_log_pi
                        target = batch["rewards"] + cfg["gamma"] * (1.0 - batch["dones"]) * next_q

                    q1_pred = q1(batch["obs"], batch["actions"])
                    q2_pred = q2(batch["obs"], batch["actions"])
                    critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

                    optim_q.zero_grad()
                    critic_loss.backward()
                    optim_q.step()

                    # actor update
                    new_a, new_log_pi = actor.sample(batch["obs"])
                    q1_new = q1(batch["obs"], new_a)
                    q2_new = q2(batch["obs"], new_a)
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss = (alpha * new_log_pi - q_new).mean()

                    optim_actor.zero_grad()
                    actor_loss.backward()
                    optim_actor.step()

                    # alpha update
                    if cfg["autotune_alpha"]:
                        alpha_loss = -(log_alpha.exp() * (new_log_pi.detach() + target_entropy)).mean()
                        optim_alpha.zero_grad()
                        alpha_loss.backward()
                        optim_alpha.step()

                    # target nets
                    _polyak_update(q1, q1_target, cfg["tau"])
                    _polyak_update(q2, q2_target, cfg["tau"])

    out = Path(log_dir) / "sac" / env.spec.id / f"seed{seed}.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "n_params_actor": count_parameters(actor),
            "n_params_critic_each": count_parameters(q1),
        },
        out,
    )
