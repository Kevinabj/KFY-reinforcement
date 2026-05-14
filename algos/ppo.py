"""PPO (Schulman et al. 2017).

Implementation notes for the diary:
  - GAE-lambda advantage estimation.
  - Clipped surrogate policy loss with eps = 0.2.
  - Optional value-loss clipping (Schulman's trick); on by default.
  - Per-mini-batch advantage normalisation: critical for stability.
  - Truncation bootstrap: on a time-limit truncation that is NOT a real
    termination, we add gamma * V(s_{t+1}) to the reward and mark the buffer
    `done` flag, so GAE correctly accounts for the value of the truncated state.
    This is the canonical fix and matters most on Pendulum where every
    episode is truncated at 200 steps.
  - Linear learning-rate annealing to 0 across iterations.
  - Single combined optimiser over actor + critic parameters.
  - The discrete actor is Categorical; the continuous actor is a diagonal
    Gaussian with state-independent log-std and NO tanh squash (the
    classic Schulman recipe). Actions are clipped to the action range
    before stepping the env.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from common.buffers import RolloutBuffer
from common.envs import env_info
from common.logger import Logger
from common.nets import (
    CategoricalActor,
    GaussianActor,
    MLPVCritic,
    count_parameters,
)
from common.seeding import get_device


DEFAULTS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "lr": 3.0e-4,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "clip_ratio": 0.2,
    "vf_coef": 0.5,
    "ent_coef_discrete": 0.0,
    "ent_coef_continuous": 0.01,
    "max_grad_norm": 0.5,
    "anneal_lr": True,
    "norm_adv": True,
    "clip_value_loss": True,
    "hidden": [64, 64],
}


def train(env: gym.Env, cfg: dict, seed: int, total_steps: int, log_dir: str) -> None:
    cfg = {**DEFAULTS, **(cfg or {})}
    device = get_device()
    info = env_info(env)
    obs_dim = info["obs_dim"]
    discrete = info["discrete"]

    if discrete:
        actor = CategoricalActor(obs_dim, info["n_actions"], hidden=tuple(cfg["hidden"])).to(device)
        ent_coef = cfg["ent_coef_discrete"]
        act_dim = 1
    else:
        actor = GaussianActor(obs_dim, info["act_dim"], hidden=tuple(cfg["hidden"])).to(device)
        ent_coef = cfg["ent_coef_continuous"]
        act_dim = info["act_dim"]

    critic = MLPVCritic(obs_dim, hidden=tuple(cfg["hidden"])).to(device)

    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])

    rollout = RolloutBuffer(
        n_steps=cfg["n_steps"],
        obs_dim=obs_dim,
        act_dim=act_dim,
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        device=device,
        discrete=discrete,
    )

    log_path = Path(log_dir) / "ppo" / env.spec.id / f"seed{seed}.csv"

    obs, _ = env.reset(seed=seed)
    ep_return, ep_length = 0.0, 0
    step_count = 0
    n_iterations = max(1, total_steps // cfg["n_steps"])

    with Logger(log_path) as logger:
        for iteration in range(n_iterations):
            if cfg["anneal_lr"]:
                frac = 1.0 - iteration / n_iterations
                for g in optimizer.param_groups:
                    g["lr"] = cfg["lr"] * frac

            # ---- rollout collection ----
            rollout.reset()
            last_done = 0.0
            for step in range(cfg["n_steps"]):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    if discrete:
                        action_t, log_prob_t, _ = actor.act(obs_t)
                        action_for_buffer = int(action_t.item())
                        action_for_env = action_for_buffer
                    else:
                        action_t, log_prob_t, _ = actor.act(obs_t)
                        action_for_buffer = action_t.squeeze(0).cpu().numpy()
                        action_for_env = np.clip(action_for_buffer, -1.0, 1.0).astype(np.float32)
                    log_prob_v = float(log_prob_t.item())
                    value_v = float(critic(obs_t).item())

                next_obs, reward, terminated, truncated, _ = env.step(action_for_env)
                step_count += 1
                ep_return += float(reward)
                ep_length += 1

                # Truncation bootstrap trick: if the env truncated but did not
                # terminate, fold gamma * V(s_{t+1}) into the reward so GAE
                # gets the right bootstrap even after we reset.
                done_for_buffer = float(terminated)
                if truncated and not terminated:
                    with torch.no_grad():
                        next_v = float(
                            critic(
                                torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                            ).item()
                        )
                    reward = float(reward) + cfg["gamma"] * next_v
                    done_for_buffer = 1.0

                rollout.add(obs, action_for_buffer, log_prob_v, reward, done_for_buffer, value_v)
                last_done = done_for_buffer

                obs = next_obs
                if terminated or truncated:
                    logger.log_episode(step_count, ep_return, ep_length)
                    obs, _ = env.reset()
                    ep_return, ep_length = 0.0, 0

            # ---- GAE ----
            with torch.no_grad():
                last_value = float(
                    critic(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
                )
            rollout.compute_returns_and_advantages(last_value, last_done)

            # ---- PPO update ----
            for _epoch in range(cfg["n_epochs"]):
                for batch in rollout.get(cfg["batch_size"]):
                    adv = batch["advantages"]
                    if cfg["norm_adv"]:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    new_log_prob, entropy = actor.log_prob_and_entropy(batch["obs"], batch["actions"])
                    new_value = critic(batch["obs"])

                    ratio = (new_log_prob - batch["old_log_probs"]).exp()
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg["clip_ratio"], 1.0 + cfg["clip_ratio"]) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    if cfg["clip_value_loss"]:
                        v_clipped = batch["old_values"] + torch.clamp(
                            new_value - batch["old_values"], -cfg["clip_ratio"], cfg["clip_ratio"]
                        )
                        v_loss1 = (new_value - batch["returns"]) ** 2
                        v_loss2 = (v_clipped - batch["returns"]) ** 2
                        value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(new_value, batch["returns"])

                    entropy_term = entropy.mean()
                    total_loss = (
                        policy_loss
                        + cfg["vf_coef"] * value_loss
                        - ent_coef * entropy_term
                    )

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                    optimizer.step()

    out = Path(log_dir) / "ppo" / env.spec.id / f"seed{seed}.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "n_params_actor": count_parameters(actor),
            "n_params_critic": count_parameters(critic),
        },
        out,
    )
