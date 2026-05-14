"""Phase-1 infrastructure smoke test.

Instantiates each network class, each buffer, runs one update through each, and
prints parameter counts. Verifies CUDA tensors land on the right device.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from common.buffers import ReplayBuffer, RolloutBuffer
from common.envs import env_info, make_env
from common.nets import (
    CategoricalActor,
    DeterministicActor,
    GaussianActor,
    MLPCritic,
    MLPQNet,
    MLPVCritic,
    SquashedGaussianActor,
    count_parameters,
)
from common.seeding import get_device, set_seed


def main() -> None:
    set_seed(0)
    device = get_device()
    print(f"device = {device}")

    # ------- discrete env (CartPole) -------
    env = make_env("CartPole-v1", seed=0)
    info = env_info(env)
    print(f"\nCartPole-v1 info: {info}")
    obs_dim = info["obs_dim"]
    n_actions = info["n_actions"]

    qnet = MLPQNet(obs_dim, n_actions).to(device)
    vnet = MLPVCritic(obs_dim).to(device)
    cat_actor = CategoricalActor(obs_dim, n_actions).to(device)
    print(f"  MLPQNet            params={count_parameters(qnet)}")
    print(f"  MLPVCritic         params={count_parameters(vnet)}")
    print(f"  CategoricalActor   params={count_parameters(cat_actor)}")

    obs = torch.randn(8, obs_dim, device=device)
    q = qnet(obs); assert q.shape == (8, n_actions) and q.device.type == device.type
    v = vnet(obs); assert v.shape == (8,) and v.device.type == device.type
    action, logp, ent = cat_actor.act(obs)
    assert action.shape == (8,) and logp.shape == (8,) and ent.shape == (8,)

    rb_d = ReplayBuffer(capacity=100, obs_dim=obs_dim, act_dim=1, device=device, discrete=True)
    for _ in range(50):
        rb_d.add(np.random.randn(obs_dim).astype(np.float32),
                 np.random.randint(n_actions),
                 np.random.randn(),
                 np.random.randn(obs_dim).astype(np.float32),
                 done=0.0)
    batch = rb_d.sample(16)
    assert batch["actions"].dtype == torch.long
    print(f"  discrete ReplayBuffer sample OK, size={len(rb_d)}, action.shape={tuple(batch['actions'].shape)}")

    rollout = RolloutBuffer(n_steps=32, obs_dim=obs_dim, act_dim=1, gamma=0.99,
                            gae_lambda=0.95, device=device, discrete=True)
    for _ in range(32):
        rollout.add(np.random.randn(obs_dim).astype(np.float32),
                    np.random.randint(n_actions),
                    log_prob=np.random.randn(),
                    reward=np.random.randn(),
                    done=0.0,
                    value=np.random.randn())
    rollout.compute_returns_and_advantages(last_value=0.0, last_done=0.0)
    nb = sum(1 for _ in rollout.get(batch_size=8))
    print(f"  RolloutBuffer GAE OK, n_minibatches={nb}")

    env.close()

    # ------- continuous env (Pendulum) -------
    env = make_env("Pendulum-v1", seed=0)
    info = env_info(env)
    print(f"\nPendulum-v1 info: {info}")
    obs_dim = info["obs_dim"]
    act_dim = info["act_dim"]

    g_actor = GaussianActor(obs_dim, act_dim).to(device)
    sq_actor = SquashedGaussianActor(obs_dim, act_dim).to(device)
    det_actor = DeterministicActor(obs_dim, act_dim).to(device)
    critic = MLPCritic(obs_dim, act_dim).to(device)
    print(f"  GaussianActor          params={count_parameters(g_actor)}")
    print(f"  SquashedGaussianActor  params={count_parameters(sq_actor)}")
    print(f"  DeterministicActor     params={count_parameters(det_actor)}")
    print(f"  MLPCritic              params={count_parameters(critic)}")

    obs = torch.randn(8, obs_dim, device=device)
    a, logp, ent = g_actor.act(obs)
    assert a.shape == (8, act_dim) and logp.shape == (8,)
    a_s, logp_s = sq_actor.sample(obs)
    assert a_s.shape == (8, act_dim) and torch.all(a_s.abs() <= 1.0)
    a_d = det_actor(obs)
    assert a_d.shape == (8, act_dim) and torch.all(a_d.abs() <= 1.0)
    q = critic(obs, a_d)
    assert q.shape == (8,)

    rb_c = ReplayBuffer(capacity=100, obs_dim=obs_dim, act_dim=act_dim, device=device, discrete=False)
    for _ in range(50):
        rb_c.add(np.random.randn(obs_dim).astype(np.float32),
                 np.random.randn(act_dim).astype(np.float32),
                 np.random.randn(),
                 np.random.randn(obs_dim).astype(np.float32),
                 done=0.0)
    batch = rb_c.sample(16)
    assert batch["actions"].dtype == torch.float32
    print(f"  continuous ReplayBuffer sample OK, action.shape={tuple(batch['actions'].shape)}")

    env.close()
    print("\nAll Phase-1 infra checks OK.")


if __name__ == "__main__":
    main()
