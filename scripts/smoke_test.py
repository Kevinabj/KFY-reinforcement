r"""Smoke test: verify all four target Gymnasium envs reset, step, and report rewards correctly.

Run from the applied_project/ root with the project venv:
    .\.venv\Scripts\python.exe scripts/smoke_test.py
"""

import numpy as np
import gymnasium as gym

ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
]

SEED = 0
N_STEPS = 50


def smoke_one(env_id: str) -> None:
    env = gym.make(env_id)
    obs, info = env.reset(seed=SEED)
    total_reward = 0.0
    n_episodes_done = 0
    for _ in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            n_episodes_done += 1
            obs, info = env.reset()
    print(
        f"  {env_id:<28} obs_shape={np.asarray(obs).shape} "
        f"act_space={env.action_space} "
        f"reward({N_STEPS}_random_steps)={total_reward:+.3f} "
        f"episodes_done={n_episodes_done}"
    )
    env.close()


def main() -> None:
    print(f"Gymnasium version: {gym.__version__}")
    print("Running smoke test on:")
    for env_id in ENVS:
        smoke_one(env_id)
    print("All four envs OK.")


if __name__ == "__main__":
    main()
