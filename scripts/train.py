"""Single-run trainer.

Usage:
    python scripts/train.py --algo ppo --env CartPole-v1 --seed 0 \
        --config configs/ppo_cartpole.yaml --total-steps 100000

The script seeds globally, builds the env, loads the YAML config, and dispatches
to `algos.<name>.train`. Each algorithm module writes its own CSV under
`<log-dir>/<algo>/<env-id>/seed<seed>.csv`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

# Make project root importable so we can do `from common...` and `from algos...`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.envs import make_env
from common.seeding import set_seed


ALGOS = {
    "random": "algos.random_policy",
    "dqn": "algos.dqn",
    "ppo": "algos.ppo",
    "sac": "algos.sac",
    "td3": "algos.td3",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=list(ALGOS.keys()))
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", default=None, help="Optional YAML config; algos provide defaults.")
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args()

    set_seed(args.seed)
    env = make_env(args.env, args.seed)

    cfg: dict = {}
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    module = importlib.import_module(ALGOS[args.algo])
    module.train(env, cfg, args.seed, args.total_steps, args.log_dir)

    env.close()


if __name__ == "__main__":
    main()
