"""Run the full experiment matrix.

Iterates over (algo, env, seed) tuples and skips any run whose CSV already
exists. Each (algo, env) pair carries its own step budget. Distribute across
three teammates by running with `--owner A/B/C`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts" / "train.py"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)  # fall back to current interpreter

SEEDS = [0, 1, 2]

# (algo, env, total_steps, owner_letter, config_filename)
# Owner assignments balance wall-clock across 3 GPUs (measured on RTX 3070
# Laptop): A ~4.1h, B ~4.0h, C ~4.3h. The heaviest single pair (SAC on
# MountainCarContinuous, ~3.9h alone) is given to owner A as a solo
# big-batch run; B and C each get a mix of medium and small pairs.
MATRIX = [
    ("dqn", "CartPole-v1",              100_000, "A", "dqn_cartpole.yaml"),
    ("dqn", "Acrobot-v1",               200_000, "B", "dqn_acrobot.yaml"),
    ("ppo", "CartPole-v1",              100_000, "C", "ppo_cartpole.yaml"),
    ("ppo", "Acrobot-v1",               200_000, "C", "ppo_acrobot.yaml"),
    ("ppo", "Pendulum-v1",              200_000, "C", "ppo_pendulum.yaml"),
    ("ppo", "MountainCarContinuous-v0", 300_000, "C", "ppo_mountaincar.yaml"),
    ("sac", "Pendulum-v1",              200_000, "B", "sac_pendulum.yaml"),
    ("sac", "MountainCarContinuous-v0", 300_000, "A", "sac_mountaincar.yaml"),
    ("td3", "Pendulum-v1",              200_000, "B", "td3_pendulum.yaml"),
    ("td3", "MountainCarContinuous-v0", 300_000, "C", "td3_mountaincar.yaml"),
]


def csv_path(log_dir: Path, algo: str, env_id: str, seed: int) -> Path:
    return log_dir / algo / env_id / f"seed{seed}.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--owner", choices=["A", "B", "C", "ALL"], default="ALL",
                   help="Run only the rows this teammate owns.")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without running them.")
    args = p.parse_args()

    log_dir = ROOT / args.log_dir
    cfg_dir = ROOT / args.configs_dir
    log_dir.mkdir(exist_ok=True)

    planned = 0
    skipped = 0
    failed = 0
    for algo, env_id, total_steps, owner, cfg_file in MATRIX:
        if args.owner != "ALL" and owner != args.owner:
            continue
        for seed in SEEDS:
            out = csv_path(log_dir, algo, env_id, seed)
            if out.exists():
                skipped += 1
                print(f"[skip] {out.relative_to(ROOT)} already exists")
                continue
            cfg_path = cfg_dir / cfg_file
            cmd = [
                str(PY), str(TRAIN),
                "--algo", algo,
                "--env", env_id,
                "--seed", str(seed),
                "--total-steps", str(total_steps),
                "--log-dir", str(log_dir),
            ]
            if cfg_path.exists():
                cmd += ["--config", str(cfg_path)]
            planned += 1
            print(f"[run ] {algo:<4} {env_id:<28} seed={seed} steps={total_steps:>7,}")
            if args.dry_run:
                continue
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                failed += 1
                print(f"[fail] {algo} {env_id} seed={seed} returned {result.returncode}")

    print(f"\nplanned={planned}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
