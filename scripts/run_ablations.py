"""Run the four pre-registered ablations on a single machine.

Sweeps (reduced 100k-step budgets on MountainCarContinuous to fit a
~3-hour-per-machine wall clock):

  --sweep dqn_target_sync
      DQN target_sync in {100, 1000, 5000} on CartPole-v1, 100k steps
      Hypothesis: too-frequent sync destabilises learning; too-rare
      slows convergence. 1000 is the canonical default.

  --sweep ppo_clip_ratio
      PPO clip_ratio in {0.1, 0.2, 0.3} on Pendulum-v1, 200k steps
      Hypothesis: 0.2 is canonical. 0.1 too conservative on dense
      continuous control; 0.3 starts breaking the trust-region intuition.

  --sweep sac_alpha
      SAC autotune-alpha vs fixed-alpha in {0.05, 0.2} on
      MountainCarContinuous-v0, 100k steps. Hypothesis: sparse reward
      stresses exploration; auto-alpha should help; fixed-low should
      fail. (We reuse the v0 result on auto from the main matrix.)

  --sweep td3_exploration_noise
      TD3 exploration_noise in {0.1, 0.3, 0.5} on
      MountainCarContinuous-v0, 100k steps. Hypothesis: standard 0.1
      is too low for sparse reward; higher noise accelerates discovery.

All ablations are run for 3 seeds (0, 1, 2). CSVs land at
  logs_ablation/<sweep>/<value>/seed<N>.csv

Usage:
  python scripts/run_ablations.py --sweep <name>
  python scripts/run_ablations.py --sweep <name> --dry-run
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (algo, env, total_steps, base_config_yaml, override_pairs_per_value, value_label)
# override_pairs_per_value: list of (label, [(key, value), ...])
SWEEPS: dict = {
    "dqn_target_sync": {
        "algo": "dqn",
        "env": "CartPole-v1",
        "total_steps": 100_000,
        "base_config": "configs/dqn_cartpole.yaml",
        "values": [
            ("100",  [("target_sync", 100)]),
            ("1000", [("target_sync", 1000)]),
            ("5000", [("target_sync", 5000)]),
        ],
    },
    "ppo_clip_ratio": {
        "algo": "ppo",
        "env": "Pendulum-v1",
        "total_steps": 200_000,
        "base_config": "configs/ppo_pendulum.yaml",
        "values": [
            ("0.10", [("clip_ratio", 0.10)]),
            ("0.20", [("clip_ratio", 0.20)]),
            ("0.30", [("clip_ratio", 0.30)]),
        ],
    },
    "sac_alpha": {
        "algo": "sac",
        "env": "MountainCarContinuous-v0",
        "total_steps": 100_000,
        "base_config": "configs/sac_mountaincar.yaml",
        # log(0.05) approx -2.996; log(0.2) approx -1.609
        "values": [
            ("auto", [("autotune_alpha", True)]),
            ("fix0.05", [("autotune_alpha", False), ("init_log_alpha", math.log(0.05))]),
            ("fix0.20", [("autotune_alpha", False), ("init_log_alpha", math.log(0.20))]),
        ],
    },
    "td3_exploration_noise": {
        "algo": "td3",
        "env": "MountainCarContinuous-v0",
        "total_steps": 100_000,
        "base_config": "configs/td3_mountaincar.yaml",
        "values": [
            ("0.1", [("exploration_noise", 0.1)]),
            ("0.3", [("exploration_noise", 0.3)]),
            ("0.5", [("exploration_noise", 0.5)]),
        ],
    },
}

SEEDS = [0, 1, 2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True, choices=list(SWEEPS.keys()))
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = p.parse_args()

    spec = SWEEPS[args.sweep]
    log_dir = ROOT / "logs_ablation" / args.sweep
    log_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for value_label, overrides in spec["values"]:
        for seed in SEEDS:
            csv_path = log_dir / value_label / f"seed{seed}.csv"
            plan.append((value_label, overrides, seed, csv_path))

    n_skipped = 0
    n_run = 0
    n_failed = 0
    for value_label, overrides, seed, csv_path in plan:
        if csv_path.exists():
            print(f"[skip] {args.sweep}/{value_label} seed={seed} (CSV exists)")
            n_skipped += 1
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--algo", spec["algo"],
            "--env", spec["env"],
            "--seed", str(seed),
            "--config", str(ROOT / spec["base_config"]),
            "--total-steps", str(spec["total_steps"]),
            "--log-dir", str(log_dir / value_label / ".."),  # algos still write to <algo>/<env>/...
        ]
        for k, v in overrides:
            cmd.extend(["--override", f"{k}={v}"])

        # Algorithms write to <log_dir>/<algo>/<env>/seed<N>.csv. We want the
        # final file at logs_ablation/<sweep>/<value_label>/seed<N>.csv.
        # Simpler approach: pass --log-dir as a temp dir, then move.
        tmp_dir = log_dir / value_label / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cmd[cmd.index("--log-dir") + 1] = str(tmp_dir)

        print(f"[run ] {args.sweep}/{value_label} seed={seed} steps={spec['total_steps']:,}")
        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[fail] return code {e.returncode}")
            n_failed += 1
            continue

        # Move the produced CSV from tmp into the canonical ablation path.
        produced = tmp_dir / spec["algo"] / spec["env"] / f"seed{seed}.csv"
        if produced.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            produced.rename(csv_path)
            # also move the .pt checkpoint if present (algorithms save one)
            pt = produced.with_suffix(".pt")
            if pt.exists():
                pt.unlink()  # ablations do not need the .pt
            # clean up empty dirs
            try:
                produced.parent.rmdir()
                produced.parent.parent.rmdir()
                tmp_dir.rmdir()
            except OSError:
                pass
        else:
            print(f"[fail] expected output {produced} not found")
            n_failed += 1
            continue

        n_run += 1

    print()
    print(f"planned={len(plan)}  ran={n_run}  skipped={n_skipped}  failed={n_failed}")


if __name__ == "__main__":
    main()
