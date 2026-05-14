"""Aggregate run CSVs into per-environment comparison plots.

Walks `logs/<algo>/<env>/seed<N>.csv`, bins by step, computes the bootstrap
95% CI of the mean episode return across seeds, and produces one PDF per
environment under `plots/`.

Usage:
    python scripts/plot.py --log-dir logs --out plots
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Empty bins early in training trigger "Mean of empty slice" / "All-NaN slice"
# warnings from numpy. Forward-fill handles the NaNs downstream, so suppress
# the warnings to keep the plot output clean.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALGO_COLORS = {
    "dqn":    "#1f77b4",   # blue
    "ppo":    "#ff7f0e",   # orange
    "sac":    "#2ca02c",   # green
    "td3":    "#d62728",   # red
    "random": "#7f7f7f",   # grey
}


def load_runs(log_dir: Path) -> pd.DataFrame:
    """Returns a DataFrame with columns [algo, env, seed, step, episode_return]."""
    rows = []
    for csv in log_dir.glob("*/*/seed*.csv"):
        try:
            parts = csv.relative_to(log_dir).parts  # (algo, env, seedN.csv)
            algo, env_id, fname = parts
            seed = int(fname.replace("seed", "").replace(".csv", ""))
        except (ValueError, IndexError):
            continue
        df = pd.read_csv(csv)
        df["algo"] = algo
        df["env"] = env_id
        df["seed"] = seed
        rows.append(df)
    if not rows:
        raise SystemExit(f"No CSVs found under {log_dir}")
    return pd.concat(rows, ignore_index=True)


def bin_and_aggregate(
    df: pd.DataFrame, n_bins: int = 50, n_bootstrap: int = 1000
) -> pd.DataFrame:
    """For one (algo, env), bin by step and bootstrap a 95% CI over seeds.

    Returns columns [step_mid, mean, lo, hi].
    """
    max_step = df["step"].max()
    edges = np.linspace(0, max_step, n_bins + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])

    per_seed_means: list[np.ndarray] = []
    for seed, sub in df.groupby("seed"):
        bin_idx = np.clip(np.searchsorted(edges, sub["step"].values, side="right") - 1, 0, n_bins - 1)
        means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                means[b] = sub["episode_return"].values[mask].mean()
        # forward-fill across empty bins so the curve does not disappear
        last = np.nan
        for i in range(n_bins):
            if np.isnan(means[i]):
                means[i] = last
            else:
                last = means[i]
        per_seed_means.append(means)
    M = np.stack(per_seed_means, axis=0)  # (n_seeds, n_bins)

    rng = np.random.default_rng(0)
    n_seeds = M.shape[0]
    boot = np.empty((n_bootstrap, n_bins))
    for b in range(n_bootstrap):
        ids = rng.integers(0, n_seeds, size=n_seeds)
        boot[b] = np.nanmean(M[ids], axis=0)
    mean = np.nanmean(M, axis=0)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return pd.DataFrame({"step_mid": mids, "mean": mean, "lo": lo, "hi": hi})


def plot_env(env_id: str, df_env: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, sub in df_env.groupby("algo"):
        if sub["seed"].nunique() < 1:
            continue
        agg = bin_and_aggregate(sub)
        color = ALGO_COLORS.get(algo, None)
        ax.plot(agg["step_mid"], agg["mean"], label=algo.upper(), color=color, linewidth=1.6)
        ax.fill_between(agg["step_mid"], agg["lo"], agg["hi"], color=color, alpha=0.20, linewidth=0)
    ax.set_title(env_id)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--out", default="plots")
    args = p.parse_args()

    log_dir = ROOT / args.log_dir
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(log_dir)
    print(f"loaded {len(df)} episodes across {df['env'].nunique()} envs, {df['algo'].nunique()} algos")

    for env_id, sub in df.groupby("env"):
        plot_env(env_id, sub, out_dir / f"{env_id}.pdf")


if __name__ == "__main__":
    main()
