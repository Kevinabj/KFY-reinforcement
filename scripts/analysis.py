"""End-to-end analysis pipeline.

Produces every figure and table needed by the report and the poster from the
CSVs under `logs/` and `logs_ablation/`. Degrades gracefully: if a run is
missing, the corresponding curve / row is skipped with a warning.

Usage (from the applied_project root):
    .\\.venv\\Scripts\\python.exe scripts\\analysis.py

Outputs (under plots/):
    main_grid.pdf               2x2 grid of envs, all applicable algos
    per_env/<env>.pdf           individual env plots (bigger version of each panel)
    bars/sample_efficiency.pdf  steps-to-threshold per (algo, env)
    bars/wall_clock.pdf         per-100k-step wall-clock per algo
    bars/param_count.pdf        actor parameter count per algo per env
    seed_grid/<algo>_<env>.pdf  per-seed thumbnails for diary section
    ablations/<sweep>.pdf       one figure per pre-registered ablation
    scoreboard.tex              LaTeX booktabs table
    scoreboard.csv              CSV mirror of scoreboard
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.envs import env_info, make_env  # noqa: E402
from common.nets import (  # noqa: E402
    CategoricalActor,
    DeterministicActor,
    GaussianActor,
    MLPQNet,
    SquashedGaussianActor,
    count_parameters,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


# ----------------------------------------------------------------------------- style
ALGO_COLOR = {
    "dqn": "#1f77b4",   # blue
    "ppo": "#ff7f0e",   # orange
    "sac": "#2ca02c",   # green
    "td3": "#d62728",   # red
}
ALGO_LABEL = {"dqn": "DQN", "ppo": "PPO", "sac": "SAC", "td3": "TD3"}
ALGO_ORDER = ["dqn", "ppo", "sac", "td3"]


def set_style(report: bool = True) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": report,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "font.family": "serif" if report else "sans-serif",
            "lines.linewidth": 1.7,
        }
    )


# ----------------------------------------------------------------------------- thresholds
# Per-env "solved" thresholds used for the steps-to-threshold metric.
# All envs in our matrix are well-known classic-control benchmarks.
ENV_THRESHOLD: Dict[str, float] = {
    "CartPole-v1":              475.0,    # near max of 500
    "Acrobot-v1":              -100.0,    # canonical "solved" threshold
    "Pendulum-v1":             -200.0,    # acceptance threshold; near-optimal ~ 0
    "MountainCarContinuous-v0":  90.0,    # 90% of the 100-reward goal bonus
}
ENV_LABEL = {
    "CartPole-v1": "CartPole-v1",
    "Acrobot-v1": "Acrobot-v1",
    "Pendulum-v1": "Pendulum-v1",
    "MountainCarContinuous-v0": "MountainCarContinuous-v0",
}
ENV_ORDER = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCarContinuous-v0"]

# (algo, env) pairs we actually train; matches scripts/run_all.py MATRIX.
ALGO_ENV_PAIRS = [
    ("dqn", "CartPole-v1"),
    ("dqn", "Acrobot-v1"),
    ("ppo", "CartPole-v1"),
    ("ppo", "Acrobot-v1"),
    ("ppo", "Pendulum-v1"),
    ("ppo", "MountainCarContinuous-v0"),
    ("sac", "Pendulum-v1"),
    ("sac", "MountainCarContinuous-v0"),
    ("td3", "Pendulum-v1"),
    ("td3", "MountainCarContinuous-v0"),
]


# ----------------------------------------------------------------------------- data loading
def load_runs(log_dir: Path) -> pd.DataFrame:
    """Walk logs/<algo>/<env>/seed<N>.csv and return a tidy DataFrame."""
    rows = []
    for csv in log_dir.glob("*/*/seed*.csv"):
        if "random" in csv.parts:
            continue
        try:
            algo, env_id, fname = csv.relative_to(log_dir).parts
            seed = int(fname.replace("seed", "").replace(".csv", ""))
        except (ValueError, IndexError):
            continue
        df = pd.read_csv(csv)
        df["algo"] = algo
        df["env"] = env_id
        df["seed"] = seed
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=["step", "episode", "episode_return", "episode_length",
                     "wall_clock_s", "algo", "env", "seed"]
        )
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- aggregation
def bin_and_bootstrap(
    df: pd.DataFrame, n_bins: int = 50, n_bootstrap: int = 1000, rng_seed: int = 0
) -> pd.DataFrame:
    """For one (algo, env), bin by step and bootstrap a 95% CI over seeds."""
    if df.empty:
        return pd.DataFrame(columns=["step_mid", "mean", "lo", "hi"])
    max_step = float(df["step"].max())
    edges = np.linspace(0.0, max_step, n_bins + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])

    per_seed: List[np.ndarray] = []
    for _, sub in df.groupby("seed"):
        bin_idx = np.clip(np.searchsorted(edges, sub["step"].values, side="right") - 1, 0, n_bins - 1)
        means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                means[b] = sub["episode_return"].values[mask].mean()
        last = np.nan
        for i in range(n_bins):
            if np.isnan(means[i]):
                means[i] = last
            else:
                last = means[i]
        per_seed.append(means)
    M = np.stack(per_seed, axis=0)

    mean = np.nanmean(M, axis=0)
    if M.shape[0] >= 2:
        rng = np.random.default_rng(rng_seed)
        boot = np.empty((n_bootstrap, n_bins))
        for b in range(n_bootstrap):
            ids = rng.integers(0, M.shape[0], size=M.shape[0])
            boot[b] = np.nanmean(M[ids], axis=0)
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
    else:
        lo = mean.copy()
        hi = mean.copy()
    return pd.DataFrame({"step_mid": mids, "mean": mean, "lo": lo, "hi": hi})


def steps_to_threshold(df_run: pd.DataFrame, thr: float, window: int = 20) -> Optional[int]:
    """First env step at which the rolling-`window` mean return crosses `thr`."""
    s = df_run.sort_values("step")
    rolling = s["episode_return"].rolling(window).mean()
    crossed = rolling >= thr
    if not crossed.any():
        return None
    first_idx = crossed.idxmax()
    return int(s.loc[first_idx, "step"])


# ----------------------------------------------------------------------------- plots
def _format_steps_axis(ax) -> None:
    """Show x-axis tick labels as e.g. '50k' instead of '50000', preventing overlap."""
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x else "0"))


def plot_main_grid(df: pd.DataFrame, out: Path) -> None:
    """2x2 grid: one subplot per env, all applicable algos overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=False)
    for ax, env_id in zip(axes.flat, ENV_ORDER):
        sub_env = df[df.env == env_id]
        if sub_env.empty:
            ax.set_title(f"{ENV_LABEL[env_id]} (no data)")
            continue
        for algo in ALGO_ORDER:
            sub = sub_env[sub_env.algo == algo]
            if sub.empty:
                continue
            agg = bin_and_bootstrap(sub)
            color = ALGO_COLOR[algo]
            ax.plot(agg["step_mid"], agg["mean"], label=ALGO_LABEL[algo], color=color)
            if sub["seed"].nunique() >= 2:
                ax.fill_between(agg["step_mid"], agg["lo"], agg["hi"],
                                color=color, alpha=0.18, linewidth=0)
        thr = ENV_THRESHOLD.get(env_id)
        if thr is not None:
            ax.axhline(thr, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                       label=f"solved={thr:g}")
        ax.set_title(ENV_LABEL[env_id])
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode return")
        ax.legend(loc="best")
        _format_steps_axis(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def plot_per_env(df: pd.DataFrame, out_dir: Path) -> None:
    """One PDF per env, larger version of each panel (used in report)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for env_id in ENV_ORDER:
        sub_env = df[df.env == env_id]
        if sub_env.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for algo in ALGO_ORDER:
            sub = sub_env[sub_env.algo == algo]
            if sub.empty:
                continue
            agg = bin_and_bootstrap(sub)
            color = ALGO_COLOR[algo]
            ax.plot(agg["step_mid"], agg["mean"], label=ALGO_LABEL[algo], color=color)
            if sub["seed"].nunique() >= 2:
                ax.fill_between(agg["step_mid"], agg["lo"], agg["hi"],
                                color=color, alpha=0.18, linewidth=0)
        thr = ENV_THRESHOLD.get(env_id)
        if thr is not None:
            ax.axhline(thr, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                       label=f"solved={thr:g}")
        ax.set_title(ENV_LABEL[env_id])
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode return")
        ax.legend(loc="best")
        _format_steps_axis(ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{env_id}.pdf")
        plt.close(fig)
    print(f"  wrote per-env plots to {out_dir.relative_to(ROOT)}")


def plot_seed_grid(df: pd.DataFrame, out_dir: Path) -> None:
    """For each (algo, env), plot all individual seed curves plus the mean.

    This is the "diary" figure showing seed-to-seed variation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for algo, env_id in ALGO_ENV_PAIRS:
        sub = df[(df.algo == algo) & (df.env == env_id)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        color = ALGO_COLOR[algo]
        for seed, run in sub.groupby("seed"):
            run = run.sort_values("step")
            rolling = run["episode_return"].rolling(20).mean()
            ax.plot(run["step"], rolling, color=color, alpha=0.35, linewidth=1.0,
                    label=f"seed {seed}")
        agg = bin_and_bootstrap(sub)
        ax.plot(agg["step_mid"], agg["mean"], color=color, linewidth=2.2, label="mean")
        thr = ENV_THRESHOLD.get(env_id)
        if thr is not None:
            ax.axhline(thr, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{ALGO_LABEL[algo]} on {env_id}: per-seed curves")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode return (20-ep rolling)")
        ax.legend(loc="best", ncol=2)
        _format_steps_axis(ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{algo}_{env_id}.pdf")
        plt.close(fig)
    print(f"  wrote seed-grid plots to {out_dir.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- bar charts
def plot_sample_efficiency(rows: pd.DataFrame, out: Path) -> None:
    """Bar chart of steps-to-threshold per (algo, env)."""
    pivot = rows.pivot_table(index="env", columns="algo", values="steps_to_thr", aggfunc="mean")
    pivot = pivot.reindex(index=ENV_ORDER, columns=ALGO_ORDER)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    n = len(ALGO_ORDER)
    x = np.arange(len(ENV_ORDER))
    w = 0.8 / n
    for i, algo in enumerate(ALGO_ORDER):
        vals = pivot[algo].values.astype(float) / 1000.0  # in thousands of steps
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=ALGO_COLOR[algo],
                      label=ALGO_LABEL[algo], edgecolor="white", linewidth=0.5)
        for xi, v in zip(x + (i - (n - 1) / 2) * w, vals):
            if np.isnan(v):
                ax.text(xi, 0.5, "n/a", ha="center", va="bottom", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=12, ha="right")
    ax.set_ylabel("Env steps to threshold (thousands)")
    ax.set_title("Sample efficiency: steps to reach the solved threshold")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def plot_wall_clock(rows: pd.DataFrame, out: Path) -> None:
    """Bar chart of seconds-per-100k-env-steps per (algo, env)."""
    pivot = rows.pivot_table(index="env", columns="algo", values="sec_per_100k", aggfunc="mean")
    pivot = pivot.reindex(index=ENV_ORDER, columns=ALGO_ORDER)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    n = len(ALGO_ORDER)
    x = np.arange(len(ENV_ORDER))
    w = 0.8 / n
    for i, algo in enumerate(ALGO_ORDER):
        vals = pivot[algo].values.astype(float) / 60.0  # to minutes
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=ALGO_COLOR[algo],
               label=ALGO_LABEL[algo], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=12, ha="right")
    ax.set_ylabel("Wall-clock minutes per 100k env steps")
    ax.set_title("Compute cost per iteration")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def plot_param_count(rows: pd.DataFrame, out: Path) -> None:
    """Bar chart of actor parameter count per (algo, env)."""
    pivot = rows.pivot_table(index="env", columns="algo", values="actor_params", aggfunc="mean")
    pivot = pivot.reindex(index=ENV_ORDER, columns=ALGO_ORDER)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    n = len(ALGO_ORDER)
    x = np.arange(len(ENV_ORDER))
    w = 0.8 / n
    for i, algo in enumerate(ALGO_ORDER):
        vals = pivot[algo].values.astype(float) / 1000.0  # in thousands
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=ALGO_COLOR[algo],
               label=ALGO_LABEL[algo], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=12, ha="right")
    ax.set_ylabel("Actor parameters (thousands)")
    ax.set_title("Deployed policy size")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- parameter counts
def actor_params_for(algo: str, env_id: str) -> Optional[int]:
    """Reconstruct the actor architecture and count parameters."""
    env = make_env(env_id, seed=0)
    info = env_info(env)
    env.close()
    obs_dim = info["obs_dim"]
    try:
        if algo == "dqn":
            net = MLPQNet(obs_dim, info["n_actions"], hidden=(64, 64))
        elif algo == "ppo":
            if info["discrete"]:
                net = CategoricalActor(obs_dim, info["n_actions"], hidden=(64, 64))
            else:
                net = GaussianActor(obs_dim, info["act_dim"], hidden=(64, 64))
        elif algo == "sac":
            net = SquashedGaussianActor(obs_dim, info["act_dim"], hidden=(256, 256))
        elif algo == "td3":
            net = DeterministicActor(obs_dim, info["act_dim"], hidden=(256, 256))
        else:
            return None
    except Exception:
        return None
    return count_parameters(net)


# ----------------------------------------------------------------------------- scoreboard
@dataclass
class Row:
    algo: str
    env: str
    n_seeds: int
    final_mean: float
    final_ci_lo: float
    final_ci_hi: float
    steps_to_thr: Optional[int]
    wall_clock_s: float
    sec_per_100k: float
    actor_params: Optional[int]
    on_policy: bool


def compute_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Row] = []
    rng = np.random.default_rng(1)
    for algo, env_id in ALGO_ENV_PAIRS:
        sub = df[(df.algo == algo) & (df.env == env_id)]
        if sub.empty:
            continue
        # final return: last 100 episodes per seed, then bootstrap mean across seeds
        finals_per_seed = []
        steps_per_seed = []
        wall_per_seed = []
        max_step_per_seed = []
        thr = ENV_THRESHOLD.get(env_id, np.nan)
        for seed, run in sub.groupby("seed"):
            run = run.sort_values("step")
            finals_per_seed.append(run["episode_return"].tail(100).mean())
            wall_per_seed.append(run["wall_clock_s"].max())
            max_step_per_seed.append(run["step"].max())
            s = steps_to_threshold(run, thr) if not np.isnan(thr) else None
            steps_per_seed.append(s)
        finals = np.asarray(finals_per_seed, dtype=float)
        if len(finals) >= 2:
            boot = np.array(
                [np.mean(finals[rng.integers(0, len(finals), size=len(finals))]) for _ in range(2000)]
            )
            lo, hi = np.percentile(boot, [2.5, 97.5])
        else:
            lo = hi = float(finals[0])
        valid_steps = [s for s in steps_per_seed if s is not None]
        median_steps = int(np.median(valid_steps)) if valid_steps else None
        mean_wall = float(np.mean(wall_per_seed))
        mean_max_step = float(np.mean(max_step_per_seed))
        sec_per_100k = mean_wall / max(mean_max_step, 1) * 100_000.0
        rows.append(
            Row(
                algo=algo,
                env=env_id,
                n_seeds=int(sub["seed"].nunique()),
                final_mean=float(np.mean(finals)),
                final_ci_lo=float(lo),
                final_ci_hi=float(hi),
                steps_to_thr=median_steps,
                wall_clock_s=mean_wall,
                sec_per_100k=sec_per_100k,
                actor_params=actor_params_for(algo, env_id),
                on_policy=(algo == "ppo"),
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])


def write_scoreboard_latex(board: pd.DataFrame, out: Path) -> None:
    """Write a booktabs LaTeX table."""
    if board.empty:
        out.write_text("% no runs yet\n", encoding="utf-8")
        return
    lines = []
    lines.append(r"\begin{tabular}{llrlrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Algo & Env & Seeds & Final return & Steps$\to$thr & "
                 r"Wall (s/100k) & Actor params & On-pol \\")
    lines.append(r"\midrule")
    for _, r in board.iterrows():
        final = f"{r.final_mean:.1f}\\ \\([{r.final_ci_lo:.0f},{r.final_ci_hi:.0f}]\\)"
        steps = f"{r.steps_to_thr/1000:.0f}k" if pd.notna(r.steps_to_thr) else "--"
        wall = f"{r.sec_per_100k:.0f}"
        params = f"{r.actor_params/1000:.1f}k" if pd.notna(r.actor_params) else "--"
        on_pol = "yes" if r.on_policy else "no"
        lines.append(
            f"{ALGO_LABEL[r.algo]} & {r.env} & {r.n_seeds} & "
            f"{final} & {steps} & {wall} & {params} & {on_pol} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}")


def write_scoreboard_csv(board: pd.DataFrame, out: Path) -> None:
    board.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    set_style(report=True)
    log_dir = ROOT / "logs"
    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "per_env").mkdir(exist_ok=True)
    (out_dir / "bars").mkdir(exist_ok=True)
    (out_dir / "seed_grid").mkdir(exist_ok=True)
    (out_dir / "ablations").mkdir(exist_ok=True)

    df = load_runs(log_dir)
    if df.empty:
        print("No CSVs found in logs/. Run the experiments first.")
        return
    print(f"Loaded {len(df)} episodes across {df['env'].nunique()} envs, "
          f"{df['algo'].nunique()} algos, {df['seed'].nunique()} unique seeds.")

    # main figures
    print("\nGenerating figures:")
    plot_main_grid(df, out_dir / "main_grid.pdf")
    plot_per_env(df, out_dir / "per_env")
    plot_seed_grid(df, out_dir / "seed_grid")

    # scoreboard
    print("\nComputing scoreboard:")
    board = compute_scoreboard(df)
    if not board.empty:
        write_scoreboard_csv(board, out_dir / "scoreboard.csv")
        write_scoreboard_latex(board, out_dir / "scoreboard.tex")

        # bar charts (derive from scoreboard)
        print("\nGenerating bar charts:")
        plot_sample_efficiency(board, out_dir / "bars" / "sample_efficiency.pdf")
        plot_wall_clock(board, out_dir / "bars" / "wall_clock.pdf")
        plot_param_count(board, out_dir / "bars" / "param_count.pdf")

    # ablation figures (only if logs_ablation/ exists)
    abl_dir = ROOT / "logs_ablation"
    if abl_dir.exists() and any(abl_dir.iterdir()):
        print("\nGenerating ablation figures (placeholder, populated in Phase 4):")
        # Implementation will land when Phase 4 CSVs are produced.
    else:
        print("\nNo ablation CSVs yet; skipping ablation figures.")

    # summary
    print("\nSummary of coverage:")
    seen = set(zip(df.algo, df.env))
    for algo, env_id in ALGO_ENV_PAIRS:
        n_seeds = df[(df.algo == algo) & (df.env == env_id)]["seed"].nunique()
        target = 3
        status = "OK " if n_seeds >= target else "..." if n_seeds > 0 else "   "
        print(f"  {status} {algo:<4} {env_id:<28}  seeds present: {n_seeds}/{target}")

    print("\nDone.")


if __name__ == "__main__":
    main()
