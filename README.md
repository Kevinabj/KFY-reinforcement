# Deep RL: DQN, PPO, SAC, TD3

A from-scratch PyTorch re-implementation of four canonical deep reinforcement learning algorithms — DQN, PPO, SAC, and TD3 — compared head-to-head on four Gymnasium classic-control benchmarks (CartPole, Acrobot, Pendulum, MountainCarContinuous). Every run uses three random seeds, and the analysis reports mean episode return with 95% bootstrap confidence intervals so the comparison is honest rather than cherry-picked. The repo also includes four pre-registered ablations, one per algorithm, that stress the canonical hyperparameter each one is most sensitive to.

## Requirements

- Python 3.11 or 3.12
- An NVIDIA GPU with CUDA 12.8+ drivers (CPU works for DQN and PPO but is slow for SAC/TD3)
- About 5 GB free disk space for the virtual environment and Torch wheel

## Setup

```powershell
git clone <this-repo> applied_project
cd applied_project

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Swap CPU torch for the CUDA build
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

On macOS or Linux, replace `.\.venv\Scripts\python.exe` with `./.venv/bin/python`.

## Running

Train a single (algorithm, environment) pair:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\sac_pendulum.yaml --seed 0
```

Logs land in `logs/<algo>/<env>/seed<N>.csv`. Once you've trained the runs you care about, regenerate every figure and the scoreboard table:

```powershell
.\.venv\Scripts\python.exe scripts\analysis.py
```

## Layout

- `algos/` — the four algorithm implementations
- `common/` — shared infrastructure (envs, buffers, networks, logging, seeding)
- `configs/` — per-(algo, env) hyperparameter YAMLs
- `scripts/` — train, analysis, and plotting entry points
- `logs/`, `logs_ablation/` — training CSVs from the main matrix and the ablation sweeps
- `plots/` — generated figures and the scoreboard table
- `report/`, `poster/` — LaTeX sources for the writeup and the A1 poster
