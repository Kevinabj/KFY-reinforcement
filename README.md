# EE-568 Applied Project 1: Deep RL

Re-implementations of DQN, PPO, SAC, TD3 in PyTorch, compared on four
Gymnasium environments (CartPole-v1, Acrobot-v1, Pendulum-v1,
MountainCarContinuous-v0) with three random seeds each.

**Team:** Kevin (owner A), Youssef (owner B), Fuad (owner C).
**Course:** EE-568 Spring 2026 (Prof. Cevher).

---

## 1. Prerequisites

- Python 3.11 or 3.12 (we used 3.12.10).
- Git.
- A CUDA-capable NVIDIA GPU. Driver supporting CUDA 12.8 or newer
  (run `nvidia-smi` and look at the `CUDA Version` field, must be >= 12.8).
- About 5 GB of free disk for the venv and Torch wheel.

If you do not have a CUDA GPU, you can still run the small algorithms
(DQN, PPO) on CPU; SAC and TD3 will be painfully slow. Tell Kevin and we
will reassign.

---

## 2. One-time setup (Windows PowerShell)

From the directory where you want the repo to live, copy-paste the
following block. Lines starting with `#` are comments, the rest is what
you actually run.

```powershell
# 1. clone the repo
git clone https://github.com/Kevinabj/KFY-reinforcement.git applied_project
cd applied_project

# 2. create the virtual environment and install dependencies (CPU torch first)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. replace CPU torch with the CUDA wheel
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. verify CUDA is detected
.\.venv\Scripts\python.exe scripts\check_torch.py
# expected output: "CUDA available: True"

# 5. verify the four environments work
.\.venv\Scripts\python.exe scripts\smoke_test.py
# expected output: "All four envs OK."

# 6. verify the infra layer
.\.venv\Scripts\python.exe scripts\smoke_infra.py
# expected output: "All Phase-1 infra checks OK." with device = cuda
```

If any of steps 4-6 prints an error, send the error to the team chat
before running training.

**macOS or Linux:** replace every `.\.venv\Scripts\python.exe` with
`./.venv/bin/python`. Everything else is identical.

---

## 3. Your assigned runs

### Kevin (owner A): about 4.0 hours, 5 runs

```powershell
.\.venv\Scripts\python.exe scripts\run_all.py --owner A
```

Trains: DQN CartPole seeds 1-2, SAC MountainCarContinuous seeds 0-2.

### Youssef (owner B): about 3.9 hours, 8 runs

```powershell
.\.venv\Scripts\python.exe scripts\run_all.py --owner B
```

Trains: DQN Acrobot seeds 1-2, SAC Pendulum seeds 0-2, TD3 Pendulum
seeds 0-2.

### Fuad (owner C): about 3.9 hours, 13 runs

```powershell
.\.venv\Scripts\python.exe scripts\run_all.py --owner C
```

Trains: PPO CartPole seeds 1-2, PPO Acrobot seeds 0-2, PPO Pendulum
seeds 1-2, PPO MountainCarContinuous seeds 0-2, TD3
MountainCarContinuous seeds 0-2.

---

## 4. While training runs

The script writes nothing to the terminal during training. That is
expected. It logs to a CSV at `logs/<algo>/<env>/seed<N>.csv` after every
episode, so you can peek at progress in a second terminal:

```powershell
# see how far the current run has gotten
Get-Content -Tail 10 logs\sac\MountainCarContinuous-v0\seed0.csv
```

Or check the current learning curve at any point with the inspector:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dqn.py --csv logs\<algo>\<env>\seed<N>.csv
```

Don't be alarmed if some runs look bad mid-training. DQN famously dips
before recovering, and SAC takes a while to reach near-optimal on
MountainCarContinuous because of the sparse reward.

The script auto-skips any seed whose CSV already exists, so if your
machine reboots, just rerun the same `run_all.py --owner X` command and
it will pick up where it left off.

---

## 5. After your runs finish

From your terminal in `applied_project/`:

```powershell
# pick up any commits teammates pushed while you were training
git pull --rebase

# add your new CSVs
git add logs/
git status                              # only your new .csv files should appear
git commit -m "owner X runs"            # replace X with your initial
git push
```

Each teammate writes to disjoint file paths (different env / algo /
seed combinations), so concurrent pushes never produce merge conflicts.
If `git push` fails because someone else pushed first, just run
`git pull --rebase` again and `git push`. Ping the team chat with
"owner X done" so we know when to merge.

---

## 6. What is in this repo

- `algos/`: the four algorithm implementations (DQN, PPO, SAC, TD3). Frozen.
- `common/`: shared infrastructure (env factory, replay/rollout buffers, networks, logger, seeding).
- `configs/`: per-(algo, env) YAML hyperparameter files.
- `scripts/`: train.py, run_all.py, plot.py, plus smoke tests and inspectors.
- `logs/`: training CSVs, one per (algo, env, seed). CSVs are tracked in git so we can share results across machines.
- `PROJECT_PLAN.md`: the full plan, phase by phase, with diary material.
- `requirements.txt`: pinned dependencies.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `CUDA available: False` | wrong torch wheel installed | rerun step 3 of section 2 |
| `ModuleNotFoundError: gymnasium` | venv not activated, or used system python | call the venv python directly: `.\.venv\Scripts\python.exe ...` |
| `torch.cuda.OutOfMemoryError` | other process holding GPU memory | close games/browsers, then `nvidia-smi` should show <100 MB used |
| Run terminates with no output | normal | check the CSV; the trainer is silent |
| Different seeds give wildly different returns | normal for DQN, expected | document it in your diary notes for the report |

If you hit something not in this table, post the full error in the team
chat with the command you ran.
