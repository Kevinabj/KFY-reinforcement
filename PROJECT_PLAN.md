# Applied Project 1: Deep RL Project Plan

**Course:** EE-568 Reinforcement Learning, EPFL Spring 2026 (Prof. Volkan Cevher)
**Team:** Kevin + 2 teammates (A, B, C)
**Deadline:** 1 week
**Stack:** Python 3.12, Gymnasium 1.3, PyTorch 2.11 (CUDA 12.8), NumPy, Matplotlib. GPU: NVIDIA RTX 3070 Laptop (8 GB).

---

## 0. Goal in one paragraph

Re-implement DQN, PPO, SAC, TD3 from scratch in PyTorch (no stable-baselines3, no TF), compare them empirically on four Gymnasium environments with three random seeds each, deliver a 6-8 page report and an A1 portrait poster. The grading signal rewards fair multi-seed comparison plus an honest engineering diary (bugs, hyperparameter sensitivity, surprises). We will exceed both reference posters (Applied_1, Applied_1_2) on rigour (bootstrap CIs, pre-registered ablations, a single scoreboard table) and on theoretical depth (a half-page bridge to the policy-gradient / OPPO content from A3).

---

## 1. Locked decisions

| Item                | Value                                                                                |
| ------------------- | ------------------------------------------------------------------------------------ |
| Algorithms          | DQN, PPO, SAC, TD3                                                                   |
| Environments        | CartPole-v1, Acrobot-v1, Pendulum-v1, MountainCarContinuous-v0                       |
| Seeds               | {0, 1, 2} (3 seeds, frozen)                                                          |
| Step budgets        | CartPole 100k, Acrobot 200k, Pendulum 200k, MountainCarContinuous 300k               |
| Algorithm coverage  | Discrete envs: DQN + PPO. Continuous envs: PPO + SAC + TD3.                          |
| Stack               | Python 3.11, Gymnasium, PyTorch 2.x, NumPy, Matplotlib                               |
| Ownership           | A: infra + DQN. B: PPO (discrete + continuous). C: SAC + TD3.                        |
| Network width       | 2x64 for discrete envs, 2x256 for continuous envs                                    |
| Log format          | CSV, one file per (algo, env, seed)                                                  |
| Confidence reporting| 95% bootstrap CIs over 3 seeds                                                       |

---

## 2. Folder layout

```
applied_project/
├── PROJECT_PLAN.md          # this file
├── README.md                # short, written last
├── requirements.txt
├── algos/
│   ├── __init__.py
│   ├── dqn.py
│   ├── ppo.py
│   ├── sac.py
│   └── td3.py
├── common/
│   ├── __init__.py
│   ├── envs.py
│   ├── buffers.py
│   ├── nets.py
│   ├── logger.py
│   └── seeding.py
├── configs/
│   ├── dqn_cartpole.yaml
│   ├── dqn_acrobot.yaml
│   ├── ppo_cartpole.yaml
│   ├── ppo_acrobot.yaml
│   ├── ppo_pendulum.yaml
│   ├── ppo_mountaincar.yaml
│   ├── sac_pendulum.yaml
│   ├── sac_mountaincar.yaml
│   ├── td3_pendulum.yaml
│   └── td3_mountaincar.yaml
├── scripts/
│   ├── train.py             # single (algo, env, seed) run
│   ├── run_all.py           # full experiment matrix
│   ├── run_ablations.py     # pre-registered ablations
│   └── plot.py              # CSV aggregation + figures
├── analysis.ipynb           # produces every report figure + scoreboard table
├── logs/                    # main run CSVs (gitignored)
├── logs_ablation/           # ablation run CSVs (gitignored)
├── plots/                   # figures (committed)
├── report/                  # LaTeX source for the 6-8 page report
└── poster/                  # adapted from poster_template_RL/
```

---

## 3. Quality bar (what makes our submission better than the example posters)

1. **Bootstrap 95% CIs** over seeds, not std bands. We compute the CI of the mean return at each timestep via 1000 resamples of the seed axis.
2. **One scoreboard table** with columns `algo, env, steps_to_50pct, steps_to_90pct, final_return_mean, final_return_ci, wallclock_s, n_params, on_policy`. This is the artefact that directly answers the four qualitative questions in the brief.
3. **Pre-registered ablations**: announced in the report (Section 5) before reporting the results, with hypotheses.
4. **Negative results reported.** If a sweep does nothing, we say so and hypothesise why.
5. **Theoretical bridge** (Section 7 of the report) tying PPO to NPG / trust-region and SAC to OPPO exploration bonuses. Half a page, costs nothing.
6. **Consistent visual language.** One color per algorithm across every plot in report and poster. One accent color in the poster. No gridlines in the poster figures.

---

## 4. Phase 0: setup (1 hour, owner A, must finish before anyone else starts)

### Task 0.1 Initialise repo

```powershell
cd "C:\Users\Kevin\Desktop\EPFL Courses\Reinforcement-Learning\applied_project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch numpy matplotlib pyyaml gymnasium[classic-control] tqdm pandas
pip freeze > requirements.txt
```

### Task 0.2 Smoke test Gymnasium

Run a 5-line script that creates each of the four envs, calls `env.reset(seed=0)`, takes 10 random actions, and prints the return. Catches version mismatches before they cost a day.

### Task 0.3 Confirm Box2D not needed

Our four envs are all classic-control. No Box2D, no MuJoCo. Saves install pain on Windows.

### Task 0.4 Add `.gitignore`

```
.venv/
__pycache__/
logs/
logs_ablation/
*.csv
*.pt
*.pdf
!report/*.pdf
!poster/*.pdf
```

---

## 5. Phase 1: shared infrastructure (Day 1, owner A)

Each task below produces one file. Acceptance: a "random policy" placeholder trains end-to-end and produces a plot before any real algorithm exists.

### Task 1.1 `common/seeding.py`

One function `set_seed(seed: int)` that seeds `random`, `numpy.random`, `torch` (CPU + CUDA), and sets `torch.backends.cudnn.deterministic = True`.

### Task 1.2 `common/envs.py`

`make_env(env_id: str, seed: int) -> gym.Env`. Wraps with `gym.wrappers.RecordEpisodeStatistics` so we get `info["episode"]` on done. For continuous envs, optionally apply `gym.wrappers.RescaleAction(env, -1, 1)`. Always pass `seed` to `env.reset` on the first call.

### Task 1.3 `common/buffers.py`

Two classes:
- `ReplayBuffer(capacity, obs_dim, act_dim, device)`: numpy-backed circular buffer. Methods: `add(s, a, r, s2, done)`, `sample(batch_size) -> dict of torch tensors`.
- `RolloutBuffer(n_steps, obs_dim, act_dim, gamma, gae_lambda, device)`: stores one rollout, computes returns and GAE advantages in `compute_returns_and_advantages(last_value, last_done)`. Yields mini-batches via `get(batch_size)`.

### Task 1.4 `common/nets.py`

PyTorch modules:
- `MLPQNet(obs_dim, act_dim, hidden=(64,64))` for DQN
- `MLPCritic(obs_dim, act_dim, hidden=(256,256))` for SAC/TD3 Q-functions (concatenates state and action)
- `MLPVCritic(obs_dim, hidden=(64,64))` for PPO value function
- `CategoricalActor(obs_dim, n_actions, hidden=(64,64))` for PPO discrete
- `GaussianActor(obs_dim, act_dim, hidden=(64,64))` for PPO continuous (state-independent log-std parameter)
- `SquashedGaussianActor(obs_dim, act_dim, hidden=(256,256))` for SAC (state-dependent log-std, tanh squash, analytic log-prob correction)
- `DeterministicActor(obs_dim, act_dim, hidden=(256,256))` for TD3 (tanh output)

All MLPs use orthogonal init, gain `sqrt(2)` for hidden ReLU layers, `0.01` for policy heads, `1.0` for value heads.

### Task 1.5 `common/logger.py`

`Logger(path: str)` writes a CSV with header `step, episode, episode_return, episode_length, wall_clock_s`. Append-only. Flush every 10 episodes.

### Task 1.6 `scripts/train.py`

```python
# scripts/train.py
import argparse, yaml, importlib, time
from common.seeding import set_seed
from common.envs import make_env

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["dqn", "ppo", "sac", "td3"])
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args()

    set_seed(args.seed)
    env = make_env(args.env, args.seed)
    cfg = yaml.safe_load(open(args.config))
    algo_mod = importlib.import_module(f"algos.{args.algo}")
    algo_mod.train(env, cfg, args.seed, args.total_steps, args.log_dir)

if __name__ == "__main__":
    main()
```

### Task 1.7 `scripts/run_all.py`

Loops over the experiment matrix (Section 7), runs each `(algo, env, seed)` triple in sequence, prints a progress bar. Skips runs whose CSV already exists.

### Task 1.8 `scripts/plot.py`

Reads every CSV in `logs/`, parses `(algo, env, seed)` from the filename, groups by `(algo, env)`, computes the bootstrap 95% CI of the mean episode return at evenly spaced step bins, plots one figure per env with all applicable algos overlaid. Saves PDF to `plots/`.

### Task 1.9 Smoke test

A "random policy" placeholder in `algos/random.py` that just samples actions and writes logs. Run `scripts/train.py --algo random --env CartPole-v1 --seed 0 --total-steps 5000`. Confirm a CSV appears and `plot.py` produces a (boring) figure. Once this works, B and C are unblocked.

---

## 6. Phase 2: algorithm implementations (Days 2-3, parallel)

Each algorithm exposes the same interface:

```python
# algos/<algo>.py
def train(env, cfg, seed, total_steps, log_dir):
    """Trains <algo> on `env`, writes CSV to `log_dir/<algo>/<env_id>/seed<seed>.csv`."""
```

After this phase: **code freeze on all four algorithm files.** Only configs and plots change after this.

### 6.1 DQN (owner A)

**Paper:** Mnih et al. 2013.

**Components:**
- ε-greedy action selection, linear decay from `eps_start=1.0` to `eps_end=0.05` over the first `eps_decay_fraction * total_steps` steps (default 0.10).
- Replay buffer of size 50000.
- Q-network with one target copy, hard-synced every `target_sync = 1000` steps.
- Huber loss (not MSE) on `Q(s,a) - (r + gamma * max_a' Q_target(s', a') * (1 - done))`.
- Adam, lr 1e-3.
- Train every 4 environment steps, batch 64.
- Warmup: `learning_starts = 1000` random steps before any gradient updates.

**Acceptance:** ≥ 475 mean return on CartPole-v1 with seed 0 within 100k steps.

**Common pitfalls (write in diary):**
- Forgetting to `.detach()` target Q values → divergence.
- Using `MSE` early in training: outliers blow up; Huber is more forgiving.
- Target sync too frequent (e.g. every 100 steps): too close to Q-learning without a target, unstable.
- Replay buffer too small (< 10k): catastrophic forgetting.
- No warmup: gradients on noise.

### 6.2 PPO (owner B)

**Paper:** Schulman et al. 2017.

**Components:**
- Rollout of `n_steps = 2048` env steps before each update (1024 on CartPole).
- GAE-λ with `gamma = 0.99`, `lam = 0.95`.
- Clipped surrogate: `L_pi = E[min(r * A, clip(r, 1-eps, 1+eps) * A)]` with `eps = 0.2`.
- Value loss with optional clipping (Schulman's trick), coefficient 0.5.
- Entropy bonus: 0.0 for discrete, 0.01 for continuous.
- `K_epochs = 10` epochs per rollout, mini-batch 64, advantage normalisation per mini-batch.
- Adam, lr 3e-4 with linear decay to 0 over training.
- For continuous: state-independent log-std (`nn.Parameter(torch.zeros(act_dim))`), Gaussian policy with NO tanh squash (the standard PPO recipe).

**Acceptance:**
- ≥ 475 on CartPole-v1 within 100k steps
- ≥ -200 on Pendulum-v1 within 200k steps

**Common pitfalls (write in diary):**
- Computing `log_prob_old` AFTER the policy update: silent disaster. Compute and store at rollout time.
- Not normalising advantages: huge variance.
- Forgetting to zero advantages at terminal states in GAE.
- Wrong sign on the entropy bonus.
- Off-by-one in the bootstrap value at the end of the rollout.

### 6.3 SAC (owner C)

**Paper:** Haarnoja et al. 2018 (use the "automatic temperature tuning" version from the second paper, 2018b).

**Components:**
- Replay buffer 1M (or 100k for our envs, capacity does not bind here).
- Twin Q-networks, both with target copies, Polyak averaging `tau = 0.005`.
- Squashed Gaussian actor: sample `u ~ N(mu, sigma)`, `a = tanh(u)`. Log-prob with the tanh correction:
  ```
  log_pi(a|s) = log N(u | mu, sigma) - sum_i log(1 - tanh(u_i)^2 + 1e-6)
  ```
- Automatic α tuning: optimise `log_alpha` so `E[log_pi + target_entropy] = 0` with `target_entropy = -|A|`.
- Adam, lr 3e-4 for all three optimisers (actor, critic, log_alpha).
- Warmup: `learning_starts = 1000` random-policy steps.
- Train every step, batch 256.

**Acceptance:** ≥ -200 on Pendulum-v1 within 100k steps.

**Common pitfalls (write in diary):**
- Forgetting the tanh correction in log-prob: training silently broken.
- Sampling with `rsample` (reparameterised) for the policy loss but `sample` for the Q targets, OR vice versa: think carefully which one needs gradients.
- Polyak direction: `target = tau * online + (1 - tau) * target` (online is the small contribution per step).
- `log_alpha` parameterisation, NOT `alpha` directly (keeps α positive without a hard constraint).

### 6.4 TD3 (owner C)

**Paper:** Fujimoto et al. 2018.

**Components:**
- Same replay + twin Q + target nets as SAC.
- Deterministic actor with tanh output, target actor with Polyak averaging.
- Exploration: Gaussian noise `N(0, 0.1)` added to actor output at env-step time, clipped to action range.
- Target policy smoothing: noise `N(0, 0.2)` clipped to ±0.5 added to target action.
- Policy delay: actor updated every 2 critic updates.
- Adam, lr 3e-4, batch 256, `tau = 0.005`.

**Acceptance:** ≥ -200 on Pendulum-v1 within 100k steps.

**Common pitfalls (write in diary):**
- Forgetting to clip the smoothed target action to the action range.
- Updating the target nets BEFORE the actor update (order matters at the delayed step).
- Sharing the noise schedule between exploration and target smoothing: keep them separate.

### 6.5 Code freeze

When all four algorithms hit their acceptance thresholds, tag the commit `algos-frozen`. From this point only configs change.

---

## 7. Phase 3: main experiments (Days 3-4, parallel)

### 7.1 Experiment matrix

13 algo-env pairs × 3 seeds = **39 runs**. Distribute 13 per teammate.

| Env                       | Step budget | Algorithms        | Runs |
| ------------------------- | ----------- | ----------------- | ---- |
| CartPole-v1               | 100k        | DQN, PPO          | 6    |
| Acrobot-v1                | 200k        | DQN, PPO          | 6    |
| Pendulum-v1               | 200k        | PPO, SAC, TD3     | 9    |
| MountainCarContinuous-v0  | 300k        | PPO, SAC, TD3     | 9    |
| **Total seeds**           |             |                   | **30** |

(Wait: 6 + 6 + 9 + 9 = 30 single-seed runs, but each line has the per-env × 3-seed expansion. The "13 algo-env pairs" count: 2 + 2 + 3 + 3 = 10, times 3 seeds = 30 runs total. Distribute 10 per teammate.)

### 7.2 CSV naming convention

`logs/<algo>/<env_id>/seed<N>.csv`

### 7.3 What to log

Every episode end: `step, episode, episode_return, episode_length, wall_clock_s`. No eval episodes (training returns suffice for this project; if time permits, add periodic eval with deterministic policy).

### 7.4 Wall-clock instrumentation

Log wall-clock seconds since start of training every episode. We will use this to answer "which algorithm is most computationally expensive per iteration" in the qualitative discussion.

### 7.5 Parameter count

After training, save the policy state dict and count parameters:
```python
n_params = sum(p.numel() for p in policy.parameters())
```
Used to answer "which stores the policy more compactly".

---

## 8. Phase 4: pre-registered ablations (Day 4, parallel)

Each algorithm owner picks **one env** for their ablation. We commit to these BEFORE running.

### 8.1 DQN on CartPole-v1 (owner A)

**Sweep:** `target_sync ∈ {100, 1000, 5000}` steps
**Hypothesis:** Too-frequent sync (100) destabilises learning; too-rare (5000) slows convergence. 1000 is the sweet spot.
**Cost:** 3 values × 3 seeds × 100k steps = 9 runs.

### 8.2 PPO on Pendulum-v1 (owner B)

**Sweep:** `clip_ratio ∈ {0.1, 0.2, 0.3}`
**Hypothesis:** 0.2 is the canonical default. 0.1 is too conservative on dense-reward continuous control. 0.3 starts to break the trust-region intuition.
**Cost:** 3 values × 3 seeds × 200k steps = 9 runs.

### 8.3 SAC on MountainCarContinuous-v0 (owner C)

**Sweep:** automatic α (target entropy = -|A|) vs fixed α ∈ {0.05, 0.2}
**Hypothesis:** Sparse-reward env stresses exploration; automatic tuning should help. Fixed low α should fail (insufficient exploration), fixed high α should help but plateau.
**Cost:** 3 settings × 3 seeds × 300k steps = 9 runs.

### 8.4 TD3 on MountainCarContinuous-v0 (owner C)

**Sweep:** exploration noise σ ∈ {0.1, 0.3, 0.5}
**Hypothesis:** Standard 0.1 is too low for sparse reward. Higher σ accelerates discovery of the goal.
**Cost:** 3 values × 3 seeds × 300k steps = 9 runs.

### 8.5 Ablation CSV naming

`logs_ablation/<algo>/<env_id>/<param>_<value>/seed<N>.csv`

---

## 9. Phase 5: analysis (Day 5 morning)

Single owner runs `analysis.ipynb` end to end. It must:

### Task 9.1 Aggregate CSVs

Walk `logs/` and `logs_ablation/`, load every CSV into a tidy `pandas.DataFrame` with columns `[algo, env, seed, step, return]`.

### Task 9.2 Main figure (Figure 1)

2x2 grid, one subplot per env. Within each subplot, plot every applicable algorithm as a line (mean over 3 seeds) with a shaded 95% bootstrap CI band. Bin the x-axis at 50 evenly spaced step bins; within each bin, take the mean return of episodes falling in that step range, then bootstrap over seeds.

### Task 9.3 Ablation figures (Figure 2-5)

One per algorithm. Same plotting style as Figure 1, but lines are the swept hyperparameter values.

### Task 9.4 Scoreboard table (Table 1)

Rows: `(algo, env)`. Columns:
- `final_return_mean ± ci`: mean of last 100 episodes, 95% CI over seeds
- `steps_to_50pct`: env-steps to reach 50% of the final return
- `steps_to_90pct`: same for 90%
- `wallclock_s`: median wall-clock to total_steps
- `n_params`: parameter count of the deployed policy (actor only for actor-critic)
- `on_policy`: yes/no

Save as `plots/scoreboard.tex` (booktabs).

### Task 9.5 Sanity check

For every row in Table 1 that disagrees with intuition, flag it for discussion in the diary section.

---

## 10. Phase 6: report (Day 6, all hands, 6-8 pages)

LaTeX skeleton lives in `report/main.tex`. Use the `article` class, 11pt, 1-inch margins. NeurIPS-like single-column or two-column, either is fine; two-column is denser and looks more professional for this length.

### Section 1: Introduction (~0.5 page)
**Author:** whoever has bandwidth.
**Content:** Why these four algorithms. The four qualitative questions from the brief, restated. Our contributions: rigorous multi-seed comparison, pre-registered ablations, theoretical bridge.

### Section 2: Background (~0.5 page)
**Author:** A (since A writes least elsewhere).
**Content:** MDP formalism. Distinction value-based vs policy-based, on-policy vs off-policy, discrete vs continuous. Citation to Sutton & Barto or course notes.

### Section 3: Methods (~2 pages, 0.5 per algorithm)
**Author:** each algorithm owner writes their own subsection.
**Content per subsection:**
- One paragraph of intuition (start with the mental model, then the math)
- The key loss / objective in display math
- Two implementation details we will return to in the diary

### Section 4: Experimental setup (~0.5 page)
**Author:** A.
**Content:** Env table (action space, obs space, reward structure, max episode return). Hyperparameter table (one column per algorithm). Seed protocol. Hardware. Library versions.

### Section 5: Main results (~1 page)
**Author:** whoever ran analysis.
**Content:** Figure 1 (2x2 main plot). Table 1 (scoreboard). One paragraph per environment commenting on the curves. Use the scoreboard to make quantitative claims.

### Section 6: Ablations (~1 page)
**Author:** each algorithm owner writes their own paragraph.
**Content:** For each ablation, state the hypothesis (from Section 8 of this plan), show the figure, comment on whether the hypothesis held. **Report negative results.**

### Section 7: Theoretical bridge (~0.5 page)
**Author:** Kevin (since A3 is fresh).
**Content:**
- PPO's clipped objective as a first-order surrogate for the trust-region KL constraint of NPG. Reference A3 Exercise 2 (NPG slow-changing property).
- SAC's entropy bonus as a regulariser analogous to OPPO's exploration bonus (A3 Exercise 3). The two come from different lineages but both encourage exploration without explicit ε.

### Section 8: Implementation diary (~1 page)
**Author:** each owner contributes 2-3 short paragraphs.
**Content:** Bugs that cost a long time to find. Hyperparameters that mattered surprisingly little (and why we think). Implementation tricks we found necessary. **This is what the brief explicitly grades on.**

### Section 9: Qualitative discussion (~0.5 page)
**Author:** whoever wrote Section 5.
**Content:** Answer the four questions from the brief, citing Table 1 numbers as evidence.
1. Most computationally expensive per iteration: <answer>
2. Most compact policy: <answer>
3. Best continuous-action scaling: <answer>
4. Best off-policy data efficiency: <answer>

### Section 10: Conclusion (~0.25 page)
**Author:** Kevin.

### Section 11: References
Use `poster.bib` (already in `poster_template_RL/`) for the four canonical papers plus Towers et al. 2023 (Gymnasium). See Section 14 of this plan.

### Report quality pass

Two-person rotation: each section is written by one person and reviewed by another. Final pass for em-dash hunting and filler removal.

---

## 11. Phase 7: poster (Day 7)

Adapt `poster_template_RL/poster.tex`. A1 portrait, gemini-cam theme.

### Layout (two-column, top-down)

**Header (red bar):** EPFL logo right, course code + group name right-aligned, title centered, authors below title.

**Left column:**
1. **Introduction block** (3 sentences max). Tie the four algorithms to the discrete/continuous and on/off-policy taxonomy.
2. **Environments block.** Row of four small env icons + one-line description each.
3. **Methods block.** Four equal-size cards (2x2 grid). Each card has algorithm name, one key equation, two-bullet intuition.

**Right column:**
4. **Main results block.** Figure 1 (2x2 grid of env plots).
5. **Scoreboard table.** Compact version of Table 1.
6. **Ablations strip.** Four small plots side by side, one per algorithm.
7. **Takeaways block.** Four crisp bullets answering the four qualitative questions.
8. **References.** Compact, 5-6 entries.

### Poster build commands

```powershell
cd applied_project/poster
pdflatex poster.tex; bibtex poster; pdflatex poster.tex; pdflatex poster.tex
```

(or use the included `Makefile`)

### Poster acceptance

Open the PDF, zoom to 25% (simulates ~2m viewing distance). Every label, every axis tick, every caption must be readable. If not: increase font size, simplify, or split the plot.

---

## 12. Phase 8: polish (final 4 hours)

| Check                                                             | Owner |
| ----------------------------------------------------------------- | ----- |
| Every plot in report and poster uses identical color per algorithm| any   |
| Every claim in text is traceable to a number in a table or figure | Kevin |
| No em dashes anywhere (regex search: `[—–]`)                       | Kevin |
| Citation check: Mnih, Schulman, Haarnoja, Fujimoto, Towers cited  | Kevin |
| Author list and affiliations correct on both PDFs                 | Kevin |
| Page count: report ≥ 6, ≤ 8                                       | any   |
| Poster compiles cleanly to A1 PDF                                 | any   |
| Repo has a 1-paragraph README pointing at how to reproduce        | A     |

---

## 13. Debugging guide (paste this into your diary as you go)

### DQN

| Symptom                              | Likely cause                                   |
| ------------------------------------ | ---------------------------------------------- |
| Q values explode to NaN              | Target Q not detached; or no target net at all |
| Reward never moves off floor          | ε decayed too fast; no warmup                  |
| Reward plateaus at random-policy level| Buffer too small; lr too high                  |
| Reward oscillates                    | Target sync too frequent                       |

### PPO

| Symptom                                          | Likely cause                                   |
| ------------------------------------------------ | ---------------------------------------------- |
| Loss looks fine but reward never improves        | `log_prob_old` recomputed after update         |
| Huge advantage spikes                            | No advantage normalisation                     |
| Continuous policy collapses to deterministic     | Entropy coef = 0 too early; log-std too low    |
| Last-step bootstrap wrong                        | Off-by-one in GAE termination handling          |

### SAC

| Symptom                                            | Likely cause                                  |
| -------------------------------------------------- | --------------------------------------------- |
| Reward stuck at warmup level                       | Tanh log-prob correction missing or wrong sign|
| Critic loss explodes                               | Q target not detached; α too large            |
| Policy collapses to one action                     | α too small; target entropy too high (closer to 0)|
| α drifts to absurd values                          | log_alpha parameterisation missing             |

### TD3

| Symptom                                | Likely cause                                       |
| -------------------------------------- | -------------------------------------------------- |
| Policy gets stuck at action boundary    | Target smoothing noise not clipped                  |
| Critic Q values diverge                | Twin-Q min taken on wrong side; or no policy delay  |
| No exploration in sparse env           | Exploration σ too low; consider colored noise       |

---

## 14. References (paste into `report/refs.bib` and `poster/poster.bib`)

```bibtex
@article{mnih2013playing,
  title={Playing Atari with Deep Reinforcement Learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  journal={arXiv preprint arXiv:1312.5602},
  year={2013}
}

@article{schulman2017proximal,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}

@inproceedings{haarnoja2018soft,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018}
}

@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and van Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018}
}

@misc{towers2023gymnasium,
  title={Gymnasium},
  author={Towers, Mark and Terry, Jordan K. and Kwiatkowski, Ariel and others},
  year={2023},
  publisher={Zenodo},
  url={https://zenodo.org/record/8127025}
}
```

Optional, only if Section 7 (theoretical bridge) is included:

```bibtex
@article{agarwal2020theory,
  title={On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift},
  author={Agarwal, Alekh and Kakade, Sham M and Lee, Jason D and Mahajan, Gaurav},
  journal={arXiv preprint arXiv:1908.00261},
  year={2020}
}

@inproceedings{mei2020softmax,
  title={On the Global Convergence Rates of Softmax Policy Gradient Methods},
  author={Mei, Jincheng and Xiao, Chenjun and Szepesvari, Csaba and Schuurmans, Dale},
  booktitle={ICML},
  year={2020}
}

@inproceedings{cai2020provably,
  title={Provably Efficient Exploration in Policy Optimization},
  author={Cai, Qi and Yang, Zhuoran and Jin, Chi and Wang, Zhaoran},
  booktitle={ICML},
  year={2020}
}
```

---

## 15. Day-zero kickoff checklist (do this in the next 30 minutes)

1. All three of us read this file end to end.
2. Confirm algorithm ownership (A: infra + DQN, B: PPO, C: SAC + TD3) by reply.
3. A executes Phase 0 (setup) and Phase 1 (infrastructure) before sleeping.
4. B and C read Schulman 2017, Haarnoja 2018, Fujimoto 2018 tonight.
5. Open shared Overleaf for the report. Open shared Overleaf for the poster (start from `poster_template_RL/`).
6. Open a shared spreadsheet for the hyperparameter table (one column per algorithm, populated as we tune).

When Phase 1 lands, Phase 2 begins. When the four algorithms hit acceptance, we freeze code and run Phase 3-4 in parallel on three laptops.
