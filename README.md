# RLSpaceInvaders

# RLSpaceInvaders

A method-based reinforcement learning (RL) analysis project using **Space Invaders** as a controlled testbed. Rather than optimizing for a single “best” agent, we compare multiple RL methods (random → DQN-family → PPO) and run **hyperparameter sweeps + component ablations** to understand what drives performance, stability, and failure modes.

This repo also includes a small GitHub Pages site (`/docs`) that hosts our proposal/status/final writeups.

## Project goals

- **Baseline first**: start from a **random policy** (and simple heuristics if feasible).
- **Battery of methods**: compare DQN, Double DQN, Rainbow-style variants, and PPO under a consistent protocol.
- **Method-based analysis**:
  - multi-seed evaluation
  - learning curves (sample efficiency)
  - stability/variance
  - compute cost
  - ablations (turn off individual components and quantify the drop)
- **Failure modes**: document divergence/instability, brittle policies, exploration issues, etc.

## Environments

We use **Gymnasium** as the environment API.

- **Primary**: **MinAtar SpaceInvaders** (10×10 grid, channel-based observations) for fast iteration.
- **Optional**: full ALE Atari Space Invaders (if time/compute allow).

> Note: MinAtar provides `SpaceInvaders-v0` (full action set) and `SpaceInvaders-v1` (minimal action set). We will choose one and keep it fixed across experiments for fair comparisons.

## Repository structure

- `docs/` — GitHub Pages site (proposal/status/final, CSS, layout)
- `src/` — (planned) training/evaluation code
- `configs/` — (planned) experiment configs / sweep definitions
- `results/` — (planned) logs, plots, checkpoints, and summary tables

If you don’t see `src/`, `configs/`, or `results/` yet, they’re placeholders for the next phase of implementation.

## Getting started

### 1) Create an environment

We’ll use a standard Python workflow. Example using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
python -m pip install --upgrade pip
```

### 2) Install dependencies

Once we finalize an RL library (e.g., Stable-Baselines3 / sb3-contrib or CleanRL), we’ll pin exact versions.

For now, MinAtar + Gymnasium can be installed with:

```bash
pip install gymnasium
pip install minatar
```

Quick sanity check:

```python
import gymnasium as gym

env = gym.make("MinAtar/SpaceInvaders-v1")
obs, info = env.reset(seed=0)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
print("OK")
```

## Experiments (planned)

### Baselines
- Random policy (uniform actions)
- Optional heuristics (e.g., always-shoot)

### Methods
- DQN (baseline)
- Double DQN
- Rainbow DQN (and ablations)
- PPO (on-policy baseline)
- Optional: A2C / lightweight model-based experiment

### Evaluation protocol
- Same environment version/settings for all runs
- Multiple random seeds
- Learning curves vs environment steps
- Aggregate metrics: mean ± std across seeds
- Record training compute (updates / wall-clock time if available)

## GitHub Pages site

The project website lives in `docs/` and is rendered by GitHub Pages.

Locally, you can preview it (one option):

```bash
cd docs
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Team

- Brian Byunghyun Kim (bkim3164)
- Zachary Ian Lai (zilai)
- Zongze Li (zongzel3)

## License

TBD