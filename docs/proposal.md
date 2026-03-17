---
layout: default
title:  Proposal
---

# Project Proposal

Last updated January 23rd, 2026

## Summary

Space Invaders is a classic fixed-shooter game where the player controls a starfighter, attacking and dodging enemies from above. Rather than treating this as a one-off application, our project uses Space Invaders as a **controlled testbed** for comparing and understanding reinforcement learning (RL) methods.


To make broad comparisons feasible, we will primarily use **MinAtar SpaceInvaders** through **Gymnasium**. MinAtar provides a simplified 10×10, channel-based observation space that trains much faster than full Atari, enabling us to run **multiple seeds, hyperparameter sweeps, and component ablations** within our compute budget. If time permits, we will validate key findings on the full ALE Space Invaders environment.

## Project Goals

### 🎯 Minimum Goal (Baseline + Harness)
- Implement a **random policy** baseline and a simple “do-nothing / always-shoot / simple heuristic” baseline (if feasible).
- Build a reproducible training + evaluation pipeline (fixed seeds, logging, learning curves, checkpointing).
- Define core metrics (episode return/score, sample efficiency, stability across seeds, compute cost).

### 🚀 Realistic Goal (Battery of Methods)
- Train and compare multiple RL methods under a consistent protocol:
  - **DQN (baseline)**
  - **Double DQN**
  - **Rainbow DQN** (as a combined method)
  - **PPO** (policy-gradient baseline)
- Perform **hyperparameter sweeps** for each method (learning rate, exploration schedule / entropy bonus, replay parameters, batch size, target update frequency, etc.).
- Produce a broad comparative analysis: learning curves, final performance, stability, and compute tradeoffs.

### 🌙 Moonshot Goal (Ablations + Deeper Insights)
- Conduct a structured **ablation study** on Rainbow components by toggling features on/off and measuring impact:
  - Double Q-learning
  - Dueling networks
  - Prioritized experience replay
  - Noisy networks / exploration variants
  - N-step returns
  - Distributional RL
- Explore one **model-based** method or model-based component (time permitting), focusing on analysis rather than “winning” the game.
- Develop language and examples for **failure modes** (e.g., oscillatory learning, catastrophic forgetting, over-exploration, reward hacking, brittle policies, sensitivity to stochasticity).

## Methods to Compare

### Baselines
- **Random policy** (uniform actions) as the minimum naive baseline.
- Optional simple heuristics (e.g., always shoot; move away from nearest threat) as an interpretable reference.

### Value-Based Methods
- **DQN** as the baseline deep RL method.
- **Double DQN** to reduce Q-value overestimation.
- **Rainbow DQN** as a strong composite method.

### Policy-Gradient Method
- **PPO** as a widely used, stable policy-gradient baseline.


### Model-Based (If Time Permits)
- A lightweight model-based approach or component (e.g., learning a dynamics model for short-horizon planning, or comparing model-free vs. model-based sensitivity to environment stochasticity).

### Practical Method-Based Comparison Menu (Planned)

To keep the project method-driven and feasible, we plan to prioritize the following menu of methods/components (in roughly this order), adding items as our implementation bandwidth allows:

1) **Random policy** (naïve baseline)
2) **DQN** (value-based baseline)
3) **Double DQN** (overestimation reduction)
4) **Rainbow-style components** (run as a full Rainbow agent, then ablate components):
   - **Dueling networks**
   - **Prioritized experience replay**
   - **N-step returns**
   - **NoisyNet exploration** (vs ε-greedy)
   - **Distributional RL** (e.g., C51 / QR-DQN)
5) **PPO** (on-policy baseline for contrast with off-policy DQN-family)
6) **Optional additions** (time permitting):
   - **A2C** (simpler actor–critic baseline)
   - A small **model-based** experiment (e.g., short-horizon Dyna-style rollouts)

This menu is designed to support clear ablation questions (e.g., “turn off prioritized replay—how much does sample efficiency or stability change?”) and to compare major algorithm families (value-based vs policy-gradient) under a consistent Gymnasium/MinAtar protocol.

## Evaluation Plan

We will evaluate methods using a consistent, reproducible protocol.

### Metrics
- **Average episode return / score** over a fixed evaluation budget.
- **Sample efficiency**: performance vs. environment steps (learning curves).
- **Stability**: variance across multiple random seeds.
- **Compute cost**: training time, number of updates, and (if measurable) wall-clock time.
- **Human-normalized score (optional)**: report performance relative to a human reference (e.g., 100% = human score), if we can obtain a consistent human baseline.

### Experimental Design
- Use the same environment version/settings across methods.
- Run each method with **multiple seeds**.
- For each method, run a small **hyperparameter sweep**, then report:
  - best settings found (within our budget)
  - sensitivity of performance to key hyperparameters
- We will report results on MinAtar first (fast iteration), then optionally validate key findings on the full ALE environment if time permits.
- We will standardize environment preprocessing and action-set choice (MinAtar v0 vs v1) across all methods to ensure comparisons reflect algorithm differences rather than interface differences.

### Ablations (Core to the Method-Based Goal)
For Rainbow-style components, we will explicitly answer:
- **“If we turn off component X, how much worse does it get?”**

We will present ablation results as:
- side-by-side learning curves
- final score comparison
- stability/variance changes

### Failure Mode Analysis
Beyond metrics, we will document failure modes with concrete artifacts:
- plots showing instability / divergence
- qualitative behavior clips or descriptions
- proposed explanations tied to algorithm design (e.g., exploration strategy, bootstrapping bias, on-policy vs off-policy dynamics)

## Tools & Resources

- **Platform**: Python
- **Environment API**: Gymnasium
- **Environments**:
  - **Primary**: **MinAtar SpaceInvaders** (10×10 grid, channel-based observations) for fast, reproducible method comparisons.
  - **Optional**: ALE / Atari Space Invaders (Gym-style) if time/compute allow.
- **MinAtar details**:
  - We will use the Gymnasium wrapper and create envs via `gym.make("MinAtar/SpaceInvaders-v1")` (or `-v0`).
  - **v0** uses the **full action set (6 actions)**; **v1** uses a **reduced minimal action set**. We will choose one version and keep it fixed across all experiments for fairness.
  - MinAtar’s simplified dynamics make it feasible to run **multiple seeds, hyperparameter sweeps, and ablations** within our compute budget.
- **RL Frameworks** (to be selected based on implementation feasibility):
  - **Stable-Baselines3 (+ sb3-contrib)** for PPO/A2C and DQN variants
  - **CleanRL** for transparent, editable reference implementations (good for ablations)
  - A lightweight **custom implementation** for targeted Rainbow-component ablations (if needed)

## Team Meetings

📅 **January 22nd, 2026 @ 1:00 PM PST**  
[Zoom Meeting Link](https://www.google.com/url?q=https://uci.zoom.us/j/98426114741&sa=D&source=calendar&ust=1769212997341317&usg=AOvVaw3CUJ3h4T7r5JGExkt2JB5U)
