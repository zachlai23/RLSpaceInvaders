"""
Analyze trained Rainbow DQN checkpoints on MinAtar Space Invaders.

Runs 20 independent test episodes for each of three checkpoints
(250k, 500k, 1M steps), then generates:
  - A per-checkpoint performance table (max / min / mean score, avg survival)
  - An action-distribution breakdown per checkpoint
  - A learning-curve plot (eval_mean vs frame) from train_log.csv

Usage:
    python analyze_rainbow_dqn.py

Outputs:
    analysis_results.csv   — raw episode data
    learning_curve.png     — ep_rew_mean plot
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from gymnasium import spaces
from minatar import Environment
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import RainbowAgent


# ---------------------------------------------------------------------------
# MinAtar wrapper  (identical to train / test scripts)
# ---------------------------------------------------------------------------

class MinAtarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_name: str = "space_invaders"):
        self.game = Environment(env_name, sticky_action_prob=0.1, difficulty_ramping=True)
        self.action_space      = spaces.Discrete(self.game.num_actions())
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.game.state_shape(), dtype=np.float32
        )

    def step(self, action):
        reward, terminated = self.game.act(action)
        obs = self.game.state().astype(np.float32)
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self.game.state().astype(np.float32), {}


ACTION_NAMES = {
    0: "No-op",
    1: "Left",
    2: "Right",
    3: "Fire",
    4: "Left+Fire",
    5: "Right+Fire",
}

N_EPISODES  = 20
N_ATOMS     = 51
V_MIN       = -10.0
V_MAX       = 10.0

# Checkpoints to evaluate  (label, filename)
CHECKPOINTS = [
    ("250k",  "rainbow_frame_250000.pt"),
    ("500k",  "rainbow_frame_500000.pt"),
    ("750k",  "rainbow_frame_750000.pt"),
    ("1M",    "rainbow_frame_1000000.pt"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agent(ckpt_path: str, state_shape, n_actions, device):
    agent = RainbowAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=device,
        n_atoms=N_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
    )
    agent.load(ckpt_path)
    agent.online_net.eval()
    return agent


def run_episodes(env, agent, n_episodes, device):
    scores, lengths = [], []
    action_counts = Counter()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        score, steps = 0.0, 0

        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = agent.online_net.get_q_values(s).argmax(dim=1).item()

            action_counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1

        scores.append(score)
        lengths.append(steps)
        print(f"  ep {ep+1:02d}: score={score:.1f}  survival={steps}")

    return scores, lengths, action_counts


def top_actions(action_counts, n=3):
    """Return the top-n action names by frequency."""
    total = sum(action_counts.values())
    ranked = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{ACTION_NAMES.get(a,'?')} {v/total*100:.0f}%" for a, v in ranked[:n]]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HERE     = os.path.dirname(os.path.abspath(__file__))
    CKPT_DIR = os.path.join(HERE, "checkpoints")
    LOG_PATH = os.path.join(HERE, "train_log.csv")
    OUT_CSV  = os.path.join(HERE, "analysis_results.csv")
    PLOT_PATH = os.path.join(HERE, "learning_curve.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    env = MinAtarEnv("space_invaders")
    state_shape = env.observation_space.shape
    n_actions   = env.action_space.n

    all_rows   = []   # for CSV
    table_rows = []   # for console table

    for label, fname in CHECKPOINTS:
        ckpt_path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {ckpt_path} not found.\n")
            continue

        print(f"=== {label} steps  ({fname}) ===")
        agent = load_agent(ckpt_path, state_shape, n_actions, device)
        scores, lengths, action_counts = run_episodes(env, agent, N_EPISODES, device)

        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        avg_surv  = float(np.mean(lengths))

        behavior = top_actions(action_counts)

        table_rows.append({
            "Version":      "Rainbow",
            "Steps":        label,
            "Avg Score":    f"{avg_score:.1f}",
            "Std Score":    f"{std_score:.1f}",
            "Max Score":    f"{max_score:.1f}",
            "Min Score":    f"{min_score:.1f}",
            "Avg Survival": f"{avg_surv:.1f} frames",
            "Key Behavior": behavior,
        })

        for i, (sc, ln) in enumerate(zip(scores, lengths)):
            all_rows.append({
                "checkpoint": label,
                "episode":    i + 1,
                "score":      sc,
                "length":     ln,
            })

        print()

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RAINBOW DQN PERFORMANCE SUMMARY  (20 episodes per checkpoint)")
    print("=" * 80)
    header = f"{'Version':<10} {'Steps':<6} {'Avg Score':>10} {'Std':>7} {'Max':>6} {'Min':>6} {'Avg Survival':>14}  Key Behavior"
    print(header)
    print("-" * 80)
    for r in table_rows:
        print(
            f"{r['Version']:<10} {r['Steps']:<6} {r['Avg Score']:>10} "
            f"{r['Std Score']:>7} {r['Max Score']:>6} {r['Min Score']:>6} "
            f"{r['Avg Survival']:>14}  {r['Key Behavior']}"
        )
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Save raw results CSV
    # -----------------------------------------------------------------------
    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"\nRaw episode data saved to {OUT_CSV}")

    # -----------------------------------------------------------------------
    # Learning curve plot  (TensorBoard-style, clipped to 1M frames)
    # -----------------------------------------------------------------------
    if not os.path.exists(LOG_PATH):
        print(f"No training log found at {LOG_PATH} — skipping plot.")
        return

    from datetime import datetime

    log = pd.read_csv(LOG_PATH)
    log = log[log["frame"] <= 1_000_000].copy()

    # Exponential moving average (TensorBoard default smoothing = 0.6)
    def ema(values, weight=0.6):
        smoothed, last = [], values.iloc[0]
        for v in values:
            last = last * weight + v * (1 - weight)
            smoothed.append(last)
        return smoothed

    raw    = log["eval_mean"].values
    smooth = ema(log["eval_mean"])

    # Header values (final point in 0-1M window)
    last_raw    = raw[-1]
    last_smooth = smooth[-1]
    last_step   = int(log["frame"].iloc[-1])
    step_str    = f"{int(last_step / 1e3)}k"
    time_str    = datetime.now().strftime("%a %b %d, %H:%M:%S")
    # Relative: estimate from ~300 fps on CPU
    rel_secs    = last_step / 300
    rel_str     = (f"{int(rel_secs//3600)}h {int((rel_secs%3600)//60)}m"
                   if rel_secs >= 3600 else f"{int(rel_secs//60)}m {int(rel_secs%60)}s")

    LINE_COLOR = "#4db6e8"
    BG_COLOR   = "#1e1e1e"
    HDR_COLOR  = "#252525"
    GRID_COLOR = "#2d2d2d"
    TEXT_COLOR = "#cccccc"
    DIM_COLOR  = "#888888"
    SEP_COLOR  = "#3a3a3a"

    # Layout: header panel on top, chart below
    fig = plt.figure(figsize=(10, 6.2))
    fig.patch.set_facecolor(BG_COLOR)

    # Header axes (top 22% of figure)
    hdr = fig.add_axes([0.0, 0.78, 1.0, 0.22])
    hdr.set_facecolor(HDR_COLOR)
    hdr.axis("off")

    # Column positions (in axes fraction)
    col_x = {"name": 0.03, "smoothed": 0.52, "value": 0.64,
              "step": 0.74, "time": 0.84, "relative": 0.95}

    # Column headers row
    for key, label in [("name", "Name"), ("smoothed", "Smoothed"),
                        ("value", "Value"), ("step", "Step"),
                        ("time", "Time"), ("relative", "Relative")]:
        hdr.text(col_x[key], 0.80, label, color=DIM_COLOR, fontsize=8,
                 transform=hdr.transAxes, va="top",
                 ha="right" if key != "name" else "left")

    # Separator line between header cols and data row
    hdr.axhline(0.52, color=SEP_COLOR, linewidth=0.6, xmin=0.01, xmax=0.99)

    # Color swatch (filled circle) + run name
    hdr.text(col_x["name"] - 0.005, 0.22, "●", color=LINE_COLOR,
             fontsize=11, transform=hdr.transAxes, va="center")
    hdr.text(col_x["name"] + 0.025, 0.22, "RainbowDQN_1M/ep_rew_mean",
             color=TEXT_COLOR, fontsize=8.5, transform=hdr.transAxes, va="center")

    # Data values
    for key, val in [("smoothed", f"{last_smooth:.3f}"),
                     ("value",    f"{last_raw:.2f}"),
                     ("step",     step_str),
                     ("time",     time_str),
                     ("relative", rel_str)]:
        hdr.text(col_x[key], 0.22, val, color=TEXT_COLOR, fontsize=8.5,
                 transform=hdr.transAxes, va="center", ha="right")

    # Separator between header panel and chart
    fig.add_artist(plt.Line2D([0, 1], [0.78, 0.78],
                              transform=fig.transFigure,
                              color=SEP_COLOR, linewidth=0.8))

    # Chart axes
    ax = fig.add_axes([0.07, 0.08, 0.91, 0.66])
    ax.set_facecolor(BG_COLOR)

    # Raw (noisy) line — thin, low alpha
    ax.plot(log["frame"], raw, color=LINE_COLOR, linewidth=0.9,
            alpha=0.35, zorder=2)

    # Smoothed line — thick, opaque
    ax.plot(log["frame"], smooth, color=LINE_COLOR, linewidth=2.0,
            alpha=1.0, zorder=3)

    # Grid
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="-")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # Axes formatting
    ax.set_xlim(0, 1_000_000)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100_000))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1e3)}k" if x > 0 else "0")
    )
    ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT_COLOR)

    # Tag label bottom-left of chart (like TensorBoard)
    ax.text(0.01, 0.02, "tag: rollout/ep_rew_mean",
            transform=ax.transAxes, fontsize=7.5,
            color=DIM_COLOR, va="bottom")

    plt.savefig(PLOT_PATH, dpi=150, facecolor=BG_COLOR)
    plt.close()
    print(f"Learning curve saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()
