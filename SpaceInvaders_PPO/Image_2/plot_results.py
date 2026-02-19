import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _resolve_existing_path(candidates, label):
    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        checked = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No {label} found. Checked: {checked}")
    return resolved


def generate_training_curve(csv_path, output_path, show_plot):
    data = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    plt.plot(
        data["iteration"],
        data["reward"],
        label="Raw Reward",
        color="#1f77b4",
        alpha=0.3,
    )

    data["smoothed"] = data["reward"].rolling(window=10, min_periods=1).mean()
    plt.plot(
        data["iteration"],
        data["smoothed"],
        label="Smoothed Trend",
        color="#d62728",
        linewidth=2,
    )

    actual_peak = data["reward"].max()
    peak_row = data[data["reward"] == actual_peak].iloc[0]

    plt.scatter(
        peak_row["iteration"],
        actual_peak,
        color="gold",
        s=100,
        edgecolors="black",
        label=f"Peak Score: {actual_peak:.2f}",
        zorder=5,
    )

    plt.title("PPO Performance: Space Invaders Training Progress", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"--- [SUCCESS] Training curve saved to: {output_path} ---")
    if show_plot:
        plt.show()
    else:
        plt.close()


def evaluate_checkpoint(model_path, episodes, max_steps, sticky_action_prob, seed):
    import torch
    import torch.nn as nn
    import numpy as np
    from minatar.environments.space_invaders import Env

    class PolicyNetwork(nn.Module):
        def __init__(self, state_shape, n_actions):
            super(PolicyNetwork, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(state_shape[2], 16, kernel_size=3, padding=1)
            )
            self.actor = nn.Sequential(
                nn.Linear(1600, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
                nn.Softmax(dim=-1),
            )
            self.critic = nn.Sequential(
                nn.Linear(1600, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.reshape(x.size(0), -1)
            return self.actor(x), self.critic(x)

    env = Env(ramping=True)
    state_shape = env.state_shape()
    n_actions = 6

    model = PolicyNetwork(state_shape, n_actions)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    random = np.random.RandomState(seed)
    last_action = 0

    rewards = []
    for ep in range(1, episodes + 1):
        env.reset()
        state = env.state()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            steps += 1
            state_ts = (
                torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float()
            )
            with torch.no_grad():
                probs, _ = model(state_ts)
            action = torch.argmax(probs, dim=-1).item()
            if random.rand() < sticky_action_prob:
                action = last_action
            last_action = action
            reward, done = env.act(action)
            episode_reward += reward
            state = env.state()
            if steps >= max_steps:
                break

        rewards.append(float(episode_reward))
        print(
            f"Episode {ep:3d}/{episodes:3d} | Steps: {steps:5d} | Reward: {episode_reward:6.2f}",
            flush=True,
        )

    return rewards


def generate_model_curve(
    model_path,
    episodes,
    max_steps,
    sticky_action_prob,
    seed,
    output_png,
    output_csv,
    show_plot,
):
    rewards = evaluate_checkpoint(
        model_path=model_path,
        episodes=episodes,
        max_steps=max_steps,
        sticky_action_prob=sticky_action_prob,
        seed=seed,
    )
    df = pd.DataFrame(
        {"episode": list(range(1, len(rewards) + 1)), "reward": rewards}
    )
    df["smoothed"] = df["reward"].rolling(window=min(10, len(df)), min_periods=1).mean()
    df.to_csv(output_csv, index=False)

    peak = df["reward"].max()
    peak_row = df[df["reward"] == peak].iloc[0]
    avg = df["reward"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df["episode"], df["reward"], label="Episode Reward", color="#1f77b4", alpha=0.35)
    plt.plot(df["episode"], df["smoothed"], label="Smoothed Trend", color="#d62728", linewidth=2)
    plt.axhline(avg, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Mean: {avg:.2f}")
    plt.scatter(
        peak_row["episode"],
        peak,
        color="gold",
        s=100,
        edgecolors="black",
        label=f"Best Episode: {peak:.2f}",
        zorder=5,
    )
    plt.title("Best PPO Checkpoint: Evaluation Reward Curve", fontsize=14)
    plt.xlabel("Evaluation Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"--- [SUCCESS] Evaluation curve saved to: {output_png} ---")
    print(f"--- [SUCCESS] Evaluation data saved to: {output_csv} ---")
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["csv", "model"],
        default="model",
        help="`model` evaluates a checkpoint and plots episode rewards; `csv` plots training CSV.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes in model mode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Max env steps per evaluation episode in model mode.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Checkpoint path (default auto-detects Models/best_ppo_model_peak.pth).",
    )
    parser.add_argument(
        "--sticky-action-prob",
        type=float,
        default=0.1,
        help="Sticky action probability for MinAtar evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sticky-action sampling.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive plt.show() (useful for terminal/headless runs).",
    )
    args = parser.parse_args()

    if args.mode == "csv":
        candidate_csvs = [
            script_dir / "ppo_progress_master.csv",
            repo_root / "Image_2/ppo_progress_master.csv",
            Path("ppo_progress_master.csv"),
        ]
        csv_path = _resolve_existing_path(candidate_csvs, "CSV")
        output_path = script_dir / "final_training_curve.png"
        generate_training_curve(
            csv_path=csv_path,
            output_path=output_path,
            show_plot=not args.no_show,
        )
        return

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if not (0.0 <= args.sticky_action_prob <= 1.0):
        raise ValueError("--sticky-action-prob must be in [0, 1]")

    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
    else:
        candidate_models = [
            repo_root / "Models/best_ppo_model_peak.pth",
            repo_root / "best_ppo_model_peak.pth",
            Path("Models/best_ppo_model_peak.pth").resolve(),
            Path("best_ppo_model_peak.pth").resolve(),
        ]
        model_path = _resolve_existing_path(candidate_models, "model checkpoint")

    output_png = script_dir / "best_model_eval_curve.png"
    output_csv = script_dir / "best_model_eval_rewards.csv"
    generate_model_curve(
        model_path=model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        sticky_action_prob=args.sticky_action_prob,
        seed=args.seed,
        output_png=output_png,
        output_csv=output_csv,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"--- [ERROR] Visualization failed: {e} ---")
