"""
Evaluate a trained Rainbow DQN checkpoint on MinAtar Space Invaders.

Usage:
    python test_rainbow_dqn.py                        # uses rainbow_best.pt
    python test_rainbow_dqn.py checkpoints/rainbow_final.pt

Prints a performance report card matching the format used by the DQN
test script so results are directly comparable.
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from minatar import Environment
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import RainbowAgent


# ---------------------------------------------------------------------------
# MinAtar wrapper  (identical to the one in train_rainbow_dqn.py)
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


# ---------------------------------------------------------------------------
# MinAtar Space Invaders action labels
# ---------------------------------------------------------------------------
ACTION_NAMES = {
    0: "No-op",
    1: "Left",
    2: "Right",
    3: "Fire",
    4: "Left+Fire",
    5: "Right+Fire",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))

    # Accept an optional CLI argument for the checkpoint path
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    else:
        checkpoint = os.path.join(HERE, "checkpoints", "rainbow_best.pt")

    N_EPISODES = 20

    # Network hyper-parameters must match what was used during training
    N_ATOMS = 51
    V_MIN   = -10.0
    V_MAX   = 10.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env         = MinAtarEnv("space_invaders")
    state_shape = env.observation_space.shape
    n_actions   = env.action_space.n

    agent = RainbowAgent(
        state_shape = state_shape,
        n_actions   = n_actions,
        device      = device,
        n_atoms     = N_ATOMS,
        v_min       = V_MIN,
        v_max       = V_MAX,
    )

    print(f"Loading checkpoint: {checkpoint}")
    if not os.path.exists(checkpoint):
        print("Checkpoint not found. Train first using train_rainbow_dqn.py")
        return

    agent.load(checkpoint)
    agent.online_net.eval()   # deterministic: uses mu weights, no noise

    # -----------------------------------------------------------------------
    # Test episodes
    # -----------------------------------------------------------------------
    scores       = []
    lengths      = []
    action_counts = Counter()

    print(f"\nRunning {N_EPISODES} test episodes...")
    print("-" * 40)

    for i in range(N_EPISODES):
        obs, _ = env.reset()
        done   = False
        score  = 0.0
        steps  = 0

        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = agent.online_net.get_q_values(s).argmax(dim=1).item()

            action_counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            score += reward
            steps += 1

        scores.append(score)
        lengths.append(steps)
        print(f"Episode {i+1:02d}: Score = {score:.1f} | Duration = {steps} steps")

    # -----------------------------------------------------------------------
    # Report card
    # -----------------------------------------------------------------------
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    avg_len   = np.mean(lengths)

    total_actions = sum(action_counts.values())
    action_dist   = {k: (v / total_actions) * 100 for k, v in action_counts.items()}

    print("\n" + "=" * 40)
    print("       PERFORMANCE REPORT CARD       ")
    print("=" * 40)
    print(f"Model:           Rainbow DQN")
    print(f"Games Played:    {N_EPISODES}")
    print(f"Average Score:   {avg_score:.2f} +/- {std_score:.2f}")
    print(f"High Score:      {max_score:.1f}")
    print(f"Low Score:       {min_score:.1f}")
    print(f"Avg Survival:    {avg_len:.1f} frames")
    print("-" * 40)
    print("Behavior (Action Distribution):")
    for act, pct in sorted(action_dist.items()):
        name = ACTION_NAMES.get(act, f"Action {act}")
        print(f"  {name:>12}: {pct:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
