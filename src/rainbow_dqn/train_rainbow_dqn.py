"""
Train Rainbow DQN on MinAtar Space Invaders.

Usage:
    python train_rainbow_dqn.py

Checkpoints are saved to src/rainbow_dqn/checkpoints/.
A CSV training log is written to src/rainbow_dqn/train_log.csv.
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minatar import Environment
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import RainbowAgent


# ---------------------------------------------------------------------------
# MinAtar wrapper
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(env: MinAtarEnv, agent: RainbowAgent, n_episodes: int = 5):
    """Run greedy episodes (eval mode — noise disabled)."""
    agent.online_net.eval()
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                action = agent.online_net.get_q_values(s).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        scores.append(total)
    agent.online_net.train()
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    TOTAL_FRAMES  = 5_000_000
    EVAL_INTERVAL = 50_000      # evaluate every N frames
    SAVE_INTERVAL = 250_000     # save periodic checkpoint every N frames
    MIN_BUFFER    = 1_600       # warm-up before training begins

    BATCH_SIZE    = 32
    LR            = 1e-4
    GAMMA         = 0.99
    N_STEP        = 3
    N_ATOMS       = 51
    V_MIN         = -10.0
    V_MAX         = 10.0
    BUFFER_SIZE   = 100_000
    ALPHA         = 0.5         # PER priority exponent
    BETA_START    = 0.4         # IS weight exponent (annealed to 1.0)
    BETA_FRAMES   = TOTAL_FRAMES
    TARGET_UPDATE = 1_000       # hard target-net sync interval (steps)

    HERE      = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR  = os.path.join(HERE, "checkpoints")
    LOG_PATH  = os.path.join(HERE, "train_log.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # -----------------------------------------------------------------------
    # Environment & agent
    # -----------------------------------------------------------------------
    env      = MinAtarEnv("space_invaders")
    eval_env = MinAtarEnv("space_invaders")

    state_shape = env.observation_space.shape   # (10, 10, C)
    n_actions   = env.action_space.n
    print(f"State shape : {state_shape}   Actions : {n_actions}")

    agent = RainbowAgent(
        state_shape  = state_shape,
        n_actions    = n_actions,
        device       = device,
        n_atoms      = N_ATOMS,
        v_min        = V_MIN,
        v_max        = V_MAX,
        buffer_size  = BUFFER_SIZE,
        batch_size   = BATCH_SIZE,
        n_step       = N_STEP,
        gamma        = GAMMA,
        alpha        = ALPHA,
        beta_start   = BETA_START,
        beta_frames  = BETA_FRAMES,
        lr           = LR,
        target_update= TARGET_UPDATE,
    )

    print(f"Training for {TOTAL_FRAMES:,} frames …\n")

    # -----------------------------------------------------------------------
    # CSV log
    # -----------------------------------------------------------------------
    with open(LOG_PATH, "w") as f:
        f.write("frame,episode,eval_mean,eval_std,avg_loss\n")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    obs, _      = env.reset()
    episode     = 0
    ep_reward   = 0.0
    recent_loss = []
    best_eval   = -float("inf")
    t0          = time.time()

    for frame in range(1, TOTAL_FRAMES + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        obs        = next_obs
        ep_reward += reward

        # Train once the buffer has enough transitions
        if agent.buffer.is_ready(MIN_BUFFER):
            loss = agent.update(frame)
            recent_loss.append(loss)

        if done:
            episode  += 1
            obs, _    = env.reset()
            ep_reward = 0.0

        # -------------------------------------------------------------------
        # Periodic evaluation & logging
        # -------------------------------------------------------------------
        if frame % EVAL_INTERVAL == 0:
            eval_mean, eval_std = evaluate(eval_env, agent)
            avg_loss = float(np.mean(recent_loss[-500:])) if recent_loss else 0.0
            fps      = frame / (time.time() - t0)

            print(
                f"Frame {frame:>9,} | Ep {episode:>5} | "
                f"Eval {eval_mean:6.2f} ± {eval_std:.2f} | "
                f"Loss {avg_loss:.5f} | {fps:.0f} fps"
            )

            with open(LOG_PATH, "a") as f:
                f.write(f"{frame},{episode},{eval_mean:.4f},{eval_std:.4f},{avg_loss:.6f}\n")

            if eval_mean > best_eval:
                best_eval = eval_mean
                agent.save(os.path.join(SAVE_DIR, "rainbow_best.pt"))
                print(f"  ↑ New best ({best_eval:.2f}) — checkpoint saved.")

        # Periodic checkpoint
        if frame % SAVE_INTERVAL == 0:
            agent.save(os.path.join(SAVE_DIR, f"rainbow_frame_{frame}.pt"))

    # Final checkpoint
    agent.save(os.path.join(SAVE_DIR, "rainbow_final.pt"))
    print(f"\nTraining complete. Final checkpoint saved to {SAVE_DIR}/rainbow_final.pt")


if __name__ == "__main__":
    main()
