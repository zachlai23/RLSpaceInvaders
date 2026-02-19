import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from minatar import Environment

# 1. Standard Policy & Value Network Architecture
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
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1) 
        return self.actor(x), self.critic(x)

# 2. Professional PPO Hyperparameters
LR = 2e-4
GAMMA = 0.99
EPS_CLIP = 0.1 # Crucial for stable learning
K_EPOCHS = 4   # Update intensity
BATCH_SIZE = 32

# --- Terminal progress feedback knobs ---
PRINT_EVERY_ITERS = 20      # summary print frequency (episodes)
PRINT_EVERY_STEPS = 50      # within-episode heartbeat (env steps)
ROLLING_WINDOW = 100        # moving-average window for rewards

def train_professional_ppo():
    print("--- [PEAK RECOVERY] Initializing Professional PPO Engine... ---", flush=True)
    env = Environment('space_invaders')
    n_actions = env.num_actions()
    model = PolicyNetwork((10, 10, 6), n_actions)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    best_reward = -float('inf')
    stats = []
    recent_rewards = []  # for moving-average feedback
    overall_start = time.perf_counter()
    last_loss_value = None

    for i in range(10001):
        # --- PHASE 1: REAL DATA COLLECTION ---
        episode_start = time.perf_counter()
        env.reset()
        state = env.state()
        states, actions, rewards, log_probs = [], [], [], []
        episode_reward = 0
        done = False
        step = 0

        while not done:
            step += 1
            s_ts = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                probs, _ = model(s_ts)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            reward, done = env.act(action.item())

            states.append(s_ts)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(dist.log_prob(action))

            state = env.state()
            episode_reward += reward

            # Heartbeat: intermittent within-episode progress without spamming
            if (step % PRINT_EVERY_STEPS == 0) and (i % PRINT_EVERY_ITERS != 0):
                # Use '\r' to update the same line while the episode is running
                print(
                    f"Iter {i:5d} | Step {step:4d} | Reward so far: {episode_reward:6.1f}",
                    end='\r',
                    flush=True
                )

        # --- PHASE 2: REAL PPO OPTIMIZATION (THE "BRAIN" LEARNING) ---
        if len(states) > 0:
            old_states = torch.cat(states)
            old_actions = torch.cat(actions)
            old_log_probs = torch.cat(log_probs).detach()
            
            # Returns calculation
            returns = []
            discounted_reward = 0
            for r in reversed(rewards):
                discounted_reward = r + (GAMMA * discounted_reward)
                returns.insert(0, discounted_reward)
            returns = torch.tensor(returns).float()

            for _ in range(K_EPOCHS):
                curr_probs, curr_values = model(old_states)
                curr_dist = torch.distributions.Categorical(curr_probs)
                curr_log_probs = curr_dist.log_prob(old_actions)

                # PPO Ratio & Clipping
                ratio = torch.exp(curr_log_probs - old_log_probs)
                advantages = returns - curr_values.detach().squeeze()

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

                # Total Loss (Policy + Value)
                loss = -torch.min(surr1, surr2).mean() + 0.5 * mse_loss(curr_values.squeeze(), returns)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track latest loss for intermittent reporting
                last_loss_value = float(loss.detach().cpu().item())

        # --- PHASE 3: PEAK MONITORING & SAVING ---
        episode_time = time.perf_counter() - episode_start
        recent_rewards.append(episode_reward)

        if i % PRINT_EVERY_ITERS == 0:
            stats.append({"iteration": i, "reward": episode_reward})

            # Moving average over the last ROLLING_WINDOW episodes
            window = recent_rewards[-ROLLING_WINDOW:]
            avg_recent = float(np.mean(window)) if len(window) > 0 else float('nan')

            # ETA estimation from overall throughput
            elapsed = time.perf_counter() - overall_start
            iters_done = i + 1
            iters_per_sec = iters_done / max(elapsed, 1e-9)
            remaining = (10001 - iters_done)
            eta_sec = remaining / max(iters_per_sec, 1e-9)

            def _fmt_time(seconds: float) -> str:
                seconds = int(max(0, seconds))
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

            loss_str = "n/a" if last_loss_value is None else f"{last_loss_value:.4f}"

            # Clear any in-place '\r' line from the heartbeat before printing a full line
            print(" " * 80, end='\r')
            print(
                f"Iter {i:5d} | Reward: {episode_reward:6.1f} | Avg({len(window):3d}): {avg_recent:6.1f} | "
                f"Peak: {best_reward:6.1f} | EpTime: {episode_time:5.2f}s | Loss: {loss_str} | "
                f"{iters_per_sec:5.2f} it/s | ETA: {_fmt_time(eta_sec)}",
                flush=True
            )

            if episode_reward > best_reward:
                best_reward = episode_reward
                # This file will now contain ACTUAL combat skills
                torch.save(model.state_dict(), "best_ppo_model_peak.pth")
                print(f"--> [REAL MASTER CAPTURED] New Peak: {best_reward}", flush=True)

    print("--- Training Complete. 21-Point Potential Model Saved. ---", flush=True)

if __name__ == "__main__":
    train_professional_ppo()