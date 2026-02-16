"""
Rainbow DQN Agent.

Integrates all six Rainbow components:
  1. Double DQN       — online net selects action, target net evaluates value
  2. Prioritized PER  — importance-sampled replay with sum-tree
  3. N-step Returns   — multi-step bootstrapped targets
  4. Dueling Networks — separate value / advantage streams
  5. Distributional   — C51 return distribution (n_atoms atoms)
  6. Noisy Networks   — parametric noise replaces epsilon-greedy
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from network import RainbowNetwork
from replay_buffer import PrioritizedReplayBuffer


class RainbowAgent:
    def __init__(
        self,
        state_shape,          # (H, W, C)
        n_actions: int,
        device: torch.device,
        # --- Distributional ---
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # --- Replay ---
        buffer_size: int = 100_000,
        batch_size: int = 32,
        # --- N-step ---
        n_step: int = 3,
        gamma: float = 0.99,
        # --- PER ---
        alpha: float = 0.5,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        # --- Optimisation ---
        lr: float = 1e-4,
        target_update: int = 1_000,
    ):
        self.n_actions = n_actions
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.target_update = target_update
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        in_channels = state_shape[2]
        self.online_net = RainbowNetwork(in_channels, n_actions, n_atoms, v_min, v_max).to(device)
        self.target_net = RainbowNetwork(in_channels, n_actions, n_atoms, v_min, v_max).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=lr, eps=1.5e-4
        )

        self.buffer = PrioritizedReplayBuffer(buffer_size, n_step, gamma, alpha)
        self.update_count = 0

    # ------------------------------------------------------------------
    # Beta annealing (importance-sampling exponent)
    # ------------------------------------------------------------------

    def _get_beta(self, frame: int) -> float:
        progress = min(frame / self.beta_frames, 1.0)
        return self.beta_start + progress * (1.0 - self.beta_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Greedy w.r.t. expected Q-values.
        Exploration comes from the noisy layers (training mode).
        """
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.online_net.get_q_values(s).argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def update(self, frame: int) -> float:
        beta = self._get_beta(frame)
        states, actions, rewards, next_states, dones, idxs, weights = \
            self.buffer.sample(self.batch_size, beta)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        weights     = torch.FloatTensor(weights).to(self.device)

        # --- Compute distributional Bellman target (C51 projection) ---
        with torch.no_grad():
            # Double DQN: online net picks actions, target net evaluates
            next_q        = self.online_net.get_q_values(next_states)
            next_actions  = next_q.argmax(dim=1)                          # (B,)

            next_dist     = self.target_net(next_states)                  # (B, A, Z)
            next_dist     = next_dist[range(self.batch_size), next_actions]  # (B, Z)

            # Project Bellman update onto the fixed support
            gamma_n = self.gamma ** self.n_step
            tz = rewards.unsqueeze(1) + gamma_n * (1 - dones.unsqueeze(1)) \
                 * self.online_net.support.unsqueeze(0)                   # (B, Z)
            tz = tz.clamp(self.v_min, self.v_max)

            b  = (tz - self.v_min) / self.delta_z                        # (B, Z)
            lo = b.floor().long()
            hi = b.ceil().long()

            # Resolve ties where lo == hi (atom sits exactly on a grid point)
            lo[(hi > 0) & (lo == hi)] -= 1
            hi[(lo < (self.n_atoms - 1)) & (lo == hi)] += 1

            proj = torch.zeros_like(next_dist)                            # (B, Z)
            proj.scatter_add_(1, lo.clamp(0, self.n_atoms - 1), next_dist * (hi.float() - b))
            proj.scatter_add_(1, hi.clamp(0, self.n_atoms - 1), next_dist * (b - lo.float()))

        # --- Current distribution for the taken actions ---
        curr_dist = self.online_net(states)[range(self.batch_size), actions]  # (B, Z)
        curr_dist = curr_dist.clamp(1e-8, 1.0)

        # Cross-entropy loss (≡ KL since proj is detached)
        loss_elem = -(proj * curr_dist.log()).sum(dim=1)                  # (B,)
        loss      = (weights * loss_elem).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update PER priorities using per-sample TD error magnitude
        new_prios = loss_elem.detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(idxs, new_prios)

        # Resample noise for the next forward pass
        self.online_net.sample_noise()
        self.target_net.sample_noise()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "online_net":    self.online_net.state_dict(),
                "target_net":    self.target_net.state_dict(),
                "optimizer":     self.optimizer.state_dict(),
                "update_count":  self.update_count,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt.get("update_count", 0)
