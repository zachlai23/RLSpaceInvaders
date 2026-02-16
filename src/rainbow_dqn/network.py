"""
Rainbow DQN Network
Combines:
  - Noisy Networks (Fortunato et al., 2017) — parametric noise for exploration
  - Dueling Architecture (Wang et al., 2016) — separate value and advantage streams
  - Distributional RL / C51 (Bellemare et al., 2017) — models return distribution
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorised Noisy Linear layer for exploration."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        """Resample factorised Gaussian noise."""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class RainbowNetwork(nn.Module):
    """
    Rainbow network for MinAtar (10×10×C input).

    Architecture:
      Conv (shared feature extractor)
      -> Dueling streams (value / advantage), each with two NoisyLinear layers
      -> C51 distributional output: softmax over N_ATOMS per action
    """

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        n_atoms: int,
        v_min: float,
        v_max: float,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        # Single conv layer — MinAtar spatial size is only 10×10
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        conv_out = 16 * 10 * 10  # 1600

        # Value stream
        self.value_hidden = NoisyLinear(conv_out, 128)
        self.value_out = NoisyLinear(128, n_atoms)

        # Advantage stream
        self.advantage_hidden = NoisyLinear(conv_out, 128)
        self.advantage_out = NoisyLinear(128, n_actions * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, H, W, C) float32 in [0, 1]
        Returns:
            prob: (batch, n_actions, n_atoms) — probability distribution over atoms
        """
        # MinAtar gives (H, W, C); PyTorch conv expects (C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x).flatten(1)

        v = F.relu(self.value_hidden(x))
        v = self.value_out(v).view(-1, 1, self.n_atoms)

        a = F.relu(self.advantage_hidden(x))
        a = self.advantage_out(a).view(-1, self.n_actions, self.n_atoms)

        # Dueling combination, then softmax over atoms
        q = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(q, dim=2)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Expected Q-value for each action: E[Z] = sum_i p_i * z_i."""
        return (self.forward(x) * self.support).sum(dim=2)

    def sample_noise(self):
        """Resample noise in all NoisyLinear layers."""
        self.value_hidden.sample_noise()
        self.value_out.sample_noise()
        self.advantage_hidden.sample_noise()
        self.advantage_out.sample_noise()
