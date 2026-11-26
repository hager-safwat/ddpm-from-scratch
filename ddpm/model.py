"""Noise-prediction network eps_theta(x_t, t)."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of integer timesteps (as in Transformer positional
    encodings), so the MLP knows which noise level it is looking at."""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: integer timesteps, shape (B,) -> (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EpsilonModel(nn.Module):
    """Predicts the noise that was added to a point, given the point and t."""

    def __init__(self, data_dim=2, t_dim=64, hidden=256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))
