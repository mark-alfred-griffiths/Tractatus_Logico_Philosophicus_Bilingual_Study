from __future__ import annotations

import torch
from torch import nn


class GRUTransition(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.gru(z_sequence)
        return self.mu(outputs), self.logvar(outputs).clamp(min=-10.0, max=10.0)

    def next_distribution(self, z_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.forward(z_prev.unsqueeze(1))
        return mu[:, -1], logvar[:, -1]
