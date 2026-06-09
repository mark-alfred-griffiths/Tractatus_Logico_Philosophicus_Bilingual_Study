from __future__ import annotations

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, latent_dim: int = 32, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        state = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.mu(state), self.logvar(state)
