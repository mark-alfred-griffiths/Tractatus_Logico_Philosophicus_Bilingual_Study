from __future__ import annotations

import torch
from torch import nn


class TextDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        pad_idx: int = 0,
        language_count: int = 1,
        language_embedding_dim: int = 8,
    ):
        super().__init__()
        self.language_count = language_count
        self.language_embedding_dim = language_embedding_dim if language_count > 1 else 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if self.language_embedding_dim:
            self.language_embedding = nn.Embedding(language_count, self.language_embedding_dim)
        self.condition_dim = latent_dim + self.language_embedding_dim
        self.z_to_hidden = nn.Linear(self.condition_dim, hidden_dim)
        self.gru = nn.GRU(embedding_dim + self.condition_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _condition(self, z: torch.Tensor, language_ids: torch.Tensor | None) -> torch.Tensor:
        if not self.language_embedding_dim:
            return z
        if language_ids is None:
            raise ValueError("language_ids are required when language_count > 1")
        language = self.language_embedding(language_ids)
        return torch.cat([z, language], dim=-1)

    def forward(self, decoder_ids: torch.Tensor, z: torch.Tensor, language_ids: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(decoder_ids)
        condition = self._condition(z, language_ids)
        condition_steps = condition.unsqueeze(1).expand(-1, decoder_ids.size(1), -1)
        hidden0 = torch.tanh(self.z_to_hidden(condition)).unsqueeze(0)
        outputs, _ = self.gru(torch.cat([embedded, condition_steps], dim=-1), hidden0)
        return self.output(outputs)
