from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .decoder import TextDecoder
from .encoder import TextEncoder


class HierarchicalRNNVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        proposition_count: int,
        max_depth: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        pad_idx: int = 0,
        language_count: int = 1,
        language_embedding_dim: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pad_idx = pad_idx
        self.encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim, latent_dim, pad_idx)
        self.decoder = TextDecoder(
            vocab_size,
            embedding_dim,
            hidden_dim,
            latent_dim,
            pad_idx,
            language_count=language_count,
            language_embedding_dim=language_embedding_dim,
        )
        self.parent_head = nn.Linear(latent_dim, proposition_count + 1)
        self.depth_head = nn.Linear(latent_dim, max_depth + 1)
        self.next_head = nn.Linear(latent_dim, proposition_count + 1)
        self.child_count_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input_ids, lengths)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        decoder_ids: torch.Tensor,
        language_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(input_ids, lengths)
        z = self.reparameterize(mu, logvar)
        return {
            "logits": self.decoder(decoder_ids, z, language_ids),
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "text_mu": mu,
            "structure_mu": mu,
            "parent_logits": self.parent_head(z),
            "depth_logits": self.depth_head(z),
            "next_logits": self.next_head(z),
            "child_count": self.child_count_head(z).squeeze(-1),
        }


class SplitLatentHierarchicalRNNVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        proposition_count: int,
        max_depth: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        text_latent_dim: int = 24,
        structure_latent_dim: int = 8,
        pad_idx: int = 0,
        language_count: int = 1,
        language_embedding_dim: int = 8,
    ):
        super().__init__()
        self.text_latent_dim = text_latent_dim
        self.structure_latent_dim = structure_latent_dim
        self.latent_dim = text_latent_dim + structure_latent_dim
        self.pad_idx = pad_idx
        self.encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim, self.latent_dim, pad_idx)
        self.decoder = TextDecoder(
            vocab_size,
            embedding_dim,
            hidden_dim,
            text_latent_dim,
            pad_idx,
            language_count=language_count,
            language_embedding_dim=language_embedding_dim,
        )
        self.parent_head = nn.Linear(structure_latent_dim, proposition_count + 1)
        self.depth_head = nn.Linear(structure_latent_dim, max_depth + 1)
        self.next_head = nn.Linear(structure_latent_dim, proposition_count + 1)
        self.child_count_head = nn.Sequential(nn.Linear(structure_latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input_ids, lengths)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _split(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value[:, : self.text_latent_dim], value[:, self.text_latent_dim :]

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        decoder_ids: torch.Tensor,
        language_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(input_ids, lengths)
        z = self.reparameterize(mu, logvar)
        z_text, z_structure = self._split(z)
        mu_text, mu_structure = self._split(mu)
        logvar_text, logvar_structure = self._split(logvar)
        return {
            "logits": self.decoder(decoder_ids, z_text, language_ids),
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "text_mu": mu_text,
            "structure_mu": mu_structure,
            "text_logvar": logvar_text,
            "structure_logvar": logvar_structure,
            "parent_logits": self.parent_head(z_structure),
            "depth_logits": self.depth_head(z_structure),
            "next_logits": self.next_head(z_structure),
            "child_count": self.child_count_head(z_structure).squeeze(-1),
        }


def vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    parent_targets: torch.Tensor | None = None,
    depth_targets: torch.Tensor | None = None,
    next_targets: torch.Tensor | None = None,
    child_count_targets: torch.Tensor | None = None,
    pad_idx: int = 0,
    beta: float = 0.1,
    beta_text: float | None = None,
    beta_structure: float | None = None,
    lambdas: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> dict[str, torch.Tensor]:
    recon = F.cross_entropy(outputs["logits"].reshape(-1, outputs["logits"].size(-1)), targets.reshape(-1), ignore_index=pad_idx)
    if beta_text is not None or beta_structure is not None:
        if "text_logvar" not in outputs or "structure_logvar" not in outputs:
            raise ValueError("beta_text/beta_structure require split-latent model outputs")
        kl_text = -0.5 * torch.mean(
            torch.sum(1 + outputs["text_logvar"] - outputs["text_mu"].pow(2) - outputs["text_logvar"].exp(), dim=-1)
        )
        kl_structure = -0.5 * torch.mean(
            torch.sum(
                1 + outputs["structure_logvar"] - outputs["structure_mu"].pow(2) - outputs["structure_logvar"].exp(),
                dim=-1,
            )
        )
        kl = kl_text + kl_structure
        total = recon + (beta if beta_text is None else beta_text) * kl_text + (beta if beta_structure is None else beta_structure) * kl_structure
        losses = {
            "loss": total,
            "reconstruction": recon.detach(),
            "kl": kl.detach(),
            "kl_text": kl_text.detach(),
            "kl_structure": kl_structure.detach(),
        }
    else:
        kl = -0.5 * torch.mean(torch.sum(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp(), dim=-1))
        total = recon + beta * kl
        losses = {"loss": total, "reconstruction": recon.detach(), "kl": kl.detach()}

    if parent_targets is not None and lambdas[0] > 0:
        parent_loss = F.cross_entropy(outputs["parent_logits"], parent_targets)
        total = total + lambdas[0] * parent_loss
        losses["parent"] = parent_loss.detach()
    if depth_targets is not None and lambdas[1] > 0:
        depth_loss = F.cross_entropy(outputs["depth_logits"], depth_targets)
        total = total + lambdas[1] * depth_loss
        losses["depth"] = depth_loss.detach()
    if next_targets is not None and lambdas[2] > 0:
        next_loss = F.cross_entropy(outputs["next_logits"], next_targets)
        total = total + lambdas[2] * next_loss
        losses["next"] = next_loss.detach()
    if child_count_targets is not None and lambdas[3] > 0:
        child_loss = F.mse_loss(outputs["child_count"], child_count_targets.float())
        total = total + lambdas[3] * child_loss
        losses["child"] = child_loss.detach()

    losses["loss"] = total
    return losses
