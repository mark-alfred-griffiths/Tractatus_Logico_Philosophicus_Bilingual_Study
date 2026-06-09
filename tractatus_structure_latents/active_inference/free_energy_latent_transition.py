from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tractatus_structure_latents.models.dynamics import GRUTransition


def free_energy(belief: torch.Tensor, predicted_next: torch.Tensor, observation: torch.Tensor, precision: float = 1.0) -> torch.Tensor:
    prediction_error = F.mse_loss(predicted_next, observation, reduction="sum")
    complexity = 0.5 * precision * torch.sum(belief.pow(2))
    return prediction_error + complexity


def infer_next_belief(transition: GRUTransition, previous_z: torch.Tensor, observed_next_z: torch.Tensor, steps: int = 64, lr: float = 0.05) -> tuple[torch.Tensor, list[float]]:
    belief = previous_z.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([belief], lr=lr)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted_mu, _ = transition.next_distribution(belief.unsqueeze(0))
        energy = free_energy(belief, predicted_mu.squeeze(0), observed_next_z)
        energy.backward()
        optimizer.step()
        history.append(float(energy.detach()))
    return belief.detach(), history


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory free-energy-style belief update over exported "
            "Tractatus latent states; not part of the canonical seed sweeps."
        )
    )
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    z = torch.load(args.latents, map_location="cpu")
    if not 0 <= args.index < len(z) - 1:
        raise ValueError("--index must select a proposition with a next proposition")
    transition = GRUTransition(latent_dim=z.size(-1))
    transition.load_state_dict(torch.load(args.dynamics, map_location="cpu"))
    transition.eval()
    belief, history = infer_next_belief(transition, z[args.index], z[args.index + 1])
    print({"initial_free_energy": history[0], "final_free_energy": history[-1], "belief_norm": float(belief.norm())})


if __name__ == "__main__":
    main()
