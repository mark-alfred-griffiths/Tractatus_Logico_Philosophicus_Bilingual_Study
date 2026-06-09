from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tractatus_structure_latents.models.dynamics import GRUTransition


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU transition prior from saved latent sequence tensor.")
    parser.add_argument("--latents", type=Path, required=True, help="Tensor shaped [T, latent_dim], e.g. exported z means.")
    parser.add_argument("--out", type=Path, default=Path("runs/dynamics.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    z = torch.load(args.latents, map_location="cpu")
    if z.ndim != 2 or z.size(0) < 2:
        raise ValueError("--latents must be a [T, latent_dim] tensor with T >= 2")
    model = GRUTransition(latent_dim=z.size(-1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x = z[:-1].unsqueeze(0)
    y = z[1:].unsqueeze(0)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        mu, _ = model(x)
        loss = F.mse_loss(mu, y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch} transition_mse={float(loss):.6f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
