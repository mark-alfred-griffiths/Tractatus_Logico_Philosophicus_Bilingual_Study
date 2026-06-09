from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tractatus_structure_latents.models.vae import HierarchicalRNNVAE, SplitLatentHierarchicalRNNVAE, vae_loss
from tractatus_structure_latents.training.data import TractatusDataset, Vocabulary, collate_batch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def checkpoint_payload(
    model: torch.nn.Module,
    dataset: TractatusDataset,
    args: argparse.Namespace,
    epoch: int | None = None,
) -> dict:
    payload = {
        "model": model.state_dict(),
        "vocab": dataset.vocab.token_to_id,
        "max_depth": dataset.max_depth,
        "count": len(dataset),
        "sample_count": len(dataset),
        "proposition_count": dataset.proposition_count,
        "latent_dim": args.latent_dim,
        "split_latent": args.split_latent,
        "text_latent_dim": args.text_latent_dim if args.split_latent else args.latent_dim,
        "structure_latent_dim": args.structure_latent_dim if args.split_latent else args.latent_dim,
        "beta": args.beta,
        "beta_text": args.beta_text,
        "beta_structure": args.beta_structure,
        "languages": dataset.languages,
        "language_to_id": dataset.language_to_id,
        "language_count": dataset.language_count,
        "language_embedding_dim": args.language_embedding_dim,
        "lambda_language_alignment": args.lambda_language_alignment,
        "seed": args.seed,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    return payload


def latent_contrastive_loss(
    z: torch.Tensor,
    indices: torch.Tensor,
    parent_targets: torch.Tensor,
    parent_weight: float = 0.0,
    sibling_weight: float = 0.0,
    unrelated_weight: float = 0.0,
    margin: float = 1.0,
) -> torch.Tensor:
    loss = z.new_tensor(0.0)
    index_to_batch = {int(index): i for i, index in enumerate(indices.detach().cpu().tolist())}

    if parent_weight > 0:
        pairs = [(i, index_to_batch[int(parent)]) for i, parent in enumerate(parent_targets.detach().cpu().tolist()) if int(parent) in index_to_batch]
        if pairs:
            left = torch.tensor([i for i, _ in pairs], device=z.device)
            right = torch.tensor([j for _, j in pairs], device=z.device)
            loss = loss + parent_weight * F.mse_loss(z[left], z[right])

    if sibling_weight > 0:
        sibling_pairs: list[tuple[int, int]] = []
        parents = parent_targets.detach().cpu().tolist()
        for i in range(len(parents)):
            if int(parents[i]) == 0:
                continue
            for j in range(i + 1, len(parents)):
                if int(parents[i]) == int(parents[j]):
                    sibling_pairs.append((i, j))
        if sibling_pairs:
            left = torch.tensor([i for i, _ in sibling_pairs], device=z.device)
            right = torch.tensor([j for _, j in sibling_pairs], device=z.device)
            loss = loss + sibling_weight * F.mse_loss(z[left], z[right])

    if unrelated_weight > 0 and z.size(0) > 1:
        rolled = torch.roll(z, shifts=max(1, z.size(0) // 2), dims=0)
        distances = torch.norm(z - rolled, dim=-1)
        loss = loss + unrelated_weight * torch.mean(F.relu(margin - distances).pow(2))

    return loss


def language_alignment_loss(
    z: torch.Tensor,
    indices: torch.Tensor,
    language_ids: torch.Tensor,
) -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    index_to_items: dict[int, list[tuple[int, int]]] = {}
    for batch_i, (index, language_id) in enumerate(zip(indices.detach().cpu().tolist(), language_ids.detach().cpu().tolist())):
        index_to_items.setdefault(int(index), []).append((batch_i, int(language_id)))

    for items in index_to_items.values():
        for left_i, (left_batch_i, left_language_id) in enumerate(items):
            for right_batch_i, right_language_id in items[left_i + 1 :]:
                if left_language_id != right_language_id:
                    pairs.append((left_batch_i, right_batch_i))

    if not pairs:
        return z.new_tensor(0.0)

    left = torch.tensor([i for i, _ in pairs], device=z.device)
    right = torch.tensor([j for _, j in pairs], device=z.device)
    return F.mse_loss(z[left], z[right])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical RNN-VAE on Tractatus propositions.")
    parser.add_argument("--data", type=Path, default=Path("tractatus_structure_latents/data/tractatus.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/vae_baseline.pt"))
    parser.add_argument("--init-checkpoint", type=Path, help="Optional checkpoint to warm-start model weights from.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--split-latent", action="store_true")
    parser.add_argument("--text-latent-dim", type=int, default=24)
    parser.add_argument("--structure-latent-dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--beta-text", type=float)
    parser.add_argument("--beta-structure", type=float)
    parser.add_argument("--lambda-parent", type=float, default=0.0)
    parser.add_argument("--lambda-depth", type=float, default=0.0)
    parser.add_argument("--lambda-next", type=float, default=0.0)
    parser.add_argument("--lambda-child", type=float, default=0.0)
    parser.add_argument("--lambda-parent-contrastive", type=float, default=0.0)
    parser.add_argument("--lambda-sibling-contrastive", type=float, default=0.0)
    parser.add_argument("--lambda-unrelated-contrastive", type=float, default=0.0)
    parser.add_argument("--lambda-language-alignment", type=float, default=0.0)
    parser.add_argument("--contrastive-margin", type=float, default=1.0)
    parser.add_argument("--languages", help="Comma-separated dataset languages to train on. Defaults to all languages in the dataset.")
    parser.add_argument("--language-embedding-dim", type=int, default=8)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save an intermediate checkpoint every N epochs.")
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model initialisation and DataLoader shuffling.")
    args = parser.parse_args()

    data_generator = set_seed(args.seed)
    print(f"using seed={args.seed}", flush=True)
    languages = [language.strip() for language in args.languages.split(",") if language.strip()] if args.languages else None
    device = resolve_device(args.device)
    print(f"using device={device}", flush=True)
    dataset = TractatusDataset(args.data, languages=languages)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_batch, pad_idx=dataset.vocab.pad_idx),
        generator=data_generator,
    )
    if args.split_latent:
        args.latent_dim = args.text_latent_dim + args.structure_latent_dim
        model = SplitLatentHierarchicalRNNVAE(
            vocab_size=len(dataset.vocab.token_to_id),
            proposition_count=dataset.proposition_count,
            max_depth=dataset.max_depth,
            text_latent_dim=args.text_latent_dim,
            structure_latent_dim=args.structure_latent_dim,
            pad_idx=dataset.vocab.pad_idx,
            language_count=dataset.language_count,
            language_embedding_dim=args.language_embedding_dim,
        )
    else:
        model = HierarchicalRNNVAE(
            vocab_size=len(dataset.vocab.token_to_id),
            proposition_count=dataset.proposition_count,
            max_depth=dataset.max_depth,
            latent_dim=args.latent_dim,
            pad_idx=dataset.vocab.pad_idx,
            language_count=dataset.language_count,
            language_embedding_dim=args.language_embedding_dim,
        )
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        checkpoint_vocab = ckpt.get("vocab")
        if checkpoint_vocab != dataset.vocab.token_to_id:
            raise ValueError("--init-checkpoint vocabulary does not match this dataset")
        if ckpt.get("latent_dim", args.latent_dim) != args.latent_dim:
            raise ValueError("--init-checkpoint latent_dim does not match --latent-dim")
        if bool(ckpt.get("split_latent", False)) != args.split_latent:
            raise ValueError("--init-checkpoint split_latent does not match --split-latent")
        if ckpt.get("language_to_id", dataset.language_to_id) != dataset.language_to_id:
            raise ValueError("--init-checkpoint language_to_id does not match this dataset/language selection")
        if ckpt.get("language_count", dataset.language_count) != dataset.language_count:
            raise ValueError("--init-checkpoint language_count does not match this dataset/language selection")
        model.load_state_dict(ckpt["model"])
        print(f"loaded initial weights from {args.init_checkpoint}", flush=True)
    model.to(device)
    if args.freeze_decoder:
        for parameter in model.decoder.parameters():
            parameter.requires_grad = False
        print("froze decoder parameters", flush=True)
    if args.freeze_embeddings:
        for parameter in model.encoder.embedding.parameters():
            parameter.requires_grad = False
        for parameter in model.decoder.embedding.parameters():
            parameter.requires_grad = False
        print("froze encoder and decoder token embeddings", flush=True)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)
    lambdas = (args.lambda_parent, args.lambda_depth, args.lambda_next, args.lambda_child)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_ids"], batch["lengths"], batch["decoder_ids"], batch["language_ids"])
            losses = vae_loss(
                outputs,
                batch["targets"],
                batch["parent"],
                batch["depth"],
                batch["next"],
                batch["child_count"],
                pad_idx=dataset.vocab.pad_idx,
                beta=args.beta,
                beta_text=args.beta_text,
                beta_structure=args.beta_structure,
                lambdas=lambdas,
            )
            contrastive = latent_contrastive_loss(
                outputs["structure_mu"],
                batch["index"],
                batch["parent"],
                parent_weight=args.lambda_parent_contrastive,
                sibling_weight=args.lambda_sibling_contrastive,
                unrelated_weight=args.lambda_unrelated_contrastive,
                margin=args.contrastive_margin,
            )
            if contrastive.requires_grad:
                losses["loss"] = losses["loss"] + contrastive
            alignment = language_alignment_loss(outputs["structure_mu"], batch["index"], batch["language_ids"])
            if args.lambda_language_alignment > 0 and alignment.requires_grad:
                losses["loss"] = losses["loss"] + args.lambda_language_alignment * alignment
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(losses["loss"].detach())
        print(f"epoch={epoch} loss={total / max(len(loader), 1):.4f}", flush=True)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = args.out.with_suffix(f".epoch{epoch}.pt")
            torch.save(checkpoint_payload(model, dataset, args, epoch=epoch), checkpoint_path)
            torch.save(checkpoint_payload(model, dataset, args, epoch=epoch), args.out.with_suffix(".latest.pt"))
            print(f"saved checkpoint {checkpoint_path}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, dataset, args, epoch=args.epochs), args.out)
    Vocabulary(dataset.vocab.token_to_id).to_json(args.out.with_suffix(".vocab.json"))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
