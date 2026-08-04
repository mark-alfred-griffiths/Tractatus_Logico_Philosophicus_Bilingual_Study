from __future__ import annotations

import argparse
import csv
from functools import partial
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tractatus_structure_latents.models.vae import HierarchicalRNNVAE, SplitLatentHierarchicalRNNVAE, vae_loss
from tractatus_structure_latents.training.data import TractatusDataset, Vocabulary, collate_batch


def resolve_device(requested: str) -> torch.device:
    if requested == "gpu":
        requested = "cuda"
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


class PairedLanguageBatchSampler:
    def __init__(
        self,
        dataset: TractatusDataset,
        batch_size: int,
        generator: torch.Generator,
        paired_languages: tuple[str, str] = ("de", "en"),
    ):
        if batch_size % 2 != 0:
            raise ValueError("--paired-language-batches requires an even --batch-size")
        self.ids_per_batch = batch_size // 2
        self.generator = generator
        left_language, right_language = paired_languages
        by_id: dict[str, dict[str, int]] = {}
        for sample_i, sample in enumerate(dataset.samples):
            by_id.setdefault(str(sample["id"]), {})[str(sample["language"])] = sample_i
        missing = [
            prop_id
            for prop_id, language_to_sample in by_id.items()
            if left_language not in language_to_sample or right_language not in language_to_sample
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Missing paired {left_language}/{right_language} samples for proposition IDs: {preview}")
        self.pairs = sorted(
            (
                dataset.id_to_index[prop_id],
                language_to_sample[left_language],
                language_to_sample[right_language],
            )
            for prop_id, language_to_sample in by_id.items()
        )
        self.corpus_pair_count = len(self.pairs)

    def __iter__(self):
        order = torch.randperm(len(self.pairs), generator=self.generator).tolist()
        for start in range(0, len(order), self.ids_per_batch):
            batch: list[int] = []
            for pair_i in order[start : start + self.ids_per_batch]:
                _prop_index, left_sample_i, right_sample_i = self.pairs[pair_i]
                batch.extend([left_sample_i, right_sample_i])
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.pairs) / self.ids_per_batch)


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
        "formal_target_shuffle_seed": args.formal_target_shuffle_seed,
        "formal_target_shuffle_fields": args.formal_target_shuffle_fields,
        "sample_ids_file": str(args.sample_ids_file) if args.sample_ids_file else None,
        "sample_ids_count": len(dataset.sample_ids) if dataset.sample_ids is not None else None,
        "paired_language_batches": args.paired_language_batches,
        "paired_languages": args.paired_languages,
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
    loss, _pair_count = language_alignment_loss_with_count(z, indices, language_ids)
    return loss


def language_alignment_loss_with_count(
    z: torch.Tensor,
    indices: torch.Tensor,
    language_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
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
        return z.new_tensor(0.0), 0

    left = torch.tensor([i for i, _ in pairs], device=z.device)
    right = torch.tensor([j for _, j in pairs], device=z.device)
    pair_mse = (z[left] - z[right]).pow(2).mean(dim=-1)
    return pair_mse.mean(), len(pairs)


def mean_same_id_distance(z: torch.Tensor, indices: torch.Tensor, language_ids: torch.Tensor) -> float:
    index_to_items: dict[int, list[tuple[int, int]]] = {}
    for row_i, (index, language_id) in enumerate(zip(indices.detach().cpu().tolist(), language_ids.detach().cpu().tolist())):
        index_to_items.setdefault(int(index), []).append((row_i, int(language_id)))
    distances: list[float] = []
    for items in index_to_items.values():
        for left_i, (left_row_i, left_language_id) in enumerate(items):
            for right_row_i, right_language_id in items[left_i + 1 :]:
                if left_language_id != right_language_id:
                    distances.append(float(torch.dist(z[left_row_i], z[right_row_i]).detach().cpu()))
    return sum(distances) / max(len(distances), 1)


def append_epoch_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--formal-target-shuffle-seed", type=int, help="Deterministically shuffle selected formal target tuples across proposition IDs before training.")
    parser.add_argument(
        "--formal-target-shuffle-fields",
        default="parent,depth,next,child_count",
        help="Comma-separated formal target fields to jointly shuffle when --formal-target-shuffle-seed is set.",
    )
    parser.add_argument("--contrastive-margin", type=float, default=1.0)
    parser.add_argument("--languages", help="Comma-separated dataset languages to train on. Defaults to all languages in the dataset.")
    parser.add_argument("--sample-ids-file", type=Path, help="Optional JSON file containing proposition IDs to include as train/eval samples.")
    parser.add_argument("--language-embedding-dim", type=int, default=8)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save an intermediate checkpoint every N epochs.")
    parser.add_argument("--paired-language-batches", action="store_true", help="Batch German/English rows with the same proposition ID together.")
    parser.add_argument("--paired-languages", default="de,en", help="Comma-separated language pair used by --paired-language-batches.")
    parser.add_argument("--epoch-metrics-out", type=Path, help="Optional CSV path for per-epoch training diagnostics.")
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
    args.formal_target_shuffle_fields = [
        field.strip()
        for field in args.formal_target_shuffle_fields.split(",")
        if field.strip()
    ]
    sample_ids = None
    if args.sample_ids_file:
        sample_ids = json.loads(args.sample_ids_file.read_text(encoding="utf-8"))
    dataset = TractatusDataset(
        args.data,
        languages=languages,
        formal_target_shuffle_seed=args.formal_target_shuffle_seed,
        formal_target_shuffle_fields=args.formal_target_shuffle_fields,
        sample_ids=sample_ids,
    )
    paired_languages = tuple(language.strip() for language in args.paired_languages.split(",") if language.strip())
    if len(paired_languages) != 2:
        raise ValueError("--paired-languages must contain exactly two comma-separated language codes")
    paired_sampler = None
    if args.paired_language_batches:
        paired_sampler = PairedLanguageBatchSampler(dataset, args.batch_size, data_generator, paired_languages=paired_languages)
        loader = DataLoader(dataset, batch_sampler=paired_sampler, collate_fn=partial(collate_batch, pad_idx=dataset.vocab.pad_idx))
    else:
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
    if args.epoch_metrics_out and args.epoch_metrics_out.exists():
        args.epoch_metrics_out.unlink()

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "parent": 0.0,
            "depth": 0.0,
            "next": 0.0,
            "child": 0.0,
            "kl": 0.0,
            "kl_text": 0.0,
            "kl_structure": 0.0,
        }
        epoch_pair_count = 0
        epoch_alignment_mse_sum = 0.0
        epoch_weighted_alignment_sum = 0.0
        epoch_grad_norms: list[float] = []
        epoch_structure_mu: list[torch.Tensor] = []
        epoch_structure_var: list[torch.Tensor] = []
        epoch_indices: list[torch.Tensor] = []
        epoch_language_ids: list[torch.Tensor] = []
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
            alignment, pair_count = language_alignment_loss_with_count(outputs["structure_mu"], batch["index"], batch["language_ids"])
            if args.lambda_language_alignment > 0 and alignment.requires_grad:
                losses["loss"] = losses["loss"] + args.lambda_language_alignment * alignment
            epoch_pair_count += pair_count
            epoch_alignment_mse_sum += float(alignment.detach()) * pair_count
            epoch_weighted_alignment_sum += float((args.lambda_language_alignment * alignment).detach()) * pair_count
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_grad_norms.append(float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm))
            for name in totals:
                if name in losses:
                    totals[name] += float(losses[name].detach())
            epoch_structure_mu.append(outputs["structure_mu"].detach().cpu())
            epoch_structure_var.append(outputs["structure_logvar"].detach().cpu().exp())
            epoch_indices.append(batch["index"].detach().cpu())
            epoch_language_ids.append(batch["language_ids"].detach().cpu())
        batch_count = max(len(loader), 1)
        total_possible_pairs = paired_sampler.corpus_pair_count if paired_sampler is not None else dataset.proposition_count
        pair_coverage = epoch_pair_count / max(total_possible_pairs, 1)
        if paired_sampler is not None and epoch_pair_count != paired_sampler.corpus_pair_count:
            raise AssertionError(
                f"Paired batch coverage failed: saw {epoch_pair_count} pairs, expected {paired_sampler.corpus_pair_count}"
            )
        structure_mu = torch.cat(epoch_structure_mu, dim=0)
        structure_var = torch.cat(epoch_structure_var, dim=0)
        epoch_index = torch.cat(epoch_indices, dim=0)
        epoch_language_id = torch.cat(epoch_language_ids, dim=0)
        epoch_row = {
            "epoch": epoch,
            "same_id_pairs_processed": epoch_pair_count,
            "pair_coverage": pair_coverage,
            "raw_alignment_mse": epoch_alignment_mse_sum / max(epoch_pair_count, 1),
            "weighted_alignment_contribution": epoch_weighted_alignment_sum / max(epoch_pair_count, 1),
            "loss": totals["loss"] / batch_count,
            "reconstruction_loss": totals["reconstruction"] / batch_count,
            "parent_loss": totals["parent"] / batch_count,
            "depth_loss": totals["depth"] / batch_count,
            "successor_loss": totals["next"] / batch_count,
            "child_count_loss": totals["child"] / batch_count,
            "kl": totals["kl"] / batch_count,
            "kl_text": totals["kl_text"] / batch_count,
            "kl_structure": totals["kl_structure"] / batch_count,
            "same_id_distance": mean_same_id_distance(structure_mu, epoch_index, epoch_language_id),
            "structure_mean_norm": float(structure_mu.norm(dim=-1).mean()),
            "posterior_variance": float(structure_var.mean()),
            "gradient_norm": sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if args.epoch_metrics_out:
            append_epoch_metrics(args.epoch_metrics_out, epoch_row)
        print(f"epoch={epoch} loss={epoch_row['loss']:.4f}", flush=True)
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
