from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tractatus_structure_latents.models.vae import HierarchicalRNNVAE, SplitLatentHierarchicalRNNVAE, vae_loss
from tractatus_structure_latents.training.data import TractatusDataset, Vocabulary, collate_batch


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _per_sample_reconstruction(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="none",
    ).reshape(targets.shape)
    mask = targets != pad_idx
    return token_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _accuracy_by_language(
    correct_by_language: dict[str, int],
    count_by_language: dict[str, int],
) -> dict[str, float]:
    return {
        language: correct_by_language.get(language, 0) / max(count, 1)
        for language, count in sorted(count_by_language.items())
    }


def _reconstruction_by_language(
    reconstruction_by_language: dict[str, list[float]],
) -> dict[str, float]:
    return {
        language: _mean(values)
        for language, values in sorted(reconstruction_by_language.items())
    }


def _cross_language_metrics(
    z: torch.Tensor,
    metadata: list[dict[str, str | int]],
    rows: list[dict],
) -> dict[str, float]:
    by_id_language = {
        (str(item["id"]), str(item["language"])): i
        for i, item in enumerate(metadata)
    }
    languages = sorted({str(item["language"]) for item in metadata})
    same_id_distances: list[float] = []
    retrieval_top1: list[float] = []
    retrieval_rr: list[float] = []
    direction_top1: dict[str, list[float]] = {}
    direction_rr: dict[str, list[float]] = {}

    for source_i, source in enumerate(metadata):
        source_id = str(source["id"])
        source_language = str(source["language"])
        target_languages = [language for language in languages if language != source_language]
        for target_language in target_languages:
            target_i = by_id_language.get((source_id, target_language))
            if target_i is None:
                continue
            same_id_distances.append(torch.dist(z[source_i], z[target_i]).item())
            candidates = [
                i
                for i, item in enumerate(metadata)
                if str(item["language"]) == target_language
            ]
            if not candidates:
                continue
            distances = torch.norm(z[candidates] - z[source_i].unsqueeze(0), dim=-1)
            order = torch.argsort(distances).detach().cpu().tolist()
            ranked_ids = [str(metadata[candidates[position]]["id"]) for position in order]
            rank = ranked_ids.index(source_id) + 1 if source_id in ranked_ids else len(ranked_ids) + 1
            top1 = 1.0 if ranked_ids and ranked_ids[0] == source_id else 0.0
            rr = 1.0 / rank
            direction = f"{source_language}_to_{target_language}"
            retrieval_top1.append(top1)
            retrieval_rr.append(rr)
            direction_top1.setdefault(direction, []).append(top1)
            direction_rr.setdefault(direction, []).append(rr)

    cross_parent_child_distances: list[float] = []
    for row in rows:
        child_id = row["id"]
        parent_id = row["parent_id"]
        if parent_id is None:
            continue
        for child_language in languages:
            child_i = by_id_language.get((child_id, child_language))
            if child_i is None:
                continue
            for parent_language in languages:
                if parent_language == child_language:
                    continue
                parent_i = by_id_language.get((parent_id, parent_language))
                if parent_i is not None:
                    cross_parent_child_distances.append(torch.dist(z[child_i], z[parent_i]).item())

    metrics: dict[str, float] = {
        "mean_same_id_cross_language_distance": _mean(same_id_distances),
        "cross_language_top1_id_accuracy": _mean(retrieval_top1),
        "cross_language_mrr": _mean(retrieval_rr),
        "mean_cross_language_parent_child_distance": _mean(cross_parent_child_distances),
    }
    for direction, values in sorted(direction_top1.items()):
        metrics[f"cross_language_top1_id_accuracy_{direction}"] = _mean(values)
    for direction, values in sorted(direction_rr.items()):
        metrics[f"cross_language_mrr_{direction}"] = _mean(values)
    return metrics


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tractatus VAE text, structure, and latent geometry metrics.")
    parser.add_argument("--data", type=Path, default=Path("tractatus_structure_latents/data/tractatus.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--export-latents", type=Path)
    parser.add_argument("--latent-part", choices=["all", "text", "structure"], default="all")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--beta-text", type=float)
    parser.add_argument("--beta-structure", type=float)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = Vocabulary(ckpt["vocab"])
    language_to_id = ckpt.get("language_to_id")
    languages = ckpt.get("languages")
    dataset = TractatusDataset(args.data, vocab=vocab, languages=languages, language_to_id=language_to_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_batch, pad_idx=vocab.pad_idx))
    latent_dim = ckpt.get("latent_dim", ckpt["model"]["encoder.mu.weight"].shape[0])
    if ckpt.get("split_latent", False):
        model = SplitLatentHierarchicalRNNVAE(
            len(vocab.token_to_id),
            ckpt.get("proposition_count", dataset.proposition_count),
            dataset.max_depth,
            text_latent_dim=ckpt["text_latent_dim"],
            structure_latent_dim=ckpt["structure_latent_dim"],
            pad_idx=vocab.pad_idx,
            language_count=ckpt.get("language_count", dataset.language_count),
            language_embedding_dim=ckpt.get("language_embedding_dim", 8),
        )
    else:
        model = HierarchicalRNNVAE(
            len(vocab.token_to_id),
            ckpt.get("proposition_count", dataset.proposition_count),
            dataset.max_depth,
            latent_dim=latent_dim,
            pad_idx=vocab.pad_idx,
            language_count=ckpt.get("language_count", dataset.language_count),
            language_embedding_dim=ckpt.get("language_embedding_dim", 8),
        )
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_kl_text = 0.0
    total_kl_structure = 0.0
    parent_ok = depth_ok = next_ok = count = 0
    count_by_language: dict[str, int] = {}
    reconstruction_by_language: dict[str, list[float]] = {}
    parent_ok_by_language: dict[str, int] = {}
    depth_ok_by_language: dict[str, int] = {}
    next_ok_by_language: dict[str, int] = {}
    latents: list[torch.Tensor] = []
    metadata: list[dict[str, str | int]] = []
    for batch in loader:
        outputs = model(batch["input_ids"], batch["lengths"], batch["decoder_ids"], batch["language_ids"])
        beta_text = args.beta_text if args.beta_text is not None else ckpt.get("beta_text")
        beta_structure = args.beta_structure if args.beta_structure is not None else ckpt.get("beta_structure")
        losses = vae_loss(
            outputs,
            batch["targets"],
            pad_idx=vocab.pad_idx,
            beta=args.beta,
            beta_text=beta_text,
            beta_structure=beta_structure,
        )
        total_loss += float(losses["loss"])
        total_reconstruction += float(losses["reconstruction"])
        total_kl += float(losses["kl"])
        total_kl_text += float(losses.get("kl_text", torch.tensor(0.0)))
        total_kl_structure += float(losses.get("kl_structure", torch.tensor(0.0)))
        parent_correct = outputs["parent_logits"].argmax(-1) == batch["parent"]
        depth_correct = outputs["depth_logits"].argmax(-1) == batch["depth"]
        next_correct = outputs["next_logits"].argmax(-1) == batch["next"]
        parent_ok += int(parent_correct.sum())
        depth_ok += int(depth_correct.sum())
        next_ok += int(next_correct.sum())
        count += len(batch["ids"])
        per_sample_reconstruction = _per_sample_reconstruction(outputs["logits"], batch["targets"], vocab.pad_idx)
        for i, language in enumerate(batch["languages"]):
            count_by_language[language] = count_by_language.get(language, 0) + 1
            reconstruction_by_language.setdefault(language, []).append(float(per_sample_reconstruction[i]))
            parent_ok_by_language[language] = parent_ok_by_language.get(language, 0) + int(parent_correct[i])
            depth_ok_by_language[language] = depth_ok_by_language.get(language, 0) + int(depth_correct[i])
            next_ok_by_language[language] = next_ok_by_language.get(language, 0) + int(next_correct[i])
        if args.latent_part == "text":
            latents.append(outputs["text_mu"])
        elif args.latent_part == "structure":
            latents.append(outputs["structure_mu"])
        else:
            latents.append(outputs["mu"])
        metadata.extend(
            {"id": prop_id, "language": language, "index": int(index)}
            for prop_id, language, index in zip(batch["ids"], batch["languages"], batch["index"])
        )

    z = torch.cat(latents, dim=0)
    rows = dataset.rows
    id_language_to_i = {
        (str(item["id"]), str(item["language"])): i
        for i, item in enumerate(metadata)
    }
    parent_child_dist = []
    sibling_dist = []
    unrelated_dist = []
    for i, item in enumerate(metadata):
        row = rows[int(item["index"]) - 1]
        language = str(item["language"])
        if row["parent_id"] is not None and (row["parent_id"], language) in id_language_to_i:
            parent_child_dist.append(torch.dist(z[i], z[id_language_to_i[(row["parent_id"], language)]]).item())
        for sibling in row["siblings"][:1]:
            if (sibling, language) in id_language_to_i:
                sibling_dist.append(torch.dist(z[i], z[id_language_to_i[(sibling, language)]]).item())
        same_language = [j for j, other in enumerate(metadata) if str(other["language"]) == language and j != i]
        if same_language:
            j = same_language[(i * 37 + 11) % len(same_language)]
            unrelated_dist.append(torch.dist(z[i], z[j]).item())

    mean_loss = total_loss / max(len(loader), 1)
    mean_reconstruction = total_reconstruction / max(len(loader), 1)
    mean_kl = total_kl / max(len(loader), 1)
    mean_kl_text = total_kl_text / max(len(loader), 1)
    mean_kl_structure = total_kl_structure / max(len(loader), 1)
    metrics = {
        "loss": mean_loss,
        "reconstruction_loss": mean_reconstruction,
        "kl": mean_kl,
        "kl_text": mean_kl_text,
        "kl_structure": mean_kl_structure,
        "perplexity": float(torch.exp(torch.tensor(mean_reconstruction)).clamp(max=1e9)),
        "parent_accuracy": parent_ok / max(count, 1),
        "depth_accuracy": depth_ok / max(count, 1),
        "next_accuracy": next_ok / max(count, 1),
        "reconstruction_loss_by_language": _reconstruction_by_language(reconstruction_by_language),
        "parent_accuracy_by_language": _accuracy_by_language(parent_ok_by_language, count_by_language),
        "depth_accuracy_by_language": _accuracy_by_language(depth_ok_by_language, count_by_language),
        "next_accuracy_by_language": _accuracy_by_language(next_ok_by_language, count_by_language),
        "mean_parent_child_distance": sum(parent_child_dist) / max(len(parent_child_dist), 1),
        "mean_sibling_distance": sum(sibling_dist) / max(len(sibling_dist), 1),
        "mean_unrelated_distance": sum(unrelated_dist) / max(len(unrelated_dist), 1),
    }
    if dataset.language_count > 1:
        metrics.update(_cross_language_metrics(z, metadata, rows))
    print(json.dumps(metrics, indent=2))
    if args.export_latents:
        args.export_latents.parent.mkdir(parents=True, exist_ok=True)
        torch.save(z, args.export_latents)
        args.export_latents.with_suffix(".ids.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
