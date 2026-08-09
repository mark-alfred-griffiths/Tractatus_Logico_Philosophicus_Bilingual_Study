#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tractatus_structure_latents.models.vae import HierarchicalRNNVAE, SplitLatentHierarchicalRNNVAE
from tractatus_structure_latents.training.data import TractatusDataset, Vocabulary, collate_batch, row_texts, tokenize

DEFAULT_OUT = ROOT / "results" / "dsh_validation" / "phase1_ablations"
DATA_PATH = ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json"
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_RESAMPLES = 10_000
PROTECTED_PATTERNS = [
    "paper/main.tex",
    "paper/references.bib",
    "paper/*.pdf",
    "paper/**/*.pdf",
]


@dataclass(frozen=True)
class Condition:
    name: str
    lambda_parent: float
    lambda_depth: float
    lambda_next: float
    lambda_child: float
    shuffle_seed: int | None = None
    shuffle_fields: tuple[str, ...] = ()


CONDITIONS: dict[str, Condition] = {
    "full_model": Condition("full_model", 0.2, 0.1, 0.2, 0.02),
    "reconstruction_only": Condition("reconstruction_only", 0.0, 0.0, 0.0, 0.0),
    "no_successor": Condition("no_successor", 0.2, 0.1, 0.0, 0.02),
    "parent_depth_only": Condition("parent_depth_only", 0.2, 0.1, 0.0, 0.0),
    "successor_only": Condition("successor_only", 0.0, 0.0, 0.2, 0.0),
    "shuffled_joint_targets": Condition(
        "shuffled_joint_targets",
        0.2,
        0.1,
        0.2,
        0.02,
        shuffle_seed=8675309,
        shuffle_fields=("parent", "depth", "next", "child_count"),
    ),
    "shuffled_no_successor": Condition(
        "shuffled_no_successor",
        0.2,
        0.1,
        0.0,
        0.02,
        shuffle_seed=314159,
        shuffle_fields=("parent", "depth", "child_count"),
    ),
}


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protected_hashes() -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in PROTECTED_PATTERNS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def git_output(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    detail = result.stderr.strip() or result.stdout.strip() or f"git exited with status {result.returncode}"
    return f"unavailable ({detail.splitlines()[0]})"


def command_line(command: list[str]) -> str:
    return " ".join(command)


def ensure_layout(out_root: Path) -> None:
    for name in ["checkpoints", "logs", "configs", "raw", "per_seed", "figures"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)


def condition_config(condition: Condition, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "condition": condition.name,
        "data": str(args.data),
        "proposition_ids": 526,
        "languages": ["en", "de"],
        "split_latent": True,
        "text_latent_dim": args.text_latent_dim,
        "structure_latent_dim": args.structure_latent_dim,
        "language_embedding_dim": args.language_embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "beta": args.beta,
        "beta_text": args.beta_text,
        "beta_structure": args.beta_structure,
        "lambda_parent": condition.lambda_parent,
        "lambda_depth": condition.lambda_depth,
        "lambda_next": condition.lambda_next,
        "lambda_child": condition.lambda_child,
        "lambda_language_alignment": 0.0,
        "lr": args.lr,
        "device": args.device,
        "formal_target_shuffle_seed": condition.shuffle_seed,
        "formal_target_shuffle_fields": list(condition.shuffle_fields),
    }


def train_command(args: argparse.Namespace, condition: Condition, seed: int, checkpoint: Path) -> list[str]:
    command = [
        "python3",
        "-m",
        "tractatus_structure_latents.training.train_vae",
        "--data",
        str(args.data),
        "--split-latent",
        "--text-latent-dim",
        str(args.text_latent_dim),
        "--structure-latent-dim",
        str(args.structure_latent_dim),
        "--language-embedding-dim",
        str(args.language_embedding_dim),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--beta",
        str(args.beta),
        "--beta-text",
        str(args.beta_text),
        "--beta-structure",
        str(args.beta_structure),
        "--lambda-parent",
        str(condition.lambda_parent),
        "--lambda-depth",
        str(condition.lambda_depth),
        "--lambda-next",
        str(condition.lambda_next),
        "--lambda-child",
        str(condition.lambda_child),
        "--lambda-language-alignment",
        "0.0",
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--out",
        str(checkpoint),
    ]
    if condition.shuffle_seed is not None:
        command.extend(["--formal-target-shuffle-seed", str(condition.shuffle_seed)])
        command.extend(["--formal-target-shuffle-fields", ",".join(condition.shuffle_fields)])
    return command


def instantiate_model(ckpt: dict[str, Any], dataset: TractatusDataset, vocab: Vocabulary) -> torch.nn.Module:
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
        latent_dim = ckpt.get("latent_dim", ckpt["model"]["encoder.mu.weight"].shape[0])
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
    return model


def per_sample_reconstruction(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="none",
    ).reshape(targets.shape)
    mask = targets != pad_idx
    return token_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def topk_rank(logits: np.ndarray, targets: np.ndarray, k_values: tuple[int, ...]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    order = np.argsort(-logits, axis=1)
    ranks = np.empty(len(targets), dtype=np.int64)
    hits = {k: np.zeros(len(targets), dtype=np.float64) for k in k_values}
    for i, target in enumerate(targets):
        loc = np.where(order[i] == target)[0]
        rank = int(loc[0]) + 1 if len(loc) else logits.shape[1] + 1
        ranks[i] = rank
        for k in k_values:
            hits[k][i] = 1.0 if rank <= k else 0.0
    return ranks, hits


def retrieval_columns(df: pd.DataFrame, z: np.ndarray, part: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ids = df["id"].astype(str).to_numpy()
    languages = df["language"].astype(str).to_numpy()
    row_lookup = {(row.id, row.language): i for i, row in df[["id", "language"]].iterrows()}
    all_top1: list[float] = []
    all_top5: list[float] = []
    all_top10: list[float] = []
    all_rr: list[float] = []
    all_dist: list[float] = []
    by_direction: dict[str, dict[str, list[float]]] = {}

    for source_i, row in df.iterrows():
        source_id = str(row["id"])
        source_language = str(row["language"])
        for target_language in sorted(set(languages) - {source_language}):
            target_i = row_lookup[(source_id, target_language)]
            candidates = np.where(languages == target_language)[0]
            distances = np.linalg.norm(z[candidates] - z[source_i], axis=1)
            order = np.argsort(distances)
            ranked_candidate_indices = candidates[order]
            ranked_ids = ids[ranked_candidate_indices]
            rank = int(np.where(ranked_ids == source_id)[0][0]) + 1
            same_distance = float(np.linalg.norm(z[source_i] - z[target_i]))
            direction = f"{source_language}_to_{target_language}"
            df.loc[source_i, f"{part}_retrieval_rank_{direction}"] = rank
            df.loc[source_i, f"{part}_same_id_distance_{direction}"] = same_distance
            df.loc[source_i, f"{part}_retrieval_top1_{direction}"] = float(rank <= 1)
            df.loc[source_i, f"{part}_retrieval_top5_{direction}"] = float(rank <= 5)
            df.loc[source_i, f"{part}_retrieval_top10_{direction}"] = float(rank <= 10)
            df.loc[source_i, f"{part}_retrieval_rr_{direction}"] = 1.0 / rank
            values = by_direction.setdefault(direction, {"top1": [], "top5": [], "top10": [], "rr": [], "rank": [], "distance": []})
            values["top1"].append(float(rank <= 1))
            values["top5"].append(float(rank <= 5))
            values["top10"].append(float(rank <= 10))
            values["rr"].append(1.0 / rank)
            values["rank"].append(float(rank))
            values["distance"].append(same_distance)
            all_top1.append(float(rank <= 1))
            all_top5.append(float(rank <= 5))
            all_top10.append(float(rank <= 10))
            all_rr.append(1.0 / rank)
            all_dist.append(same_distance)

    for key, values in {
        "top1": all_top1,
        "top5": all_top5,
        "top10": all_top10,
        "mrr": all_rr,
        "mean_rank": [1.0 / rr for rr in all_rr],
        "same_id_distance": all_dist,
    }.items():
        metrics[f"{part}_cross_language_{key}"] = mean(values)
    for direction, values in by_direction.items():
        metrics[f"{part}_cross_language_top1_{direction}"] = mean(values["top1"])
        metrics[f"{part}_cross_language_top5_{direction}"] = mean(values["top5"])
        metrics[f"{part}_cross_language_top10_{direction}"] = mean(values["top10"])
        metrics[f"{part}_cross_language_mrr_{direction}"] = mean(values["rr"])
        metrics[f"{part}_cross_language_mean_rank_{direction}"] = mean(values["rank"])
        metrics[f"{part}_same_id_distance_{direction}"] = mean(values["distance"])
    return metrics


def neighbourhood_jaccard(df: pd.DataFrame, z: np.ndarray, part: str, k_values: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ids = df["id"].astype(str).to_numpy()
    languages = df["language"].astype(str).to_numpy()
    row_lookup = {(row.id, row.language): i for i, row in df[["id", "language"]].iterrows()}
    for k in k_values:
        all_scores: list[float] = []
        by_direction: dict[str, list[float]] = {}
        for source_i, row in df.iterrows():
            source_id = str(row["id"])
            source_language = str(row["language"])
            source_candidates = np.where((languages == source_language) & (ids != source_id))[0]
            source_order = np.argsort(np.linalg.norm(z[source_candidates] - z[source_i], axis=1))[:k]
            source_neighbours = set(ids[source_candidates[source_order]])
            for target_language in sorted(set(languages) - {source_language}):
                target_i = row_lookup[(source_id, target_language)]
                target_candidates = np.where((languages == target_language) & (ids != source_id))[0]
                target_order = np.argsort(np.linalg.norm(z[target_candidates] - z[target_i], axis=1))[:k]
                target_neighbours = set(ids[target_candidates[target_order]])
                denom = len(source_neighbours | target_neighbours)
                score = len(source_neighbours & target_neighbours) / denom if denom else 0.0
                direction = f"{source_language}_to_{target_language}"
                df.loc[source_i, f"{part}_neighbourhood_jaccard_k{k}_{direction}"] = score
                all_scores.append(score)
                by_direction.setdefault(direction, []).append(score)
        metrics[f"{part}_wider_neighbourhood_jaccard_k{k}"] = mean(all_scores)
        for direction, values in by_direction.items():
            metrics[f"{part}_wider_neighbourhood_jaccard_k{k}_{direction}"] = mean(values)
    return metrics


def lexical_reference_metrics(dataset: TractatusDataset) -> dict[str, float]:
    rows = []
    for row in dataset.rows:
        for language, text in row_texts(row).items():
            rows.append({"id": row["id"], "language": language, "text": text})
    if len({row["language"] for row in rows}) < 2:
        return {}
    ids = np.array([row["id"] for row in rows])
    languages = np.array([row["language"] for row in rows])
    texts = [row["text"] for row in rows]
    metrics: dict[str, float] = {}

    def retrieval_from_distance(name: str, distances: np.ndarray) -> None:
        top1: list[float] = []
        rr: list[float] = []
        for i, row in enumerate(rows):
            candidates = np.where(languages != row["language"])[0]
            ordered = candidates[np.argsort(distances[i, candidates])]
            ranked_ids = ids[ordered]
            rank = int(np.where(ranked_ids == row["id"])[0][0]) + 1
            top1.append(float(rank == 1))
            rr.append(1.0 / rank)
        metrics[f"reference_{name}_top1"] = mean(top1)
        metrics[f"reference_{name}_mrr"] = mean(rr)

    token_sets = [set(tokenize(text)[:96]) for text in texts]
    jaccard_distance = np.zeros((len(rows), len(rows)), dtype=np.float64)
    for i, left in enumerate(token_sets):
        for j, right in enumerate(token_sets):
            union = left | right
            jaccard_distance[i, j] = 1.0 - (len(left & right) / len(union) if union else 0.0)
    retrieval_from_distance("exact_token_jaccard", jaccard_distance)

    word = TfidfVectorizer(tokenizer=lambda text: tokenize(text)[:96], token_pattern=None, lowercase=False)
    word_x = word.fit_transform(texts)
    retrieval_from_distance("word_tfidf", cosine_distances(word_x))

    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True)
    char_x = char.fit_transform(texts)
    retrieval_from_distance("char_3_5_tfidf", cosine_distances(char_x))
    return metrics


def relation_metrics(df: pd.DataFrame, z: np.ndarray, part: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    lookup = {
        (str(item["id"]), str(item["language"])): i
        for i, item in enumerate(df[["id", "language"]].to_dict("records"))
    }
    sibling: list[float] = []
    parent_child: list[float] = []
    unrelated: list[float] = []
    df[f"{part}_sibling_distance"] = np.nan
    df[f"{part}_parent_child_distance"] = np.nan
    df[f"{part}_unrelated_distance"] = np.nan
    language_positions: dict[str, list[int]] = {}
    for i, language in enumerate(df["language"].astype(str).tolist()):
        language_positions.setdefault(language, []).append(i)
    for i, item in enumerate(df.to_dict("records")):
        row = rows[int(item["index"]) - 1]
        language = str(item["language"])
        if row["parent_id"] is not None and (row["parent_id"], language) in lookup:
            distance = float(np.linalg.norm(z[i] - z[lookup[(row["parent_id"], language)]]))
            parent_child.append(distance)
            df.loc[i, f"{part}_parent_child_distance"] = distance
        for sibling_id in row["siblings"][:1]:
            if (sibling_id, language) in lookup:
                distance = float(np.linalg.norm(z[i] - z[lookup[(sibling_id, language)]]))
                sibling.append(distance)
                df.loc[i, f"{part}_sibling_distance"] = distance
        same_language = [j for j in language_positions[language] if j != i]
        if same_language:
            j = same_language[(i * 37 + 11) % len(same_language)]
            distance = float(np.linalg.norm(z[i] - z[j]))
            unrelated.append(distance)
            df.loc[i, f"{part}_unrelated_distance"] = distance
    return {
        f"{part}_mean_sibling_distance": mean(sibling),
        f"{part}_mean_parent_child_distance": mean(parent_child),
        f"{part}_mean_unrelated_distance": mean(unrelated),
        f"{part}_sibling_vs_unrelated_contrast": mean(unrelated) - mean(sibling),
    }


@torch.no_grad()
def evaluate_checkpoint(condition: str, seed: int, checkpoint: Path, data_path: Path, out_root: Path) -> dict[str, float]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    vocab = Vocabulary(ckpt["vocab"])
    dataset = TractatusDataset(data_path, vocab=vocab, languages=ckpt.get("languages"), language_to_id=ckpt.get("language_to_id"))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: collate_batch(batch, pad_idx=vocab.pad_idx))
    model = instantiate_model(ckpt, dataset, vocab)

    records: list[dict[str, Any]] = []
    text_mu: list[torch.Tensor] = []
    structure_mu: list[torch.Tensor] = []
    for batch in loader:
        outputs = model(batch["input_ids"], batch["lengths"], batch["decoder_ids"], batch["language_ids"])
        recon = per_sample_reconstruction(outputs["logits"], batch["targets"], vocab.pad_idx)
        kl_text = -0.5 * torch.sum(1 + outputs["text_logvar"] - outputs["text_mu"].pow(2) - outputs["text_logvar"].exp(), dim=-1)
        kl_structure = -0.5 * torch.sum(
            1 + outputs["structure_logvar"] - outputs["structure_mu"].pow(2) - outputs["structure_logvar"].exp(),
            dim=-1,
        )
        parent_logits = outputs["parent_logits"].detach().cpu().numpy()
        depth_logits = outputs["depth_logits"].detach().cpu().numpy()
        next_logits = outputs["next_logits"].detach().cpu().numpy()
        parent_pred = parent_logits.argmax(axis=1)
        depth_pred = depth_logits.argmax(axis=1)
        next_pred = next_logits.argmax(axis=1)
        child_pred = outputs["child_count"].detach().cpu().numpy()
        parent_rank, parent_hits = topk_rank(parent_logits, batch["parent"].numpy(), (5,))
        next_rank, next_hits = topk_rank(next_logits, batch["next"].numpy(), (5, 10))
        t_mu = outputs["text_mu"].detach().cpu()
        s_mu = outputs["structure_mu"].detach().cpu()
        text_mu.append(t_mu)
        structure_mu.append(s_mu)
        for i, prop_id in enumerate(batch["ids"]):
            records.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "id": prop_id,
                    "language": batch["languages"][i],
                    "index": int(batch["index"][i]),
                    "parent_true": int(batch["parent"][i]),
                    "parent_pred": int(parent_pred[i]),
                    "parent_rank": int(parent_rank[i]),
                    "parent_top5": float(parent_hits[5][i]),
                    "depth_true": int(batch["depth"][i]),
                    "depth_pred": int(depth_pred[i]),
                    "depth_abs_error": abs(int(depth_pred[i]) - int(batch["depth"][i])),
                    "successor_true": int(batch["next"][i]),
                    "successor_pred": int(next_pred[i]),
                    "successor_rank": int(next_rank[i]),
                    "successor_top5": float(next_hits[5][i]),
                    "successor_top10": float(next_hits[10][i]),
                    "successor_rr": 1.0 / int(next_rank[i]),
                    "child_count_true": float(batch["child_count"][i]),
                    "child_count_pred": float(child_pred[i]),
                    "child_count_abs_error": abs(float(child_pred[i]) - float(batch["child_count"][i])),
                    "child_count_sq_error": (float(child_pred[i]) - float(batch["child_count"][i])) ** 2,
                    "reconstruction_loss": float(recon[i]),
                    "kl_text": float(kl_text[i]),
                    "kl_structure": float(kl_structure[i]),
                    "structure_mean_norm": float(torch.norm(s_mu[i])),
                    "text_posterior_variance": float(outputs["text_logvar"][i].exp().mean()),
                    "structure_posterior_variance": float(outputs["structure_logvar"][i].exp().mean()),
                    "text_mu": json.dumps([float(x) for x in t_mu[i]]),
                    "structure_mu": json.dumps([float(x) for x in s_mu[i]]),
                }
            )

    df = pd.DataFrame(records)
    text_z = torch.cat(text_mu, dim=0).numpy()
    structure_z = torch.cat(structure_mu, dim=0).numpy()
    metrics: dict[str, float] = {}
    metrics.update(retrieval_columns(df, text_z, "text"))
    metrics.update(retrieval_columns(df, structure_z, "structure"))
    metrics.update(neighbourhood_jaccard(df, text_z, "text"))
    metrics.update(neighbourhood_jaccard(df, structure_z, "structure"))
    metrics.update(relation_metrics(df, text_z, "text", dataset.rows))
    metrics.update(relation_metrics(df, structure_z, "structure", dataset.rows))
    metrics.update(lexical_reference_metrics(dataset))

    y_parent = df["parent_true"].to_numpy()
    p_parent = df["parent_pred"].to_numpy()
    y_depth = df["depth_true"].to_numpy()
    p_depth = df["depth_pred"].to_numpy()
    y_next = df["successor_true"].to_numpy()
    p_next = df["successor_pred"].to_numpy()
    metrics.update(
        {
            "parent_accuracy": accuracy_score(y_parent, p_parent),
            "parent_balanced_accuracy": balanced_accuracy_score(y_parent, p_parent),
            "parent_macro_f1": f1_score(y_parent, p_parent, average="macro", zero_division=0),
            "parent_top5": float(df["parent_top5"].mean()),
            "parent_mean_rank": float(df["parent_rank"].mean()),
            "depth_accuracy": accuracy_score(y_depth, p_depth),
            "depth_balanced_accuracy": balanced_accuracy_score(y_depth, p_depth),
            "depth_absolute_error": float(df["depth_abs_error"].mean()),
            "successor_top1": accuracy_score(y_next, p_next),
            "successor_top5": float(df["successor_top5"].mean()),
            "successor_top10": float(df["successor_top10"].mean()),
            "successor_mrr": float(df["successor_rr"].mean()),
            "successor_mean_rank": float(df["successor_rank"].mean()),
            "child_count_mae": float(df["child_count_abs_error"].mean()),
            "child_count_rmse": math.sqrt(float(df["child_count_sq_error"].mean())),
            "reconstruction_loss": float(df["reconstruction_loss"].mean()),
            "perplexity": float(min(math.exp(float(df["reconstruction_loss"].mean())), 1e9)),
            "kl_text": float(df["kl_text"].mean()),
            "kl_structure": float(df["kl_structure"].mean()),
            "structure_mean_norm": float(df["structure_mean_norm"].mean()),
            "text_posterior_variance": float(df["text_posterior_variance"].mean()),
            "structure_posterior_variance": float(df["structure_posterior_variance"].mean()),
        }
    )
    # Backward-compatible aliases for the primary, manuscript-style structure latent.
    metrics["cross_language_top1_id_accuracy"] = metrics["structure_cross_language_top1"]
    metrics["cross_language_mrr"] = metrics["structure_cross_language_mrr"]
    metrics["mean_same_id_cross_language_distance"] = metrics["structure_cross_language_same_id_distance"]
    metrics["wider_neighbourhood_jaccard_k10"] = metrics["structure_wider_neighbourhood_jaccard_k10"]
    metrics["sibling_vs_unrelated_contrast"] = metrics["structure_sibling_vs_unrelated_contrast"]

    raw_path = out_root / "raw" / condition / f"seed{seed:03d}.per_proposition.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)
    metrics_path = out_root / "per_seed" / condition / f"seed{seed:03d}.metrics.json"
    write_json(metrics_path, metrics)
    return metrics


def summarise_seed_results(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for path in sorted((out_root / "per_seed").glob("*/*.metrics.json")):
        condition = path.parent.name
        seed = int(path.stem.split(".")[0].replace("seed", ""))
        metrics = read_json(path)
        rows.append({"condition": condition, "seed": seed, **metrics})
    seed_df = pd.DataFrame(rows).sort_values(["condition", "seed"])
    seed_df.to_csv(out_root / "phase1_seed_level_results.csv", index=False)
    combined = pd.concat([pd.read_parquet(path) for path in sorted((out_root / "raw").glob("*/*.parquet"))], ignore_index=True)
    combined.to_parquet(out_root / "phase1_per_proposition_results.parquet", index=False)

    summary_rows: list[dict[str, Any]] = []
    metric_cols = [col for col in seed_df.columns if col not in {"condition", "seed"}]
    for condition, group in seed_df.groupby("condition", sort=True):
        for metric in metric_cols:
            values = group[metric].dropna().astype(float).tolist()
            if not values:
                continue
            summary_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": mean(values),
                    "sample_sd": stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "phase1_ablation_summary.csv", index=False)
    return seed_df, summary_df


def bootstrap_ci(diffs: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        draws.append(mean(sample))
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def paired_differences(out_root: Path, seed_df: pd.DataFrame, baseline: str = "full_model") -> pd.DataFrame:
    metrics = [
        "structure_cross_language_top1",
        "structure_cross_language_mrr",
        "structure_cross_language_same_id_distance",
        "structure_wider_neighbourhood_jaccard_k10",
        "structure_sibling_vs_unrelated_contrast",
    ]
    rows: list[dict[str, Any]] = []
    base = seed_df[seed_df["condition"] == baseline].set_index("seed")
    for condition in sorted(set(seed_df["condition"]) - {baseline}):
        group = seed_df[seed_df["condition"] == condition].set_index("seed")
        common = sorted(set(base.index) & set(group.index))
        for metric in metrics:
            diffs = [float(group.loc[seed, metric]) - float(base.loc[seed, metric]) for seed in common]
            low, high = bootstrap_ci(diffs)
            rows.append(
                {
                    "baseline": baseline,
                    "condition": condition,
                    "metric": metric,
                    "paired_seed_count": len(diffs),
                    "paired_mean_difference": mean(diffs),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )
    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(out_root / "phase1_paired_differences.csv", index=False)
    return diff_df


def write_figures(out_root: Path, seed_df: pd.DataFrame) -> None:
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ordered = list(CONDITIONS)
    metrics = [
        ("structure_cross_language_top1", "Structure Top-1"),
        ("text_cross_language_top1", "Text Top-1"),
        ("structure_wider_neighbourhood_jaccard_k10", "Structure k=10 Jaccard"),
        ("structure_sibling_vs_unrelated_contrast", "Sibling contrast"),
    ]
    numeric_cols = [col for col in seed_df.columns if col not in {"condition"}]
    summary = seed_df.groupby("condition")[numeric_cols].agg(["mean", "std"])
    for metric, label in metrics:
        plt.figure(figsize=(9, 4.6))
        means = [summary.loc[condition, (metric, "mean")] for condition in ordered if condition in summary.index]
        errs = [summary.loc[condition, (metric, "std")] for condition in ordered if condition in summary.index]
        labels = [condition for condition in ordered if condition in summary.index]
        plt.bar(np.arange(len(labels)), means, yerr=errs, capsize=4, color="#4C78A8")
        plt.xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}.png", dpi=600)
        plt.close()


def metrics_from_raw_group(
    group: pd.DataFrame,
    dataset: TractatusDataset,
    lexical_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    df = group.reset_index(drop=True).copy()
    text_z = np.array([json.loads(value) for value in df["text_mu"]], dtype=np.float64)
    structure_z = np.array([json.loads(value) for value in df["structure_mu"]], dtype=np.float64)
    metrics: dict[str, float] = {}
    metrics.update(retrieval_columns(df, text_z, "text"))
    metrics.update(retrieval_columns(df, structure_z, "structure"))
    metrics.update(neighbourhood_jaccard(df, text_z, "text"))
    metrics.update(neighbourhood_jaccard(df, structure_z, "structure"))
    metrics.update(relation_metrics(df, text_z, "text", dataset.rows))
    metrics.update(relation_metrics(df, structure_z, "structure", dataset.rows))
    metrics.update(lexical_metrics if lexical_metrics is not None else lexical_reference_metrics(dataset))

    y_parent = df["parent_true"].to_numpy()
    p_parent = df["parent_pred"].to_numpy()
    y_depth = df["depth_true"].to_numpy()
    p_depth = df["depth_pred"].to_numpy()
    y_next = df["successor_true"].to_numpy()
    p_next = df["successor_pred"].to_numpy()
    metrics.update(
        {
            "parent_accuracy": accuracy_score(y_parent, p_parent),
            "parent_balanced_accuracy": balanced_accuracy_score(y_parent, p_parent),
            "parent_macro_f1": f1_score(y_parent, p_parent, average="macro", zero_division=0),
            "parent_top5": float(df["parent_top5"].mean()),
            "parent_mean_rank": float(df["parent_rank"].mean()),
            "depth_accuracy": accuracy_score(y_depth, p_depth),
            "depth_balanced_accuracy": balanced_accuracy_score(y_depth, p_depth),
            "depth_absolute_error": float(df["depth_abs_error"].mean()),
            "successor_top1": accuracy_score(y_next, p_next),
            "successor_top5": float(df["successor_top5"].mean()),
            "successor_top10": float(df["successor_top10"].mean()),
            "successor_mrr": float(df["successor_rr"].mean()),
            "successor_mean_rank": float(df["successor_rank"].mean()),
            "child_count_mae": float(df["child_count_abs_error"].mean()),
            "child_count_rmse": math.sqrt(float(df["child_count_sq_error"].mean())),
            "reconstruction_loss": float(df["reconstruction_loss"].mean()),
            "perplexity": float(min(math.exp(float(df["reconstruction_loss"].mean())), 1e9)),
            "kl_text": float(df["kl_text"].mean()),
            "kl_structure": float(df["kl_structure"].mean()),
            "structure_mean_norm": float(df["structure_mean_norm"].mean()),
            "text_posterior_variance": float(df["text_posterior_variance"].mean()),
            "structure_posterior_variance": float(df["structure_posterior_variance"].mean()),
        }
    )
    metrics["cross_language_top1_id_accuracy"] = metrics["structure_cross_language_top1"]
    metrics["cross_language_mrr"] = metrics["structure_cross_language_mrr"]
    metrics["mean_same_id_cross_language_distance"] = metrics["structure_cross_language_same_id_distance"]
    metrics["wider_neighbourhood_jaccard_k10"] = metrics["structure_wider_neighbourhood_jaccard_k10"]
    metrics["sibling_vs_unrelated_contrast"] = metrics["structure_sibling_vs_unrelated_contrast"]
    return metrics


def value(summary_df: pd.DataFrame, condition: str, metric: str) -> tuple[float, float]:
    row = summary_df[(summary_df["condition"] == condition) & (summary_df["metric"] == metric)].iloc[0]
    return float(row["mean"]), float(row["sample_sd"])


def optional_value(summary_df: pd.DataFrame, condition: str, metric: str) -> float | None:
    rows = summary_df[(summary_df["condition"] == condition) & (summary_df["metric"] == metric)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["mean"])


def fmt(mean_value: float, sd_value: float) -> str:
    return f"{mean_value:.4f} +/- {sd_value:.4f}"


def fmt_optional(value: float | None) -> str:
    return "not run" if value is None else f"{value:.4f}"


def write_report(out_root: Path, seed_df: pd.DataFrame, summary_df: pd.DataFrame, diff_df: pd.DataFrame) -> None:
    lines = [
        "# Phase 1 Ablation Report",
        "",
        "All conditions used the shared bilingual vocabulary/parameters, 24-dimensional text latent, 8-dimensional structure latent, GRU encoder/decoder, manuscript KL/reconstruction settings, lambda_language_alignment=0.00, and seeds 0-9.",
        "",
        "## Seed-level summary",
        "",
        "| condition | structure Top-1 | structure MRR | text Top-1 | text MRR | k=10 Jaccard | sibling contrast | successor Top-1 | parent acc. | depth acc. |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        if condition not in set(summary_df["condition"]):
            continue
        vals = {
            metric: value(summary_df, condition, metric)
            for metric in [
                "structure_cross_language_top1",
                "structure_cross_language_mrr",
                "text_cross_language_top1",
                "text_cross_language_mrr",
                "structure_wider_neighbourhood_jaccard_k10",
                "structure_sibling_vs_unrelated_contrast",
                "successor_top1",
                "parent_accuracy",
                "depth_accuracy",
            ]
        }
        lines.append(
            f"| `{condition}` | {fmt(*vals['structure_cross_language_top1'])} | {fmt(*vals['structure_cross_language_mrr'])} | "
            f"{fmt(*vals['text_cross_language_top1'])} | {fmt(*vals['text_cross_language_mrr'])} | "
            f"{fmt(*vals['structure_wider_neighbourhood_jaccard_k10'])} | {fmt(*vals['structure_sibling_vs_unrelated_contrast'])} | "
            f"{fmt(*vals['successor_top1'])} | {fmt(*vals['parent_accuracy'])} | {fmt(*vals['depth_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences versus full_model",
            "",
            "| condition | metric | mean diff | 95% bootstrap CI |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, row in diff_df.iterrows():
        lines.append(
            f"| `{row['condition']}` | `{row['metric']}` | {float(row['paired_mean_difference']):.4f} | "
            f"[{float(row['bootstrap_ci95_low']):.4f}, {float(row['bootstrap_ci95_high']):.4f}] |"
        )

    recon_top1 = optional_value(summary_df, "reconstruction_only", "structure_cross_language_top1")
    recon_text_top1 = optional_value(summary_df, "reconstruction_only", "text_cross_language_top1")
    no_succ_top1 = optional_value(summary_df, "no_successor", "structure_cross_language_top1")
    succ_only_top1 = optional_value(summary_df, "successor_only", "structure_cross_language_top1")
    shuffled_top1 = optional_value(summary_df, "shuffled_joint_targets", "structure_cross_language_top1")
    full_sibling = optional_value(summary_df, "full_model", "structure_sibling_vs_unrelated_contrast")
    shuffled_sibling = optional_value(summary_df, "shuffled_joint_targets", "structure_sibling_vs_unrelated_contrast")
    parent_depth_top1 = optional_value(summary_df, "parent_depth_only", "structure_cross_language_top1")

    lines.extend(
        [
            "",
            "## Required questions",
            "",
            f"1. Reconstruction alone produced structure-latent Top-1 {fmt_optional(recon_top1)}; text-latent Top-1 was {fmt_optional(recon_text_top1)}. This is the controlled estimate for retrieval available without formal supervision.",
            f"2. Retrieval primarily resides in the latent with higher Top-1/MRR within each condition; in reconstruction_only the text-vs-structure split is {fmt_optional(recon_text_top1)} versus {fmt_optional(recon_top1)}.",
            f"3. Removing successor left structure-latent Top-1 {fmt_optional(no_succ_top1)}. Interpret this against full_model using the paired CI table, not by a mean comparison alone.",
            f"4. Successor alone reached structure-latent Top-1 {fmt_optional(succ_only_top1)}, directly testing whether near proposition-specific labels can reproduce the retrieval effect.",
            f"5. Shuffled shared targets reached structure-latent Top-1 {fmt_optional(shuffled_top1)}; because German and English share each shuffled target tuple, this tests target-sharing without true formal position.",
            f"6. Sibling cohesion is supported only if true-target conditions retain a positive sibling-versus-unrelated contrast beyond shuffled controls. The full_model contrast is {fmt_optional(full_sibling)}; shuffled_joint_targets is {fmt_optional(shuffled_sibling)}.",
            f"7. Parent/depth without successor reached Top-1 {fmt_optional(parent_depth_top1)}; successor_only and no_successor identify which formal objectives add evidence beyond proposition-specific identity.",
            "",
            "Small numerical differences should be treated as descriptive unless their paired bootstrap intervals and the task design make the direction stable and interpretable.",
            "",
            "## Implications for the current manuscript",
            "",
            "Claims about same-ID bilingual retrieval remain supported only to the extent that the full_model exceeds reconstruction_only and shuffled shared-target controls. Claims that retrieval reflects hierarchy rather than proposition-specific formal labels require qualification if successor_only or shuffled_joint_targets approach the full_model. Sibling/family-organisation claims remain supported when true hierarchy-supervised conditions show stronger sibling-versus-unrelated contrast than shuffled controls; otherwise they should be narrowed. No manuscript text was edited in this phase.",
        ]
    )
    (out_root / "phase1_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    packages = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        packages["scikit_learn"] = sklearn.__version__
    except Exception as exc:  # pragma: no cover
        packages["scikit_learn"] = f"unavailable: {exc}"
    try:
        import pyarrow

        packages["pyarrow"] = pyarrow.__version__
    except Exception as exc:  # pragma: no cover
        packages["pyarrow"] = f"unavailable: {exc}"
    return packages


def gpu_info() -> dict[str, Any]:
    info = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def run(args: argparse.Namespace) -> None:
    out_root = args.out_root.resolve()
    ensure_layout(out_root)
    selected = [CONDITIONS[name] for name in args.conditions.split(",") if name]
    seeds = parse_ints(args.seeds)
    manifest_path = out_root / "phase1_config_manifest.json"
    baseline_hashes_path = out_root / "phase1_protected_hashes_before.json"
    if not baseline_hashes_path.exists():
        write_json(baseline_hashes_path, protected_hashes())

    command_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    manifest = {
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_at_start": git_output(["status", "--short"]),
        "git_branch": git_output(["branch", "--show-current"]),
        "package_versions": package_versions(),
        "gpu_info": gpu_info(),
        "seeds": seeds,
        "conditions": {},
        "outputs": str(out_root),
    }

    for condition in selected:
        config = condition_config(condition, args)
        manifest["conditions"][condition.name] = config
        write_json(out_root / "configs" / f"{condition.name}.json", config)
        for seed in seeds:
            checkpoint = out_root / "checkpoints" / condition.name / f"seed{seed:03d}.pt"
            log_path = out_root / "logs" / condition.name / f"seed{seed:03d}.train.log"
            metrics_path = out_root / "per_seed" / condition.name / f"seed{seed:03d}.metrics.json"
            command = train_command(args, condition, seed, checkpoint)
            command_lines.append(command_line(command))
            command_lines.append(
                command_line(
                    [
                        "python3",
                        "tools/phase1_ablations.py",
                        "eval-one",
                        "--condition",
                        condition.name,
                        "--seed",
                        str(seed),
                        "--checkpoint",
                        str(checkpoint),
                        "--data",
                        str(args.data),
                        "--out-root",
                        str(out_root),
                    ]
                )
            )
            if args.skip_existing and checkpoint.exists() and metrics_path.exists():
                print(f"skipping {condition.name}/seed{seed:03d}")
                continue
            print(f"training {condition.name}/seed{seed:03d}", flush=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=handle, stderr=subprocess.STDOUT)
            evaluate_checkpoint(condition.name, seed, checkpoint, args.data, out_root)

    write_json(manifest_path, manifest)
    (out_root / "phase1_commands.sh").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    seed_df, summary_df = summarise_seed_results(out_root)
    diff_df = paired_differences(out_root, seed_df)
    write_figures(out_root, seed_df)
    write_report(out_root, seed_df, summary_df, diff_df)


def eval_one(args: argparse.Namespace) -> None:
    evaluate_checkpoint(args.condition, args.seed, args.checkpoint, args.data, args.out_root.resolve())


def verify(args: argparse.Namespace) -> None:
    out_root = args.out_root.resolve()
    raw_paths = sorted((out_root / "raw").glob("*/*.parquet"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw per-seed parquet files under {out_root / 'raw'}")
    combined = pd.concat([pd.read_parquet(path) for path in raw_paths], ignore_index=True)
    reported_seed = pd.read_csv(out_root / "phase1_seed_level_results.csv")
    dataset = TractatusDataset(DATA_PATH)
    lexical_metrics = lexical_reference_metrics(dataset)
    checks: list[str] = []
    recomputed_rows: list[dict[str, Any]] = []
    for (condition, seed), group in combined.groupby(["condition", "seed"], sort=True):
        metrics = metrics_from_raw_group(group, dataset, lexical_metrics=lexical_metrics)
        row = {"condition": condition, "seed": int(seed), **metrics}
        recomputed_rows.append(row)
        reported = reported_seed[(reported_seed["condition"] == condition) & (reported_seed["seed"] == int(seed))].iloc[0]
        for key, value_recomputed in row.items():
            if key in {"condition", "seed"}:
                continue
            if key not in reported:
                raise AssertionError(f"{condition}/seed{seed} metric missing from reported seed CSV: {key}")
            value_reported = float(reported[key])
            if abs(value_reported - value_recomputed) > 1e-6:
                raise AssertionError(f"{condition}/seed{seed} {key}: reported {value_reported}, recomputed {value_recomputed}")
    recomputed = pd.DataFrame(recomputed_rows)
    recomputed.to_csv(out_root / "phase1_recomputed_seed_level_results.csv", index=False)
    summary = pd.read_csv(out_root / "phase1_ablation_summary.csv")
    for (condition, metric), group in summary.groupby(["condition", "metric"]):
        if metric not in recomputed.columns:
            raise AssertionError(f"summary metric cannot be recomputed from raw outputs: {condition}/{metric}")
        reported_values = recomputed[recomputed["condition"] == condition][metric].dropna().astype(float).tolist()
        if not reported_values:
            continue
        mean_value = mean(reported_values)
        sd_value = stdev(reported_values) if len(reported_values) > 1 else 0.0
        row = group.iloc[0]
        if abs(float(row["mean"]) - mean_value) > 1e-6 or abs(float(row["sample_sd"]) - sd_value) > 1e-6:
            raise AssertionError(f"summary mismatch for {condition}/{metric}")
    checks.append(f"Verified {len(recomputed)} condition-seed rows and {len(recomputed.columns) - 2} metrics per row from raw parquet files.")
    checks.append(f"Verified {len(summary)} reported mean/sample-SD summary aggregates from recomputed seed metrics.")
    before_path = out_root / "phase1_protected_hashes_before.json"
    if before_path.exists():
        before = read_json(before_path)
        after = protected_hashes()
        changed = [path for path, digest in before.items() if after.get(path) != digest]
        if changed:
            raise AssertionError(f"Protected canonical files changed: {changed[:20]}")
        checks.append(f"Verified {len(before)} protected canonical file hashes unchanged.")
    status = git_output(["status", "--short"])
    diff_stat = git_output(["diff", "--stat"])
    lines = [
        "# Phase 1 Verification Report",
        "",
        *[f"- {check}" for check in checks],
        f"- Raw per-seed parquet files: {len(raw_paths)}.",
        f"- Combined raw rows: {len(combined)}.",
        f"- Git branch: {git_output(['branch', '--show-current'])}.",
        f"- Git commit during verification: {git_output(['rev-parse', 'HEAD'])}.",
        "",
        "## Git status",
        "",
        "```",
        status,
        "```",
        "",
        "## Git diff stat",
        "",
        "```",
        diff_stat,
        "```",
    ]
    (out_root / "phase1_verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(checks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 formal-objective and bilingual retrieval ablations.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    run_parser.add_argument("--data", type=Path, default=DATA_PATH)
    run_parser.add_argument("--conditions", default=",".join(CONDITIONS))
    run_parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    run_parser.add_argument("--epochs", type=int, default=80)
    run_parser.add_argument("--batch-size", type=int, default=32)
    run_parser.add_argument("--text-latent-dim", type=int, default=24)
    run_parser.add_argument("--structure-latent-dim", type=int, default=8)
    run_parser.add_argument("--language-embedding-dim", type=int, default=8)
    run_parser.add_argument("--beta", type=float, default=0.01)
    run_parser.add_argument("--beta-text", type=float, default=0.01)
    run_parser.add_argument("--beta-structure", type=float, default=0.05)
    run_parser.add_argument("--lr", type=float, default=0.001)
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--skip-existing", action="store_true")
    run_parser.set_defaults(func=run)

    eval_parser = sub.add_parser("eval-one")
    eval_parser.add_argument("--condition", required=True)
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--data", type=Path, default=DATA_PATH)
    eval_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    eval_parser.set_defaults(func=eval_one)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
