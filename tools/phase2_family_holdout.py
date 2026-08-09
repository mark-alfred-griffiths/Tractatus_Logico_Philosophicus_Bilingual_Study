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
from collections import Counter
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
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase1_ablations import instantiate_model
from tractatus_structure_latents.training.data import TractatusDataset, Vocabulary, collate_batch, row_texts, tokenize

DATA_PATH = ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json"
DEFAULT_OUT = ROOT / "results" / "dsh_validation" / "phase2_family_holdout"
MATCHED_RESAMPLES = 5_000
MATCHED_SEED = 20260805
PROTECTED_PATTERNS = [
    "paper/main.tex",
    "paper/references.bib",
    "paper/*.pdf",
    "paper/**/*.pdf",
    "results/dsh_validation/phase1_ablations/**/*",
]


@dataclass(frozen=True)
class Condition:
    name: str
    lambda_parent: float
    lambda_depth: float
    lambda_next: float
    lambda_child: float


CONDITIONS = {
    "full_model": Condition("full_model", 0.2, 0.1, 0.2, 0.02),
    "no_successor": Condition("no_successor", 0.2, 0.1, 0.0, 0.02),
    "reconstruction_only": Condition("reconstruction_only", 0.0, 0.0, 0.0, 0.0),
}


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def normalise_device(raw: str) -> str:
    return "cuda" if raw == "gpu" else raw


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
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


def top_branch(prop_id: str) -> str:
    return prop_id.split(".", 1)[0]


def family_id(row: dict[str, Any]) -> str:
    return str(row["parent_id"]) if row["parent_id"] is not None else f"ROOT:{row['id']}"


def text_len(row: dict[str, Any]) -> float:
    texts = row_texts(row)
    return mean(len(tokenize(text)[:96]) for text in texts.values()) if texts else 0.0


def make_fold_manifest(data_path: Path, out: Path, fold_count: int = 5) -> pd.DataFrame:
    rows = read_json(data_path)
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        families.setdefault(family_id(row), []).append(row)
    family_stats = []
    for fam_id, items in families.items():
        depths = Counter(int(item["depth"]) for item in items)
        branches = Counter(top_branch(item["id"]) for item in items)
        family_stats.append(
            {
                "family_id": fam_id,
                "items": items,
                "size": len(items),
                "depths": depths,
                "branches": branches,
                "sort_key": (-len(items), sorted(branches), sorted(depths.items()), fam_id),
            }
        )
    folds = [
        {"size": 0, "depths": Counter(), "branches": Counter(), "family_sizes": Counter(), "families": []}
        for _ in range(fold_count)
    ]
    for family in sorted(family_stats, key=lambda item: item["sort_key"]):
        best_fold = None
        best_score = None
        for fold_i, fold in enumerate(folds):
            size_after = [folds[i]["size"] + (family["size"] if i == fold_i else 0) for i in range(fold_count)]
            score = max(size_after) - min(size_after)
            score += 0.1 * sum(
                abs((fold["depths"] + family["depths"])[depth] - mean([f["depths"][depth] for f in folds]))
                for depth in set(family["depths"])
            )
            score += 0.05 * sum(
                abs((fold["branches"] + family["branches"])[branch] - mean([f["branches"][branch] for f in folds]))
                for branch in set(family["branches"])
            )
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_i
        assert best_fold is not None
        fold = folds[best_fold]
        fold["size"] += family["size"]
        fold["depths"].update(family["depths"])
        fold["branches"].update(family["branches"])
        fold["family_sizes"][family["size"]] += 1
        fold["families"].append(family["family_id"])
    family_to_fold = {
        fam_id: fold_i
        for fold_i, fold in enumerate(folds)
        for fam_id in fold["families"]
    }
    manifest_rows = []
    for row in rows:
        fam_id = family_id(row)
        manifest_rows.append(
            {
                "id": row["id"],
                "parent_id": "" if row["parent_id"] is None else row["parent_id"],
                "family_id": fam_id,
                "fold": family_to_fold[fam_id],
                "depth": int(row["depth"]),
                "top_level_branch": top_branch(row["id"]),
                "family_size": len(families[fam_id]),
                "mean_text_length_96_tokens": text_len(row),
            }
        )
    df = pd.DataFrame(manifest_rows).sort_values(["fold", "family_id", "id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_fold_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "id": str,
            "parent_id": str,
            "family_id": str,
            "top_level_branch": str,
        },
    )
    df["fold"] = df["fold"].astype(int)
    df["depth"] = df["depth"].astype(int)
    df["family_size"] = df["family_size"].astype(int)
    return df


def ensure_layout(out_root: Path) -> None:
    for name in ["checkpoints", "logs", "configs", "raw", "per_seed", "figures", "ids"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)


def sample_ids_for_fold(manifest: pd.DataFrame, fold: int) -> tuple[list[str], list[str]]:
    test_ids = sorted(manifest[manifest["fold"] == fold]["id"].astype(str).tolist())
    train_ids = sorted(manifest[manifest["fold"] != fold]["id"].astype(str).tolist())
    return train_ids, test_ids


def train_command(args: argparse.Namespace, condition: Condition, fold: int, seed: int, checkpoint: Path, ids_file: Path) -> list[str]:
    return [
        "python3",
        "-m",
        "tractatus_structure_latents.training.train_vae",
        "--data",
        str(args.data),
        "--sample-ids-file",
        str(ids_file),
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


@torch.no_grad()
def encode_rows(checkpoint: Path, data_path: Path, sample_ids: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, TractatusDataset]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    vocab = Vocabulary(ckpt["vocab"])
    dataset = TractatusDataset(
        data_path,
        vocab=vocab,
        languages=ckpt.get("languages"),
        language_to_id=ckpt.get("language_to_id"),
        sample_ids=sample_ids,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: collate_batch(batch, pad_idx=vocab.pad_idx))
    model = instantiate_model(ckpt, dataset, vocab)
    records = []
    text_mu = []
    structure_mu = []
    for batch in loader:
        outputs = model(batch["input_ids"], batch["lengths"], batch["decoder_ids"], batch["language_ids"])
        recon = F.cross_entropy(
            outputs["logits"].reshape(-1, outputs["logits"].size(-1)),
            batch["targets"].reshape(-1),
            ignore_index=vocab.pad_idx,
            reduction="none",
        ).reshape(batch["targets"].shape)
        mask = batch["targets"] != vocab.pad_idx
        per_recon = recon.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        kl_text = -0.5 * torch.sum(1 + outputs["text_logvar"] - outputs["text_mu"].pow(2) - outputs["text_logvar"].exp(), dim=-1)
        kl_structure = -0.5 * torch.sum(
            1 + outputs["structure_logvar"] - outputs["structure_mu"].pow(2) - outputs["structure_logvar"].exp(),
            dim=-1,
        )
        parent_logits = outputs["parent_logits"].detach().cpu().numpy()
        next_logits = outputs["next_logits"].detach().cpu().numpy()
        depth_logits = outputs["depth_logits"].detach().cpu().numpy()
        child_pred = outputs["child_count"].detach().cpu().numpy()
        t_mu = outputs["text_mu"].detach().cpu()
        s_mu = outputs["structure_mu"].detach().cpu()
        text_mu.append(t_mu)
        structure_mu.append(s_mu)
        for i, prop_id in enumerate(batch["ids"]):
            records.append(
                {
                    "id": prop_id,
                    "language": batch["languages"][i],
                    "index": int(batch["index"][i]),
                    "parent_true": int(batch["parent"][i]),
                    "parent_pred": int(parent_logits[i].argmax()),
                    "successor_true": int(batch["next"][i]),
                    "successor_pred": int(next_logits[i].argmax()),
                    "depth_true": int(batch["depth"][i]),
                    "depth_pred": int(depth_logits[i].argmax()),
                    "depth_abs_error": abs(int(depth_logits[i].argmax()) - int(batch["depth"][i])),
                    "child_count_true": float(batch["child_count"][i]),
                    "child_count_pred": float(child_pred[i]),
                    "child_count_abs_error": abs(float(child_pred[i]) - float(batch["child_count"][i])),
                    "child_count_sq_error": (float(child_pred[i]) - float(batch["child_count"][i])) ** 2,
                    "reconstruction_loss": float(per_recon[i]),
                    "kl_text": float(kl_text[i]),
                    "kl_structure": float(kl_structure[i]),
                    "structure_mean_norm": float(torch.norm(s_mu[i])),
                    "text_posterior_variance": float(outputs["text_logvar"][i].exp().mean()),
                    "structure_posterior_variance": float(outputs["structure_logvar"][i].exp().mean()),
                    "text_mu": json.dumps([float(x) for x in t_mu[i]]),
                    "structure_mu": json.dumps([float(x) for x in s_mu[i]]),
                }
            )
    return pd.DataFrame(records), torch.cat(text_mu, dim=0).numpy(), torch.cat(structure_mu, dim=0).numpy(), dataset


def retrieval_metrics(
    source_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    source_z: np.ndarray,
    candidate_z: np.ndarray,
    part: str,
    candidate_scope: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    directions = sorted(
        (source_language, target_language)
        for source_language in set(source_df["language"])
        for target_language in set(candidate_df["language"])
        if source_language != target_language
    )
    all_values = {"top1": [], "top5": [], "top10": [], "rr": [], "rank": [], "distance": []}
    for source_language, target_language in directions:
        values = {"top1": [], "top5": [], "top10": [], "rr": [], "rank": [], "distance": []}
        source_indices = [i for i, language in enumerate(source_df["language"]) if language == source_language]
        candidate_indices = [i for i, language in enumerate(candidate_df["language"]) if language == target_language]
        candidate_ids = candidate_df.iloc[candidate_indices]["id"].astype(str).to_numpy()
        for source_i in source_indices:
            source_id = str(source_df.iloc[source_i]["id"])
            distances = np.linalg.norm(candidate_z[candidate_indices] - source_z[source_i], axis=1)
            order = np.argsort(distances)
            ranked_ids = candidate_ids[order]
            loc = np.where(ranked_ids == source_id)[0]
            if not len(loc):
                continue
            rank = int(loc[0]) + 1
            same_distance = float(distances[order[rank - 1]])
            row_key = f"{part}_{candidate_scope}_retrieval"
            source_df.loc[source_i, f"{row_key}_rank_{source_language}_to_{target_language}"] = rank
            source_df.loc[source_i, f"{row_key}_top1_{source_language}_to_{target_language}"] = float(rank <= 1)
            source_df.loc[source_i, f"{row_key}_top5_{source_language}_to_{target_language}"] = float(rank <= 5)
            source_df.loc[source_i, f"{row_key}_top10_{source_language}_to_{target_language}"] = float(rank <= 10)
            source_df.loc[source_i, f"{row_key}_mrr_{source_language}_to_{target_language}"] = 1.0 / rank
            source_df.loc[source_i, f"{part}_{candidate_scope}_same_id_distance_{source_language}_to_{target_language}"] = same_distance
            values["top1"].append(float(rank <= 1))
            values["top5"].append(float(rank <= 5))
            values["top10"].append(float(rank <= 10))
            values["rr"].append(1.0 / rank)
            values["rank"].append(float(rank))
            values["distance"].append(same_distance)
        direction = f"{source_language}_to_{target_language}"
        for key, vals in values.items():
            if vals:
                metrics[f"{part}_{candidate_scope}_{key if key != 'rr' else 'mrr'}_{direction}"] = mean(vals)
                all_values[key].extend(vals)
    for key, vals in all_values.items():
        if vals:
            metrics[f"{part}_{candidate_scope}_{key if key != 'rr' else 'mrr'}"] = mean(vals)
    return metrics


def neighbourhood_jaccard(df: pd.DataFrame, z: np.ndarray, part: str, k_values: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    metrics = {}
    ids = df["id"].astype(str).to_numpy()
    languages = df["language"].astype(str).to_numpy()
    lookup = {(row.id, row.language): i for i, row in df[["id", "language"]].iterrows()}
    for k in k_values:
        scores = []
        for i, row in df.iterrows():
            same_lang = np.where((languages == row["language"]) & (ids != row["id"]))[0]
            if len(same_lang) == 0:
                continue
            own_order = np.argsort(np.linalg.norm(z[same_lang] - z[i], axis=1))[:k]
            own_neighbours = set(ids[same_lang[own_order]])
            for target_language in sorted(set(languages) - {row["language"]}):
                target_i = lookup.get((row["id"], target_language))
                if target_i is None:
                    continue
                target_lang = np.where((languages == target_language) & (ids != row["id"]))[0]
                target_order = np.argsort(np.linalg.norm(z[target_lang] - z[target_i], axis=1))[:k]
                target_neighbours = set(ids[target_lang[target_order]])
                denom = len(own_neighbours | target_neighbours)
                score = len(own_neighbours & target_neighbours) / denom if denom else 0.0
                df.loc[i, f"{part}_test_neighbourhood_jaccard_k{k}_{row['language']}_to_{target_language}"] = score
                scores.append(score)
        metrics[f"{part}_test_wider_neighbourhood_jaccard_k{k}"] = mean(scores) if scores else 0.0
    return metrics


def relation_metrics(df: pd.DataFrame, z: np.ndarray, part: str, rows_by_id: dict[str, dict[str, Any]], rng: random.Random) -> dict[str, float]:
    lookup = {(row.id, row.language): i for i, row in df[["id", "language"]].iterrows()}
    language_positions: dict[str, list[int]] = {}
    for i, language in enumerate(df["language"].astype(str).tolist()):
        language_positions.setdefault(language, []).append(i)
    sibling = []
    unrelated = []
    for i, item in df.iterrows():
        row = rows_by_id[str(item["id"])]
        language = str(item["language"])
        sib_dists = []
        for sibling_id in row["siblings"]:
            j = lookup.get((sibling_id, language))
            if j is not None:
                sib_dists.append(float(np.linalg.norm(z[i] - z[j])))
        if sib_dists:
            value = mean(sib_dists)
            sibling.append(value)
            df.loc[i, f"{part}_heldout_sibling_distance"] = value
            candidates = [
                j
                for j in language_positions[language]
                if str(df.iloc[j]["id"]) != str(item["id"])
                and rows_by_id[str(df.iloc[j]["id"])]["parent_id"] != row["parent_id"]
                and int(df.iloc[j]["depth_true"]) == int(item["depth_true"])
            ]
            if not candidates:
                candidates = [
                    j
                    for j in language_positions[language]
                    if str(df.iloc[j]["id"]) != str(item["id"])
                    and rows_by_id[str(df.iloc[j]["id"])]["parent_id"] != row["parent_id"]
                ]
            if candidates:
                j = rng.choice(candidates)
                value = float(np.linalg.norm(z[i] - z[j]))
                unrelated.append(value)
                df.loc[i, f"{part}_matched_unrelated_distance"] = value
    return {
        f"{part}_mean_heldout_sibling_distance": mean(sibling) if sibling else 0.0,
        f"{part}_mean_matched_unrelated_distance": mean(unrelated) if unrelated else 0.0,
        f"{part}_sibling_vs_matched_unrelated_contrast": (mean(unrelated) - mean(sibling)) if sibling and unrelated else 0.0,
    }


def lexical_references(rows: list[dict[str, Any]], train_ids: set[str], test_ids: set[str]) -> dict[str, float]:
    train_texts = [text for row in rows if row["id"] in train_ids for text in row_texts(row).values()]
    records = [{"id": row["id"], "language": lang, "text": text} for row in rows for lang, text in row_texts(row).items()]
    if len({record["language"] for record in records}) < 2:
        return {}
    df = pd.DataFrame(records)
    test_indices = [i for i, row in df.iterrows() if row["id"] in test_ids]
    metrics: dict[str, float] = {}

    def from_distance(name: str, distances: np.ndarray) -> None:
        for scope in ["test", "complete"]:
            values = {"top1": [], "top5": [], "top10": [], "rr": [], "rank": []}
            for i in test_indices:
                source = df.iloc[i]
                candidates = [
                    j
                    for j, candidate in df.iterrows()
                    if candidate["language"] != source["language"] and (scope == "complete" or candidate["id"] in test_ids)
                ]
                candidate_ids = df.iloc[candidates]["id"].astype(str).to_numpy()
                ranked = candidate_ids[np.argsort(distances[i, candidates])]
                rank = int(np.where(ranked == source["id"])[0][0]) + 1
                values["top1"].append(float(rank <= 1))
                values["top5"].append(float(rank <= 5))
                values["top10"].append(float(rank <= 10))
                values["rr"].append(1.0 / rank)
                values["rank"].append(float(rank))
            for key, vals in values.items():
                metrics[f"reference_{name}_{scope}_{key if key != 'rr' else 'mrr'}"] = mean(vals)

    token_sets = [set(tokenize(text)[:96]) for text in df["text"]]
    jaccard = np.zeros((len(df), len(df)), dtype=np.float64)
    for i, left in enumerate(token_sets):
        for j, right in enumerate(token_sets):
            union = left | right
            jaccard[i, j] = 1.0 - (len(left & right) / len(union) if union else 0.0)
    from_distance("exact_token_jaccard", jaccard)

    word = TfidfVectorizer(tokenizer=lambda text: tokenize(text)[:96], token_pattern=None, lowercase=False)
    word.fit(train_texts)
    from_distance("word_tfidf", cosine_distances(word.transform(df["text"])))
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True)
    char.fit(train_texts)
    from_distance("char_3_5_tfidf", cosine_distances(char.transform(df["text"])))
    return metrics


def matched_family_analysis(
    df: pd.DataFrame,
    z: np.ndarray,
    fold_manifest: pd.DataFrame,
    rows_by_id: dict[str, dict[str, Any]],
    condition: str,
    fold: int,
    seed: int,
    matched_resamples: int = MATCHED_RESAMPLES,
    part: str = "structure",
) -> list[dict[str, Any]]:
    rng = random.Random(MATCHED_SEED + fold * 1000 + seed)
    id_to_position = {(row.id, row.language): i for i, row in df[["id", "language"]].iterrows()}
    available_ids = sorted(set(df["id"].astype(str)))
    length_by_id = {prop_id: text_len(rows_by_id[prop_id]) for prop_id in available_ids}
    depth_by_id = {prop_id: int(rows_by_id[prop_id]["depth"]) for prop_id in available_ids}
    family_rows = []
    for family, group in fold_manifest.groupby("family_id"):
        ids = group["id"].astype(str).tolist()
        if len(ids) < 2:
            continue
        observed_distances = []
        for language in sorted(set(df["language"])):
            positions = [id_to_position[(prop_id, language)] for prop_id in ids if (prop_id, language) in id_to_position]
            for left_i, left in enumerate(positions):
                for right in positions[left_i + 1 :]:
                    observed_distances.append(float(np.linalg.norm(z[left] - z[right])))
        if not observed_distances:
            continue
        observed = mean(observed_distances)
        depth_profile = sorted(int(rows_by_id[prop_id]["depth"]) for prop_id in ids)
        branch = top_branch(ids[0])
        lengths = [length_by_id[prop_id] for prop_id in ids]
        branch_pool = [prop_id for prop_id in available_ids if prop_id not in ids and top_branch(prop_id) == branch]
        all_pool = [prop_id for prop_id in available_ids if prop_id not in ids]
        matched = []
        for _ in range(matched_resamples):
            depth_matched = []
            for depth in depth_profile:
                available = [prop_id for prop_id in branch_pool if prop_id not in depth_matched]
                candidates = [prop_id for prop_id in available if depth_by_id[prop_id] == depth]
                if not candidates:
                    candidates = [prop_id for prop_id in all_pool if prop_id not in depth_matched and depth_by_id[prop_id] == depth]
                if not candidates:
                    candidates = [prop_id for prop_id in all_pool if prop_id not in depth_matched]
                if not candidates:
                    break
                target_len = lengths[len(depth_matched)]
                near = sorted(candidates, key=lambda prop_id: abs(length_by_id[prop_id] - target_len))[:50]
                chosen = rng.choice(near)
                depth_matched.append(chosen)
            if len(depth_matched) != len(ids):
                continue
            dists = []
            for language in sorted(set(df["language"])):
                positions = [id_to_position[(prop_id, language)] for prop_id in depth_matched if (prop_id, language) in id_to_position]
                for left_i, left in enumerate(positions):
                    for right in positions[left_i + 1 :]:
                        dists.append(float(np.linalg.norm(z[left] - z[right])))
            if dists:
                matched.append(mean(dists))
        tail = sum(1 for value in matched if value <= observed) / max(len(matched), 1)
        family_rows.append(
            {
                "condition": condition,
                "fold": fold,
                "seed": seed,
                "family_id": family,
                "family_size": len(ids),
                "top_level_branch": branch,
                "depth_profile": json.dumps(depth_profile),
                "observed_cohesion_distance": observed,
                "matched_mean_distance": mean(matched) if matched else 0.0,
                "matched_resamples": len(matched),
                "empirical_lower_tail_probability": tail,
            }
        )
    return family_rows


def evaluate_one(
    condition: str,
    fold: int,
    seed: int,
    checkpoint: Path,
    data_path: Path,
    out_root: Path,
    matched_resamples: int = MATCHED_RESAMPLES,
) -> dict[str, float]:
    manifest = load_fold_manifest(out_root / "phase2_fold_manifest.csv")
    train_ids, test_ids = sample_ids_for_fold(manifest, fold)
    all_df, text_z_all, structure_z_all, dataset = encode_rows(checkpoint, data_path, sample_ids=None)
    test_mask = all_df["id"].isin(test_ids)
    test_df = all_df[test_mask].reset_index(drop=True).copy()
    text_z_test = text_z_all[np.where(test_mask.to_numpy())[0]]
    structure_z_test = structure_z_all[np.where(test_mask.to_numpy())[0]]
    rows_by_id = {row["id"]: row for row in dataset.rows}
    train_target_classes = {
        "parent": {dataset._index_or_zero(rows_by_id[prop_id]["parent_id"]) for prop_id in train_ids},
        "successor": {dataset._index_or_zero(rows_by_id[prop_id]["next_id"]) for prop_id in train_ids},
    }
    test_df["parent_target_seen_in_training"] = test_df["parent_true"].isin(train_target_classes["parent"])
    test_df["successor_target_seen_in_training"] = test_df["successor_true"].isin(train_target_classes["successor"])
    rng = random.Random(MATCHED_SEED + fold * 1000 + seed)
    metrics: dict[str, float] = {
        "test_proposition_ids": float(len(test_ids)),
        "train_proposition_ids": float(len(train_ids)),
        "parent_seen_class_coverage": float(test_df["parent_target_seen_in_training"].mean()),
        "successor_seen_class_coverage": float(test_df["successor_target_seen_in_training"].mean()),
        "parent_seen_class_accuracy": accuracy_score(
            test_df[test_df["parent_target_seen_in_training"]]["parent_true"],
            test_df[test_df["parent_target_seen_in_training"]]["parent_pred"],
        )
        if test_df["parent_target_seen_in_training"].any()
        else 0.0,
        "successor_seen_class_accuracy": accuracy_score(
            test_df[test_df["successor_target_seen_in_training"]]["successor_true"],
            test_df[test_df["successor_target_seen_in_training"]]["successor_pred"],
        )
        if test_df["successor_target_seen_in_training"].any()
        else 0.0,
        "depth_accuracy": accuracy_score(test_df["depth_true"], test_df["depth_pred"]),
        "depth_absolute_error": float(test_df["depth_abs_error"].mean()),
        "child_count_mae": float(test_df["child_count_abs_error"].mean()),
        "child_count_rmse": math.sqrt(float(test_df["child_count_sq_error"].mean())),
        "reconstruction_loss": float(test_df["reconstruction_loss"].mean()),
        "perplexity": float(min(math.exp(float(test_df["reconstruction_loss"].mean())), 1e9)),
        "kl_text": float(test_df["kl_text"].mean()),
        "kl_structure": float(test_df["kl_structure"].mean()),
        "structure_mean_norm": float(test_df["structure_mean_norm"].mean()),
        "text_posterior_variance": float(test_df["text_posterior_variance"].mean()),
        "structure_posterior_variance": float(test_df["structure_posterior_variance"].mean()),
    }
    for part, test_z, all_z in [("text", text_z_test, text_z_all), ("structure", structure_z_test, structure_z_all)]:
        metrics.update(retrieval_metrics(test_df, test_df, test_z, test_z, part, "test_candidates"))
        metrics.update(retrieval_metrics(test_df, all_df, test_z, all_z, part, "complete_candidates"))
        metrics.update(neighbourhood_jaccard(test_df, test_z, part))
        metrics.update(relation_metrics(test_df, test_z, part, rows_by_id, rng))
    metrics.update(lexical_references(dataset.rows, set(train_ids), set(test_ids)))
    metrics["primary_structure_test_top1"] = metrics.get("structure_test_candidates_top1", 0.0)
    metrics["primary_structure_test_mrr"] = metrics.get("structure_test_candidates_mrr", 0.0)
    metrics["primary_text_test_top1"] = metrics.get("text_test_candidates_top1", 0.0)
    fold_manifest = manifest[manifest["fold"] == fold]
    family_rows = matched_family_analysis(
        test_df,
        structure_z_test,
        fold_manifest,
        rows_by_id,
        condition,
        fold,
        seed,
        matched_resamples=matched_resamples,
    )
    metrics["matched_family_count"] = float(len(family_rows))
    metrics["matched_family_mean_lower_tail_probability"] = mean([row["empirical_lower_tail_probability"] for row in family_rows]) if family_rows else 0.0
    raw_path = out_root / "raw" / condition / f"fold{fold}_seed{seed:03d}.per_proposition.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.insert(0, "condition", condition)
    test_df.insert(1, "fold", fold)
    test_df.insert(2, "seed", seed)
    test_df.to_parquet(raw_path, index=False)
    family_path = out_root / "raw" / condition / f"fold{fold}_seed{seed:03d}.matched_families.csv"
    write_csv(family_path, family_rows)
    write_json(out_root / "per_seed" / condition / f"fold{fold}_seed{seed:03d}.metrics.json", metrics)
    return metrics


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
    except Exception as exc:
        packages["scikit_learn"] = f"unavailable: {exc}"
    try:
        import pyarrow

        packages["pyarrow"] = pyarrow.__version__
    except Exception as exc:
        packages["pyarrow"] = f"unavailable: {exc}"
    return packages


def gpu_info() -> dict[str, Any]:
    info = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def write_configs(out_root: Path, args: argparse.Namespace, selected_conditions: list[Condition], folds: list[int], seeds: list[int]) -> None:
    for condition in selected_conditions:
        write_json(
            out_root / "configs" / f"{condition.name}.json",
            {
                "condition": condition.name,
                "lambda_parent": condition.lambda_parent,
                "lambda_depth": condition.lambda_depth,
                "lambda_next": condition.lambda_next,
                "lambda_child": condition.lambda_child,
                "lambda_language_alignment": 0.0,
                "folds": folds,
                "seeds": seeds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "text_latent_dim": args.text_latent_dim,
                "structure_latent_dim": args.structure_latent_dim,
                "language_embedding_dim": args.language_embedding_dim,
                "beta": args.beta,
                "beta_text": args.beta_text,
                "beta_structure": args.beta_structure,
                "lr": args.lr,
                "device": args.device,
                "matched_resamples": args.matched_resamples,
            },
        )


def summarise(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted((out_root / "per_seed").glob("*/*.metrics.json")):
        condition = path.parent.name
        parts = path.name.replace(".metrics.json", "").split("_")
        fold = int(parts[0].replace("fold", ""))
        seed = int(parts[1].replace("seed", ""))
        rows.append({"condition": condition, "fold": fold, "seed": seed, **read_json(path)})
    seed_df = pd.DataFrame(rows).sort_values(["condition", "fold", "seed"])
    seed_df.to_csv(out_root / "phase2_seed_fold_results.csv", index=False)
    raw = pd.concat([pd.read_parquet(path) for path in sorted((out_root / "raw").glob("*/*.per_proposition.parquet"))], ignore_index=True)
    raw.to_parquet(out_root / "phase2_per_proposition_results.parquet", index=False)
    family_paths = sorted((out_root / "raw").glob("*/*.matched_families.csv"))
    if family_paths:
        family_df = pd.concat([pd.read_csv(path) for path in family_paths], ignore_index=True)
        family_df.to_csv(out_root / "phase2_matched_family_results.csv", index=False)
    summary_rows = []
    metric_cols = [col for col in seed_df.columns if col not in {"condition", "fold", "seed"}]
    for condition, group in seed_df.groupby("condition", sort=True):
        for metric in metric_cols:
            values = group[metric].dropna().astype(float).tolist()
            if values:
                summary_rows.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "run_count": len(values),
                        "mean": mean(values),
                        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values),
                        "max": max(values),
                    }
                )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "phase2_summary.csv", index=False)
    return seed_df, summary_df


def write_figures(out_root: Path, seed_df: pd.DataFrame) -> None:
    metrics = [
        ("structure_test_candidates_top1", "Structure Top-1, test candidates"),
        ("structure_complete_candidates_top1", "Structure Top-1, complete candidates"),
        ("text_test_candidates_top1", "Text Top-1, test candidates"),
        ("structure_sibling_vs_matched_unrelated_contrast", "Sibling contrast"),
    ]
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in metrics:
        if metric not in seed_df:
            continue
        groups = seed_df.groupby("condition")[metric]
        labels = sorted(groups.groups)
        means = [groups.mean()[label_name] for label_name in labels]
        errs = [groups.std()[label_name] for label_name in labels]
        plt.figure(figsize=(7.5, 4.2))
        plt.bar(range(len(labels)), means, yerr=errs, capsize=4, color="#4C78A8")
        plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}.png", dpi=600)
        plt.close()


def summary_value(summary: pd.DataFrame, condition: str, metric: str) -> tuple[float, float]:
    row = summary[(summary["condition"] == condition) & (summary["metric"] == metric)].iloc[0]
    return float(row["mean"]), float(row["sample_sd"])


def optional_summary_mean(summary: pd.DataFrame, condition: str, metric: str) -> float | None:
    rows = summary[(summary["condition"] == condition) & (summary["metric"] == metric)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["mean"])


def fmt(value: tuple[float, float]) -> str:
    return f"{value[0]:.4f} +/- {value[1]:.4f}"


def fmt_optional(value: float | None) -> str:
    return "not run" if value is None else f"{value:.4f}"


def write_report(out_root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 2 Family-Held-Out Generalisation Report",
        "",
        "This evaluation uses deterministic five-fold immediate-parent-family holdout. German and English samples for each proposition ID remain in the same fold. Vocabularies are built from training-fold texts only; held-out-only tokens use the normal `<unk>` path.",
        "",
        "## Summary",
        "",
        "| condition | structure test Top-1 | structure complete Top-1 | structure MRR | text test Top-1 | depth acc. | child MAE | sibling contrast |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        if condition not in set(summary["condition"]):
            continue
        lines.append(
            f"| `{condition}` | {fmt(summary_value(summary, condition, 'structure_test_candidates_top1'))} | "
            f"{fmt(summary_value(summary, condition, 'structure_complete_candidates_top1'))} | "
            f"{fmt(summary_value(summary, condition, 'structure_test_candidates_mrr'))} | "
            f"{fmt(summary_value(summary, condition, 'text_test_candidates_top1'))} | "
            f"{fmt(summary_value(summary, condition, 'depth_accuracy'))} | "
            f"{fmt(summary_value(summary, condition, 'child_count_mae'))} | "
            f"{fmt(summary_value(summary, condition, 'structure_sibling_vs_matched_unrelated_contrast'))} |"
        )
    full = optional_summary_mean(summary, "full_model", "structure_test_candidates_top1")
    no_succ = optional_summary_mean(summary, "no_successor", "structure_test_candidates_top1")
    full_text = optional_summary_mean(summary, "full_model", "text_test_candidates_top1")
    sibling = optional_summary_mean(summary, "full_model", "structure_sibling_vs_matched_unrelated_contrast")
    lines.extend(
        [
            "",
            "## Required questions",
            "",
            f"1. Same-ID bilingual retrieval under unseen-family splitting is estimated by full_model structure test-candidate Top-1 = {fmt_optional(full)}.",
            f"2. Removing successor gives no_successor structure test-candidate Top-1 = {fmt_optional(no_succ)}; compare with full_model before treating any difference as meaningful.",
            f"3. The full_model text-latent Top-1 is {fmt_optional(full_text)}; structure-vs-text generalisation should be read from the paired fold/seed table.",
            f"4. Sibling cohesion in unseen families is summarised by full_model structure sibling-versus-matched-unrelated contrast = {fmt_optional(sibling)}; matched family tail probabilities are in `phase2_matched_family_results.csv`.",
            f"5. Retained-corpus performance is overstated where Phase 1 full_model retrieval exceeds held-out full_model retrieval; Phase 2 tests unseen texts and immediate-parent families.",
            "6. Exact parent and successor classification cannot be validly interpreted for held-out targets whose class IDs were absent from training; the report separates seen-class coverage and seen-class accuracy.",
            "",
            "## Implications for publication claims",
            "",
            "Retained-corpus findings: Phase 1 remains the fitted-corpus analysis of objective contributions.",
            "",
            "Demonstrated held-out generalisation: claims should be limited to metrics that survive this unseen-family split, especially same-ID retrieval, depth, child-count, and sibling-cohesion contrasts.",
            "",
            "Unresolved generalisation questions: exact parent/successor prediction for unseen class IDs is not a valid held-out classification task without a different label formulation; larger seed sweeps can refine uncertainty without changing the fixed folds.",
        ]
    )
    (out_root / "phase2_holdout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def leakage_checks(out_root: Path, data_path: Path) -> list[str]:
    manifest = load_fold_manifest(out_root / "phase2_fold_manifest.csv")
    rows = read_json(data_path)
    rows_by_id = {row["id"]: row for row in rows}
    checks = []
    for fold in sorted(manifest["fold"].unique()):
        train_ids, test_ids = sample_ids_for_fold(manifest, int(fold))
        train_set = set(train_ids)
        test_set = set(test_ids)
        if train_set & test_set:
            raise AssertionError(f"Fold {fold}: train/test overlap")
        checks.append(f"Fold {fold}: no proposition ID occurs in both train and test.")
        family_counts = manifest.groupby("family_id")["fold"].nunique()
        split_families = family_counts[family_counts > 1]
        if not split_families.empty:
            raise AssertionError(f"Immediate-parent families split across folds: {split_families.index.tolist()[:10]}")
        checks.append(f"Fold {fold}: no immediate-parent family is divided across folds.")
        train_texts = [text for prop_id in train_set for text in row_texts(rows_by_id[prop_id]).values()]
        expected_vocab = Vocabulary.build(train_texts).token_to_id
        for vocab_path in sorted((out_root / "checkpoints").glob(f"*/fold{fold}_seed*.vocab.json")):
            observed = read_json(vocab_path)
            if observed != expected_vocab:
                raise AssertionError(f"Vocabulary mismatch for {vocab_path}")
        test_tokens = {token for prop_id in test_set for text in row_texts(rows_by_id[prop_id]).values() for token in tokenize(text)}
        train_tokens = {token for text in train_texts for token in tokenize(text)}
        checks.append(f"Fold {fold}: {len(test_tokens - train_tokens)} held-out-only tokens map through `<unk>`.")
    checks.append("German and English counterparts are represented by one proposition ID and therefore share the same fold.")
    checks.append("Parent/successor heads use a fixed global proposition-ID class space; seen/unseen target coverage is reported separately.")
    checks.append("No normalisation/calibration objects are fitted outside the training vocabulary builders; model selection is not performed from test metrics.")
    (out_root / "phase2_leakage_checks.md").write_text("# Phase 2 Leakage Checks\n\n" + "\n".join(f"- {line}" for line in checks) + "\n", encoding="utf-8")
    return checks


def run(args: argparse.Namespace) -> None:
    out_root = args.out_root.resolve()
    args.device = normalise_device(args.device)
    ensure_layout(out_root)
    if not (out_root / "phase2_protected_hashes_before.json").exists():
        write_json(out_root / "phase2_protected_hashes_before.json", protected_hashes())
    manifest_path = out_root / "phase2_fold_manifest.csv"
    if manifest_path.exists():
        manifest = load_fold_manifest(manifest_path)
    else:
        manifest = make_fold_manifest(args.data, manifest_path)
    folds = parse_ints(args.folds)
    seeds = parse_ints(args.seeds)
    selected = [CONDITIONS[name] for name in args.conditions.split(",") if name]
    write_configs(out_root, args, selected, folds, seeds)
    command_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    write_json(
        out_root / "phase2_config_manifest.json",
        {
            "git_commit": git_output(["rev-parse", "HEAD"]),
            "git_status_at_start": git_output(["status", "--short"]),
            "git_branch": git_output(["branch", "--show-current"]),
            "package_versions": package_versions(),
            "gpu_info": gpu_info(),
            "conditions": [condition.name for condition in selected],
            "folds": folds,
            "seeds": seeds,
            "matched_resamples": args.matched_resamples,
            "fold_manifest": str(manifest_path),
        },
    )
    for fold in folds:
        train_ids, _test_ids = sample_ids_for_fold(manifest, fold)
        ids_file = out_root / "ids" / f"fold{fold}_train_ids.json"
        write_json(ids_file, train_ids)
        for condition in selected:
            for seed in seeds:
                checkpoint = out_root / "checkpoints" / condition.name / f"fold{fold}_seed{seed:03d}.pt"
                metric_path = out_root / "per_seed" / condition.name / f"fold{fold}_seed{seed:03d}.metrics.json"
                command = train_command(args, condition, fold, seed, checkpoint, ids_file)
                command_lines.append(" ".join(command))
                command_lines.append(
                    " ".join(
                        [
                            "python3",
                            "tools/phase2_family_holdout.py",
                            "eval-one",
                            "--condition",
                            condition.name,
                            "--fold",
                            str(fold),
                            "--seed",
                            str(seed),
                            "--checkpoint",
                            str(checkpoint),
                            "--data",
                            str(args.data),
                            "--out-root",
                            str(out_root),
                            "--matched-resamples",
                            str(args.matched_resamples),
                        ]
                    )
                )
                if args.skip_existing and checkpoint.exists() and metric_path.exists():
                    print(f"skipping {condition.name}/fold{fold}/seed{seed:03d}", flush=True)
                    continue
                print(f"training {condition.name}/fold{fold}/seed{seed:03d}", flush=True)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                log_path = out_root / "logs" / condition.name / f"fold{fold}_seed{seed:03d}.train.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as handle:
                    subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=handle, stderr=subprocess.STDOUT)
                evaluate_one(condition.name, fold, seed, checkpoint, args.data, out_root, matched_resamples=args.matched_resamples)
    (out_root / "phase2_commands.sh").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    leakage_checks(out_root, args.data)
    seed_df, summary_df = summarise(out_root)
    write_figures(out_root, seed_df)
    write_report(out_root, summary_df)


def eval_one(args: argparse.Namespace) -> None:
    evaluate_one(
        args.condition,
        args.fold,
        args.seed,
        args.checkpoint,
        args.data,
        args.out_root.resolve(),
        matched_resamples=args.matched_resamples,
    )


def verify(args: argparse.Namespace) -> None:
    out_root = args.out_root.resolve()
    manifest = load_fold_manifest(out_root / "phase2_fold_manifest.csv")
    leakage = leakage_checks(out_root, DATA_PATH)
    raw_paths = sorted((out_root / "raw").glob("*/*.per_proposition.parquet"))
    if not raw_paths:
        raise FileNotFoundError("No raw per-proposition files found")
    raw = pd.concat([pd.read_parquet(path) for path in raw_paths], ignore_index=True)
    reported = pd.read_csv(out_root / "phase2_seed_fold_results.csv")
    checks = [*leakage]
    recomputed_rows = []
    for (condition, fold, seed), group in raw.groupby(["condition", "fold", "seed"], sort=True):
        row = {
            "condition": condition,
            "fold": int(fold),
            "seed": int(seed),
            "depth_accuracy": float((group["depth_true"] == group["depth_pred"]).mean()),
            "depth_absolute_error": float(group["depth_abs_error"].mean()),
            "child_count_mae": float(group["child_count_abs_error"].mean()),
            "child_count_rmse": math.sqrt(float(group["child_count_sq_error"].mean())),
            "reconstruction_loss": float(group["reconstruction_loss"].mean()),
            "structure_test_candidates_top1": float(group.filter(like="structure_test_candidates_retrieval_top1_").stack().mean()),
            "structure_test_candidates_mrr": float(group.filter(like="structure_test_candidates_retrieval_mrr_").stack().mean()),
            "text_test_candidates_top1": float(group.filter(like="text_test_candidates_retrieval_top1_").stack().mean()),
            "structure_test_wider_neighbourhood_jaccard_k10": float(group.filter(like="structure_test_neighbourhood_jaccard_k10_").stack().mean()),
        }
        recomputed_rows.append(row)
        report_row = reported[(reported["condition"] == condition) & (reported["fold"] == int(fold)) & (reported["seed"] == int(seed))].iloc[0]
        for key, value in row.items():
            if key in {"condition", "fold", "seed"}:
                continue
            if abs(float(report_row[key]) - value) > 1e-6:
                raise AssertionError(f"Mismatch for {condition}/fold{fold}/seed{seed}/{key}")
    summary = pd.read_csv(out_root / "phase2_summary.csv")
    for (condition, metric), group in summary.groupby(["condition", "metric"]):
        if metric not in reported.columns:
            continue
        values = reported[reported["condition"] == condition][metric].dropna().astype(float).tolist()
        if not values:
            continue
        if abs(float(group.iloc[0]["mean"]) - mean(values)) > 1e-6:
            raise AssertionError(f"Summary mean mismatch for {condition}/{metric}")
        sd_value = stdev(values) if len(values) > 1 else 0.0
        if abs(float(group.iloc[0]["sample_sd"]) - sd_value) > 1e-6:
            raise AssertionError(f"Summary SD mismatch for {condition}/{metric}")
    before = read_json(out_root / "phase2_protected_hashes_before.json")
    after = protected_hashes()
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    if changed:
        raise AssertionError(f"Protected canonical files changed: {changed[:20]}")
    checks.append(f"Verified {len(recomputed_rows)} condition-fold-seed rows from raw parquet files.")
    checks.append(f"Verified {len(summary)} reported summary rows where raw/seed metrics apply.")
    checks.append(f"Verified {len(before)} protected canonical file hashes unchanged.")
    lines = [
        "# Phase 2 Verification Report",
        "",
        *[f"- {check}" for check in checks],
        f"- Fold manifest rows: {len(manifest)}.",
        f"- Raw per-proposition rows: {len(raw)}.",
        f"- Git branch: {git_output(['branch', '--show-current'])}.",
        f"- Git commit during verification: {git_output(['rev-parse', 'HEAD'])}.",
        "",
        "## Git status",
        "",
        "```",
        git_output(["status", "--short"]),
        "```",
        "",
        "## Git diff stat",
        "",
        "```",
        git_output(["diff", "--stat"]),
        "```",
    ]
    (out_root / "phase2_verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified {len(recomputed_rows)} condition-fold-seed rows from raw parquet files.")
    print(f"Verified {len(before)} protected canonical file hashes unchanged.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 family-held-out generalisation evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    run_parser.add_argument("--data", type=Path, default=DATA_PATH)
    run_parser.add_argument("--conditions", default="full_model,no_successor,reconstruction_only")
    run_parser.add_argument("--folds", default="0,1,2,3,4")
    run_parser.add_argument("--seeds", default="0,1,2")
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
    run_parser.add_argument("--matched-resamples", type=int, default=MATCHED_RESAMPLES)
    run_parser.set_defaults(func=run)
    eval_parser = sub.add_parser("eval-one")
    eval_parser.add_argument("--condition", required=True)
    eval_parser.add_argument("--fold", type=int, required=True)
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--data", type=Path, default=DATA_PATH)
    eval_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    eval_parser.add_argument("--matched-resamples", type=int, default=MATCHED_RESAMPLES)
    eval_parser.set_defaults(func=eval_one)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
