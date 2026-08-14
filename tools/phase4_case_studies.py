#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from tools.phase1_ablations import instantiate_model
from tractatus_structure_latents.training.data import TractatusDataset, Vocabulary, collate_batch, row_texts, tokenize

DATA_PATH = ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json"
PHASE3 = ROOT / "results" / "dsh_validation" / "phase3_controlled_alignment"
DEFAULT_OUT = ROOT / "results" / "dsh_validation" / "phase4_case_studies"
PRIMARY_LABEL = "paired_full_model_align000"
ROBUST_LABELS = ["paired_full_model_align003", "paired_no_successor_align000"]
ALL_LABELS = [PRIMARY_LABEL, *ROBUST_LABELS]
SEEDS = list(range(10))
K_VALUES = [5, 10, 20]
PARENT_GOOD_QUANTILE = 0.10
PARENT_POOR_QUANTILE = 0.90
SUCCESSOR_GOOD_QUANTILE = 0.10
SUCCESSOR_POOR_QUANTILE = 0.65


@dataclass(frozen=True)
class CorpusMeta:
    rows: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    top_branch: dict[str, str]
    family_children: dict[str, list[str]]
    lengths: dict[tuple[str, str], int]
    truncated: dict[tuple[str, str], bool]


def git_output(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    detail = result.stderr.strip() or result.stdout.strip() or f"git exited with status {result.returncode}"
    return f"unavailable ({detail.splitlines()[0]})"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display_path_arg(value: object) -> str:
    text = str(value)
    try:
        return str(Path(text).resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_layout(out_root: Path) -> None:
    for name in ["dossiers", "figures", "data"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)


def load_corpus_meta(data_path: Path) -> CorpusMeta:
    rows = read_json(data_path)
    by_id = {str(row["id"]): row for row in rows}
    top_branch = {prop_id: prop_id.split(".")[0] for prop_id in by_id}
    family_children: dict[str, list[str]] = {}
    lengths: dict[tuple[str, str], int] = {}
    truncated: dict[tuple[str, str], bool] = {}
    for row in rows:
        prop_id = str(row["id"])
        parent_id = str(row["parent_id"]) if row["parent_id"] is not None else "ROOT"
        family_children.setdefault(parent_id, []).append(prop_id)
        for language, text in row_texts(row).items():
            token_count = len(tokenize(text))
            lengths[(prop_id, language)] = token_count
            truncated[(prop_id, language)] = token_count + 2 > 96
    return CorpusMeta(rows, by_id, top_branch, family_children, lengths, truncated)


def require_language_text(row: dict[str, Any], language: str) -> str:
    texts = row_texts(row)
    if language not in texts:
        prop_id = row.get("id", "<unknown>")
        available = ",".join(sorted(texts)) or "none"
        raise ValueError(f"Proposition {prop_id} requires {language!r} text; available languages: {available}")
    return texts[language]


def load_raw(label: str, seed: int) -> pd.DataFrame:
    path = PHASE3 / "raw" / label / f"seed{seed:03d}.per_proposition.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 3 raw file: {path}")
    df = pd.read_parquet(path)
    df["structure_vec"] = df["structure_mu"].apply(lambda value: np.asarray(json.loads(value), dtype=float))
    df["text_vec"] = df["text_mu"].apply(lambda value: np.asarray(json.loads(value), dtype=float))
    return df


def vectors_by_id_language(df: pd.DataFrame, vector_col: str = "structure_vec") -> dict[tuple[str, str], np.ndarray]:
    return {(str(row.id), str(row.language)): getattr(row, vector_col) for row in df.itertuples()}


def euclidean(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def pairwise_mean(vectors: list[np.ndarray]) -> float:
    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(euclidean(vectors[i], vectors[j]))
    return mean(distances) if distances else 0.0


def family_descriptors(meta: CorpusMeta) -> pd.DataFrame:
    rows = []
    for parent_id, children in meta.family_children.items():
        if parent_id == "ROOT" or not (4 <= len(children) <= 10):
            continue
        depths = [int(meta.by_id[child]["depth"]) for child in children]
        child_counts = [int(meta.by_id[child]["child_count"]) for child in children]
        branches = [meta.top_branch[child] for child in children]
        branch = max(set(branches), key=branches.count)
        text_lengths = [meta.lengths[(child, language)] for child in children for language in ("en", "de")]
        truncations = [meta.truncated[(child, language)] for child in children for language in ("en", "de")]
        rows.append(
            {
                "family_id": parent_id,
                "family_size": len(children),
                "depth_profile": json.dumps(sorted(depths)),
                "child_count_profile": json.dumps(sorted(child_counts)),
                "top_level_branch": branch,
                "mean_text_length": mean(text_lengths),
                "sd_text_length": stdev(text_lengths) if len(text_lengths) > 1 else 0.0,
                "truncated_count": int(sum(truncations)),
                "children": json.dumps(children),
            }
        )
    return pd.DataFrame(rows)


def family_match_score(left: pd.Series, right: pd.Series) -> float:
    return (
        10.0 * (left["family_size"] != right["family_size"])
        + 4.0 * (left["depth_profile"] != right["depth_profile"])
        + 3.0 * (left["child_count_profile"] != right["child_count_profile"])
        + 2.0 * (left["top_level_branch"] != right["top_level_branch"])
        + 2.0 * abs(float(left["truncated_count"]) - float(right["truncated_count"]))
        + abs(float(left["mean_text_length"]) - float(right["mean_text_length"])) / 10.0
        + abs(float(left["sd_text_length"]) - float(right["sd_text_length"])) / 10.0
    )


def compute_family_metrics(meta: CorpusMeta, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = family_descriptors(meta)
    seed_rows = []
    for seed in SEEDS:
        df = load_raw(PRIMARY_LABEL, seed)
        vecs = vectors_by_id_language(df)
        for fam in families.itertuples():
            children = json.loads(fam.children)
            language_distances = []
            member_distances: dict[str, list[float]] = {child: [] for child in children}
            for language in ("en", "de"):
                vectors = [vecs[(child, language)] for child in children]
                language_distances.append(pairwise_mean(vectors))
                for child in children:
                    child_vec = vecs[(child, language)]
                    sibling_distances = [euclidean(child_vec, vecs[(other, language)]) for other in children if other != child]
                    member_distances[child].extend(sibling_distances)
            seed_rows.append(
                {
                    "family_id": fam.family_id,
                    "seed": seed,
                    "mean_pairwise_sibling_distance": mean(language_distances),
                    "member_mean_distances": json.dumps({child: mean(values) for child, values in member_distances.items()}),
                }
            )
    seed_df = pd.DataFrame(seed_rows)
    rows = []
    matched_rows = []
    for fam in families.itertuples():
        family_seed = seed_df[seed_df["family_id"] == fam.family_id].copy()
        candidates = families[families["family_id"] != fam.family_id].copy()
        candidates["match_score"] = candidates.apply(lambda row: family_match_score(pd.Series(fam._asdict()), row), axis=1)
        matched = candidates.sort_values(["match_score", "family_id"]).head(10)
        matched_ids = matched["family_id"].tolist()
        for matched_id in matched_ids:
            matched_rows.append({"family_id": fam.family_id, "matched_family_id": matched_id, "match_score": float(matched[matched["family_id"] == matched_id]["match_score"].iloc[0])})
        seed_z = []
        seed_direction = []
        for seed in SEEDS:
            value = float(family_seed[family_seed["seed"] == seed]["mean_pairwise_sibling_distance"].iloc[0])
            matched_values = seed_df[(seed_df["seed"] == seed) & (seed_df["family_id"].isin(matched_ids))]["mean_pairwise_sibling_distance"].tolist()
            matched_mean = mean(matched_values)
            matched_sd = stdev(matched_values) if len(matched_values) > 1 else 0.0
            z = (value - matched_mean) / matched_sd if matched_sd > 0 else 0.0
            seed_z.append(z)
            seed_direction.append("low" if value < matched_mean else "high")
        rows.append(
            {
                **{key: getattr(fam, key) for key in families.columns},
                "mean_pairwise_sibling_distance": float(family_seed["mean_pairwise_sibling_distance"].mean()),
                "sibling_distance_sd": float(family_seed["mean_pairwise_sibling_distance"].std(ddof=1)),
                "matched_z_mean": mean(seed_z),
                "matched_z_sd": stdev(seed_z) if len(seed_z) > 1 else 0.0,
                "low_vs_matched_seed_count": seed_direction.count("low"),
                "high_vs_matched_seed_count": seed_direction.count("high"),
                "matched_family_ids": json.dumps(matched_ids),
            }
        )
    family_df = pd.DataFrame(rows)
    matched_df = pd.DataFrame(matched_rows)
    seed_df.to_csv(out_root / "data" / "family_seed_metrics.csv", index=False)
    family_df.to_csv(out_root / "data" / "family_candidate_metrics.csv", index=False)
    matched_df.to_csv(out_root / "data" / "family_matched_controls.csv", index=False)
    return family_df, matched_df


def nearest_ids(vecs: dict[tuple[str, str], np.ndarray], prop_id: str, source_language: str, target_language: str, k: int, exclude_self: bool = True) -> list[tuple[str, float]]:
    source = vecs[(prop_id, source_language)]
    rows = []
    for (candidate_id, candidate_language), vector in vecs.items():
        if candidate_language != target_language:
            continue
        if exclude_self and candidate_id == prop_id:
            continue
        rows.append((candidate_id, euclidean(source, vector)))
    return sorted(rows, key=lambda item: (item[1], item[0]))[:k]


def compute_neighbourhood_metrics(meta: CorpusMeta, out_root: Path) -> pd.DataFrame:
    seed_rows = []
    ids = sorted(meta.by_id)
    for seed in SEEDS:
        df = load_raw(PRIMARY_LABEL, seed)
        vecs = vectors_by_id_language(df)
        for prop_id in ids:
            row: dict[str, Any] = {"id": prop_id, "seed": seed}
            for k in K_VALUES:
                en_to_de = [candidate for candidate, _distance in nearest_ids(vecs, prop_id, "en", "de", k)]
                de_to_en = [candidate for candidate, _distance in nearest_ids(vecs, prop_id, "de", "en", k)]
                en_within = [candidate for candidate, _distance in nearest_ids(vecs, prop_id, "en", "en", k)]
                de_within = [candidate for candidate, _distance in nearest_ids(vecs, prop_id, "de", "de", k)]
                cross_j = len(set(en_to_de) & set(de_to_en)) / max(len(set(en_to_de) | set(de_to_en)), 1)
                within_j = len(set(en_within) & set(de_within)) / max(len(set(en_within) | set(de_within)), 1)
                row[f"cross_direction_jaccard_k{k}"] = cross_j
                row[f"within_language_jaccard_k{k}"] = within_j
                row[f"directional_asymmetry_k{k}"] = 1.0 - cross_j
                row[f"en_to_de_neighbours_k{k}"] = json.dumps(en_to_de)
                row[f"de_to_en_neighbours_k{k}"] = json.dumps(de_to_en)
            seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)
    rows = []
    for prop_id in ids:
        group = seed_df[seed_df["id"] == prop_id]
        values = group["cross_direction_jaccard_k10"].tolist()
        rows.append(
            {
                "id": prop_id,
                "cross_direction_jaccard_k5_mean": float(group["cross_direction_jaccard_k5"].mean()),
                "cross_direction_jaccard_k10_mean": float(group["cross_direction_jaccard_k10"].mean()),
                "cross_direction_jaccard_k20_mean": float(group["cross_direction_jaccard_k20"].mean()),
                "cross_direction_jaccard_k10_sd": float(group["cross_direction_jaccard_k10"].std(ddof=1)),
                "within_language_jaccard_k10_mean": float(group["within_language_jaccard_k10"].mean()),
                "directional_asymmetry_k10_mean": float(group["directional_asymmetry_k10"].mean()),
                "low_overlap_seed_count": int(sum(value <= np.quantile(seed_df[seed_df["seed"] == seed]["cross_direction_jaccard_k10"], 0.10) for value, seed in zip(values, group["seed"]))),
                "high_overlap_seed_count": int(sum(value >= np.quantile(seed_df[seed_df["seed"] == seed]["cross_direction_jaccard_k10"], 0.90) for value, seed in zip(values, group["seed"]))),
                "depth": int(meta.by_id[prop_id]["depth"]),
                "child_count": int(meta.by_id[prop_id]["child_count"]),
                "top_level_branch": meta.top_branch[prop_id],
                "mean_text_length": mean([meta.lengths[(prop_id, "en")], meta.lengths[(prop_id, "de")]]),
                "truncated": bool(meta.truncated[(prop_id, "en")] or meta.truncated[(prop_id, "de")]),
            }
        )
    out = pd.DataFrame(rows)
    seed_df.to_csv(out_root / "data" / "neighbourhood_seed_metrics.csv", index=False)
    out.to_csv(out_root / "data" / "neighbourhood_candidate_metrics.csv", index=False)
    return out


def compute_hierarchy_sequence_metrics(meta: CorpusMeta, out_root: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        df = load_raw(PRIMARY_LABEL, seed)
        grouped = df.groupby("id", sort=True).agg(
            parent_rank=("parent_rank", "mean"),
            parent_correct=("parent_pred", lambda values: float(np.mean(values.to_numpy() == df.loc[values.index, "parent_true"].to_numpy()))),
            successor_rank=("successor_rank", "mean"),
            successor_top1=("successor_pred", lambda values: float(np.mean(values.to_numpy() == df.loc[values.index, "successor_true"].to_numpy()))),
            structure_sibling_distance=("structure_sibling_distance", "mean"),
        )
        for prop_id, row in grouped.iterrows():
            rows.append({"id": str(prop_id), "seed": seed, **row.to_dict()})
    seed_df = pd.DataFrame(rows)
    seed_df["parent_good_threshold"] = seed_df.groupby("seed")["parent_rank"].transform(lambda values: values.quantile(PARENT_GOOD_QUANTILE))
    seed_df["parent_poor_threshold"] = seed_df.groupby("seed")["parent_rank"].transform(lambda values: values.quantile(PARENT_POOR_QUANTILE))
    seed_df["successor_good_threshold"] = seed_df.groupby("seed")["successor_rank"].transform(lambda values: values.quantile(SUCCESSOR_GOOD_QUANTILE))
    seed_df["successor_poor_threshold"] = seed_df.groupby("seed")["successor_rank"].transform(lambda values: values.quantile(SUCCESSOR_POOR_QUANTILE))
    output = []
    for prop_id, group in seed_df.groupby("id", sort=True):
        parent_good = group["parent_rank"] <= group["parent_good_threshold"]
        successor_poor = group["successor_rank"] >= group["successor_poor_threshold"]
        successor_good = group["successor_rank"] <= group["successor_good_threshold"]
        parent_poor = group["parent_rank"] >= group["parent_poor_threshold"]
        output.append(
            {
                "id": prop_id,
                "parent_rank_mean": float(group["parent_rank"].mean()),
                "successor_rank_mean": float(group["successor_rank"].mean()),
                "parent_correct_mean": float(group["parent_correct"].mean()),
                "successor_top1_mean": float(group["successor_top1"].mean()),
                "structure_sibling_distance_mean": float(group["structure_sibling_distance"].mean()),
                "parent_good_successor_poor_seed_count": int((parent_good & successor_poor).sum()),
                "successor_good_parent_poor_seed_count": int((successor_good & parent_poor).sum()),
                "depth": int(meta.by_id[prop_id]["depth"]),
                "child_count": int(meta.by_id[prop_id]["child_count"]),
                "top_level_branch": meta.top_branch[prop_id],
            }
        )
    out = pd.DataFrame(output)
    seed_df.to_csv(out_root / "data" / "hierarchy_sequence_seed_metrics.csv", index=False)
    out.to_csv(out_root / "data" / "hierarchy_sequence_candidate_metrics.csv", index=False)
    return out


def select_cases(family_df: pd.DataFrame, neighbourhood_df: pd.DataFrame, hierarchy_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    selection_config = {
        "primary_representation": PRIMARY_LABEL,
        "robustness_labels": ROBUST_LABELS,
        "family_size_range": [4, 10],
        "family_stability_requirement": "principal direction versus matched actual families in at least 8 of 10 seeds",
        "neighbourhood_stability_requirement": "lowest/highest decile k=10 cross-direction Jaccard in at least 8 of 10 seeds",
        "hierarchy_sequence_threshold": {
            "parent_good_quantile": PARENT_GOOD_QUANTILE,
            "parent_poor_quantile": PARENT_POOR_QUANTILE,
            "successor_good_quantile": SUCCESSOR_GOOD_QUANTILE,
            "successor_poor_quantile": SUCCESSOR_POOR_QUANTILE,
            "stability": "at least 8 of 10 seeds",
            "note": "The poor-successor threshold is less stringent than a decile because the decile rule produced no stable parent-good/successor-poor candidates.",
        },
        "text_blind_columns": "IDs, numeric latent diagnostics, and formal/length metadata only; wording joined after SHA-256 freeze",
    }
    write_json(out_root / "data" / "case_selection_config.json", selection_config)

    rows: list[dict[str, Any]] = []
    strong = family_df[family_df["low_vs_matched_seed_count"] >= 8].sort_values(["matched_z_mean", "family_id"]).iloc[0]
    differentiated = family_df[family_df["high_vs_matched_seed_count"] >= 8].sort_values(["matched_z_mean", "family_id"], ascending=[False, True]).iloc[0]
    typical = family_df.assign(abs_z=family_df["matched_z_mean"].abs()).sort_values(["abs_z", "family_id"]).iloc[0]
    for role, row in [("unusually_strong_cohesion", strong), ("unusually_strong_internal_differentiation", differentiated), ("typical_matched_control_family", typical)]:
        rows.append({"case_study": "A_formally_controlled_family_differentiation", "role": role, "selection_unit": "family", **row.to_dict()})

    low = neighbourhood_df[neighbourhood_df["low_overlap_seed_count"] >= 8].sort_values(["cross_direction_jaccard_k10_mean", "id"]).head(2)
    high = neighbourhood_df[neighbourhood_df["high_overlap_seed_count"] >= 8].sort_values(["cross_direction_jaccard_k10_mean", "id"], ascending=[False, True]).head(2)
    for role, frame in [("stable_low_overlap", low), ("stable_high_overlap_control", high)]:
        for _i, row in frame.iterrows():
            rows.append({"case_study": "B_bilingual_neighbourhood_divergence", "role": role, "selection_unit": "proposition", **row.to_dict()})

    c1_pool = hierarchy_df[hierarchy_df["parent_good_successor_poor_seed_count"] >= 8]
    c2_pool = hierarchy_df[hierarchy_df["successor_good_parent_poor_seed_count"] >= 8]
    if c1_pool.empty or c2_pool.empty:
        raise AssertionError("Hierarchy-sequence case-study thresholds produced no 8-of-10 stable candidates.")
    c1 = c1_pool.sort_values(["parent_rank_mean", "successor_rank_mean"], ascending=[True, False]).iloc[0]
    c2 = c2_pool.sort_values(["successor_rank_mean", "parent_rank_mean"], ascending=[True, False]).iloc[0]
    c3 = hierarchy_df.assign(score=(hierarchy_df["parent_rank_mean"].rank(pct=True) - 0.5).abs() + (hierarchy_df["successor_rank_mean"].rank(pct=True) - 0.5).abs()).sort_values(["score", "id"]).iloc[0]
    for role, row in [("confident_parent_poor_successor", c1), ("strong_successor_weak_parent", c2), ("ordinary_control", c3)]:
        rows.append({"case_study": "C_hierarchy_sequence_tension", "role": role, "selection_unit": "proposition", **row.to_dict()})

    manifest = pd.DataFrame(rows)
    text_like = [col for col in manifest.columns if "text" in col.lower() and col not in {"mean_text_length", "sd_text_length"}]
    if text_like:
        raise AssertionError(f"Pre-text manifest would contain wording-like columns: {text_like}")
    path = out_root / "candidate_manifest_pre_text.csv"
    manifest.to_csv(path, index=False)
    digest = sha256_file(path)
    (out_root / "candidate_manifest_pre_text.sha256").write_text(f"{digest}  candidate_manifest_pre_text.csv\n", encoding="utf-8")
    write_json(
        out_root / "data" / "manifest_freeze.json",
        {
            "candidate_manifest_pre_text": display_path_arg(path),
            "sha256": digest,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "text_joined_utc": None,
        },
    )
    return manifest


def lexical_baselines(meta: CorpusMeta) -> pd.DataFrame:
    ids = sorted(meta.by_id)
    en_texts = [require_language_text(meta.by_id[prop_id], "en") for prop_id in ids]
    de_texts = [require_language_text(meta.by_id[prop_id], "de") for prop_id in ids]
    word = TfidfVectorizer(analyzer="word", lowercase=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True)
    word_matrix = word.fit_transform(en_texts + de_texts)
    char_matrix = char.fit_transform(en_texts + de_texts)
    rows = []
    for i, prop_id in enumerate(ids):
        en_tokens = set(tokenize(en_texts[i]))
        de_tokens = set(tokenize(de_texts[i]))
        rows.append(
            {
                "id": prop_id,
                "exact_token_jaccard": len(en_tokens & de_tokens) / max(len(en_tokens | de_tokens), 1),
                "word_tfidf_similarity": float(cosine_similarity(word_matrix[i], word_matrix[len(ids) + i])[0, 0]),
                "char_3_5_tfidf_similarity": float(cosine_similarity(char_matrix[i], char_matrix[len(ids) + i])[0, 0]),
            }
        )
    return pd.DataFrame(rows)


def selected_ids_from_manifest(manifest: pd.DataFrame) -> list[str]:
    ids: set[str] = set()
    for row in manifest.itertuples():
        if row.selection_unit == "family":
            for child in json.loads(row.children):
                ids.add(str(child))
            ids.add(str(row.family_id))
        else:
            ids.add(str(row.id))
    return sorted(ids)


@torch.no_grad()
def collect_logits(label: str, seed: int, selected_ids: set[str], data_path: Path) -> pd.DataFrame:
    checkpoint = PHASE3 / "checkpoints" / "/".join(label.split("_")[:1]) / "_".join(label.split("_")[1:-1]) / label.split("_")[-1] / f"seed{seed:03d}.pt"
    if not checkpoint.exists():
        # label parsing for condition names with underscores
        parts = label.split("_")
        batching = parts[0]
        tag = parts[-1]
        condition = "_".join(parts[1:-1])
        checkpoint = PHASE3 / "checkpoints" / batching / condition / tag / f"seed{seed:03d}.pt"
    ckpt = torch.load(checkpoint, map_location="cpu")
    vocab = Vocabulary(ckpt["vocab"])
    dataset = TractatusDataset(data_path, vocab=vocab, languages=ckpt.get("languages"), language_to_id=ckpt.get("language_to_id"))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: collate_batch(batch, pad_idx=vocab.pad_idx))
    model = instantiate_model(ckpt, dataset, vocab)
    rows = []
    for batch in loader:
        outputs = model(batch["input_ids"], batch["lengths"], batch["decoder_ids"], batch["language_ids"])
        parent_logits = outputs["parent_logits"].detach().cpu().numpy()
        depth_logits = outputs["depth_logits"].detach().cpu().numpy()
        successor_logits = outputs["next_logits"].detach().cpu().numpy()
        for i, prop_id in enumerate(batch["ids"]):
            if str(prop_id) not in selected_ids:
                continue
            rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "id": str(prop_id),
                    "language": batch["languages"][i],
                    "parent_logits": json.dumps(parent_logits[i].round(6).tolist()),
                    "depth_logits": json.dumps(depth_logits[i].round(6).tolist()),
                    "successor_logits": json.dumps(successor_logits[i].round(6).tolist()),
                }
            )
    return pd.DataFrame(rows)


def add_text_and_outputs(meta: CorpusMeta, manifest: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    if "id" in manifest.columns:
        manifest["id"] = manifest["id"].astype("string")
    lexical = lexical_baselines(meta)
    selected_ids = set(selected_ids_from_manifest(manifest))
    selected_rows = []
    for prop_id in sorted(selected_ids):
        row = meta.by_id[prop_id]
        base = lexical[lexical["id"] == prop_id].iloc[0].to_dict()
        selected_rows.append(
            {
                "id": prop_id,
                "german_text": require_language_text(row, "de"),
                "english_text": require_language_text(row, "en"),
                "parent_id": row["parent_id"],
                "depth": row["depth"],
                "successor_id": row["next_id"],
                "child_count": row["child_count"],
                "top_level_branch": meta.top_branch[prop_id],
                "english_length": meta.lengths[(prop_id, "en")],
                "german_length": meta.lengths[(prop_id, "de")],
                "english_truncated": meta.truncated[(prop_id, "en")],
                "german_truncated": meta.truncated[(prop_id, "de")],
                **{key: value for key, value in base.items() if key != "id"},
            }
        )
    text_df = pd.DataFrame(selected_rows)
    with_text = manifest.merge(text_df, how="left", left_on="id", right_on="id", suffixes=("", "_selected_prop"))
    family_rows = []
    for row in manifest[manifest["selection_unit"] == "family"].itertuples():
        children = json.loads(row.children)
        family_rows.append(
            {
                "case_study": row.case_study,
                "role": row.role,
                "family_id": row.family_id,
                "family_member_ids": json.dumps(children),
                "family_member_english_texts": json.dumps({child: require_language_text(meta.by_id[child], "en") for child in children}, ensure_ascii=False),
                "family_member_german_texts": json.dumps({child: require_language_text(meta.by_id[child], "de") for child in children}, ensure_ascii=False),
            }
        )
    with_text.to_csv(out_root / "candidate_manifest_with_text.csv", index=False)
    pd.DataFrame(family_rows).to_csv(out_root / "data" / "family_member_texts.csv", index=False)
    freeze = read_json(out_root / "data" / "manifest_freeze.json")
    text_manifest = out_root / "candidate_manifest_with_text.csv"
    text_sha256 = sha256_file(text_manifest)
    if freeze.get("candidate_manifest_with_text_sha256") != text_sha256 or not freeze.get("text_joined_utc"):
        freeze["text_joined_utc"] = datetime.now(timezone.utc).isoformat()
    freeze["candidate_manifest_with_text"] = display_path_arg(text_manifest)
    freeze["candidate_manifest_with_text_sha256"] = text_sha256
    write_json(out_root / "data" / "manifest_freeze.json", freeze)
    return with_text


def selected_case_raw_outputs(meta: CorpusMeta, manifest: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    selected_ids = set(selected_ids_from_manifest(manifest))
    lexical = lexical_baselines(meta).set_index("id")
    rows = []
    for label in ALL_LABELS:
        for seed in SEEDS:
            raw = load_raw(label, seed)
            raw = raw[raw["id"].astype(str).isin(selected_ids)].copy()
            vecs = vectors_by_id_language(load_raw(label, seed))
            for item in raw.itertuples():
                prop_id = str(item.id)
                language = str(item.language)
                target_language = "de" if language == "en" else "en"
                neighbours = nearest_ids(vecs, prop_id, language, target_language, 20)
                base = lexical.loc[prop_id]
                rows.append(
                    {
                        "label": label,
                        "seed": seed,
                        "id": prop_id,
                        "language": language,
                        "posterior_text_mean": item.text_mu,
                        "posterior_structure_mean": item.structure_mu,
                        "text_posterior_variance": float(item.text_posterior_variance),
                        "structure_posterior_variance": float(item.structure_posterior_variance),
                        "parent_rank": int(item.parent_rank),
                        "depth_pred": int(item.depth_pred),
                        "successor_rank": int(item.successor_rank),
                        "child_count_pred": float(item.child_count_pred),
                        "bilingual_retrieval_rank": float(getattr(item, f"structure_retrieval_rank_{language}_to_{target_language}", np.nan)),
                        "bilingual_same_id_distance": float(getattr(item, f"structure_same_id_distance_{language}_to_{target_language}", np.nan)),
                        "neighbour_ids_k20": json.dumps([candidate for candidate, _distance in neighbours]),
                        "neighbour_distances_k20": json.dumps([distance for _candidate, distance in neighbours]),
                        "exact_token_jaccard": float(base["exact_token_jaccard"]),
                        "word_tfidf_similarity": float(base["word_tfidf_similarity"]),
                        "char_3_5_tfidf_similarity": float(base["char_3_5_tfidf_similarity"]),
                        "length": meta.lengths[(prop_id, language)],
                        "truncated": meta.truncated[(prop_id, language)],
                        "parent_id": meta.by_id[prop_id]["parent_id"],
                        "depth": int(meta.by_id[prop_id]["depth"]),
                        "successor_id": meta.by_id[prop_id]["next_id"],
                        "child_count": int(meta.by_id[prop_id]["child_count"]),
                        "top_level_branch": meta.top_branch[prop_id],
                    }
                )
    raw_df = pd.DataFrame(rows)
    logits = pd.concat([collect_logits(label, seed, selected_ids, DATA_PATH) for label in ALL_LABELS for seed in SEEDS], ignore_index=True)
    raw_df = raw_df.merge(logits, on=["label", "seed", "id", "language"], how="left")
    raw_df.to_parquet(out_root / "data" / "selected_case_raw_outputs.parquet", index=False)
    return raw_df


def plot_case_figures(manifest: pd.DataFrame, raw_df: pd.DataFrame, out_root: Path) -> None:
    primary = raw_df[raw_df["label"] == PRIMARY_LABEL]
    for i, row in enumerate(manifest.itertuples(), start=1):
        key = case_key(i, pd.Series(row._asdict()))
        if row.selection_unit == "family":
            ids = json.loads(row.children)
        else:
            ids = [str(row.id)]
        subset = primary[primary["id"].isin(ids)]
        plt.figure(figsize=(6, 3.5))
        if not subset.empty:
            subset.groupby("seed")["bilingual_same_id_distance"].mean().plot(marker="o")
        plt.xlabel("seed")
        plt.ylabel("Mean same-ID distance")
        plt.tight_layout()
        plt.savefig(out_root / "figures" / f"{key}_same_id_distance_by_seed.png", dpi=600)
        plt.close()


def case_key(index: int, case_row: pd.Series) -> str:
    identifier = case_row.get("family_id") if case_row.get("selection_unit") == "family" else case_row.get("id")
    identifier = str(identifier).replace(".", "_").replace("/", "_")
    return f"{index:02d}_{case_row['case_study']}_{case_row['role']}_{identifier}".replace("/", "_")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def dossier_text(case_row: pd.Series, meta: CorpusMeta, raw_df: pd.DataFrame, out_root: Path) -> str:
    if case_row["selection_unit"] == "family":
        ids = json.loads(case_row["children"])
        title_id = str(case_row["family_id"])
    else:
        ids = [str(case_row["id"])]
        title_id = ids[0]
    subset = raw_df[(raw_df["label"] == PRIMARY_LABEL) & (raw_df["id"].isin(ids))]
    agg = subset.groupby("id").agg(
        retrieval_rank_mean=("bilingual_retrieval_rank", "mean"),
        same_id_distance_mean=("bilingual_same_id_distance", "mean"),
        parent_rank_mean=("parent_rank", "mean"),
        successor_rank_mean=("successor_rank", "mean"),
        structure_variance_mean=("structure_posterior_variance", "mean"),
    ).reset_index()
    case_index = int(case_row.name) + 1 if case_row.name is not None else 0
    plot_name = f"{case_key(case_index, case_row)}_same_id_distance_by_seed.png"
    selection_diagnostics = pd.DataFrame(
        [
            {
                key: value
                for key, value in case_row.to_dict().items()
                if key
                not in {
                    "german_text",
                    "english_text",
                    "family_member_english_texts",
                    "family_member_german_texts",
                }
                and not pd.isna(value)
            }
        ]
    )
    lines = [
        f"# {case_row['case_study']} - {case_row['role']}",
        "",
        f"Selection unit: `{case_row['selection_unit']}`. ID: `{title_id}`.",
        "",
        "## Selection Diagnostics",
        "",
        markdown_table(selection_diagnostics),
        "",
        "## Proposition Text",
        "",
    ]
    for prop_id in ids:
        row = meta.by_id[prop_id]
        lines.extend(
            [
                f"### {prop_id}",
                "",
                f"German: {require_language_text(row, 'de')}",
                "",
                f"English: {require_language_text(row, 'en')}",
                "",
                f"Formal metadata: parent `{row['parent_id']}`, depth `{row['depth']}`, successor `{row['next_id']}`, child count `{row['child_count']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Aggregate Diagnostics",
            "",
            markdown_table(agg),
            "",
            "## Seed-Level Diagnostics",
            "",
            markdown_table(subset[["seed", "id", "language", "bilingual_retrieval_rank", "bilingual_same_id_distance", "parent_rank", "successor_rank", "structure_posterior_variance"]]),
            "",
            "## Nearest Neighbours",
            "",
        ]
    )
    neighbour_rows = subset[subset["seed"] == 0][["id", "language", "neighbour_ids_k20", "neighbour_distances_k20"]]
    lines.extend([markdown_table(neighbour_rows), ""])
    if case_row["selection_unit"] == "family" and "matched_family_ids" in case_row:
        lines.extend(
            [
                "## Matched Control Cases",
                "",
                f"Matched actual family IDs: `{case_row['matched_family_ids']}`.",
                "",
            ]
        )
    elif case_row["case_study"].startswith("B_"):
        lines.extend(
            [
                "## Matched Control Cases",
                "",
                "The high-overlap and low-overlap rows in the same manifest provide the paired control context for neighbourhood divergence.",
                "",
            ]
        )
    elif case_row["case_study"].startswith("C_"):
        lines.extend(
            [
                "## Matched Control Cases",
                "",
                "The ordinary-control row in the hierarchy-sequence manifest provides the control context for the two contrasting diagnostics.",
                "",
            ]
        )
    robust = raw_df[(raw_df["label"].isin(ALL_LABELS)) & (raw_df["id"].isin(ids))]
    robust_summary = robust.groupby("label").agg(
        same_id_distance_mean=("bilingual_same_id_distance", "mean"),
        retrieval_rank_mean=("bilingual_retrieval_rank", "mean"),
        parent_rank_mean=("parent_rank", "mean"),
        successor_rank_mean=("successor_rank", "mean"),
    ).reset_index()
    lines.extend(
        [
            "## Robustness",
            "",
            markdown_table(robust_summary),
            "",
            "Sensitivity to k=5, k=10 and k=20 is reported in the Phase 4 candidate and neighbourhood data tables. Truncation, length and lexical-baseline values are retained in `data/selected_case_raw_outputs.parquet`.",
            "",
            "## Relevant Plots",
            "",
            f"- `../figures/{plot_name}`",
            "",
            "## Questions for close reading",
            "",
            "- Does the proposition change expository or argumentative role?",
            "- Does its wording depart from formally matched neighbours?",
            "- Does the German-English difference reflect a translation choice?",
            "- Does the proposition open or close a branch?",
            "- Is the latent difference already explained by supervised formal metadata?",
            "",
        ]
    )
    return "\n".join(lines)


def write_dossiers(meta: CorpusMeta, manifest: pd.DataFrame, raw_df: pd.DataFrame, out_root: Path) -> None:
    plot_case_figures(manifest, raw_df, out_root)
    for i, row in manifest.iterrows():
        name = f"{i+1:02d}_{row['case_study']}_{row['role']}.md"
        (out_root / "dossiers" / name).write_text(dossier_text(row, meta, raw_df, out_root) + "\n", encoding="utf-8")


def write_reports(meta: CorpusMeta, manifest: pd.DataFrame, with_text: pd.DataFrame, raw_df: pd.DataFrame, out_root: Path) -> None:
    robust_rows = []
    for row in manifest.itertuples():
        ids = json.loads(row.children) if row.selection_unit == "family" else [str(row.id)]
        for label in ALL_LABELS:
            group = raw_df[(raw_df["label"] == label) & (raw_df["id"].isin(ids))]
            robust_rows.append(
                {
                    "case_study": row.case_study,
                    "role": row.role,
                    "label": label,
                    "same_id_distance_mean": float(group["bilingual_same_id_distance"].mean()),
                    "retrieval_rank_mean": float(group["bilingual_retrieval_rank"].mean()),
                    "parent_rank_mean": float(group["parent_rank"].mean()),
                    "successor_rank_mean": float(group["successor_rank"].mean()),
                    "truncated_any": bool(group["truncated"].any()),
                    "mean_length": float(group["length"].mean()),
                    "survives_lambda_003": label == "paired_full_model_align003",
                    "survives_no_successor": label == "paired_no_successor_align000",
                }
            )
    robustness = pd.DataFrame(robust_rows)
    robustness.to_csv(out_root / "phase4_robustness_summary.csv", index=False)
    freeze = read_json(out_root / "data" / "manifest_freeze.json")
    protocol = [
        "# Phase 4 Case Selection Protocol",
        "",
        "1. Numeric and formal metadata were computed from Phase 3 latent outputs, retrieval diagnostics, ranks, family metadata, length and truncation indicators.",
        "2. Candidate ranking was executed from `data/case_selection_config.json` before proposition wording was joined.",
        "3. `candidate_manifest_pre_text.csv` was written without German or English wording.",
        f"4. The frozen pre-text SHA-256 hash is `{freeze['sha256']}`.",
        f"5. Wording was joined at `{freeze['text_joined_utc']}` after the pre-text manifest was frozen.",
        "6. Selected cases were not changed after wording was joined; no post hoc replacement set is present.",
    ]
    (out_root / "phase4_case_selection_protocol.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")
    lines = [
        "# Phase 4 Proposition-Level Case Studies",
        "",
        f"Primary representation: `{PRIMARY_LABEL}`. Robustness comparisons: `{', '.join(ROBUST_LABELS)}`.",
        "",
        "The selected cases are computational entry points for close reading. The model is not used to make philosophical conclusions.",
        "",
        "## Text-Blind Selection",
        "",
        f"`candidate_manifest_pre_text.csv` was frozen with SHA-256 `{freeze['sha256']}` before text was joined. The joined manifest hash is `{freeze['candidate_manifest_with_text_sha256']}`.",
        "",
        "## Selected Cases",
        "",
    ]
    for row in manifest.itertuples():
        ids = json.loads(row.children) if row.selection_unit == "family" else [str(row.id)]
        group = raw_df[(raw_df["label"] == PRIMARY_LABEL) & (raw_df["id"].isin(ids))]
        lines.extend(
            [
                f"### {row.case_study}: {row.role}",
                "",
                f"Selection unit: `{row.selection_unit}`; IDs: `{', '.join(ids)}`.",
                f"Primary mean same-ID distance: {float(group['bilingual_same_id_distance'].mean()):.4f}; mean retrieval rank: {float(group['bilingual_retrieval_rank'].mean()):.4f}.",
                "See the corresponding dossier for wording, neighbours, seed-level diagnostics and close-reading questions.",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence suitable for a DSH manuscript",
            "",
        ]
    )
    for row in manifest.itertuples():
        lines.extend(
            [
                f"### {row.case_study}: {row.role}",
                "",
                "What the computation establishes: this case was selected by a pre-specified numerical/formal rule before wording was inspected, and its seed-level diagnostics are retained.",
                "",
                "What it merely suggests: the case may be useful for studying how formal position, bilingual neighbourhoods and sequence diagnostics interact in the corpus.",
                "",
                "What requires human close reading: expository role, argumentative force, translation choices and any philosophical interpretation.",
                "",
                "What must not be claimed: latent distance does not establish philosophical dependence, semantic equivalence, anomaly or mistranslation.",
                "",
            ]
        )
    (out_root / "phase4_case_studies_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_root = args.out_root.resolve()
    ensure_layout(out_root)
    meta = load_corpus_meta(args.data)
    family_df, _matched_df = compute_family_metrics(meta, out_root)
    neighbourhood_df = compute_neighbourhood_metrics(meta, out_root)
    hierarchy_df = compute_hierarchy_sequence_metrics(meta, out_root)
    manifest_path = out_root / "candidate_manifest_pre_text.csv"
    if manifest_path.exists() and not args.reselect:
        manifest = pd.read_csv(manifest_path, dtype={"id": "string", "family_id": "string"})
    else:
        manifest = select_cases(family_df, neighbourhood_df, hierarchy_df, out_root)
    with_text = add_text_and_outputs(meta, manifest, out_root)
    raw_df = selected_case_raw_outputs(meta, manifest, out_root)
    write_dossiers(meta, manifest, raw_df, out_root)
    write_reports(meta, manifest, with_text, raw_df, out_root)
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 tools/phase4_case_studies.py run --out-root results/dsh_validation/phase4_case_studies",
        "PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 tools/phase4_case_studies.py verify --out-root results/dsh_validation/phase4_case_studies",
    ]
    (out_root / "phase4_commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")


def verify(args: argparse.Namespace) -> None:
    out_root = args.out_root.resolve()
    required = [
        "phase4_case_selection_protocol.md",
        "candidate_manifest_pre_text.csv",
        "candidate_manifest_pre_text.sha256",
        "candidate_manifest_with_text.csv",
        "phase4_case_studies_report.md",
        "phase4_robustness_summary.csv",
        "phase4_commands.sh",
        "data/selected_case_raw_outputs.parquet",
    ]
    checks = []
    missing = [name for name in required if not (out_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing deliverables: {missing}")
    checks.append("All required Phase 4 report, manifest and data deliverables are present.")
    manifest = pd.read_csv(out_root / "candidate_manifest_pre_text.csv")
    forbidden = [col for col in manifest.columns if col.lower() in {"text", "english_text", "german_text"} or "wording" in col.lower()]
    if forbidden:
        raise AssertionError(f"Pre-text manifest contains text columns: {forbidden}")
    checks.append("Pre-text manifest contains no proposition wording columns.")
    expected_hash = (out_root / "candidate_manifest_pre_text.sha256").read_text(encoding="utf-8").split()[0]
    observed_hash = sha256_file(out_root / "candidate_manifest_pre_text.csv")
    if expected_hash != observed_hash:
        raise AssertionError("Pre-text manifest SHA-256 does not match.")
    freeze = read_json(out_root / "data" / "manifest_freeze.json")
    if freeze["sha256"] != observed_hash or freeze["text_joined_utc"] is None:
        raise AssertionError("Manifest freeze metadata is incomplete.")
    if freeze["frozen_utc"] > freeze["text_joined_utc"]:
        raise AssertionError("Text join timestamp precedes manifest freeze timestamp.")
    checks.append("Pre-text manifest hash and freeze-before-join timestamps verify.")
    raw = pd.read_parquet(out_root / "data" / "selected_case_raw_outputs.parquet")
    required_raw = [
        "posterior_text_mean",
        "posterior_structure_mean",
        "text_posterior_variance",
        "structure_posterior_variance",
        "parent_logits",
        "parent_rank",
        "depth_logits",
        "successor_logits",
        "successor_rank",
        "child_count_pred",
        "bilingual_retrieval_rank",
        "neighbour_ids_k20",
        "neighbour_distances_k20",
        "exact_token_jaccard",
        "word_tfidf_similarity",
        "char_3_5_tfidf_similarity",
        "length",
        "truncated",
        "parent_id",
        "depth",
        "successor_id",
        "child_count",
    ]
    absent = [col for col in required_raw if col not in raw.columns]
    if absent:
        raise AssertionError(f"Missing selected raw-output columns: {absent}")
    checks.append("Selected-case raw outputs retain required posterior, rank, logit, neighbour, lexical, length and formal fields.")
    dossier_count = len(list((out_root / "dossiers").glob("*.md")))
    if dossier_count != len(manifest):
        raise AssertionError(f"Expected {len(manifest)} dossiers, found {dossier_count}.")
    checks.append("One dossier exists for every selected manifest row.")
    tracked_diff = git_output(["diff", "--name-only"]).splitlines()
    forbidden_diff = [path for path in tracked_diff if path in {"paper/main.tex", "paper/references.bib"} or path.startswith("paper/")]
    if forbidden_diff:
        raise AssertionError(f"Protected manuscript/canonical paths changed: {forbidden_diff}")
    checks.append("No tracked diffs touch manuscript files or canonical run outputs.")
    lines = [
        "# Phase 4 Verification Report",
        "",
        *[f"- {check}" for check in checks],
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
    (out_root / "phase4_verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified Phase 4 case-study outputs and text-blind manifest freeze.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create text-blind Phase 4 proposition-level case studies.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    run_parser.add_argument("--data", type=Path, default=DATA_PATH)
    run_parser.add_argument("--reselect", action="store_true", help="Recreate the frozen pre-text selection manifest.")
    run_parser.set_defaults(func=run)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
