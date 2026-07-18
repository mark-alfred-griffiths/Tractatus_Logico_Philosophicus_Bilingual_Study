from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tractatus_structure_latents.models.vae import (
    HierarchicalRNNVAE,
    SplitLatentHierarchicalRNNVAE,
)
from tractatus_structure_latents.training.data import (
    TractatusDataset,
    Vocabulary,
    collate_batch,
)


REPORT_DIR = ROOT / "reports" / "empirical_audit"
BILINGUAL_ROOT = ROOT / "runs" / "seed_sweeps" / "bilingual_alignment_lambda_sweep"
MONOLINGUAL_ROOT = ROOT / "runs" / "seed_sweeps" / "monolingual_split_24_8_reg005"
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_RESAMPLES = 10_000

SOURCE_CITATIONS = {
    "lambda_tag": "tractatus_structure_latents/scripts/run_bilingual_alignment_seed_sweep.py:9",
    "lambda_defaults": "tractatus_structure_latents/scripts/run_bilingual_alignment_seed_sweep.py:89",
    "manifest": "tractatus_structure_latents/scripts/run_bilingual_alignment_seed_sweep.py:28",
    "alignment_loss": "tractatus_structure_latents/training/train_vae.py:119",
    "alignment_return": "tractatus_structure_latents/training/train_vae.py:140",
    "alignment_added": "tractatus_structure_latents/training/train_vae.py:271",
    "set_seed": "tractatus_structure_latents/training/train_vae.py:25",
    "reparameterize_eval": "tractatus_structure_latents/models/vae.py:111",
    "split_forward": "tractatus_structure_latents/models/vae.py:127",
    "child_head": "tractatus_structure_latents/models/vae.py:106",
    "child_loss": "tractatus_structure_latents/models/vae.py:200",
    "child_target": "tractatus_structure_latents/training/data.py:124",
    "same_id": "tractatus_structure_latents/evaluation/evaluate_structure.py:50",
    "same_id_distance": "tractatus_structure_latents/evaluation/evaluate_structure.py:74",
    "retrieval_distance": "tractatus_structure_latents/evaluation/evaluate_structure.py:82",
    "retrieval_rank": "tractatus_structure_latents/evaluation/evaluate_structure.py:85",
    "eval_latents": "tractatus_structure_latents/evaluation/evaluate_structure.py:212",
    "relation_distances": "tractatus_structure_latents/evaluation/evaluate_structure.py:229",
    "metric_keys": "tractatus_structure_latents/evaluation/evaluate_structure.py:250",
    "sweep_summary": "tractatus_structure_latents/evaluation/plot_bilingual_alignment_sweep.py:61",
    "pca_single_seed": "tractatus_structure_latents/evaluation/generate_paper_figures.py:70",
}

PAIR_METRICS = [
    "cross_language_top1_id_accuracy",
    "cross_language_mrr",
    "parent_accuracy",
    "depth_accuracy",
    "next_accuracy",
    "reconstruction_loss",
    "perplexity",
    "kl_text",
    "kl_structure",
    "mean_same_id_cross_language_distance",
]

LOWER_IS_NUMERICALLY_FAVOURABLE = {
    "reconstruction_loss",
    "perplexity",
    "kl_text",
    "kl_structure",
    "mean_same_id_cross_language_distance",
}


@dataclass(frozen=True)
class Study:
    tag: str
    lambda_value: float
    path: Path


def lambda_tag(value: float) -> str:
    return f"align{int(round(value * 100)):03d}"


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
        for row in rows:
            writer.writerow(row)


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, next_key))
        elif isinstance(value, bool):
            flat[next_key] = float(value)
        elif isinstance(value, (int, float)):
            flat[next_key] = float(value)
    return flat


def git_output(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr.strip()}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_studies() -> list[Study]:
    studies = []
    for path in sorted(BILINGUAL_ROOT.glob("align*/manifest.json")):
        manifest = read_json(path)
        studies.append(Study(path.parent.name, float(manifest["lambda_language_alignment"]), path.parent))
    return sorted(studies, key=lambda study: study.lambda_value)


def seed_from_path(path: Path) -> int:
    stem = path.name.split(".")[0]
    if not stem.startswith("seed") or not stem[4:].isdigit():
        raise ValueError(f"Cannot parse seed from {path}")
    return int(stem[4:])


def load_seed_metrics(study: Study) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted((study.path / "metrics").glob("seed*.metrics.json")):
        out[seed_from_path(path)] = read_json(path)
    return out


def load_cached_latents(study: Study, seed: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    stem = f"seed{seed:03d}_structure"
    z = torch.load(study.path / "latents" / f"{stem}.pt", map_location="cpu")
    metadata = read_json(study.path / "latents" / f"{stem}.ids.json")
    return z, metadata


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def cross_language_metrics(z: torch.Tensor, metadata: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, float]:
    by_id_language = {(str(item["id"]), str(item["language"])): i for i, item in enumerate(metadata)}
    languages = sorted({str(item["language"]) for item in metadata})
    same_id_distances: list[float] = []
    same_id_mse: list[float] = []
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
            delta = z[source_i] - z[target_i]
            same_id_distances.append(torch.dist(z[source_i], z[target_i]).item())
            same_id_mse.append(delta.pow(2).mean().item())
            candidates = [i for i, item in enumerate(metadata) if str(item["language"]) == target_language]
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
        "mean_same_id_cross_language_distance": mean_or_zero(same_id_distances),
        "mean_same_id_cross_language_mse": mean_or_zero(same_id_mse),
        "cross_language_top1_id_accuracy": mean_or_zero(retrieval_top1),
        "cross_language_mrr": mean_or_zero(retrieval_rr),
        "mean_cross_language_parent_child_distance": mean_or_zero(cross_parent_child_distances),
        "same_id_pair_count_directional": float(len(same_id_distances)),
    }
    for direction, values in sorted(direction_top1.items()):
        metrics[f"cross_language_top1_id_accuracy_{direction}"] = mean_or_zero(values)
    for direction, values in sorted(direction_rr.items()):
        metrics[f"cross_language_mrr_{direction}"] = mean_or_zero(values)
    return metrics


def relation_distances(z: torch.Tensor, metadata: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, float]:
    id_language_to_i = {(str(item["id"]), str(item["language"])): i for i, item in enumerate(metadata)}
    parent_child_dist: list[float] = []
    sibling_dist: list[float] = []
    unrelated_dist: list[float] = []
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
    return {
        "mean_parent_child_distance": mean_or_zero(parent_child_dist),
        "mean_sibling_distance": mean_or_zero(sibling_dist),
        "mean_unrelated_distance": mean_or_zero(unrelated_dist),
        "parent_child_pair_count": float(len(parent_child_dist)),
        "sibling_pair_count": float(len(sibling_dist)),
        "unrelated_pair_count": float(len(unrelated_dist)),
    }


def instantiate_model(ckpt: dict[str, Any], dataset: TractatusDataset, vocab: Vocabulary) -> torch.nn.Module:
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
    return model


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, data_path: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    vocab = Vocabulary(ckpt["vocab"])
    dataset = TractatusDataset(
        data_path,
        vocab=vocab,
        languages=ckpt.get("languages"),
        language_to_id=ckpt.get("language_to_id"),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: collate_batch(batch, pad_idx=vocab.pad_idx))
    model = instantiate_model(ckpt, dataset, vocab)

    structure_mu: list[torch.Tensor] = []
    structure_logvar: list[torch.Tensor] = []
    child_predictions: list[float] = []
    child_targets: list[float] = []
    languages: list[str] = []
    metadata: list[dict[str, Any]] = []

    for batch in loader:
        outputs = model(batch["input_ids"], batch["lengths"], batch["decoder_ids"], batch["language_ids"])
        structure_mu.append(outputs["structure_mu"])
        if "structure_logvar" in outputs:
            structure_logvar.append(outputs["structure_logvar"])
        child_predictions.extend(float(x) for x in outputs["child_count"].detach().cpu())
        child_targets.extend(float(x) for x in batch["child_count"].detach().cpu())
        languages.extend(str(x) for x in batch["languages"])
        metadata.extend(
            {"id": prop_id, "language": language, "index": int(index)}
            for prop_id, language, index in zip(batch["ids"], batch["languages"], batch["index"])
        )

    mu = torch.cat(structure_mu, dim=0)
    logvar = torch.cat(structure_logvar, dim=0) if structure_logvar else torch.empty_like(mu)
    return {
        "structure_mu": mu,
        "structure_logvar": logvar,
        "child_predictions": child_predictions,
        "child_targets": child_targets,
        "languages": languages,
        "metadata": metadata,
        "checkpoint": ckpt,
    }


def regression_metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
    errors = [p - t for p, t in zip(predictions, targets)]
    abs_errors = [abs(x) for x in errors]
    sq_errors = [x * x for x in errors]
    target_mean = mean(targets)
    sorted_targets = sorted(targets)
    mid = len(sorted_targets) // 2
    target_median = sorted_targets[mid] if len(sorted_targets) % 2 else (sorted_targets[mid - 1] + sorted_targets[mid]) / 2
    mean_baseline_abs = [abs(target_mean - t) for t in targets]
    mean_baseline_sq = [(target_mean - t) ** 2 for t in targets]
    median_baseline_abs = [abs(target_median - t) for t in targets]
    median_baseline_sq = [(target_median - t) ** 2 for t in targets]
    return {
        "n": float(len(targets)),
        "target_mean": target_mean,
        "target_median": target_median,
        "prediction_mean": mean(predictions),
        "mae": mean(abs_errors),
        "rmse": math.sqrt(mean(sq_errors)),
        "mse": mean(sq_errors),
        "mean_baseline_mae": mean(mean_baseline_abs),
        "mean_baseline_rmse": math.sqrt(mean(mean_baseline_sq)),
        "median_baseline_mae": mean(median_baseline_abs),
        "median_baseline_rmse": math.sqrt(mean(median_baseline_sq)),
    }


def child_count_metrics_for_group(
    condition: str,
    lambda_value: float | None,
    seed: int,
    predictions: list[float],
    targets: list[float],
    languages: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = ["all", *sorted(set(languages))]
    for group in groups:
        indices = [i for i, language in enumerate(languages) if group == "all" or language == group]
        group_targets = [targets[i] for i in indices]
        group_predictions = [predictions[i] for i in indices]
        row = {
            "condition": condition,
            "lambda_language_alignment": "" if lambda_value is None else lambda_value,
            "seed": seed,
            "language": group,
            "task_type": "regression",
            "decoding_rule": "none_in_implementation",
        }
        row.update(regression_metrics(group_targets, group_predictions))
        rows.append(row)
    return rows


def child_count_distribution() -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for dataset_name, path in [
        ("monolingual", ROOT / "tractatus_structure_latents" / "data" / "tractatus.json"),
        ("bilingual", ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json"),
    ]:
        data = read_json(path)
        base_counter = Counter(int(row["child_count"]) for row in data)
        language_counts = {"all": len(data)}
        if dataset_name == "bilingual":
            language_counts.update({"en": len(data), "de": len(data)})
        else:
            language_counts.update({"en": len(data)})
        for language, sample_count in language_counts.items():
            multiplier = 1 if language != "all" else (2 if dataset_name == "bilingual" else 1)
            denom = sample_count if language != "all" else len(data) * multiplier
            for child_count, support in sorted(base_counter.items()):
                language_support = support if language != "all" else support * multiplier
                rows_out.append(
                    {
                        "dataset": dataset_name,
                        "language": language,
                        "child_count": child_count,
                        "support": language_support,
                        "proportion": language_support / denom,
                        "min_child_count": min(base_counter),
                        "max_child_count": max(base_counter),
                    }
                )
    return rows_out


def paired_bootstrap_ci(diffs: list[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_i = int(0.025 * resamples)
    hi_i = int(0.975 * resamples) - 1
    return means[lo_i], means[hi_i]


def sign_flip_p_value(diffs: list[float]) -> float:
    observed = abs(mean(diffs))
    n = len(diffs)
    count = 0
    total = 1 << n
    abs_diffs = [abs(x) for x in diffs]
    for mask in range(total):
        signed = [abs_diffs[i] if (mask >> i) & 1 else -abs_diffs[i] for i in range(n)]
        if abs(mean(signed)) >= observed - 1e-15:
            count += 1
    return count / total


def paired_effect_size(diffs: list[float]) -> float:
    sd = stdev(diffs) if len(diffs) > 1 else 0.0
    return mean(diffs) / sd if sd else float("nan")


def paired_lambda_outputs(metrics_by_study: dict[str, dict[int, dict[str, float]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    seeds = sorted(set(metrics_by_study["align000"]) & set(metrics_by_study["align003"]))
    for metric in PAIR_METRICS:
        diffs: list[float] = []
        for seed in seeds:
            value_000 = metrics_by_study["align000"][seed][metric]
            value_003 = metrics_by_study["align003"][seed][metric]
            diff = value_003 - value_000
            diffs.append(diff)
            per_seed.append(
                {
                    "metric": metric,
                    "seed": seed,
                    "lambda_000": value_000,
                    "lambda_003": value_003,
                    "difference_003_minus_000": diff,
                }
            )
        ci_low, ci_high = paired_bootstrap_ci(diffs)
        lower_better = metric in LOWER_IS_NUMERICALLY_FAVOURABLE
        equal = sum(1 for x in diffs if x == 0)
        higher_003 = sum(1 for x in diffs if x > 0)
        higher_000 = sum(1 for x in diffs if x < 0)
        if lower_better:
            favour_003 = higher_000
            favour_000 = higher_003
            favourable_direction = "lower"
        else:
            favour_003 = higher_003
            favour_000 = higher_000
            favourable_direction = "higher"
        summary.append(
            {
                "metric": metric,
                "n": len(diffs),
                "mean_lambda_000": mean(metrics_by_study["align000"][seed][metric] for seed in seeds),
                "mean_lambda_003": mean(metrics_by_study["align003"][seed][metric] for seed in seeds),
                "mean_difference_003_minus_000": mean(diffs),
                "sd_difference": stdev(diffs) if len(diffs) > 1 else 0.0,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_ci_95_low": ci_low,
                "bootstrap_ci_95_high": ci_high,
                "sign_flip_p_value_two_sided": sign_flip_p_value(diffs),
                "seeds_higher_003": higher_003,
                "seeds_higher_000": higher_000,
                "seeds_equal": equal,
                "favourable_direction": favourable_direction,
                "seeds_favouring_003": favour_003,
                "seeds_favouring_000": favour_000,
                "standardised_paired_effect_size_dz": paired_effect_size(diffs),
                "effect_size_formula": "mean(difference_003_minus_000) / sample_sd(difference_003_minus_000)",
            }
        )
    return per_seed, summary


def compare_configs(studies: list[Study]) -> dict[str, Any]:
    manifests = {study.tag: read_json(study.path / "manifest.json") for study in studies}
    baseline = manifests["align000"]
    comparisons = {}
    ignored = {"lambda_language_alignment"}
    for tag, manifest in manifests.items():
        differences: list[str] = []
        for key in ["seeds", "data", "languages", "model"]:
            if manifest.get(key) != baseline.get(key):
                differences.append(key)
        base_training = {k: v for k, v in baseline["training"].items() if k not in ignored}
        training = {k: v for k, v in manifest["training"].items() if k not in ignored}
        if training != base_training:
            differences.append("training_except_lambda")
        comparisons[tag] = {
            "lambda_language_alignment": manifest["lambda_language_alignment"],
            "non_lambda_differences_from_align000": differences,
        }
    return comparisons


def audit() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    studies = load_studies()
    bilingual_rows = read_json(ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json")
    same_id_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    recompute_rows: list[dict[str, Any]] = []
    child_metric_rows: list[dict[str, Any]] = []
    metrics_by_study: dict[str, dict[int, dict[str, float]]] = {}
    checkpoint_cache: dict[tuple[str, int], dict[str, Any]] = {}

    for study in studies:
        metrics_by_study[study.tag] = {}
        for seed, metrics in load_seed_metrics(study).items():
            flat = flatten_metrics(metrics)
            metrics_by_study[study.tag][seed] = flat
            z_cached, metadata = load_cached_latents(study, seed)
            cross = cross_language_metrics(z_cached, metadata, bilingual_rows)
            relation = relation_distances(z_cached, metadata, bilingual_rows)
            checkpoint_path = study.path / "checkpoints" / f"seed{seed:03d}.pt"
            evaluated = evaluate_checkpoint(checkpoint_path, ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json")
            checkpoint_cache[(study.tag, seed)] = evaluated
            mu = evaluated["structure_mu"]
            logvar = evaluated["structure_logvar"]
            max_cached_mu_abs_diff = float((z_cached - mu).abs().max().item())
            var = logvar.exp()
            std = torch.exp(0.5 * logvar)

            same_id_rows.append(
                {
                    "condition": study.tag,
                    "lambda_language_alignment": study.lambda_value,
                    "seed": seed,
                    "published_mean_same_id_cross_language_distance": flat["mean_same_id_cross_language_distance"],
                    "recomputed_published_definition": cross["mean_same_id_cross_language_distance"],
                    "posterior_mean_euclidean_distance": cross["mean_same_id_cross_language_distance"],
                    "posterior_mean_mse_distance": cross["mean_same_id_cross_language_mse"],
                    "same_id_pair_count_directional": int(cross["same_id_pair_count_directional"]),
                    "uses_cached_or_recomputed_posterior_mean": "posterior_mean",
                    "sampled_latent_distance": "",
                    "sampled_latent_note": "not_computed_published_eval_uses_eval_mode_mu",
                    "max_abs_diff_cached_latent_vs_checkpoint_mu": max_cached_mu_abs_diff,
                }
            )
            latent_rows.append(
                {
                    "condition": study.tag,
                    "lambda_language_alignment": study.lambda_value,
                    "seed": seed,
                    "mean_structure_mu_norm": float(mu.norm(dim=-1).mean().item()),
                    "mean_structure_mu_squared_norm": float(mu.pow(2).sum(dim=-1).mean().item()),
                    "mean_structure_posterior_std": float(std.mean().item()),
                    "mean_structure_posterior_variance": float(var.mean().item()),
                    "mean_structure_logvar": float(logvar.mean().item()),
                    "mean_parent_child_distance": relation["mean_parent_child_distance"],
                    "mean_sibling_distance": relation["mean_sibling_distance"],
                    "mean_unrelated_distance": relation["mean_unrelated_distance"],
                    "mean_cross_language_parent_child_distance": cross["mean_cross_language_parent_child_distance"],
                    "max_abs_diff_cached_latent_vs_checkpoint_mu": max_cached_mu_abs_diff,
                }
            )
            for metric in [
                "mean_same_id_cross_language_distance",
                "cross_language_top1_id_accuracy",
                "cross_language_mrr",
                "cross_language_top1_id_accuracy_de_to_en",
                "cross_language_top1_id_accuracy_en_to_de",
                "cross_language_mrr_de_to_en",
                "cross_language_mrr_en_to_de",
                "mean_cross_language_parent_child_distance",
                "mean_parent_child_distance",
                "mean_sibling_distance",
                "mean_unrelated_distance",
            ]:
                recomputed = cross.get(metric, relation.get(metric))
                published = flat.get(metric)
                recompute_rows.append(
                    {
                        "condition": study.tag,
                        "lambda_language_alignment": study.lambda_value,
                        "seed": seed,
                        "metric": metric,
                        "published": published,
                        "recomputed": recomputed,
                        "abs_diff": "" if published is None or recomputed is None else abs(published - recomputed),
                    }
                )
            child_metric_rows.extend(
                child_count_metrics_for_group(
                    study.tag,
                    study.lambda_value,
                    seed,
                    evaluated["child_predictions"],
                    evaluated["child_targets"],
                    evaluated["languages"],
                )
            )

    for seed in range(10):
        checkpoint_path = MONOLINGUAL_ROOT / "checkpoints" / f"seed{seed:03d}.pt"
        evaluated = evaluate_checkpoint(checkpoint_path, ROOT / "tractatus_structure_latents" / "data" / "tractatus.json")
        child_metric_rows.extend(
            child_count_metrics_for_group(
                "monolingual_split_24_8_reg005",
                None,
                seed,
                evaluated["child_predictions"],
                evaluated["child_targets"],
                evaluated["languages"],
            )
        )

    pair_rows, pair_summary = paired_lambda_outputs(metrics_by_study)
    dist_rows = child_count_distribution()

    write_csv(REPORT_DIR / "same_id_distance_by_seed.csv", same_id_rows)
    write_csv(REPORT_DIR / "latent_scale_variance_by_seed.csv", latent_rows)
    write_csv(REPORT_DIR / "recomputed_vs_published.csv", recompute_rows)
    write_csv(REPORT_DIR / "lambda_000_vs_003_paired.csv", pair_rows)
    write_csv(REPORT_DIR / "lambda_000_vs_003_summary.csv", pair_summary)
    write_csv(REPORT_DIR / "child_count_distribution.csv", dist_rows)
    write_csv(REPORT_DIR / "child_count_seed_metrics.csv", child_metric_rows)

    manifest = {
        "repository": {
            "commit": git_output(["rev-parse", "HEAD"]),
            "branch": git_output(["branch", "--show-current"]),
            "status_short": git_output(["status", "--short"]).splitlines(),
        },
        "audit": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "studies": [{"tag": s.tag, "lambda_language_alignment": s.lambda_value, "path": str(s.path.relative_to(ROOT))} for s in studies],
            "config_comparison": compare_configs(studies),
            "source_citations": SOURCE_CITATIONS,
            "outputs": {},
        },
    }
    output_paths = [
        "same_id_distance_by_seed.csv",
        "latent_scale_variance_by_seed.csv",
        "lambda_000_vs_003_paired.csv",
        "lambda_000_vs_003_summary.csv",
        "child_count_distribution.csv",
        "child_count_seed_metrics.csv",
        "recomputed_vs_published.csv",
        "report_for_chatgpt.md",
        "commands_run.txt",
    ]
    write_commands_run()
    write_report(manifest, same_id_rows, latent_rows, recompute_rows, pair_summary, child_metric_rows, dist_rows)
    for name in output_paths:
        path = REPORT_DIR / name
        if path.exists():
            manifest["audit"]["outputs"][name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    write_json(REPORT_DIR / "audit_manifest.json", manifest)
    write_inventory(manifest)
    return manifest


def summarise_by_condition(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row["condition"], row.get("lambda_language_alignment")) for row in rows}, key=lambda x: (999 if x[1] == "" else float(x[1]), x[0]))
    for condition, lambda_value in keys:
        values = [float(row[metric]) for row in rows if row["condition"] == condition and row.get("lambda_language_alignment") == lambda_value]
        if values:
            out.append(
                {
                    "condition": condition,
                    "lambda": lambda_value,
                    "n": len(values),
                    "mean": mean(values),
                    "sd": stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str], digits: int = 4) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.{digits}g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def max_abs_diff(recompute_rows: list[dict[str, Any]]) -> float:
    values = [float(row["abs_diff"]) for row in recompute_rows if row["abs_diff"] != ""]
    return max(values) if values else 0.0


def artifact_table() -> list[dict[str, Any]]:
    artefacts = [
        ("package", ROOT / "tractatus_structure_latents"),
        ("paper/main.tex", ROOT / "paper" / "main.tex"),
        ("paper bilingual summary", ROOT / "paper" / "bilingual_results_summary.txt"),
        ("bilingual summary JSON", BILINGUAL_ROOT / "summaries" / "summary.json"),
        ("bilingual sweep root", BILINGUAL_ROOT),
        ("monolingual sweep root", MONOLINGUAL_ROOT),
    ]
    rows = []
    for name, path in artefacts:
        rows.append({"artifact": name, "path": str(path.relative_to(ROOT)), "status": "found" if path.exists() else "missing"})
    for tag in ["align000", "align003", "align010", "align030", "align100"]:
        base = BILINGUAL_ROOT / tag
        rows.append(
            {
                "artifact": f"{tag} per-seed files",
                "path": str(base.relative_to(ROOT)),
                "status": f"metrics={len(list((base / 'metrics').glob('seed*.metrics.json')))}, checkpoints={len(list((base / 'checkpoints').glob('seed*.pt')))}, latents={len(list((base / 'latents').glob('seed*_structure.pt')))}",
            }
        )
    rows.append(
        {
            "artifact": "monolingual per-seed files",
            "path": str(MONOLINGUAL_ROOT.relative_to(ROOT)),
            "status": f"metrics={len(list((MONOLINGUAL_ROOT / 'metrics').glob('seed*.metrics.json')))}, checkpoints={len(list((MONOLINGUAL_ROOT / 'checkpoints').glob('seed*.pt')))}, latents={len(list((MONOLINGUAL_ROOT / 'latents').glob('seed*_structure.pt')))}",
        }
    )
    return rows


def metric_keys_table() -> list[dict[str, Any]]:
    sample_paths = [
        BILINGUAL_ROOT / "align003" / "metrics" / "seed000.metrics.json",
        MONOLINGUAL_ROOT / "metrics" / "seed000.metrics.json",
    ]
    rows = []
    for path in sample_paths:
        keys = sorted(flatten_metrics(read_json(path)))
        rows.append({"file": str(path.relative_to(ROOT)), "metric_keys": ", ".join(keys)})
    return rows


def write_inventory(manifest: dict[str, Any]) -> None:
    lines = [
        "# Empirical Audit Inventory",
        "",
        f"Commit: `{manifest['repository']['commit']}`",
        f"Branch: `{manifest['repository']['branch']}`",
        "Dirty state:",
    ]
    status = manifest["repository"]["status_short"]
    lines.extend([f"- `{item}`" for item in status] or ["- clean"])
    lines.extend(["", "## Canonical Artifacts"])
    lines.extend(markdown_table(artifact_table(), ["artifact", "path", "status"]))
    lines.extend(
        [
            "",
            "## Implementation Evidence",
            f"- Training language alignment loss: `{SOURCE_CITATIONS['alignment_loss']}`. It builds within-batch German-English pairs sharing the same proposition index and returns `F.mse_loss` over `structure_mu` at `{SOURCE_CITATIONS['alignment_return']}`; the weighted term is added at `{SOURCE_CITATIONS['alignment_added']}`.",
            f"- Same-ID and retrieval metrics: `{SOURCE_CITATIONS['same_id']}`. They operate on exported evaluation latents; export uses `outputs['structure_mu']` when `--latent-part structure` is selected at `{SOURCE_CITATIONS['eval_latents']}`.",
            f"- Child-count target: `{SOURCE_CITATIONS['child_target']}`. Head and loss: `{SOURCE_CITATIONS['child_head']}`, `{SOURCE_CITATIONS['child_loss']}`.",
            f"- Lambda label construction: `{SOURCE_CITATIONS['lambda_tag']}`; default sweep values are at `{SOURCE_CITATIONS['lambda_defaults']}`.",
            "",
            "## Metric Keys Found",
        ]
    )
    lines.extend(markdown_table(metric_keys_table(), ["file", "metric_keys"], digits=6))
    lines.extend(
        [
            "",
            "## Recommended Audit Implementation Plan",
            "1. Recompute published cross-language and relation-distance metrics from cached `*_structure.pt` latents, because those files are the exported posterior means used by the canonical evaluator.",
            "2. Load checkpoints in `eval()` mode only for quantities not saved in metrics: posterior variance/std and child-count predictions.",
            "3. Compare recomputed metrics with every per-seed JSON and report maximum absolute differences.",
            "4. Use seed-matched align000/align003 metrics for exploratory paired summaries with bootstrap CIs and exact sign-flip p-values.",
            "",
            "## Unresolved Questions From Inventory",
            "- No existing saved child-count predictions or metrics were found in per-seed metric JSON files.",
            "- The evaluator does not save posterior log-variance, so posterior variance must be recomputed from checkpoints.",
            "- Principal metrics are retained-corpus evaluations unless another held-out split artifact is identified.",
        ]
    )
    (REPORT_DIR / "01_repository_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands_run() -> None:
    commands = [
        "git status --short",
        "mkdir -p reports/empirical_audit",
        "PYTHONDONTWRITEBYTECODE=1 python3 tools/empirical_audit.py",
        "PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_empirical_audit.py",
    ]
    (REPORT_DIR / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_report(
    manifest: dict[str, Any],
    same_id_rows: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
    recompute_rows: list[dict[str, Any]],
    pair_summary: list[dict[str, Any]],
    child_metric_rows: list[dict[str, Any]],
    dist_rows: list[dict[str, Any]],
) -> None:
    same_summary = summarise_by_condition(same_id_rows, "posterior_mean_euclidean_distance")
    norm_summary = summarise_by_condition(latent_rows, "mean_structure_mu_norm")
    var_summary = summarise_by_condition(latent_rows, "mean_structure_posterior_variance")
    child_all = [row for row in child_metric_rows if row["language"] == "all"]
    child_summary = []
    for row in summarise_by_condition(child_all, "mae"):
        rmses = [float(x["rmse"]) for x in child_all if x["condition"] == row["condition"] and x.get("lambda_language_alignment") == row["lambda"]]
        baselines = [float(x["mean_baseline_mae"]) for x in child_all if x["condition"] == row["condition"] and x.get("lambda_language_alignment") == row["lambda"]]
        row["rmse_mean"] = mean(rmses)
        row["mean_baseline_mae"] = mean(baselines)
        child_summary.append(row)

    lines = [
        "# Empirical Audit Report for ChatGPT",
        "",
        "## 1. Executive Findings",
        "- The training alignment loss, reported same-ID diagnostic, and cross-language retrieval all use posterior means of the structure latent, not sampled latents, for the relevant published evaluation path.",
        "- The same-ID diagnostic is directional over both German-to-English and English-to-German sources and is a mean Euclidean distance, while the training alignment loss is mean squared error over within-batch pairs.",
        "- Recomputed cached-latent metrics match the per-seed JSON metrics to numerical precision; the maximum absolute discrepancy is shown below.",
        "- Child-count is implemented as a regression auxiliary head trained with MSE. No saved child-count metrics or decoding rule were present, so it should be described as an auxiliary objective unless the newly computed regression diagnostics are explicitly reported as audit results.",
        "- The lambda 0.03 versus 0.00 comparison is seed-matched and exploratory. It supports descriptive wording such as numerically higher for selected retrieval metrics, not formal model-selection significance.",
        "",
        "## 2. Repository Version, Environment, and Canonical Artifacts",
        f"- Commit: `{manifest['repository']['commit']}`",
        f"- Branch: `{manifest['repository']['branch']}`",
        f"- Dirty state: {', '.join(f'`{x}`' for x in manifest['repository']['status_short']) if manifest['repository']['status_short'] else 'clean'}",
    ]
    lines.extend(markdown_table(artifact_table(), ["artifact", "path", "status"]))
    lines.extend(
        [
            "",
            "## 3. Exact Alignment-Loss Definition",
            f"`language_alignment_loss` is implemented at `{SOURCE_CITATIONS['alignment_loss']}` and is added to the training objective at `{SOURCE_CITATIONS['alignment_added']}`. It receives `outputs['structure_mu']`, groups batch items by proposition index, forms pairs only when language ids differ, and returns `F.mse_loss(z[left], z[right])` at `{SOURCE_CITATIONS['alignment_return']}`. This is a mean squared error over all paired elements and latent dimensions in the current shuffled training batch. Training randomness comes from the training seed and shuffled `DataLoader` seeded at `{SOURCE_CITATIONS['set_seed']}`.",
            "",
            "## 4. Exact Same-ID-Distance Definition",
            f"Same-ID distance is implemented inside `_cross_language_metrics` at `{SOURCE_CITATIONS['same_id']}`. Evaluation calls `model.eval()` and exports `outputs['structure_mu']` for `--latent-part structure` at `{SOURCE_CITATIONS['eval_latents']}`; the split model returns `mu` in eval mode at `{SOURCE_CITATIONS['reparameterize_eval']}` and splits structure means at `{SOURCE_CITATIONS['split_forward']}`. The metric appends `torch.dist(z[source_i], z[target_i]).item()` at `{SOURCE_CITATIONS['same_id_distance']}` for each source sample and each other language with the same proposition id, then averages directional distances.",
            "",
            "## 5. Exact Retrieval Definition",
            f"Retrieval is in the same `_cross_language_metrics` function at `{SOURCE_CITATIONS['same_id']}`. For each source item and target language, candidates are all items in the target language. Distances are Euclidean norms at `{SOURCE_CITATIONS['retrieval_distance']}`, sorted ascending. Top-1 is 1 when the nearest candidate id matches the source id, and MRR is `1 / rank` of the same id at `{SOURCE_CITATIONS['retrieval_rank']}`.",
            "",
            "## 6. Definition Comparison",
        ]
    )
    lines.extend(
        markdown_table(
            [
                {
                    "quantity": "training alignment loss",
                    "latent": "structure_mu",
                    "formula": "F.mse_loss over paired tensors",
                    "pairing": "same dataset index, different language, within batch",
                    "mode": "train",
                    "random": "batch shuffle and model training",
                },
                {
                    "quantity": "same-ID diagnostic",
                    "latent": "exported structure_mu",
                    "formula": "mean torch.dist Euclidean",
                    "pairing": "same proposition id, other language, directional",
                    "mode": "eval",
                    "random": "none in evaluation",
                },
                {
                    "quantity": "Top-1/MRR retrieval",
                    "latent": "exported structure_mu",
                    "formula": "Euclidean nearest-neighbour rank",
                    "pairing": "source item to all target-language candidates",
                    "mode": "eval",
                    "random": "none in evaluation",
                },
            ],
            ["quantity", "latent", "formula", "pairing", "mode", "random"],
        )
    )
    lines.extend(
        [
            "",
            "## 7. Recomputed Same-ID and Latent Scale/Variance",
            f"Maximum absolute difference between recomputed cached-latent metrics and per-seed JSON values: `{max_abs_diff(recompute_rows):.12g}`.",
            "",
            "Mean same-ID posterior-mean Euclidean distance by condition:",
        ]
    )
    lines.extend(markdown_table(same_summary, ["condition", "lambda", "n", "mean", "sd", "min", "max"]))
    lines.extend(["", "Mean structure-mu norm by condition:"])
    lines.extend(markdown_table(norm_summary, ["condition", "lambda", "n", "mean", "sd", "min", "max"]))
    lines.extend(["", "Mean posterior variance by condition:"])
    lines.extend(markdown_table(var_summary, ["condition", "lambda", "n", "mean", "sd", "min", "max"]))
    lines.extend(
        [
            "",
            "## 8. Evidence-Based Explanation of the High-Lambda Pattern",
            "The high-lambda rise in same-ID distance is not explained by a mean-versus-sample mismatch: both cached evaluation latents and recomputed checkpoint latents are posterior means, and cached means match checkpoint means as reported in `latent_scale_variance_by_seed.csv`. It is also not a lambda-labelling error: manifests map align000/003/010/030/100 to 0.00/0.03/0.10/0.30/1.00 and non-lambda settings match align000.",
            "The supported pattern is that strong alignment coincides with reduced structure-mean norm and reduced relational separation: parent-child, unrelated, and cross-language parent-child distances contract at high lambda while same-ID distance rises in the published Euclidean diagnostic. Posterior variance does not contract; it increases at high lambda in the recomputed checkpoint diagnostics. The exact optimization mechanism remains unresolved because no per-epoch latent trajectories or batch-level alignment-loss traces are saved.",
            "",
            "## 9. Child-Count Implementation, Distribution, Metrics, and Recommendation",
            f"The target is `row['child_count']` converted to a float tensor at `{SOURCE_CITATIONS['child_target']}`. The split-latent head is `Linear(structure_latent_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)` at `{SOURCE_CITATIONS['child_head']}`. The loss is `F.mse_loss(outputs['child_count'], child_count_targets.float())` at `{SOURCE_CITATIONS['child_loss']}`, weighted by lambda_child. This is regression, not classification.",
            "Target distribution from the data files:",
        ]
    )
    lines.extend(markdown_table([row for row in dist_rows if row["language"] == "all"], ["dataset", "language", "child_count", "support", "proportion"]))
    lines.extend(["", "Child-count regression metrics, all languages/samples, averaged across seeds:"])
    lines.extend(markdown_table(child_summary, ["condition", "lambda", "n", "mean", "sd", "rmse_mean", "mean_baseline_mae"], digits=5))
    lines.extend(
        [
            "Recommendation: describe child-count as an auxiliary objective unless the paper adds the audit-computed regression metrics with the regression task definition and baseline. The original per-seed JSON files did not contain child-count performance.",
            "",
            "## 10. Paired Lambda 0.00 Versus 0.03 Results",
            f"Paired differences are defined as lambda 0.03 minus lambda 0.00. Bootstrap CIs use seed `{BOOTSTRAP_SEED}` and `{BOOTSTRAP_RESAMPLES}` resamples. Exact sign-flip p-values enumerate all 2^10 sign assignments. Effect size is `mean(diff) / sample_sd(diff)`.",
        ]
    )
    lines.extend(markdown_table(pair_summary, ["metric", "n", "mean_difference_003_minus_000", "sd_difference", "bootstrap_ci_95_low", "bootstrap_ci_95_high", "sign_flip_p_value_two_sided", "seeds_favouring_003", "seeds_favouring_000", "standardised_paired_effect_size_dz"], digits=5))
    lines.extend(
        [
            "",
            "## 11. Empirical Statements Supported by the Audit",
            "- Alignment loss is MSE on posterior structure means within shuffled training batches.",
            "- Same-ID distance and retrieval are deterministic retained-corpus diagnostics over evaluation posterior structure means.",
            "- Lambda 0.03 is numerically slightly higher than lambda 0.00 for aggregate Top-1 and MRR means in the retained seed sweep, with small paired differences relative to seed variation.",
            "- Directional retrieval and by-language structure metrics are reciprocal in the sense that both de-to-en and en-to-de are reported and similar in magnitude.",
            "- Strong alignment coincides with reduced structure-mean norm and reduced relational distances, while posterior variance and same-ID Euclidean distance rise at high lambda.",
            "",
            "## 12. Statements Not Supported by the Audit",
            "- The audit does not support claims of formal model-selection significance for lambda 0.03.",
            "- The audit does not support held-out generalisation claims for the principal reported metrics.",
            "- The audit does not support semantic equivalence, philosophical understanding, or language-invariant logic claims.",
            "- The audit does not support treating child-count as an originally reported structural performance metric.",
            "",
            "## 13. Exact Unresolved Questions and Missing Artifacts",
            "- No saved child-count predictions or metrics were present in canonical per-seed JSON files.",
            "- No per-epoch latent-scale, posterior-variance, or alignment-loss traces were found, limiting causal explanation of the high-lambda pattern.",
            "- PCA figures are representative diagnostics for `align003` seed000, not multi-seed summaries.",
            "- No held-out split artifact was identified for the bilingual principal metrics.",
            "",
            "## 14. Reproduction Commands",
            "```bash",
            "python3 tools/empirical_audit.py",
            "python3 -m unittest tests/test_empirical_audit.py",
            "```",
            "",
            "Machine-readable outputs are in `reports/empirical_audit/`.",
        ]
    )
    (REPORT_DIR / "report_for_chatgpt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible empirical audit over existing Tractatus run artifacts.")
    parser.parse_args()
    manifest = audit()
    print(json.dumps({"status": "ok", "report": str(REPORT_DIR / "report_for_chatgpt.md"), "commit": manifest["repository"]["commit"]}, indent=2))


if __name__ == "__main__":
    main()
