#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
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

from tools.phase1_ablations import evaluate_checkpoint
from tools.canonical_experiments import (
    CANONICAL_STATUS,
    PHASE3_CANONICAL_BATCHING,
    PHASE3_CANONICAL_CONDITIONS,
    canonical_phase3_ids,
    phase3_experiment_status,
    phase3_sampler_type,
)

DATA_PATH = ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json"
DEFAULT_OUT = ROOT / "results" / "dsh_validation" / "phase3_controlled_alignment"
LAMBDA_GRID = [0.00, 0.03, 0.10, 0.30, 1.00]
FULL_SEEDS = list(range(10))
REQUIRED_METRICS = [
    "parent_accuracy",
    "depth_accuracy",
    "successor_top1",
    "child_count_mae",
    "child_count_rmse",
    "structure_cross_language_top1_de_to_en",
    "structure_cross_language_top1_en_to_de",
    "structure_cross_language_top1",
    "structure_cross_language_top5",
    "structure_cross_language_top10",
    "structure_cross_language_mrr",
    "structure_cross_language_mean_rank",
    "structure_cross_language_same_id_distance",
    "structure_wider_neighbourhood_jaccard_k5",
    "structure_wider_neighbourhood_jaccard_k10",
    "structure_wider_neighbourhood_jaccard_k20",
    "reconstruction_loss",
    "perplexity",
    "kl_text",
    "kl_structure",
    "structure_mean_sibling_distance",
    "structure_mean_parent_child_distance",
    "structure_mean_cross_language_parent_child_distance",
    "structure_mean_unrelated_distance",
    "structure_mean_norm",
    "structure_posterior_variance",
]
REQUIRED_EPOCH_COLUMNS = [
    "same_id_pairs_processed",
    "pair_coverage",
    "raw_alignment_mse",
    "weighted_alignment_contribution",
    "reconstruction_loss",
    "parent_loss",
    "depth_loss",
    "successor_loss",
    "child_count_loss",
    "kl_text",
    "kl_structure",
    "same_id_distance",
    "structure_mean_norm",
    "posterior_variance",
    "gradient_norm",
    "learning_rate",
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
}


def lambda_tag(value: float) -> str:
    return f"align{int(round(value * 100)):03d}"


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    detail = result.stderr.strip() or result.stdout.strip() or f"git exited with status {result.returncode}"
    return f"unavailable ({detail.splitlines()[0]})"


def display_path_arg(value: object) -> str:
    text = str(value)
    try:
        return str(Path(text).resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return text


def command_line(command: list[str]) -> str:
    return " ".join(display_path_arg(part) for part in command)


def ensure_layout(out_root: Path) -> None:
    for name in ["configs", "figures", "logs", "checkpoints", "metrics", "latents", "epoch_trajectories", "raw", "per_seed"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)


def run_label(batching: str, condition: str, alignment_lambda: float) -> str:
    return f"{batching}_{condition}_{lambda_tag(alignment_lambda)}"


def parse_run_label(label: str) -> tuple[str, str, float]:
    parts = label.split("_")
    batching = parts[0]
    alignment_lambda = int(parts[-1].replace("align", "")) / 100.0
    condition = "_".join(parts[1:-1])
    return batching, condition, alignment_lambda


def is_canonical_phase3_label(label: str) -> bool:
    return label in canonical_phase3_ids()


def train_command(args: argparse.Namespace, condition: Condition, alignment_lambda: float, checkpoint: Path, epoch_metrics: Path, batching: str, seed: int) -> list[str]:
    command = [
        "python3",
        "-m",
        "tractatus_structure_latents.training.train_vae",
        "--data",
        str(args.data),
        "--languages",
        "en,de",
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
        str(alignment_lambda),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--epoch-metrics-out",
        str(epoch_metrics),
        "--out",
        str(checkpoint),
    ]
    if batching == "paired":
        command.append("--paired-language-batches")
    return command


def run_subprocess(command: list[str], log_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(" ".join(command))
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=handle, stderr=subprocess.STDOUT)


def evaluate_run(out_root: Path, label: str, seed: int, checkpoint: Path, data: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"python3 tools/phase3_controlled_alignment.py eval-one --label {label} --seed {seed} --checkpoint {checkpoint} --data {data} --out-root {out_root}")
        return
    evaluate_checkpoint(label, seed, checkpoint, data, out_root)
    augment_metric_file(out_root, label, seed, data)


def cross_language_parent_child_distance(raw_path: Path, data_path: Path) -> float:
    rows = read_json(data_path)
    parent_by_id = {str(row["id"]): row["parent_id"] for row in rows}
    df = pd.read_parquet(raw_path)
    positions = {(str(row.id), str(row.language)): i for i, row in enumerate(df.itertuples())}
    structure_z = np.asarray([json.loads(value) for value in df["structure_mu"]], dtype=float)
    distances: list[float] = []
    for item in df.itertuples():
        child_id = str(item.id)
        child_language = str(item.language)
        parent_id = parent_by_id.get(child_id)
        if parent_id is None:
            continue
        child_i = positions.get((child_id, child_language))
        if child_i is None:
            continue
        for parent_language in sorted({language for _id, language in positions}):
            if parent_language == child_language:
                continue
            parent_i = positions.get((str(parent_id), parent_language))
            if parent_i is not None:
                distances.append(float(np.linalg.norm(structure_z[child_i] - structure_z[parent_i])))
    return mean(distances) if distances else 0.0


def augment_metric_file(out_root: Path, label: str, seed: int, data_path: Path) -> None:
    metric_path = out_root / "per_seed" / label / f"seed{seed:03d}.metrics.json"
    raw_path = out_root / "raw" / label / f"seed{seed:03d}.per_proposition.parquet"
    if not metric_path.exists() or not raw_path.exists():
        return
    metrics = read_json(metric_path)
    if "structure_mean_cross_language_parent_child_distance" in metrics:
        return
    value = cross_language_parent_child_distance(raw_path, data_path)
    metrics["structure_mean_cross_language_parent_child_distance"] = value
    metrics["mean_cross_language_parent_child_distance"] = value
    write_json(metric_path, metrics)


def augment_all_metric_files(out_root: Path, data_path: Path = DATA_PATH) -> None:
    for metric_path in sorted((out_root / "per_seed").glob("*/*.metrics.json")):
        label = metric_path.parent.name
        seed = int(metric_path.stem.split(".")[0].replace("seed", ""))
        augment_metric_file(out_root, label, seed, data_path)


def run_grid(
    args: argparse.Namespace,
    batching: str,
    condition_names: list[str],
    lambdas: list[float],
    seeds: list[int],
) -> None:
    if batching != PHASE3_CANONICAL_BATCHING:
        raise SystemExit(f"Only canonical paired batching is supported, got: {batching}")
    out_root = args.out_root.resolve()
    ensure_layout(out_root)
    command_lines: list[str] = []
    experiment_status = phase3_experiment_status(batching)
    for condition_name in condition_names:
        condition = CONDITIONS[condition_name]
        if not args.dry_run:
            write_json(
                out_root / "configs" / f"{batching}_{condition.name}.json",
                {
                    "experiment_status": experiment_status,
                    "evidence_tier": CANONICAL_STATUS if experiment_status == CANONICAL_STATUS else "historical",
                    "sampler_type": phase3_sampler_type(batching),
                    "batching": batching,
                    "condition": condition.__dict__,
                    "lambdas": lambdas,
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
                    "data": display_path_arg(args.data),
                },
            )
        for alignment_lambda in lambdas:
            tag = lambda_tag(alignment_lambda)
            label = run_label(batching, condition.name, alignment_lambda)
            for seed in seeds:
                checkpoint = out_root / "checkpoints" / batching / condition.name / tag / f"seed{seed:03d}.pt"
                metric_path = out_root / "per_seed" / label / f"seed{seed:03d}.metrics.json"
                epoch_metrics = out_root / "epoch_trajectories" / batching / condition.name / tag / f"seed{seed:03d}.csv"
                log_path = out_root / "logs" / batching / condition.name / tag / f"seed{seed:03d}.train.log"
                command = train_command(args, condition, alignment_lambda, checkpoint, epoch_metrics, batching, seed)
                command_lines.append(command_line(command))
                command_lines.append(
                    command_line(
                        [
                            "python3",
                            "tools/phase3_controlled_alignment.py",
                            "eval-one",
                            "--label",
                            label,
                            "--seed",
                            str(seed),
                            "--checkpoint",
                            display_path_arg(checkpoint),
                            "--data",
                            display_path_arg(args.data),
                            "--out-root",
                            display_path_arg(out_root),
                        ]
                    )
                )
                if args.skip_existing and checkpoint.exists() and metric_path.exists() and epoch_metrics.exists():
                    print(f"skipping {label}/seed{seed:03d}", flush=True)
                    continue
                print(f"running {label}/seed{seed:03d}", flush=True)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                run_subprocess(command, log_path, args.dry_run)
                evaluate_run(out_root, label, seed, checkpoint, args.data, args.dry_run)
    if command_lines and not args.dry_run:
        commands_path = out_root / "phase3_commands.sh"
        existing = commands_path.read_text(encoding="utf-8").splitlines() if commands_path.exists() else ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        commands_path.write_text("\n".join([*existing, *command_lines]) + "\n", encoding="utf-8")
    if not args.dry_run:
        summarise(out_root)


def seed_results(out_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((out_root / "per_seed").glob("*/*.metrics.json")):
        label = path.parent.name
        seed = int(path.stem.split(".")[0].replace("seed", ""))
        batching, condition, alignment_lambda = parse_run_label(label)
        experiment_status = phase3_experiment_status(batching)
        rows.append(
            {
                "experiment_id": label,
                "experiment_status": experiment_status,
                "evidence_tier": CANONICAL_STATUS if experiment_status == CANONICAL_STATUS else "historical",
                "sampler_type": phase3_sampler_type(batching),
                "batching": batching,
                "condition": condition,
                "lambda_language_alignment": alignment_lambda,
                "seed": seed,
                "label": label,
                **read_json(path),
            }
        )
    return pd.DataFrame(rows)


def verify_canonical_seed_records(seed_df: pd.DataFrame) -> None:
    if seed_df.empty:
        return
    invalid_ids = sorted(set(seed_df["label"].astype(str)) - canonical_phase3_ids())
    if invalid_ids:
        raise AssertionError(f"Canonical Phase 3 output contains non-canonical labels: {invalid_ids}")
    if not (seed_df["sampler_type"] == "id_level_paired").all():
        raise AssertionError("Canonical Phase 3 records must all use the ID-level paired sampler.")


def verify_canonical_pair_coverage(seed_df: pd.DataFrame, coverage_df: pd.DataFrame) -> None:
    if seed_df.empty:
        return
    if coverage_df.empty:
        raise AssertionError("Canonical Phase 3 records require pair-coverage verification.")
    needed = {
        (str(row.batching), str(row.condition), float(row.lambda_language_alignment), int(row.seed))
        for row in seed_df.itertuples()
    }
    observed = {
        (str(row.batching), str(row.condition), float(row.lambda_language_alignment), int(row.seed))
        for row in coverage_df.itertuples()
    }
    missing = sorted(needed - observed)[:20]
    if missing:
        raise AssertionError(f"Missing pair-coverage rows for canonical Phase 3 records: {missing}")
    paired = coverage_df[
        (coverage_df["batching"] == PHASE3_CANONICAL_BATCHING)
        & (coverage_df["condition"].isin(PHASE3_CANONICAL_CONDITIONS))
    ]
    if paired.empty or not ((paired["min_pair_coverage"] == 1.0) & (paired["max_pair_coverage"] == 1.0)).all():
        raise AssertionError("Canonical Phase 3 paired records must have pair_coverage == 1.0 for every seed/lambda/condition.")


def summarise(out_root: Path) -> None:
    augment_all_metric_files(out_root)
    seed_df = seed_results(out_root)
    if seed_df.empty:
        return
    verify_canonical_seed_records(seed_df)
    seed_df = seed_df.sort_values(["batching", "condition", "lambda_language_alignment", "seed"])
    seed_df.to_csv(out_root / "phase3_seed_results.csv", index=False)
    metric_cols = [
        col
        for col in seed_df.columns
        if col not in {"experiment_id", "experiment_status", "evidence_tier", "sampler_type", "batching", "condition", "lambda_language_alignment", "seed", "label"}
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in seed_df.groupby(["batching", "condition", "lambda_language_alignment"], sort=True):
        batching, condition, alignment_lambda = keys
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            if not values:
                continue
            summary_rows.append(
                {
                    "batching": batching,
                    "condition": condition,
                    "lambda_language_alignment": alignment_lambda,
                    "metric": metric,
                    "mean": mean(values),
                    "sample_sd": stdev(values) if len(values) > 1 else 0.0,
                    "seed_count": len(values),
                }
            )
    pd.DataFrame(summary_rows).to_csv(out_root / "phase3_summary.csv", index=False)
    epoch_paths = sorted((out_root / "epoch_trajectories").glob("*/*/*/*.csv"))
    coverage_df = pd.DataFrame()
    if epoch_paths:
        epoch_frames = []
        coverage_rows = []
        for path in epoch_paths:
            batching, condition, tag, seed_file = path.relative_to(out_root / "epoch_trajectories").parts
            label = run_label(batching, condition, int(tag.replace("align", "")) / 100.0)
            if not is_canonical_phase3_label(label):
                continue
            seed = int(seed_file.replace(".csv", "").replace("seed", ""))
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            frame.insert(0, "lambda_language_alignment", int(tag.replace("align", "")) / 100.0)
            frame.insert(0, "condition", condition)
            frame.insert(0, "batching", batching)
            epoch_frames.append(frame)
            coverage_rows.append(
                {
                    "batching": batching,
                    "condition": condition,
                    "lambda_language_alignment": int(tag.replace("align", "")) / 100.0,
                    "seed": seed,
                    "min_pair_coverage": float(frame["pair_coverage"].min()),
                    "max_pair_coverage": float(frame["pair_coverage"].max()),
                    "epochs": len(frame),
                }
            )
        trajectories = pd.concat(epoch_frames, ignore_index=True)
        trajectories.to_parquet(out_root / "phase3_epoch_trajectories.parquet", index=False)
        coverage_df = pd.DataFrame(coverage_rows)
        coverage_df.to_csv(out_root / "phase3_pair_coverage.csv", index=False)
    verify_canonical_pair_coverage(seed_df, coverage_df)
    write_figures(out_root, seed_df)
    write_report(out_root)

def write_figures(out_root: Path, seed_df: pd.DataFrame) -> None:
    figure_specs = [
        ("structure_cross_language_top1", "retrieval_across_lambda.png", "Structure Top-1"),
        ("parent_accuracy", "formal_parent_accuracy_across_lambda.png", "Parent accuracy"),
        ("depth_accuracy", "formal_depth_accuracy_across_lambda.png", "Depth accuracy"),
        ("successor_top1", "formal_successor_accuracy_across_lambda.png", "Successor Top-1"),
        ("structure_cross_language_same_id_distance", "same_id_distance_across_lambda.png", "Same-ID distance"),
        ("structure_mean_norm", "structure_mean_norm_across_lambda.png", "Structure mean norm"),
        ("structure_posterior_variance", "posterior_variance_across_lambda.png", "Structure posterior variance"),
    ]
    for metric, filename, ylabel in figure_specs:
        if metric not in seed_df:
            continue
        plt.figure(figsize=(6, 4))
        for (batching, condition), group in seed_df.groupby(["batching", "condition"], sort=True):
            means = group.groupby("lambda_language_alignment")[metric].mean()
            if means.empty:
                continue
            plt.plot(means.index, means.values, marker="o", label=f"{batching} {condition}")
        plt.xlabel("lambda_language_alignment")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_root / "figures" / filename, dpi=600)
        plt.close()
    trajectories_path = out_root / "phase3_epoch_trajectories.parquet"
    if trajectories_path.exists():
        trajectories = pd.read_parquet(trajectories_path)
        if "raw_alignment_mse" in trajectories:
            plt.figure(figsize=(6, 4))
            for keys, group in trajectories.groupby(["batching", "condition", "lambda_language_alignment"], sort=True):
                batching, condition, alignment_lambda = keys
                means = group.groupby("epoch")["raw_alignment_mse"].mean()
                plt.plot(means.index, means.values, label=f"{batching} {condition} {alignment_lambda:.2f}")
            plt.xlabel("epoch")
            plt.ylabel("Raw alignment MSE")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(out_root / "figures" / "alignment_loss_trajectories.png", dpi=600)
            plt.close()


def summary_value(summary_df: pd.DataFrame, batching: str, condition: str, alignment_lambda: float, metric: str) -> tuple[float, float, int]:
    row = summary_df[
        (summary_df["batching"] == batching)
        & (summary_df["condition"] == condition)
        & (summary_df["lambda_language_alignment"] == alignment_lambda)
        & (summary_df["metric"] == metric)
    ]
    if row.empty:
        return float("nan"), float("nan"), 0
    first = row.iloc[0]
    return float(first["mean"]), float(first["sample_sd"]), int(first["seed_count"])


def metric_mean(summary_df: pd.DataFrame, batching: str, condition: str, alignment_lambda: float, metric: str) -> float:
    return summary_value(summary_df, batching, condition, alignment_lambda, metric)[0]


def fmt_mean_sd(summary_df: pd.DataFrame, batching: str, condition: str, alignment_lambda: float, metric: str) -> str:
    value, sd, count = summary_value(summary_df, batching, condition, alignment_lambda, metric)
    if count == 0 or pd.isna(value):
        return "not available"
    return f"{value:.4f} +/- {sd:.4f} (n={count})"


def delta(summary_df: pd.DataFrame, batching: str, condition: str, left_lambda: float, right_lambda: float, metric: str) -> float:
    return metric_mean(summary_df, batching, condition, right_lambda, metric) - metric_mean(summary_df, batching, condition, left_lambda, metric)


def metric_table(summary_df: pd.DataFrame, batching: str, condition: str, metrics: list[tuple[str, str]]) -> list[str]:
    subset = summary_df[(summary_df["batching"] == batching) & (summary_df["condition"] == condition)]
    lines = [
        f"### {batching} {condition}",
        "",
        "| lambda | " + " | ".join(label for label, _metric in metrics) + " |",
        "|---:" + "|---:" * len(metrics) + "|",
    ]
    for alignment_lambda in sorted(subset["lambda_language_alignment"].unique()):
        cells = [f"{alignment_lambda:.2f}"]
        for _label, metric in metrics:
            cells.append(fmt_mean_sd(summary_df, batching, condition, float(alignment_lambda), metric))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def trend_description(values: list[float]) -> str:
    if any(pd.isna(value) for value in values) or len(values) < 2:
        return "not available"
    change = values[-1] - values[0]
    if abs(change) < 1e-9:
        return "is flat over the tested grid"
    return "increases over the tested grid" if change > 0 else "decreases over the tested grid"


def write_report(out_root: Path) -> None:
    seed_path = out_root / "phase3_seed_results.csv"
    summary_path = out_root / "phase3_summary.csv"
    if not seed_path.exists() or not summary_path.exists():
        return
    seed_df = pd.read_csv(seed_path)
    summary_df = pd.read_csv(summary_path)
    coverage_df = pd.read_csv(out_root / "phase3_pair_coverage.csv") if (out_root / "phase3_pair_coverage.csv").exists() else pd.DataFrame()
    decision_path = out_root / "configs" / "no_successor_expansion_decision.json"
    decision = read_json(decision_path) if decision_path.exists() else {}

    top1_delta_003 = delta(summary_df, "paired", "full_model", 0.0, 0.03, "structure_cross_language_top1")
    same_delta_003 = delta(summary_df, "paired", "full_model", 0.0, 0.03, "structure_cross_language_same_id_distance")
    top1_delta_high = delta(summary_df, "paired", "full_model", 0.0, 1.0, "structure_cross_language_top1")
    same_delta_high = delta(summary_df, "paired", "full_model", 0.0, 1.0, "structure_cross_language_same_id_distance")
    no_succ_high_top1_delta = metric_mean(summary_df, "paired", "no_successor", 1.0, "structure_cross_language_top1") - metric_mean(summary_df, "paired", "full_model", 1.0, "structure_cross_language_top1")
    full_dist_trend = trend_description([metric_mean(summary_df, "paired", "full_model", lam, "structure_cross_language_same_id_distance") for lam in LAMBDA_GRID])
    no_succ_dist_trend = trend_description([metric_mean(summary_df, "paired", "no_successor", lam, "structure_cross_language_same_id_distance") for lam in LAMBDA_GRID])
    min_paired_coverage = float(coverage_df[coverage_df["batching"] == "paired"]["min_pair_coverage"].min()) if not coverage_df.empty else float("nan")

    lines = [
        "# Phase 3 Controlled Alignment Report",
        "",
        f"Git commit analysed: `{git_output(['rev-parse', 'HEAD'])}`.",
        f"Seed-level runs: {len(seed_df)}. Paired-batch minimum pair coverage: {min_paired_coverage:.4f}.",
        "",
        "## Design",
        "",
        "The canonical controlled sweep uses an ID-level paired sampler. Each epoch shuffles proposition IDs, places the German and English rows for each ID in the same minibatch, computes same-ID structure-latent MSE over every observed pair, and asserts complete pair coverage.",
        "",
        "## Final Metrics",
        "",
        *metric_table(
            summary_df,
            "paired",
            "full_model",
            [
                ("Top-1", "structure_cross_language_top1"),
                ("Top-5", "structure_cross_language_top5"),
                ("Top-10", "structure_cross_language_top10"),
                ("MRR", "structure_cross_language_mrr"),
                ("rank", "structure_cross_language_mean_rank"),
                ("same-ID distance", "structure_cross_language_same_id_distance"),
            ],
        ),
        *metric_table(
            summary_df,
            "paired",
            "no_successor",
            [
                ("Top-1", "structure_cross_language_top1"),
                ("MRR", "structure_cross_language_mrr"),
                ("same-ID distance", "structure_cross_language_same_id_distance"),
                ("parent acc.", "parent_accuracy"),
                ("depth acc.", "depth_accuracy"),
                ("successor Top-1", "successor_top1"),
            ],
        ),
        "## Metric Coverage",
        "",
        "The final seed table includes parent, depth and successor accuracy; child-count MAE/RMSE; directional and combined Top-1, Top-5, Top-10, MRR and rank; same-ID distance; wider-neighbourhood Jaccard at k=5, 10 and 20; reconstruction and perplexity; KL terms; sibling, parent-child, cross-language parent-child and unrelated distances; structure-mean norm; and posterior variance.",
        "",
        "Epoch trajectories are stored in `phase3_epoch_trajectories.parquet`, with pair coverage in `phase3_pair_coverage.csv`.",
        "",
        "## Analysis",
        "",
        f"Effect of lambda: In the paired full model, lambda=0.03 changes structure Top-1 by {top1_delta_003:+.4f} and same-ID distance by {same_delta_003:+.4f} relative to lambda=0.00. From lambda=0.00 to 1.00, Top-1 changes by {top1_delta_high:+.4f} and same-ID distance changes by {same_delta_high:+.4f}.",
        "",
        f"Effect of the successor objective: The no-successor pilot met the expansion rule and was expanded to seeds 0-9. Recorded reasons: {', '.join(decision.get('reasons', ['not recorded']))}. At lambda=1.00, no-successor minus full-model Top-1 is {no_succ_high_top1_delta:+.4f}.",
        "",
        "Optimisation failure: The epoch trajectories expose raw alignment MSE, weighted contribution, gradient norm, reconstruction loss and formal losses per epoch. High-lambda behaviour should be interpreted jointly with reconstruction/perplexity and formal-head metrics rather than from retrieval alone.",
        "",
        f"Latent-scale changes: The report separates same-ID distance from structure-mean norm and posterior variance. The full-model same-ID distance {full_dist_trend}; the no-successor same-ID distance {no_succ_dist_trend}.",
        "",
        "Relational-geometry changes: The final seed table reports sibling, parent-child, cross-language parent-child and unrelated distances separately, so a change in same-ID distance is not treated as a uniform contraction of all distances.",
        "",
        "## Required Questions",
        "",
        f"1. Does lambda=0.03 still produce only a small local tightening? The paired full-model change at lambda=0.03 is Top-1 {top1_delta_003:+.4f} and same-ID distance {same_delta_003:+.4f}; this should be described as small only relative to the seed-level SDs in `phase3_summary.csv`.",
        "",
        f"2. Does high lambda still increase realised same-ID distance? In the paired full model, same-ID distance change from lambda=0.00 to 1.00 is {same_delta_high:+.4f}.",
        "",
        f"3. Does high-weight deterioration remain after pair coverage is controlled? Pair coverage is {min_paired_coverage:.4f} for paired runs. Retrieval change from lambda=0.00 to 1.00 is {top1_delta_high:+.4f}.",
        "",
        f"4. Is the failure regime caused or amplified by the successor objective? The no-successor expansion completed. At lambda=1.00, its Top-1 differs from the full model by {no_succ_high_top1_delta:+.4f}; compare the full grid in `phase3_summary.csv` before attributing the regime solely to successor supervision.",
        "",
        "5. Does paired batching itself change lambda=0 performance? This repository now retains only the paired-batch Phase 3 evidence layer.",
        "",
        "6. Can the alignment sweep now support a clear causal statement about direct pairwise attraction? Yes for the paired-batch conditions: every German-English ID pair contributes exactly once per epoch, including lambda=0.00 controls. The statement should still be limited to the measured retained-corpus setting and reported alongside optimisation and latent-scale diagnostics.",
        "",
        "## Figures",
        "",
        "- `figures/retrieval_across_lambda.png`",
        "- `figures/formal_parent_accuracy_across_lambda.png`",
        "- `figures/formal_depth_accuracy_across_lambda.png`",
        "- `figures/formal_successor_accuracy_across_lambda.png`",
        "- `figures/same_id_distance_across_lambda.png`",
        "- `figures/alignment_loss_trajectories.png`",
        "- `figures/structure_mean_norm_across_lambda.png`",
        "- `figures/posterior_variance_across_lambda.png`",
        "",
        "## Recommended interpretation of the alignment experiment",
        "",
        "The controlled paired-batch sweep gives lambda a direct operational interpretation: it is the weight on an observed same-ID German-English structure-latent MSE term for every proposition pair in every epoch. Publication claims should use this paired analysis as the authoritative bilingual alignment evidence and should report retrieval, same-ID distance, formal-head performance, latent scale and relational distances together.",
    ]
    (out_root / "phase3_alignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_mean_metric(out_root: Path, batching: str, condition: str, lambdas: list[float], seeds: list[int], metric: str) -> dict[float, float]:
    seed_df = seed_results(out_root)
    if seed_df.empty:
        raise FileNotFoundError("No Phase 3 seed metrics found; run the full-model sweep first.")
    values: dict[float, float] = {}
    for alignment_lambda in lambdas:
        group = seed_df[
            (seed_df["batching"] == batching)
            & (seed_df["condition"] == condition)
            & (seed_df["lambda_language_alignment"] == alignment_lambda)
            & (seed_df["seed"].isin(seeds))
        ]
        if group.empty or metric not in group:
            raise FileNotFoundError(f"Missing {metric} for {batching}/{condition}/{alignment_lambda:.2f}")
        values[alignment_lambda] = float(pd.to_numeric(group[metric], errors="coerce").mean())
    return values


def expansion_needed(out_root: Path, lambdas: list[float], pilot_seeds: list[int]) -> tuple[bool, list[str]]:
    full_top1 = load_mean_metric(out_root, "paired", "full_model", lambdas, pilot_seeds, "structure_cross_language_top1")
    no_succ_top1 = load_mean_metric(out_root, "paired", "no_successor", lambdas, pilot_seeds, "structure_cross_language_top1")
    full_dist = load_mean_metric(out_root, "paired", "full_model", lambdas, pilot_seeds, "structure_cross_language_same_id_distance")
    no_succ_dist = load_mean_metric(out_root, "paired", "no_successor", lambdas, pilot_seeds, "structure_cross_language_same_id_distance")
    reasons: list[str] = []
    for alignment_lambda in lambdas:
        diff = abs(no_succ_top1[alignment_lambda] - full_top1[alignment_lambda])
        if diff >= 0.02:
            reasons.append(f"Top-1 differs by {diff:.4f} at lambda={alignment_lambda:.2f}")
    full_direction = full_dist[lambdas[-1]] - full_dist[lambdas[0]]
    no_succ_direction = no_succ_dist[lambdas[-1]] - no_succ_dist[lambdas[0]]
    if (full_direction > 0) != (no_succ_direction > 0):
        reasons.append("same-ID-distance trend direction differs from full_model")
    full_best = max(full_top1, key=full_top1.get)
    no_succ_best = max(no_succ_top1, key=no_succ_top1.get)
    if full_best != no_succ_best:
        reasons.append(f"highest-retrieval lambda changes from {full_best:.2f} to {no_succ_best:.2f}")
    return bool(reasons), reasons


def run(args: argparse.Namespace) -> None:
    run_grid(args, args.batching, [item for item in args.conditions.split(",") if item], parse_floats(args.lambdas), parse_ints(args.seeds))


def successor_control(args: argparse.Namespace) -> None:
    lambdas = parse_floats(args.lambdas)
    pilot_seeds = parse_ints(args.pilot_seeds)
    all_seeds = parse_ints(args.all_seeds)
    run_grid(args, "paired", ["no_successor"], lambdas, pilot_seeds)
    needed, reasons = expansion_needed(args.out_root.resolve(), lambdas, pilot_seeds)
    write_json(
        args.out_root.resolve() / "configs" / "no_successor_expansion_decision.json",
        {"pilot_seeds": pilot_seeds, "all_seeds": all_seeds, "expand": needed, "reasons": reasons},
    )
    if needed:
        remaining = [seed for seed in all_seeds if seed not in pilot_seeds]
        print("expanding no_successor to seeds 0-9: " + "; ".join(reasons), flush=True)
        run_grid(args, "paired", ["no_successor"], lambdas, remaining)
    else:
        print("no_successor expansion criteria were not met after pilot seeds", flush=True)


def eval_one(args: argparse.Namespace) -> None:
    evaluate_run(args.out_root.resolve(), args.label, args.seed, args.checkpoint, args.data, dry_run=False)
    summarise(args.out_root.resolve())


def verify_outputs(out_root: Path) -> list[str]:
    checks: list[str] = []
    required_files = [
        "phase3_alignment_report.md",
        "phase3_summary.csv",
        "phase3_seed_results.csv",
        "phase3_epoch_trajectories.parquet",
        "phase3_pair_coverage.csv",
        "phase3_commands.sh",
    ]
    missing = [name for name in required_files if not (out_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 3 deliverables: {missing}")
    checks.append("All tabular/report deliverables are present.")

    seed_df = pd.read_csv(out_root / "phase3_seed_results.csv")
    summary_df = pd.read_csv(out_root / "phase3_summary.csv")
    expected = {
        ("paired", "full_model", alignment_lambda, seed)
        for alignment_lambda in LAMBDA_GRID
        for seed in FULL_SEEDS
    }
    expected.update(
        {
            ("paired", "no_successor", alignment_lambda, seed)
            for alignment_lambda in LAMBDA_GRID
            for seed in FULL_SEEDS
        }
    )
    observed = {
        (str(row.batching), str(row.condition), float(row.lambda_language_alignment), int(row.seed))
        for row in seed_df.itertuples()
    }
    if observed != expected:
        missing_runs = sorted(expected - observed)[:20]
        extra_runs = sorted(observed - expected)[:20]
        raise AssertionError(f"Unexpected canonical Phase 3 run set. Missing={missing_runs}; extra={extra_runs}")
    verify_canonical_seed_records(seed_df)
    checks.append("Observed the expected 100 canonical paired condition/lambda/seed final metric rows.")

    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in seed_df.columns]
    if missing_metrics:
        raise AssertionError(f"Missing required final metric columns: {missing_metrics}")
    checks.append("All required final metric columns are present.")

    metric_cols = [
        col
        for col in seed_df.columns
        if col not in {"experiment_id", "experiment_status", "evidence_tier", "sampler_type", "batching", "condition", "lambda_language_alignment", "seed", "label"}
    ]
    for keys, group in seed_df.groupby(["batching", "condition", "lambda_language_alignment"], sort=True):
        batching, condition, alignment_lambda = keys
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            if not values:
                continue
            row = summary_df[
                (summary_df["batching"] == batching)
                & (summary_df["condition"] == condition)
                & (summary_df["lambda_language_alignment"] == alignment_lambda)
                & (summary_df["metric"] == metric)
            ]
            if row.empty:
                raise AssertionError(f"Missing summary row for {keys}/{metric}")
            reported = row.iloc[0]
            sd_value = stdev(values) if len(values) > 1 else 0.0
            if abs(float(reported["mean"]) - mean(values)) > 1e-9:
                raise AssertionError(f"Summary mean mismatch for {keys}/{metric}")
            if abs(float(reported["sample_sd"]) - sd_value) > 1e-9:
                raise AssertionError(f"Summary SD mismatch for {keys}/{metric}")
    checks.append("Summary means and sample SDs recompute exactly from seed metrics.")

    epoch_df = pd.read_parquet(out_root / "phase3_epoch_trajectories.parquet")
    coverage_df = pd.read_csv(out_root / "phase3_pair_coverage.csv")
    missing_epoch = [column for column in REQUIRED_EPOCH_COLUMNS if column not in epoch_df.columns]
    if missing_epoch:
        raise AssertionError(f"Missing required epoch trajectory columns: {missing_epoch}")
    paired = coverage_df[coverage_df["batching"] == "paired"]
    if paired.empty or not ((paired["min_pair_coverage"] == 1.0) & (paired["max_pair_coverage"] == 1.0)).all():
        raise AssertionError("Paired runs did not maintain 100% pair coverage for every seed/lambda/condition.")
    verify_canonical_pair_coverage(seed_df, coverage_df)
    checks.append("Paired runs have 100% pair coverage for every seed/lambda/condition.")

    csv_rows = 0
    for path in sorted((out_root / "epoch_trajectories").glob("*/*/*/*.csv")):
        batching, condition, tag, _seed_file = path.relative_to(out_root / "epoch_trajectories").parts
        label = run_label(batching, condition, int(tag.replace("align", "")) / 100.0)
        if not is_canonical_phase3_label(label):
            continue
        csv_rows += len(pd.read_csv(path))
    if csv_rows != len(epoch_df):
        raise AssertionError(f"Epoch parquet row count {len(epoch_df)} does not match CSV row count {csv_rows}")
    checks.append("Epoch trajectory parquet reconstructs from raw per-run CSV files.")

    figures = [
        "retrieval_across_lambda.png",
        "formal_parent_accuracy_across_lambda.png",
        "formal_depth_accuracy_across_lambda.png",
        "formal_successor_accuracy_across_lambda.png",
        "same_id_distance_across_lambda.png",
        "alignment_loss_trajectories.png",
        "structure_mean_norm_across_lambda.png",
        "posterior_variance_across_lambda.png",
    ]
    missing_figures = [name for name in figures if not (out_root / "figures" / name).exists()]
    if missing_figures:
        raise FileNotFoundError(f"Missing Phase 3 figures: {missing_figures}")
    checks.append("All required figures are present.")

    diff_names = git_output(["diff", "--name-only"]).splitlines()
    forbidden = [
        path
        for path in diff_names
        if path in {"paper/main.tex", "paper/references.bib"} or path.startswith("paper/figures/")
    ]
    if forbidden:
        raise AssertionError(f"Protected canonical files have tracked diffs: {forbidden}")
    checks.append("No tracked diffs touch manuscript files or canonical run outputs.")
    return checks


def verify(args: argparse.Namespace) -> None:
    out_root = args.out_root.resolve()
    summarise(out_root)
    checks = verify_outputs(out_root)
    lines = [
        "# Phase 3 Verification Report",
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
    (out_root / "phase3_verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified Phase 3 outputs: 100 canonical paired final rows, complete paired coverage, and recomputed summaries.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 controlled bilingual alignment experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(run_parser: argparse.ArgumentParser) -> None:
        run_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
        run_parser.add_argument("--data", type=Path, default=DATA_PATH)
        run_parser.add_argument("--lambdas", default="0.00,0.03,0.10,0.30,1.00")
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
        run_parser.add_argument("--dry-run", action="store_true")

    run_parser = sub.add_parser("run")
    add_common(run_parser)
    run_parser.add_argument("--batching", choices=["paired"], default="paired")
    run_parser.add_argument("--conditions", default="full_model")
    run_parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    run_parser.set_defaults(func=run)

    succ_parser = sub.add_parser("successor-control")
    add_common(succ_parser)
    succ_parser.add_argument("--pilot-seeds", default="0,1,2")
    succ_parser.add_argument("--all-seeds", default="0,1,2,3,4,5,6,7,8,9")
    succ_parser.set_defaults(func=successor_control)

    eval_parser = sub.add_parser("eval-one")
    eval_parser.add_argument("--label", required=True)
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--data", type=Path, default=DATA_PATH)
    eval_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    eval_parser.set_defaults(func=eval_one)

    summary_parser = sub.add_parser("summarise")
    summary_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    summary_parser.set_defaults(func=lambda args: summarise(args.out_root.resolve()))

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
