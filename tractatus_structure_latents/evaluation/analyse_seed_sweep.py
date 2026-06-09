from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

DEFAULT_METRICS = [
    "loss",
    "reconstruction_loss",
    "kl",
    "kl_text",
    "kl_structure",
    "perplexity",
    "parent_accuracy",
    "depth_accuracy",
    "next_accuracy",
    "mean_parent_child_distance",
    "mean_sibling_distance",
    "mean_unrelated_distance",
    "mean_same_id_cross_language_distance",
    "cross_language_top1_id_accuracy",
    "cross_language_mrr",
    "cross_language_top1_id_accuracy_de_to_en",
    "cross_language_top1_id_accuracy_en_to_de",
    "cross_language_mrr_de_to_en",
    "cross_language_mrr_en_to_de",
]

PLOT_METRICS = [
    "parent_accuracy",
    "depth_accuracy",
    "next_accuracy",
    "reconstruction_loss",
    "perplexity",
    "kl_structure",
    "cross_language_top1_id_accuracy",
    "cross_language_mrr",
]


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(next_prefix, nested, out)
        return
    if isinstance(value, bool):
        out[prefix] = float(value)
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    _flatten("", metrics, flat)
    return flat


def seed_from_path(path: Path) -> int | None:
    stem = path.name.split(".")[0]
    if stem.startswith("seed") and stem[4:].isdigit():
        return int(stem[4:])
    return None


def load_metrics(study_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = study_dir / "metrics"
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("seed*.metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        flat = flatten_metrics(metrics)
        rows.append({"seed": seed_from_path(path), "path": str(path), "metrics": metrics, "flat": flat})
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted({name for row in rows for name in row["flat"]})
    summary: dict[str, Any] = {
        "seed_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "metrics": {},
    }
    for metric in metric_names:
        values = [row["flat"][metric] for row in rows if metric in row["flat"]]
        if not values:
            continue
        summary["metrics"][metric] = {
            "count": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def write_per_seed_csv(rows: list[dict[str, Any]], out: Path) -> None:
    metric_names = sorted({name for row in rows for name in row["flat"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow({"seed": row["seed"], **row["flat"]})


def write_summary_csv(summary: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "count", "mean", "std", "min", "max"])
        writer.writeheader()
        for metric, stats in sorted(summary["metrics"].items()):
            writer.writerow({"metric": metric, **stats})


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ").replace(".", " ")


def plot_study(rows: list[dict[str, Any]], out: Path, metrics: list[str]) -> None:
    available = [metric for metric in metrics if any(metric in row["flat"] for row in rows)]
    if not available:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(available), 1, figsize=(9, max(3, 2.6 * len(available))), squeeze=False)
    seeds = [row["seed"] for row in rows]
    for ax, metric in zip(axes[:, 0], available):
        values = [row["flat"].get(metric) for row in rows]
        ax.plot(seeds, values, marker="o", linewidth=1.8)
        ax.set_title(_metric_label(metric))
        ax.set_xlabel("seed")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_comparison(studies: dict[str, list[dict[str, Any]]], out: Path, metrics: list[str]) -> None:
    available = [metric for metric in metrics if any(metric in row["flat"] for rows in studies.values() for row in rows)]
    if not available:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(available), 1, figsize=(10, max(3, 2.8 * len(available))), squeeze=False)
    for ax, metric in zip(axes[:, 0], available):
        labels: list[str] = []
        values_by_study: list[list[float]] = []
        for label, rows in studies.items():
            values = [row["flat"][metric] for row in rows if metric in row["flat"]]
            if values:
                labels.append(label)
                values_by_study.append(values)
        if not values_by_study:
            continue
        ax.boxplot(values_by_study, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1), labels=labels)
        ax.set_title(_metric_label(metric))
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(study_name: str, summary: dict[str, Any], out: Path, preferred_metrics: list[str]) -> None:
    lines = [f"# {study_name} Seed Sweep", "", f"Seeds: {summary['seed_count']}", ""]
    lines.extend(["| metric | mean | std | min | max |", "| --- | ---: | ---: | ---: | ---: |"])
    metrics = summary["metrics"]
    ordered = [metric for metric in preferred_metrics if metric in metrics]
    ordered.extend(metric for metric in sorted(metrics) if metric not in ordered)
    for metric in ordered:
        stats = metrics[metric]
        lines.append(
            f"| `{metric}` | {stats['mean']:.6g} | {stats['std']:.6g} | {stats['min']:.6g} | {stats['max']:.6g} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyse_study(study_dir: Path, metrics: list[str]) -> dict[str, Any]:
    rows = load_metrics(study_dir)
    if not rows:
        raise FileNotFoundError(f"No seed metric files found in {study_dir / 'metrics'}")
    summary = summarise(rows)
    summaries_dir = study_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (summaries_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_per_seed_csv(rows, summaries_dir / "per_seed_metrics.csv")
    write_summary_csv(summary, summaries_dir / "summary.csv")
    plot_study(rows, summaries_dir / "seed_metric_trends.png", metrics)
    write_report(study_dir.name, summary, summaries_dir / "summary.md", metrics)
    return {"name": study_dir.name, "dir": str(study_dir), "rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot seed-sweep metrics.")
    parser.add_argument("study_dirs", nargs="+", type=Path, help="One or more study directories under runs/seed_sweeps.")
    parser.add_argument("--out", type=Path, default=Path("runs/seed_sweeps/summary"), help="Output directory for cross-study comparison plots.")
    parser.add_argument("--metrics", help="Comma-separated metrics to prioritize in reports and plots.")
    args = parser.parse_args()

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()] if args.metrics else PLOT_METRICS
    analyses = [analyse_study(study_dir, metrics) for study_dir in args.study_dirs]
    print(json.dumps({item["name"]: item["summary"]["seed_count"] for item in analyses}, indent=2))

    if len(analyses) > 1:
        studies = {item["name"]: item["rows"] for item in analyses}
        args.out.mkdir(parents=True, exist_ok=True)
        plot_comparison(studies, args.out / "study_metric_comparison.png", metrics)
        comparison = {item["name"]: item["summary"] for item in analyses}
        (args.out / "summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
