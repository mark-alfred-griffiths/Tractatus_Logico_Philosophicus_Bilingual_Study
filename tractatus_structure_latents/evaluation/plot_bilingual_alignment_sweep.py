from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

DPI = 450
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _flatten(prefix: str, value, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), nested, out)
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)


def flatten_metrics(metrics: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    _flatten("", metrics, flat)
    return flat


def load_seed_sweep_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for study_dir in sorted(root.glob("align*")):
        if not study_dir.is_dir():
            continue
        manifest_path = study_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            alignment_lambda = float(manifest["lambda_language_alignment"])
        else:
            alignment_lambda = int(study_dir.name.replace("align", "")) / 100.0
        metric_rows = []
        for path in sorted((study_dir / "metrics").glob("seed*.metrics.json")):
            metric_rows.append(flatten_metrics(json.loads(path.read_text(encoding="utf-8"))))
        if not metric_rows:
            continue
        metric_names = sorted({name for row in metric_rows for name in row})
        summary: dict[str, float | int | str] = {
            "lambda_language_alignment": alignment_lambda,
            "tag": study_dir.name.replace("align", ""),
            "seed_count": len(metric_rows),
        }
        for metric in metric_names:
            vals = [row[metric] for row in metric_rows if metric in row]
            if not vals:
                continue
            summary[metric] = mean(vals)
            summary[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
            summary[f"{metric}_min"] = min(vals)
            summary[f"{metric}_max"] = max(vals)
        rows.append(summary)
    return sorted(rows, key=lambda row: float(row["lambda_language_alignment"]))


def xs(rows: list[dict]) -> list[float]:
    return [float(row["lambda_language_alignment"]) for row in rows]


def x_positions(rows: list[dict]) -> list[int]:
    return list(range(len(rows)))


def lambda_tick_labels(rows: list[dict]) -> list[str]:
    labels: list[str] = []
    for value in xs(rows):
        if value == 0:
            labels.append("0")
        elif value < 0.1:
            labels.append(f"{value:.2f}")
        else:
            labels.append(f"{value:.2f}".rstrip("0").rstrip("."))
    return labels


def values(rows: list[dict], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows]


def std_values(rows: list[dict], metric: str) -> list[float] | None:
    std_key = f"{metric}_std"
    if not any(std_key in row for row in rows):
        return None
    return [float(row.get(std_key, 0.0)) for row in rows]


def format_axes(title: str, ylabel: str, rows: list[dict]) -> None:
    plt.title(title, pad=10)
    plt.xlabel("Language-alignment weight")
    plt.ylabel(ylabel)
    plt.xticks(x_positions(rows), lambda_tick_labels(rows))
    plt.grid(axis="y", alpha=0.25, linewidth=0.8)
    plt.legend(loc="best")
    plt.tight_layout()


def plot_lines(rows: list[dict], out: Path, title: str, ylabel: str, metrics: Iterable[tuple[str, str]], ylim: tuple[float, float] | None = None) -> None:
    set_publication_style()
    plt.figure(figsize=(7.0, 4.2))
    positions = x_positions(rows)
    for i, (metric, label) in enumerate(metrics):
        y = values(rows, metric)
        std = std_values(rows, metric)
        color = COLORS[i % len(COLORS)]
        if std is not None:
            plt.errorbar(
                positions,
                y,
                yerr=std,
                marker="o",
                markersize=5,
                linewidth=1.8,
                capsize=4,
                capthick=1.1,
                elinewidth=1.1,
                color=color,
                label=f"{label} (mean +/- 1 SD)",
            )
        else:
            plt.plot(positions, y, marker="o", markersize=5, linewidth=1.8, color=color, label=label)
    if ylim is not None:
        plt.ylim(*ylim)
    format_axes(title, ylabel, rows)
    save_figure(out)


def hierarchy_mean(row: dict) -> float:
    return (float(row["parent_accuracy"]) + float(row["depth_accuracy"]) + float(row["next_accuracy"])) / 3.0


def hierarchy_std(row: dict) -> float:
    parts = [
        float(row.get("parent_accuracy_std", 0.0)),
        float(row.get("depth_accuracy_std", 0.0)),
        float(row.get("next_accuracy_std", 0.0)),
    ]
    return sum(parts) / len(parts)


def plot_tradeoff(rows: list[dict], out: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(6.2, 4.6))
    lambdas = xs(rows)
    top1 = values(rows, "cross_language_top1_id_accuracy")
    top1_std = std_values(rows, "cross_language_top1_id_accuracy") or [0.0 for _ in rows]
    hierarchy = [hierarchy_mean(row) for row in rows]
    hierarchy_spread = [hierarchy_std(row) for row in rows]
    scatter = plt.scatter(hierarchy, top1, c=lambdas, s=90, cmap="viridis", edgecolor="black", linewidth=0.6, zorder=3)
    plt.errorbar(hierarchy, top1, xerr=hierarchy_spread, yerr=top1_std, fmt="none", ecolor="0.35", alpha=0.65, capsize=4, elinewidth=1.1, zorder=2)
    label_offsets = {
        0.00: (-22, 7),
        0.03: (6, 6),
        0.10: (7, -9),
        0.30: (7, 6),
        1.00: (7, 7),
    }
    for row, x_value, y_value in zip(rows, hierarchy, top1):
        lambda_value = float(row["lambda_language_alignment"])
        offset = label_offsets.get(round(lambda_value, 2), (5, 5))
        label = "0" if lambda_value == 0 else f"{lambda_value:.2f}".rstrip("0").rstrip(".")
        plt.annotate(label, (x_value, y_value), xytext=offset, textcoords="offset points", fontsize=8)
    plt.colorbar(scatter, label="Language-alignment weight")
    plt.title("Retrieval-structure tradeoff (seeds 0-9)", pad=10)
    plt.xlabel("Mean structure prediction accuracy")
    plt.ylabel("Cross-language top-1 ID accuracy")
    plt.grid(axis="both", alpha=0.25, linewidth=0.8)
    plt.tight_layout()
    save_figure(out)


def write_summary(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate paper figures for the multi-seed bilingual alignment sweep.")
    parser.add_argument("--seed-sweep-dir", type=Path, required=True, help="Multi-seed sweep root containing align*/metrics/seed*.metrics.json.")
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--summary-out", type=Path, help="Optional JSON path for aggregated plotted values.")
    args = parser.parse_args()

    rows = load_seed_sweep_rows(args.seed_sweep_dir)
    if not rows:
        raise FileNotFoundError(f"No sweep rows found in {args.seed_sweep_dir}")
    plot_lines(
        rows,
        args.out_dir / "bilingual_alignment_retrieval_sweep.png",
        "Cross-language retrieval across alignment weights (seeds 0-9)",
        "Retrieval metric",
        [
            ("cross_language_top1_id_accuracy", "Top-1 ID accuracy"),
            ("cross_language_mrr", "MRR"),
        ],
        ylim=(0.7, 1.0),
    )
    plot_lines(
        rows,
        args.out_dir / "bilingual_structure_accuracy_sweep.png",
        "Structure prediction across alignment weights (seeds 0-9)",
        "Accuracy",
        [
            ("parent_accuracy", "Parent"),
            ("depth_accuracy", "Depth"),
            ("next_accuracy", "Next"),
        ],
        ylim=(0.3, 1.0),
    )
    plot_tradeoff(rows, args.out_dir / "bilingual_retrieval_structure_tradeoff.png")
    plot_lines(
        rows,
        args.out_dir / "bilingual_reconstruction_sweep.png",
        "Reconstruction across alignment weights (seeds 0-9)",
        "Loss / perplexity",
        [
            ("reconstruction_loss", "Reconstruction loss"),
            ("perplexity", "Perplexity"),
        ],
    )
    if args.summary_out:
        write_summary(rows, args.summary_out)
    print(f"saved bilingual alignment sweep figures to {args.out_dir}")


if __name__ == "__main__":
    main()
