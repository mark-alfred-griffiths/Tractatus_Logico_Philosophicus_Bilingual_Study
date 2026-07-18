from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from tractatus_structure_latents.evaluation.plot_bilingual_alignment_sweep import (
    load_seed_sweep_rows,
    plot_lines,
    plot_tradeoff,
    write_summary,
)
from tractatus_structure_latents.evaluation.visualise_latents import plot_latents


def alignment_display_label(alignment: str) -> str:
    if alignment.startswith("align") and alignment[5:].isdigit():
        return f"align {int(alignment[5:]) / 100:.2f}"
    return alignment


def seed_display_label(seed: int) -> str:
    return f"seed {seed}"


def write_csv_copy(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "column_id", "mean_distance_across_seeds"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def generate_family_distance_figure(source_data: Path, out_dir: Path) -> None:
    if not source_data.exists():
        raise FileNotFoundError(f"Family distance matrix data not found: {source_data}")
    with source_data.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = sorted({row["row_id"] for row in rows})
    values = {(row["row_id"], row["column_id"]): float(row["mean_distance_across_seeds"]) for row in rows}
    matrix = [[values[(row_id, col_id)] for col_id in ids] for row_id in ids]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_copy(rows, out_dir / "family_case_distance_matrix_data.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    plt.figure(figsize=(6.4, 5.4))
    image = plt.imshow(matrix, cmap="viridis")
    plt.colorbar(image, label="Seed-averaged Euclidean distance")
    plt.xticks(range(len(ids)), ids, rotation=45, ha="right")
    plt.yticks(range(len(ids)), ids)
    plt.title("Family 2.2 structure-latent distances")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            colour = "white" if value > 2.0 else "black"
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=colour, fontsize=8)
    plt.tight_layout()
    png = out_dir / "family_case_distance_matrix.png"
    pdf = out_dir / "family_case_distance_matrix.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def generate_metric_figures(seed_sweep_dir: Path, out_dir: Path, summary_out: Path | None) -> None:
    rows = load_seed_sweep_rows(seed_sweep_dir)
    if not rows:
        raise FileNotFoundError(f"No seed metrics found in {seed_sweep_dir}")
    plot_lines(
        rows,
        out_dir / "bilingual_alignment_retrieval_sweep.png",
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
        out_dir / "bilingual_structure_accuracy_sweep.png",
        "Structure prediction across alignment weights (seeds 0-9)",
        "Accuracy",
        [
            ("parent_accuracy", "Parent"),
            ("depth_accuracy", "Depth"),
            ("next_accuracy", "Next"),
        ],
        ylim=(0.3, 1.0),
    )
    plot_tradeoff(rows, out_dir / "bilingual_retrieval_structure_tradeoff.png")
    plot_lines(
        rows,
        out_dir / "bilingual_reconstruction_sweep.png",
        "Reconstruction across alignment weights (seeds 0-9)",
        "Loss / perplexity",
        [
            ("reconstruction_loss", "Reconstruction loss"),
            ("perplexity", "Perplexity"),
        ],
    )
    if summary_out is not None:
        write_summary(rows, summary_out)


def generate_pca_figures(args: argparse.Namespace) -> None:
    seed = f"seed{args.representative_seed:03d}"
    display_seed = seed_display_label(args.representative_seed)
    display_alignment = alignment_display_label(args.representative_alignment)
    align_dir = args.seed_sweep_dir / args.representative_alignment
    bilingual_latents = align_dir / "latents" / f"{seed}_structure.pt"
    bilingual_ids = align_dir / "latents" / f"{seed}_structure.ids.json"
    monolingual_latents = args.monolingual_dir / "latents" / f"{seed}_structure.pt"
    monolingual_ids = args.monolingual_dir / "latents" / f"{seed}_structure.ids.json"

    plot_latents(
        latents=bilingual_latents,
        ids=bilingual_ids,
        data=args.bilingual_data,
        method="pca",
        colour_by="language",
        seed_label=display_seed,
        title=f"Bilingual structure latent PCA ({display_alignment})",
        out=args.out_dir / f"bilingual_latent_pca_language_{args.representative_alignment}_{seed}.png",
    )
    plot_latents(
        latents=bilingual_latents,
        ids=bilingual_ids,
        data=args.bilingual_data,
        method="pca",
        colour_by="depth",
        seed_label=display_seed,
        title=f"Bilingual structure latent PCA by depth ({display_alignment})",
        out=args.out_dir / f"bilingual_latent_pca_depth_{args.representative_alignment}_{seed}.png",
    )
    plot_latents(
        latents=monolingual_latents,
        ids=monolingual_ids,
        data=args.monolingual_data,
        method="pca",
        colour_by="depth",
        seed_label=display_seed,
        title="Monolingual structure latent PCA by depth",
        out=args.out_dir / f"monolingual_latent_pca_depth_reg005_{seed}.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical paper figures from seed-sweep outputs.")
    parser.add_argument("--seed-sweep-dir", type=Path, default=Path("runs/seed_sweeps/bilingual_alignment_lambda_sweep"))
    parser.add_argument("--monolingual-dir", type=Path, default=Path("runs/seed_sweeps/monolingual_split_24_8_reg005"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--summary-out", type=Path, default=Path("runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json"))
    parser.add_argument("--representative-alignment", default="align003", help="Alignment folder used for representative PCA figures.")
    parser.add_argument("--representative-seed", type=int, default=0, help="Seed used for representative PCA figures.")
    parser.add_argument("--bilingual-data", type=Path, default=Path("tractatus_structure_latents/data/tractatus_bilingual.json"))
    parser.add_argument("--monolingual-data", type=Path, default=Path("tractatus_structure_latents/data/tractatus.json"))
    parser.add_argument("--family-distance-data", type=Path, default=Path("paper/figures/family_case_distance_matrix_data.csv"))
    parser.add_argument("--skip-pca", action="store_true", help="Only generate metric sweep figures.")
    parser.add_argument("--skip-family-figure", action="store_true", help="Skip the family 2.2 distance matrix figure.")
    args = parser.parse_args()

    generate_metric_figures(args.seed_sweep_dir, args.out_dir, args.summary_out)
    if not args.skip_pca:
        generate_pca_figures(args)
    if not args.skip_family_figure:
        generate_family_distance_figure(args.family_distance_data, args.out_dir)
    print(f"saved paper figures to {args.out_dir}")


if __name__ == "__main__":
    main()
