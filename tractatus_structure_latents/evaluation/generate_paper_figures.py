from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


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
    plt.savefig(out_dir / "family_case_distance_matrix.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "family_case_distance_matrix.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-paper figures.")
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--family-distance-data", type=Path, default=Path("paper/figures/family_case_distance_matrix_data.csv"))
    args = parser.parse_args()

    generate_family_distance_figure(args.family_distance_data, args.out_dir)
    print(f"saved final-paper figure to {args.out_dir}")


if __name__ == "__main__":
    main()
