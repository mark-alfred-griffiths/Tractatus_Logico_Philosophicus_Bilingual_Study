from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch

DPI = 450
LANGUAGE_COLORS = {"de": "#D55E00", "en": "#0072B2"}


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


def project(z, method: str):
    z_np = z.detach().cpu().numpy()
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(z_np)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, len(z_np) // 5))
        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=0).fit_transform(z_np)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=7).fit_transform(z_np)
    raise ValueError(f"unknown projection method: {method}")


def infer_seed_label(path: Path) -> str | None:
    for part in [path.stem, *path.parts]:
        if "seed" not in part:
            continue
        start = part.find("seed")
        candidate = part[start : start + 7]
        if len(candidate) == 7 and candidate.startswith("seed") and candidate[4:].isdigit():
            return candidate
    return None


def plot_language_scatter(xy, metadata: list[dict]) -> None:
    languages = sorted({str(item["language"]) for item in metadata})
    for language in languages:
        idx = [i for i, item in enumerate(metadata) if str(item["language"]) == language]
        plt.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=18,
            alpha=0.72,
            color=LANGUAGE_COLORS.get(language, "#666666"),
            edgecolors="none",
            label=language,
        )
    plt.legend(title="Language", loc="best")


def plot_numeric_scatter(xy, colors: list[float | int], colour_by: str) -> None:
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=colors, s=18, cmap="viridis", alpha=0.78, edgecolors="none")
    plt.colorbar(scatter, label=colour_by)


def plot_latents(
    latents: Path,
    data: Path,
    out: Path,
    ids: Path | None = None,
    method: str = "pca",
    colour_by: str = "depth",
    seed_label: str | None = None,
    title: str | None = None,
) -> None:
    z = torch.load(latents, map_location="cpu")
    rows = json.loads(data.read_text(encoding="utf-8"))
    metadata_path = ids or latents.with_suffix(".ids.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    if metadata is not None:
        by_id = {row["id"]: row for row in rows}
        plot_rows = [by_id[item["id"]] for item in metadata]
    else:
        plot_rows = rows
    if len(plot_rows) != len(z):
        raise ValueError(f"latent count ({len(z)}) does not match plot metadata rows ({len(plot_rows)})")

    xy = project(z, method)
    resolved_seed_label = seed_label or infer_seed_label(latents)
    plot_title = title or f"Tractatus latent space ({method}, colour={colour_by})"
    if resolved_seed_label:
        plot_title = f"{plot_title} - {resolved_seed_label}"

    set_publication_style()
    plt.figure(figsize=(6.5, 5.0))
    if colour_by == "depth":
        plot_numeric_scatter(xy, [row["depth"] for row in plot_rows], "Depth")
    elif colour_by == "top":
        plot_numeric_scatter(xy, [int(row["id"].split(".")[0]) for row in plot_rows], "Top-level section")
    elif colour_by == "language":
        if metadata is None:
            raise ValueError("--colour-by language requires exported ids metadata")
        plot_language_scatter(xy, metadata)
    elif colour_by == "parent":
        parent_ids = {row["parent_id"]: i for i, row in enumerate(plot_rows) if row["parent_id"] is not None}
        plot_numeric_scatter(xy, [parent_ids.get(row["parent_id"], 0) for row in plot_rows], "Parent group")
    else:
        raise ValueError(f"unknown colour_by: {colour_by}")

    plt.title(plot_title, pad=10)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.grid(alpha=0.18, linewidth=0.7)
    plt.tight_layout()
    save_figure(out)
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latent propositions with PCA, t-SNE, or UMAP.")
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("tractatus_structure_latents/data/tractatus.json"))
    parser.add_argument("--ids", type=Path, help="Optional exported .ids.json metadata from evaluate_structure.")
    parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    parser.add_argument("--colour-by", choices=["depth", "top", "parent", "language"], default="depth")
    parser.add_argument("--out", type=Path, default=Path("runs/latent_plot.png"))
    parser.add_argument("--seed-label", help="Explicit seed label to include in the plot title, e.g. seed000.")
    parser.add_argument("--title", help="Custom plot title. The seed label is appended when available.")
    args = parser.parse_args()

    plot_latents(
        latents=args.latents,
        data=args.data,
        ids=args.ids,
        method=args.method,
        colour_by=args.colour_by,
        out=args.out,
        seed_label=args.seed_label,
        title=args.title,
    )


if __name__ == "__main__":
    main()
