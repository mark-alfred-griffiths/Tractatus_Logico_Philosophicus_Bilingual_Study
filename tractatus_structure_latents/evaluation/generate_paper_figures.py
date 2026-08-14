from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
PHASE1_PER_SEED = ROOT / "results" / "dsh_validation" / "phase1_ablations" / "per_seed"
PHASE2_PER_SEED = ROOT / "results" / "dsh_validation" / "phase2_family_holdout" / "per_seed"
RETAINED_LEXICAL = ROOT / "results" / "dsh_validation" / "canonical_reports" / "retained_lexical_references.csv"
EXPECTED_SEEDS = set(range(10))
EXPECTED_FOLDS = set(range(5))
EXPECTED_HOLDOUT_SEEDS = set(range(3))
RANDOM_526_TOP1 = 1 / 526

BLUE = "#1f4e79"
ORANGE = "#d55e00"
PURPLE = "#7b3294"
GREEN = "#007f5f"
GREY = "#6f6f6f"
LIGHT_GREY = "#b7b7b7"

FIGURE1_CONDITIONS = [
    ("full_model", "Full model"),
    ("no_successor", "No successor"),
    ("parent_depth_only", "Parent + depth only"),
    ("reconstruction_only", "Reconstruction only"),
    ("successor_only", "Successor only"),
    ("shuffled_joint_targets", "Shuffled joint targets"),
    ("shuffled_no_successor", "Shuffled no-successor"),
]

FIGURE2_REPRESENTATIONS = [
    ("full_model", "Full model\n(structure latent)", "o", BLUE),
    ("no_successor", "No successor\n(structure latent)", "s", BLUE),
    ("reconstruction_only", "Reconstruction only\n(structure latent)", "^", GREY),
    ("char_3_5_tfidf", "Character 3–5-gram TF–IDF\n(lexical reference)", "D", GREEN),
]


@dataclass(frozen=True)
class Summary:
    condition: str
    metric: str
    mean: float
    sd: float | None
    n: int | None
    source: str


def write_csv_copy(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "column_id", "mean_distance_across_seeds"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(label: str, observed: float, expected: float, tol: float = 5e-4) -> None:
    if abs(observed - expected) > tol:
        raise AssertionError(f"{label}: observed {observed:.10f}, expected {expected:.10f}")


def summarise_values(condition: str, metric: str, values: list[float], source: str) -> Summary:
    if not values:
        raise ValueError(f"No values for {condition}/{metric}")
    return Summary(condition, metric, mean(values), stdev(values) if len(values) > 1 else None, len(values), source)


def load_phase1_metrics() -> dict[str, dict[int, dict[str, float]]]:
    expected_conditions = {condition for condition, _label in FIGURE1_CONDITIONS}
    found_conditions = {path.name for path in PHASE1_PER_SEED.iterdir() if path.is_dir()}
    if found_conditions != expected_conditions:
        raise AssertionError(
            f"Phase 1 conditions mismatch: found {sorted(found_conditions)}, expected {sorted(expected_conditions)}"
        )
    metrics: dict[str, dict[int, dict[str, float]]] = {}
    for condition in sorted(expected_conditions):
        by_seed: dict[int, dict[str, float]] = {}
        for path in sorted((PHASE1_PER_SEED / condition).glob("seed*.metrics.json")):
            match = re.fullmatch(r"seed(\d{3})\.metrics", path.stem)
            if not match:
                raise AssertionError(f"Unexpected Phase 1 metrics filename: {path}")
            seed = int(match.group(1))
            if seed in by_seed:
                raise AssertionError(f"Duplicate Phase 1 seed {seed} for {condition}")
            by_seed[seed] = read_json(path)
        if set(by_seed) != EXPECTED_SEEDS:
            raise AssertionError(f"{condition}: found seeds {sorted(by_seed)}, expected {sorted(EXPECTED_SEEDS)}")
        metrics[condition] = by_seed
    return metrics


def load_phase2_metrics() -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    expected_conditions = {"full_model", "no_successor", "reconstruction_only"}
    found_conditions = {path.name for path in PHASE2_PER_SEED.iterdir() if path.is_dir()}
    if found_conditions != expected_conditions:
        raise AssertionError(
            f"Phase 2 conditions mismatch: found {sorted(found_conditions)}, expected {sorted(expected_conditions)}"
        )
    expected_runs = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_HOLDOUT_SEEDS}
    metrics: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    for condition in sorted(expected_conditions):
        by_run: dict[tuple[int, int], dict[str, float]] = {}
        for path in sorted((PHASE2_PER_SEED / condition).glob("fold*_seed*.metrics.json")):
            match = re.fullmatch(r"fold(\d+)_seed(\d{3})\.metrics", path.stem)
            if not match:
                raise AssertionError(f"Unexpected Phase 2 metrics filename: {path}")
            run = (int(match.group(1)), int(match.group(2)))
            if run in by_run:
                raise AssertionError(f"Duplicate Phase 2 run {run} for {condition}")
            by_run[run] = read_json(path)
        if set(by_run) != expected_runs:
            raise AssertionError(f"{condition}: found runs {sorted(by_run)}, expected {sorted(expected_runs)}")
        metrics[condition] = by_run
    return metrics


def load_retained_lexical_char_top1() -> Summary:
    matches = [row for row in read_csv_rows(RETAINED_LEXICAL) if row["method"] == "Character 3-5 TF-IDF"]
    if len(matches) != 1:
        raise AssertionError(f"Expected one retained Character 3-5 TF-IDF row, found {len(matches)}")
    return Summary(
        "char_3_5_tfidf",
        "retained_char_3_5_tfidf_top1",
        float(matches[0]["top1_mean"]),
        None,
        None,
        str(RETAINED_LEXICAL.relative_to(ROOT)),
    )


def build_figure1_data(phase1: dict[str, dict[int, dict[str, float]]]) -> dict[tuple[str, str], Summary]:
    summaries: dict[tuple[str, str], Summary] = {}
    for condition, _label in FIGURE1_CONDITIONS:
        for metric in ["structure_cross_language_top1", "structure_sibling_vs_unrelated_contrast"]:
            values = [phase1[condition][seed][metric] for seed in sorted(EXPECTED_SEEDS)]
            summaries[(condition, metric)] = summarise_values(
                condition,
                metric,
                values,
                f"results/dsh_validation/phase1_ablations/per_seed/{condition}/seed*.metrics.json",
            )
    assert len({condition for condition, _label in FIGURE1_CONDITIONS}) == 7
    assert_close("Figure 1 full-model Top-1", summaries[("full_model", "structure_cross_language_top1")].mean, 0.9360)
    assert_close("Figure 1 successor-only Top-1", summaries[("successor_only", "structure_cross_language_top1")].mean, 0.9944)
    assert_close(
        "Figure 1 full-model sibling contrast",
        summaries[("full_model", "structure_sibling_vs_unrelated_contrast")].mean,
        4.0361,
    )
    if summaries[("full_model", "structure_sibling_vs_unrelated_contrast")].mean <= 0:
        raise AssertionError("Sibling contrast sign check failed: expected unrelated - sibling > 0")
    for condition in ["shuffled_joint_targets", "shuffled_no_successor"]:
        if abs(summaries[(condition, "structure_sibling_vs_unrelated_contrast")].mean) >= 0.10:
            raise AssertionError(f"{condition} sibling contrast is not near zero")
    return summaries


def build_figure2_data(
    phase1: dict[str, dict[int, dict[str, float]]],
    phase2: dict[str, dict[tuple[int, int], dict[str, float]]],
) -> dict[tuple[str, str, str], Summary]:
    summaries: dict[tuple[str, str, str], Summary] = {}
    for condition in ["full_model", "no_successor", "reconstruction_only"]:
        values = [phase1[condition][seed]["structure_cross_language_top1"] for seed in sorted(EXPECTED_SEEDS)]
        summaries[(condition, "retained", "top1")] = summarise_values(
            condition,
            "structure_cross_language_top1",
            values,
            f"results/dsh_validation/phase1_ablations/per_seed/{condition}/seed*.metrics.json",
        )
        for setting, metric in [
            ("family_holdout_complete_candidates", "structure_complete_candidates_top1"),
            ("family_holdout_test_candidates", "structure_test_candidates_top1"),
        ]:
            values = [phase2[condition][run][metric] for run in sorted(phase2[condition])]
            summaries[(condition, setting, "top1")] = summarise_values(
                condition,
                metric,
                values,
                f"results/dsh_validation/phase2_family_holdout/per_seed/{condition}/fold*_seed*.metrics.json",
            )

    summaries[("char_3_5_tfidf", "retained", "top1")] = load_retained_lexical_char_top1()
    lexical_values: dict[str, tuple[list[float], list[float]]] = {}
    for condition in ["full_model", "no_successor", "reconstruction_only"]:
        complete = [phase2[condition][run]["reference_char_3_5_tfidf_complete_top1"] for run in sorted(phase2[condition])]
        test = [phase2[condition][run]["reference_char_3_5_tfidf_test_top1"] for run in sorted(phase2[condition])]
        lexical_values[condition] = (complete, test)
    reference_complete, reference_test = lexical_values["full_model"]
    for condition, (complete, test) in lexical_values.items():
        if complete != reference_complete or test != reference_test:
            raise AssertionError(f"Held-out lexical references differ unexpectedly for {condition}")
    summaries[("char_3_5_tfidf", "family_holdout_complete_candidates", "top1")] = summarise_values(
        "char_3_5_tfidf",
        "reference_char_3_5_tfidf_complete_top1",
        reference_complete,
        "results/dsh_validation/phase2_family_holdout/per_seed/full_model/fold*_seed*.metrics.json",
    )
    summaries[("char_3_5_tfidf", "family_holdout_test_candidates", "top1")] = summarise_values(
        "char_3_5_tfidf",
        "reference_char_3_5_tfidf_test_top1",
        reference_test,
        "results/dsh_validation/phase2_family_holdout/per_seed/full_model/fold*_seed*.metrics.json",
    )

    if RANDOM_526_TOP1 != 1 / 526:
        raise AssertionError("Random 526-candidate Top-1 was not calculated as exactly 1 / 526")
    assert_close("Figure 2 full retained Top-1", summaries[("full_model", "retained", "top1")].mean, 0.9360)
    assert_close(
        "Figure 2 full holdout complete Top-1",
        summaries[("full_model", "family_holdout_complete_candidates", "top1")].mean,
        0.0577,
    )
    assert_close(
        "Figure 2 full holdout test Top-1",
        summaries[("full_model", "family_holdout_test_candidates", "top1")].mean,
        0.0739,
    )
    assert_close(
        "Figure 2 held-out character TF-IDF test Top-1",
        summaries[("char_3_5_tfidf", "family_holdout_test_candidates", "top1")].mean,
        0.5619,
    )
    assert_close(
        "Figure 2 retained character TF-IDF Top-1",
        summaries[("char_3_5_tfidf", "retained", "top1")].mean,
        0.4097,
    )
    if summaries[("char_3_5_tfidf", "retained", "top1")].sd is not None:
        raise AssertionError("Retained TF-IDF has a fabricated SD")
    return summaries


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


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


def write_figure1_csv(summaries: dict[tuple[str, str], Summary], out_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for condition, label in FIGURE1_CONDITIONS:
        for metric, setting in [
            ("structure_cross_language_top1", "retained_same_id_retrieval"),
            ("structure_sibling_vs_unrelated_contrast", "retained_sibling_contrast"),
        ]:
            summary = summaries[(condition, metric)]
            rows.append(
                {
                    "figure": "Validation Figure 1",
                    "condition": condition,
                    "condition_label": label,
                    "metric": metric,
                    "evaluation_setting": setting,
                    "mean": summary.mean,
                    "sample_sd": "" if summary.sd is None else summary.sd,
                    "n": "" if summary.n is None else summary.n,
                    "source": summary.source,
                }
            )
    path = out_dir / "figure_1_plot_data.csv"
    write_csv(
        path,
        rows,
        ["figure", "condition", "condition_label", "metric", "evaluation_setting", "mean", "sample_sd", "n", "source"],
    )
    return path


def write_figure2_csv(summaries: dict[tuple[str, str, str], Summary], out_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    labels = {key: label for key, label, _marker, _colour in FIGURE2_REPRESENTATIONS}
    for representation, _label, _marker, _colour in FIGURE2_REPRESENTATIONS:
        for setting in ["retained", "family_holdout_complete_candidates", "family_holdout_test_candidates"]:
            summary = summaries[(representation, setting, "top1")]
            rows.append(
                {
                    "figure": "Validation Figure 2",
                    "representation": representation,
                    "representation_label": labels[representation].replace("\n", " "),
                    "metric": summary.metric,
                    "evaluation_setting": setting,
                    "mean": summary.mean,
                    "sample_sd": "" if summary.sd is None else summary.sd,
                    "n": "" if summary.n is None else summary.n,
                    "source": summary.source,
                }
            )
    rows.append(
        {
            "figure": "Validation Figure 2",
            "representation": "random_reference",
            "representation_label": "Random 526-candidate reference",
            "metric": "top1",
            "evaluation_setting": "common_526_candidate_reference",
            "mean": RANDOM_526_TOP1,
            "sample_sd": "",
            "n": 526,
            "source": "computed exactly as 1 / 526",
        }
    )
    path = out_dir / "figure_2_plot_data.csv"
    write_csv(
        path,
        rows,
        [
            "figure",
            "representation",
            "representation_label",
            "metric",
            "evaluation_setting",
            "mean",
            "sample_sd",
            "n",
            "source",
        ],
    )
    return path


def figure1_style(condition: str) -> dict[str, object]:
    if condition in {"full_model", "no_successor", "parent_depth_only"}:
        return {"marker": "o", "color": BLUE, "mfc": BLUE, "mec": BLUE}
    if condition == "reconstruction_only":
        return {"marker": "s", "color": GREY, "mfc": "white", "mec": GREY}
    if condition == "successor_only":
        return {"marker": "^", "color": ORANGE, "mfc": "white", "mec": ORANGE}
    return {"marker": "D", "color": PURPLE, "mfc": "white", "mec": PURPLE}


def draw_error_point(ax: plt.Axes, x: float, y: float, sd: float | None, style: dict[str, object], zorder: int = 3) -> None:
    ax.errorbar(
        x,
        y,
        xerr=sd,
        fmt=style["marker"],
        markersize=6.8,
        markerfacecolor=style["mfc"],
        markeredgecolor=style["mec"],
        markeredgewidth=1.35,
        ecolor=LIGHT_GREY,
        elinewidth=1.15,
        capsize=3.0,
        capthick=1.0,
        color=style["color"],
        linestyle="none",
        zorder=zorder,
    )


def apply_common_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#e4e4e4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png", out_dir / f"{stem}.tiff"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_validation_figure1(
    summaries: dict[tuple[str, str], Summary],
    out_dir: Path,
    suptitle: str | None,
) -> list[Path]:
    labels = [label for _condition, label in FIGURE1_CONDITIONS]
    y_positions = list(range(len(labels)))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharey=True, constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, y=1.04)
    for ax in axes:
        ax.set_yticks(y_positions, labels)
        ax.set_ylim(-0.65, len(labels) - 0.35)
        apply_common_axis_style(ax)
    axes[0].set_title("(a) Retained same-ID retrieval")
    axes[0].set_xlabel("Structure-latent Top-1 retrieval")
    axes[1].set_title("(b) Sibling cohesion")
    axes[1].set_xlabel("Sibling contrast (unrelated − sibling distance)")
    axes[1].axvline(0, color="#666666", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    for y, (condition, _label) in zip(y_positions, FIGURE1_CONDITIONS):
        style = figure1_style(condition)
        draw_error_point(axes[0], summaries[(condition, "structure_cross_language_top1")].mean, y, summaries[(condition, "structure_cross_language_top1")].sd, style)
        draw_error_point(
            axes[1],
            summaries[(condition, "structure_sibling_vs_unrelated_contrast")].mean,
            y,
            summaries[(condition, "structure_sibling_vs_unrelated_contrast")].sd,
            style,
        )
    axes[0].set_xlim(-0.04, 1.035)
    sibling_lows = [
        summaries[(condition, "structure_sibling_vs_unrelated_contrast")].mean
        - (summaries[(condition, "structure_sibling_vs_unrelated_contrast")].sd or 0.0)
        for condition, _label in FIGURE1_CONDITIONS
    ]
    sibling_highs = [
        summaries[(condition, "structure_sibling_vs_unrelated_contrast")].mean
        + (summaries[(condition, "structure_sibling_vs_unrelated_contrast")].sd or 0.0)
        for condition, _label in FIGURE1_CONDITIONS
    ]
    axes[1].set_xlim(min(-0.25, min(sibling_lows) - 0.05), max(sibling_highs) + 0.2)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=7, label="Genuine hierarchy-supervised condition"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=GREY, markeredgewidth=1.4, markersize=7, label="Reconstruction-only negative control"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.4, markersize=7, label="Successor-only control"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=PURPLE, markeredgewidth=1.4, markersize=7, label="Shuffled-target control"),
        Line2D([0], [0], color=LIGHT_GREY, marker="|", markersize=8, linewidth=1.15, label="Whiskers: ±1 sample SD"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.10), columnspacing=1.4, handlelength=1.8)
    return save_figure(fig, out_dir, "Figure_1_Retained_Ablation_Diagnostics_publication")


def draw_status_point(ax: plt.Axes, x: float, y: float, sd: float | None, marker: str, colour: str, filled: bool) -> None:
    style = {"marker": marker, "color": colour, "mfc": colour if filled else "white", "mec": colour}
    draw_error_point(ax, x, y, sd, style, zorder=4)


def plot_validation_figure2(
    summaries: dict[tuple[str, str, str], Summary],
    out_dir: Path,
    suptitle: str | None,
) -> list[Path]:
    labels = [label for _key, label, _marker, _colour in FIGURE2_REPRESENTATIONS]
    y_positions = list(range(len(labels)))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True, constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, y=1.04)
    for ax in axes:
        ax.set_yticks(y_positions, labels)
        ax.set_ylim(-0.65, len(labels) - 0.35)
        apply_common_axis_style(ax)
    axes[0].set_title("(a) Common 526-candidate comparison")
    axes[0].set_xlabel("Top-1 retrieval")
    axes[1].set_title("(b) Held-out test-candidate comparison")
    axes[1].set_xlabel("Top-1 retrieval")
    axes[0].axvline(RANDOM_526_TOP1, color="#666666", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    axes[0].annotate(
        "Random = 0.0019",
        xy=(RANDOM_526_TOP1, 0.04),
        xycoords=("data", "axes fraction"),
        xytext=(6, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    for y, (key, _label, marker, colour) in zip(y_positions, FIGURE2_REPRESENTATIONS):
        retained = summaries[(key, "retained", "top1")]
        complete = summaries[(key, "family_holdout_complete_candidates", "top1")]
        test = summaries[(key, "family_holdout_test_candidates", "top1")]
        axes[0].plot([retained.mean, complete.mean], [y, y], color=LIGHT_GREY, linewidth=1.0, zorder=2)
        draw_status_point(axes[0], retained.mean, y, retained.sd, marker, colour, filled=True)
        draw_status_point(axes[0], complete.mean, y, complete.sd, marker, colour, filled=False)
        draw_status_point(axes[1], test.mean, y, test.sd, marker, colour, filled=False)
    axes[0].set_xlim(-0.025, 1.02)
    test_high = max(
        summaries[(key, "family_holdout_test_candidates", "top1")].mean
        + (summaries[(key, "family_holdout_test_candidates", "top1")].sd or 0.0)
        for key, _label, _marker, _colour in FIGURE2_REPRESENTATIONS
    )
    axes[1].set_xlim(-0.025, max(0.63, test_high + 0.04))

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=7, label="Circle: full-model structure latent"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=7, label="Square: no-successor structure latent"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=GREY, markeredgecolor=GREY, markersize=7, label="Triangle: reconstruction-only structure latent"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GREEN, markeredgecolor=GREEN, markersize=7, label="Diamond: character 3–5-gram TF–IDF"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#444444", markeredgecolor="#444444", markersize=7, label="Filled marker: retained-corpus result"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#444444", markeredgewidth=1.4, markersize=7, label="Open marker: family-held-out result"),
        Line2D([0], [0], color=LIGHT_GREY, linewidth=1.0, label="Grey connector: same representation"),
        Line2D([0], [0], color="#666666", linestyle=(0, (4, 3)), linewidth=1.0, label="Dashed line: random Top-1 reference"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.16), columnspacing=1.15, handlelength=1.8)
    return save_figure(fig, out_dir, "Figure_2_Retained_vs_Holdout_Retrieval_publication")


def generate_validation_figures(out_dir: Path, figure1_title: str | None, figure2_title: str | None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    phase1 = load_phase1_metrics()
    phase2 = load_phase2_metrics()
    figure1_data = build_figure1_data(phase1)
    figure2_data = build_figure2_data(phase1, phase2)
    csv1 = write_figure1_csv(figure1_data, out_dir)
    csv2 = write_figure2_csv(figure2_data, out_dir)
    fig1 = plot_validation_figure1(figure1_data, out_dir, figure1_title)
    fig2 = plot_validation_figure2(figure2_data, out_dir, figure2_title)
    return [csv1, csv2, *fig1, *fig2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-paper figures.")
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--family-distance-data", type=Path, default=Path("paper/figures/family_case_distance_matrix_data.csv"))
    parser.add_argument("--skip-family-distance", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validation-subdir", default="validation")
    parser.add_argument("--validation-figure1-title", default="Figure 1. Retained-corpus ablation diagnostics")
    parser.add_argument("--validation-figure2-title", default="Figure 2. Retained-corpus versus family-held-out retrieval")
    parser.add_argument("--no-validation-suptitles", action="store_true")
    args = parser.parse_args()

    configure_matplotlib()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    family_distance_data = (
        args.family_distance_data if args.family_distance_data.is_absolute() else ROOT / args.family_distance_data
    )
    outputs: list[Path] = []
    if not args.skip_family_distance:
        generate_family_distance_figure(family_distance_data, out_dir)
        outputs.extend(
            [
                out_dir / "family_case_distance_matrix.png",
                out_dir / "family_case_distance_matrix.pdf",
                out_dir / "family_case_distance_matrix_data.csv",
            ]
        )
    if not args.skip_validation:
        outputs.extend(
            generate_validation_figures(
                out_dir / args.validation_subdir,
                None if args.no_validation_suptitles else args.validation_figure1_title,
                None if args.no_validation_suptitles else args.validation_figure2_title,
            )
        )
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
