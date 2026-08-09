#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.canonical_experiments import RESULTS, ROOT


DEFAULT_OUT = RESULTS / "canonical_reports"
NONCANONICAL_MARKERS = ("random_full_model", "paired_versus_random", "bilingual_alignment_lambda_sweep")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def tex_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["\\begin{tabular}{" + "l" * len(fields) + "}", " \\hline"]
    lines.append(" & ".join(fields).replace("_", "\\_") + " \\\\")
    lines.append(" \\hline")
    for row in rows:
        lines.append(" & ".join(fmt(row.get(field, "")) for field in fields).replace("_", "\\_") + " \\\\")
    lines.extend([" \\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def summary_lookup(rows: Iterable[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[field] for field in key_fields): row for row in rows}


def copy_metric_rows(summary_path: Path, out_csv: Path, out_tex: Path, fields: list[str], wanted: list[tuple[str, str]]) -> None:
    source = summary_lookup(read_csv(summary_path), ("condition", "metric"))
    rows = []
    for condition, metric in wanted:
        row = source[(condition, metric)]
        rows.append(
            {
                "condition": condition,
                "metric": metric,
                "mean": float(row["mean"]),
                "sample_sd": float(row["sample_sd"]),
                "seed_count": int(float(row.get("seed_count", row.get("count", 0)))),
            }
        )
    write_csv(out_csv, rows, fields)
    out_tex.write_text(tex_table(rows, fields), encoding="utf-8")


def build_phase1(out_dir: Path) -> None:
    wanted = [
        ("full_model", "structure_cross_language_top1"),
        ("full_model", "structure_cross_language_mrr"),
        ("full_model", "text_cross_language_top1"),
        ("full_model", "structure_sibling_vs_unrelated_contrast"),
        ("full_model", "parent_accuracy"),
        ("full_model", "depth_accuracy"),
        ("full_model", "successor_top1"),
        ("successor_only", "structure_cross_language_top1"),
        ("successor_only", "structure_cross_language_mrr"),
        ("successor_only", "structure_sibling_vs_unrelated_contrast"),
        ("shuffled_joint_targets", "structure_cross_language_top1"),
        ("shuffled_joint_targets", "structure_cross_language_mrr"),
        ("shuffled_joint_targets", "structure_sibling_vs_unrelated_contrast"),
    ]
    fields = ["condition", "metric", "mean", "sample_sd", "seed_count"]
    copy_metric_rows(RESULTS / "phase1_ablations" / "phase1_ablation_summary.csv", out_dir / "phase1_ablation_summary.csv", out_dir / "phase1_ablation_table.tex", fields, wanted)


def build_phase2(out_dir: Path) -> None:
    wanted = [
        ("full_model", "structure_test_candidates_top1"),
        ("full_model", "structure_test_candidates_mrr"),
        ("full_model", "structure_complete_candidates_top1"),
        ("full_model", "text_test_candidates_top1"),
        ("full_model", "depth_accuracy"),
        ("full_model", "child_count_mae"),
        ("full_model", "structure_sibling_vs_matched_unrelated_contrast"),
        ("no_successor", "structure_test_candidates_top1"),
        ("no_successor", "structure_test_candidates_mrr"),
        ("no_successor", "structure_complete_candidates_top1"),
        ("reconstruction_only", "structure_test_candidates_top1"),
        ("reconstruction_only", "structure_test_candidates_mrr"),
        ("reconstruction_only", "structure_complete_candidates_top1"),
        ("reconstruction_only", "structure_sibling_vs_matched_unrelated_contrast"),
        ("full_model", "reference_char_3_5_tfidf_test_top1"),
        ("full_model", "reference_char_3_5_tfidf_test_mrr"),
        ("full_model", "reference_char_3_5_tfidf_complete_top1"),
    ]
    fields = ["condition", "metric", "mean", "sample_sd", "seed_count"]
    copy_metric_rows(RESULTS / "phase2_family_holdout" / "phase2_summary.csv", out_dir / "phase2_family_holdout_summary.csv", out_dir / "phase2_family_holdout_table.tex", fields, wanted)


def build_phase3(out_dir: Path) -> None:
    source = read_csv(RESULTS / "phase3_controlled_alignment" / "phase3_summary.csv")
    rows = [
        row
        for row in source
        if row["batching"] == "paired"
        and row["condition"] in {"full_model", "no_successor"}
        and not any(marker in ",".join(row.values()) for marker in NONCANONICAL_MARKERS)
    ]
    if any(row.get("sampler_type") == "random_row" for row in rows):
        raise SystemExit("Refusing to build canonical Phase 3 table with removed random-batching records.")
    wanted_metrics = {
        "structure_cross_language_top1",
        "structure_cross_language_mrr",
        "structure_cross_language_same_id_distance",
        "parent_accuracy",
        "depth_accuracy",
        "structure_wider_neighbourhood_jaccard_k10",
    }
    report_rows = [
        {
            "condition": row["condition"],
            "lambda_language_alignment": float(row["lambda_language_alignment"]),
            "metric": row["metric"],
            "mean": float(row["mean"]),
            "sample_sd": float(row["sample_sd"]),
            "seed_count": int(float(row["seed_count"])),
        }
        for row in rows
        if row["metric"] in wanted_metrics and (row["condition"] == "full_model" or float(row["lambda_language_alignment"]) in {0.0, 1.0})
    ]
    fields = ["condition", "lambda_language_alignment", "metric", "mean", "sample_sd", "seed_count"]
    write_csv(out_dir / "phase3_paired_alignment_summary.csv", report_rows, fields)
    (out_dir / "phase3_paired_alignment_table.tex").write_text(tex_table(report_rows, fields), encoding="utf-8")


def build_lexical(out_dir: Path) -> None:
    phase1 = summary_lookup(read_csv(RESULTS / "phase1_ablations" / "phase1_ablation_summary.csv"), ("condition", "metric"))
    rows = [
        {"method": "Random ranking", "top1_mean": 1 / 526, "mrr_mean": 0.0130, "source": "deterministic candidate-count reference"},
        {"method": "Exact-token Jaccard", "top1_mean": 0.17680608365019013, "mrr_mean": 0.25393282503948766, "source": "retained canonical lexical reference value"},
        {"method": "Word TF-IDF", "top1_mean": 0.12927756653992395, "mrr_mean": 0.1941290691602779, "source": "retained canonical lexical reference value"},
        {"method": "Character 3-5 TF-IDF", "top1_mean": 0.40969581749049433, "mrr_mean": 0.5239247080137711, "source": "retained canonical lexical reference value"},
    ]
    rows.append(
        {
            "method": "Phase 1 full-model structure latent",
            "top1_mean": float(phase1[("full_model", "structure_cross_language_top1")]["mean"]),
            "mrr_mean": float(phase1[("full_model", "structure_cross_language_mrr")]["mean"]),
            "source": "results/dsh_validation/phase1_ablations/phase1_ablation_summary.csv",
        }
    )
    fields = ["method", "top1_mean", "mrr_mean", "source"]
    write_csv(out_dir / "retained_lexical_references.csv", rows, fields)
    (out_dir / "retained_lexical_reference_table.tex").write_text(tex_table(rows, fields), encoding="utf-8")


def build_phase4(out_dir: Path) -> None:
    source = RESULTS / "phase4_case_studies" / "candidate_manifest_pre_text.csv"
    rows = read_csv(source)
    slim_rows = [
        {
            "case_study": row["case_study"],
            "role": row["role"],
            "selection_unit": row["selection_unit"],
            "selected_id": row["id"] or row["family_id"],
        }
        for row in rows
    ]
    fields = ["case_study", "role", "selection_unit", "selected_id"]
    write_csv(out_dir / "phase4_case_manifest_summary.csv", slim_rows, fields)
    (out_dir / "phase4_case_table.tex").write_text(tex_table(slim_rows, fields), encoding="utf-8")


def build_reports(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    build_phase1(out_dir)
    build_lexical(out_dir)
    build_phase2(out_dir)
    build_phase3(out_dir)
    build_phase4(out_dir)
    index = "\n".join(f"- `{path.name}`" for path in sorted(out_dir.iterdir()) if path.is_file())
    (out_dir / "canonical_report_index.md").write_text("# Canonical Report Index\n\n" + index + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical empirical Phase 1-4 report tables from stored outputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_reports(args.out_dir.resolve())
    print(f"wrote canonical reports to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
