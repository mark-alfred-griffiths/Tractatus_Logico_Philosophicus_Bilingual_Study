#!/usr/bin/env python3
"""Build the final-paper artifact manifest from existing repository files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "paper" / "final_paper_manifest.csv"

COLUMNS = [
    "path",
    "artifact_type",
    "canonical_status",
    "used_by_final_manuscript",
    "source_artifact",
    "generation_script",
    "generation_command",
    "verification_command",
    "sha256",
    "bytes",
    "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_paths(row: dict[str, str]) -> list[Path]:
    paths: list[Path] = []
    for column in ("figure_file", "companion_file", "source_data"):
        for value in row.get(column, "").split(";"):
            value = value.strip()
            if value:
                paths.append(ROOT / value)
    return paths


def add(
    rows: list[dict[str, str]],
    path: Path,
    artifact_type: str,
    canonical_status: str,
    used: str,
    source: str,
    script: str,
    command: str,
    verify: str,
    notes: str,
) -> None:
    if not path.is_file():
        return
    rel = path.relative_to(ROOT).as_posix()
    rows.append(
        {
            "path": rel,
            "artifact_type": artifact_type,
            "canonical_status": canonical_status,
            "used_by_final_manuscript": used,
            "source_artifact": source,
            "generation_script": script,
            "generation_command": command,
            "verification_command": verify,
            "sha256": sha256(path),
            "bytes": str(path.stat().st_size),
            "notes": notes,
        }
    )


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    figure_manifest = ROOT / "paper" / "figures" / "figure_manifest.csv"
    add(
        rows,
        figure_manifest,
        "figure_manifest",
        "audit_only",
        "no",
        "final manuscript and paper/figures/",
        "manual manifest construction from final manuscript figure files",
        "TBD",
        "python3 tools/validate_paper_figure_manifest.py",
        "Maps final manuscript figure use and cleanup status.",
    )
    final_figure_paths = sorted(
        {
            path
            for row in read_csv(figure_manifest)
            if row.get("used_in_final_manuscript") == "yes"
            for path in manifest_paths(row)
        }
    )
    for path in final_figure_paths:
        artifact_type = "paper_figure_source_data" if path.suffix.lower() == ".csv" else "paper_figure"
        add(
            rows,
            path,
            artifact_type,
            "derived_paper_output",
            "yes",
            "results/dsh_validation/phase4_case_studies/",
            "tractatus_structure_latents.evaluation.generate_paper_figures",
            "python3 -m tractatus_structure_latents.evaluation.generate_paper_figures",
            "python3 tools/validate_paper_figure_manifest.py",
            "Final Figure 1 artifact.",
        )

    validation_dir = ROOT / "paper" / "figures" / "validation"
    for path in sorted(validation_dir.glob("*")):
        if not path.is_file():
            continue
        artifact_type = "validation_figure_source_data" if path.suffix.lower() == ".csv" else "validation_figure"
        add(
            rows,
            path,
            artifact_type,
            "derived_paper_output",
            "candidate",
            "results/dsh_validation/phase1_ablations/; results/dsh_validation/phase2_family_holdout/; results/dsh_validation/canonical_reports/retained_lexical_references.csv",
            "tractatus_structure_latents.evaluation.generate_paper_figures",
            "python3 -m tractatus_structure_latents.evaluation.generate_paper_figures --skip-family-distance",
            "python3 tools/validate_paper_figure_manifest.py --require-validation",
            "Publication-quality validation figure artifact.",
        )

    for path in sorted((ROOT / "paper" / "tables").glob("*.csv")):
        if not path.is_file():
            continue
        add(
            rows,
            path,
            "paper_table_manifest" if path.name == "table_manifest.csv" else "paper_table",
            "audit_only" if path.name == "table_manifest.csv" else "derived_paper_output",
            "no" if path.name == "table_manifest.csv" else "yes",
            "results/dsh_validation/canonical_reports/",
            "tools/export_paper_tables.py",
            "python3 tools/export_paper_tables.py",
            "python3 -m unittest tests/test_paper_tables.py",
            "Table-ready canonical-report export.",
        )

    for path in sorted((ROOT / "tractatus_structure_latents" / "data").glob("*.json")):
        add(
            rows,
            path,
            "dataset_json",
            "source_data",
            "yes",
            "Tractatus source corpus and dataset builder",
            "tractatus_structure_latents/scripts/build_dataset.py",
            "python3 -m tractatus_structure_latents.scripts.build_dataset",
            "python3 tools/verify_canonical_evidence.py",
            "Retained generated dataset used by canonical empirical scripts.",
        )

    for path in [
        ROOT / "results" / "dsh_validation" / "CANONICAL_VERIFICATION_REPORT.md",
        ROOT / "results" / "dsh_validation" / "canonical_verification.json",
    ]:
        artifact_type = {
            ".md": "canonical_verification_report",
            ".json": "canonical_verification_json",
            ".zip": "validation_bundle_zip",
        }.get(path.suffix, "validation_artifact")
        add(
            rows,
            path,
            artifact_type,
            "audit_only" if path.suffix in {".md", ".json"} else "derived_paper_output",
            "candidate",
            "results/dsh_validation/phase1_ablations/; results/dsh_validation/phase2_family_holdout/; results/dsh_validation/phase3_controlled_alignment/; results/dsh_validation/phase4_case_studies/",
            "tools/verify_canonical_evidence.py" if path.suffix in {".md", ".json"} else "tools/create_dsh_validation_bundle.py",
            "python3 tools/verify_canonical_evidence.py" if path.suffix in {".md", ".json"} else "python3 tools/create_dsh_validation_bundle.py",
            "python3 tools/verify_canonical_evidence.py",
            "Canonical validation artifact.",
        )

    add(
        rows,
        ROOT / "docs" / "heavy_artifacts_manifest.csv",
        "external_artifact_manifest",
        "audit_only",
        "candidate",
        "external release/archive artifact",
        "generated from SHA256SUMS before Git history cleanup",
        "TBD",
        "manual archive hash and per-file SHA256 verification",
        "Lists heavy retained artifacts removed from Git history and expected in the external archive.",
    )

    for path in sorted((ROOT / "results" / "dsh_validation" / "canonical_reports").glob("*")):
        if path.is_file():
            add(
                rows,
                path,
                "canonical_report_table" if path.suffix in {".csv", ".tex"} else "canonical_report_index",
                "derived_paper_output",
                "candidate",
                "results/dsh_validation/phase1_ablations/; results/dsh_validation/phase2_family_holdout/; results/dsh_validation/phase3_controlled_alignment/; results/dsh_validation/phase4_case_studies/",
                "tools/build_canonical_reports.py",
                "python3 tools/build_canonical_reports.py",
                "python3 tools/verify_canonical_evidence.py",
                "Canonical report or table artifact supporting final manuscript tables.",
            )

    return sorted(rows, key=lambda row: row["path"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
