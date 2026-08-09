#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.canonical_experiments import RESULTS, ROOT


NONCANONICAL_MARKERS = ("random_full_model", "paired_versus_random", "bilingual_alignment_lambda_sweep")

CANONICAL_FILES = [
    "canonical_reports/phase1_ablation_summary.csv",
    "canonical_reports/phase1_ablation_table.tex",
    "canonical_reports/retained_lexical_references.csv",
    "canonical_reports/retained_lexical_reference_table.tex",
    "canonical_reports/phase2_family_holdout_summary.csv",
    "canonical_reports/phase2_family_holdout_table.tex",
    "canonical_reports/phase3_paired_alignment_summary.csv",
    "canonical_reports/phase3_paired_alignment_table.tex",
    "canonical_reports/phase4_case_manifest_summary.csv",
    "canonical_reports/phase4_case_table.tex",
    "canonical_reports/canonical_report_index.md",
    "phase1_ablations/phase1_ablation_report.md",
    "phase1_ablations/phase1_verification_report.md",
    "phase1_ablations/phase1_config_manifest.json",
    "phase1_ablations/phase1_ablation_summary.csv",
    "phase2_family_holdout/phase2_holdout_report.md",
    "phase2_family_holdout/phase2_leakage_checks.md",
    "phase2_family_holdout/phase2_verification_report.md",
    "phase2_family_holdout/phase2_fold_manifest.csv",
    "phase2_family_holdout/phase2_summary.csv",
    "phase3_controlled_alignment/phase3_alignment_report.md",
    "phase3_controlled_alignment/phase3_verification_report.md",
    "phase3_controlled_alignment/phase3_summary.csv",
    "phase3_controlled_alignment/phase3_seed_results.csv",
    "phase3_controlled_alignment/phase3_pair_coverage.csv",
    "phase4_case_studies/phase4_case_selection_protocol.md",
    "phase4_case_studies/phase4_case_studies_report.md",
    "phase4_case_studies/phase4_verification_report.md",
    "phase4_case_studies/candidate_manifest_pre_text.csv",
    "phase4_case_studies/candidate_manifest_pre_text.sha256",
    "phase4_case_studies/candidate_manifest_with_text.csv",
    "canonical_verification.json",
    "CANONICAL_VERIFICATION_REPORT.md",
]


def is_noncanonical_removed_artifact(path: Path | str) -> bool:
    text = path.as_posix() if isinstance(path, Path) else path
    return any(marker in text for marker in NONCANONICAL_MARKERS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256", "status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_files() -> tuple[list[Path], list[str]]:
    included: list[Path] = []
    excluded: list[str] = []
    for relative in CANONICAL_FILES:
        if is_noncanonical_removed_artifact(relative):
            excluded.append(relative)
            continue
        source = RESULTS / relative
        if source.exists():
            included.append(source)
    for source in [
        *sorted((RESULTS / "phase3_controlled_alignment" / "figures").glob("*.png")),
        *sorted((RESULTS / "phase4_case_studies" / "figures").glob("*.png")),
    ]:
        relative = source.relative_to(RESULTS).as_posix()
        if is_noncanonical_removed_artifact(relative):
            excluded.append(relative)
        else:
            included.append(source)
    return sorted(set(included)), sorted(set(excluded))


def export(destination: Path, dry_run: bool, quiet: bool = False) -> dict[str, object]:
    included, excluded = collect_files()
    manifest_rows: list[dict[str, str]] = []
    conflicts: list[str] = []
    for source in included:
        relative = source.relative_to(RESULTS)
        target = destination / relative
        source_hash = sha256(source)
        if target.exists() and sha256(target) != source_hash:
            conflicts.append(str(relative))
            continue
        manifest_rows.append({"path": str(relative), "bytes": str(source.stat().st_size), "sha256": source_hash, "status": "dry-run" if dry_run else "exported"})
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if conflicts:
        raise SystemExit(f"Refusing to overwrite differing destination files: {conflicts[:20]}")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(ROOT),
        "destination": str(destination),
        "dry_run": dry_run,
        "exported_count": len(manifest_rows),
        "excluded_noncanonical_artifacts": excluded,
        "files": manifest_rows,
    }
    if dry_run:
        if not quiet:
            print(json.dumps(manifest, indent=2))
    else:
        write_csv(destination / "canonical_export_manifest.csv", manifest_rows)
        (destination / "canonical_export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical empirical evidence only.")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    export(args.destination.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
