#!/usr/bin/env python3
"""Single safe non-training entry point for final-paper outputs."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NONCANONICAL_MARKERS = (
    "random_full_model",
    "paired_versus_random",
    "random_row",
    "bilingual_alignment_lambda_sweep",
)
FORBIDDEN_COMMAND_MARKERS = (
    "train_vae",
    "run_bilingual_alignment_seed_sweep",
    "phase1_ablations.py",
    "phase2_family_holdout.py",
    "phase3_controlled_alignment.py",
    "phase4_case_studies.py",
)


@dataclass(frozen=True)
class Step:
    label: str
    command: tuple[str, ...]
    env: dict[str, str] | None = None

    def display(self) -> str:
        prefix = ""
        if self.env:
            prefix = " ".join(f"{key}={value}" for key, value in sorted(self.env.items())) + " "
        return prefix + " ".join(self.command)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fail_if_removed_artifact_is_canonical() -> None:
    failures: list[str] = []

    figure_manifest = ROOT / "paper" / "figures" / "figure_manifest.csv"
    for row in read_csv(figure_manifest):
        joined = ",".join(row.values())
        canonical = row.get("used_in_final_docx") == "yes" or row.get("canonical_status") == "canonical_evidence"
        if canonical and any(marker in joined for marker in NONCANONICAL_MARKERS):
            failures.append(f"{figure_manifest}: {row.get('figure_id', '<unknown>')} uses a removed/noncanonical marker")

    table_manifest = ROOT / "paper" / "tables" / "table_manifest.csv"
    for row in read_csv(table_manifest):
        joined = ",".join(row.values())
        if row.get("canonical_status") in {"canonical_evidence", "derived_paper_output"} and any(marker in joined for marker in NONCANONICAL_MARKERS):
            failures.append(f"{table_manifest}: {row.get('table_id', '<unknown>')} uses a removed/noncanonical marker")

    final_manifest = ROOT / "paper" / "final_paper_manifest.csv"
    for row in read_csv(final_manifest):
        joined = ",".join(row.values())
        canonical = row.get("used_by_tractatus_final_docx") == "yes" or row.get("canonical_status") == "canonical_evidence"
        if canonical and any(marker in joined for marker in NONCANONICAL_MARKERS):
            failures.append(f"{final_manifest}: {row.get('path', '<unknown>')} uses a removed/noncanonical marker")

    phase3 = ROOT / "results" / "dsh_validation" / "canonical_reports" / "phase3_paired_alignment_summary.csv"
    for row in read_csv(phase3):
        joined = ",".join(row.values())
        if any(marker in joined for marker in NONCANONICAL_MARKERS):
            failures.append(f"{phase3}: canonical Phase 3 row contains a removed/noncanonical marker")

    if failures:
        raise SystemExit("Refusing to use removed/noncanonical random-batching paths as canonical evidence:\n" + "\n".join(failures))


def build_steps(include_bundle: bool) -> list[Step]:
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    steps = [
        Step("Build canonical reports", ("python3", "tools/build_canonical_reports.py"), env),
        Step("Verify canonical evidence", ("python3", "tools/verify_canonical_evidence.py"), env),
        Step("Export paper tables", ("python3", "tools/export_paper_tables.py"), env),
        Step("Validate paper tables", ("python3", "-m", "unittest", "tests/test_paper_tables.py"), env),
        Step("Validate paper figure manifest and 600dpi used PNGs", ("python3", "tools/validate_paper_figure_manifest.py"), env),
        Step("Build final-paper manifest", ("python3", "tools/build_final_paper_manifest.py"), env),
        Step("Validate final-paper manifest", ("python3", "tools/validate_final_paper_manifest.py"), env),
    ]
    if include_bundle:
        steps.append(Step("Create DSH validation bundle", ("python3", "tools/create_dsh_validation_bundle.py"), env))
    return steps


def validate_step_safety(steps: list[Step]) -> None:
    failures = []
    for step in steps:
        display = step.display()
        if any(marker in display for marker in FORBIDDEN_COMMAND_MARKERS):
            failures.append(f"{step.label}: {display}")
    if failures:
        raise SystemExit("Refusing to run training or empirical phase commands:\n" + "\n".join(failures))


def run_steps(steps: list[Step], dry_run: bool) -> None:
    base_env = os.environ.copy()
    for step in steps:
        print(f"\n# {step.label}")
        print(step.display(), flush=True)
        if dry_run:
            continue
        env = base_env.copy()
        if step.env:
            env.update(step.env)
        subprocess.run(step.command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce final-paper outputs without retraining.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-bundle", action="store_true", help="Accepted for compatibility; bundle generation is skipped unless --include-bundle is supplied.")
    parser.add_argument("--include-bundle", action="store_true", help="Also create the local ignored validation bundle copy.")
    args = parser.parse_args()

    steps = build_steps(include_bundle=args.include_bundle and not args.skip_bundle)
    validate_step_safety(steps)
    fail_if_removed_artifact_is_canonical()
    run_steps(steps, dry_run=args.dry_run)
    if not args.dry_run:
        fail_if_removed_artifact_is_canonical()
    if args.dry_run:
        print("\nDry run only. Review the command list before running without --dry-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
