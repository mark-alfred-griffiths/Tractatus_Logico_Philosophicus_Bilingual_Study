#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.canonical_experiments import PHASE3_LAMBDA_GRID, PHASE3_SEEDS, RESULTS, ROOT, canonical_phase3_ids
from tools.export_canonical_evidence import export as dry_run_export


NONCANONICAL_MARKERS = ("random_full_model", "paired_versus_random", "bilingual_alignment_lambda_sweep")
PHASE1_CONDITIONS = {"full_model", "reconstruction_only", "no_successor", "parent_depth_only", "successor_only", "shuffled_joint_targets", "shuffled_no_successor"}
PHASE2_CONDITIONS = {"full_model", "no_successor", "reconstruction_only"}
PHASE4_HASH = "c4c6ac3c5473f47d181fdb8f1e155eab2d938a9bd674f2435c7fb48bc29e5ffc"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def verify(out_json: Path, out_md: Path) -> int:
    checks: list[dict[str, Any]] = []

    corpus = read_json(ROOT / "tractatus_structure_latents" / "data" / "tractatus_bilingual.json")
    ids = [str(row["id"]) for row in corpus]
    add(checks, "corpus proposition count", len(ids) == 526, str(len(ids)))
    add(checks, "corpus row count", sum(len(row.get("texts", {})) for row in corpus) == 1052, str(sum(len(row.get("texts", {})) for row in corpus)))
    add(checks, "no duplicate proposition IDs", len(ids) == len(set(ids)), str(len(ids) - len(set(ids))))
    add(checks, "bilingual pairing completeness", all({"en", "de"} <= set(row.get("texts", {})) for row in corpus), "all rows have en/de text")

    phase1 = read_csv(RESULTS / "phase1_ablations" / "phase1_ablation_summary.csv")
    add(checks, "Phase 1 condition completeness", {row["condition"] for row in phase1} >= PHASE1_CONDITIONS, ",".join(sorted({row["condition"] for row in phase1})))
    phase1_seed = read_csv(RESULTS / "phase1_ablations" / "phase1_seed_level_results.csv")
    phase1_runs = {(row["condition"], int(row["seed"])) for row in phase1_seed}
    add(checks, "Phase 1 seed completeness", all((condition, seed) in phase1_runs for condition in PHASE1_CONDITIONS for seed in range(10)), str(len(phase1_runs)))

    phase2 = read_csv(RESULTS / "phase2_family_holdout" / "phase2_summary.csv")
    add(checks, "Phase 2 condition completeness", {row["condition"] for row in phase2} >= PHASE2_CONDITIONS, ",".join(sorted({row["condition"] for row in phase2})))
    fold_rows = read_csv(RESULTS / "phase2_family_holdout" / "phase2_fold_manifest.csv")
    id_to_fold = {row["id"]: row["fold"] for row in fold_rows}
    family_to_folds: dict[str, set[str]] = {}
    for row in fold_rows:
        family_to_folds.setdefault(row["family_id"], set()).add(row["fold"])
    add(checks, "fold ID uniqueness", len(id_to_fold) == len(fold_rows), str(len(fold_rows)))
    add(checks, "fold family integrity", all(len(folds) == 1 for folds in family_to_folds.values()), str(sum(1 for folds in family_to_folds.values() if len(folds) > 1)))
    phase2_seed = read_csv(RESULTS / "phase2_family_holdout" / "phase2_seed_fold_results.csv")
    phase2_runs = {(row["condition"], int(row["fold"]), int(row["seed"])) for row in phase2_seed}
    add(checks, "Phase 2 seed-fold completeness", all((condition, fold, seed) in phase2_runs for condition in PHASE2_CONDITIONS for fold in range(5) for seed in range(3)), str(len(phase2_runs)))

    phase3_seed = read_csv(RESULTS / "phase3_controlled_alignment" / "phase3_seed_results.csv")
    phase3_ids = {row["experiment_id"] for row in phase3_seed}
    add(checks, "Phase 3 canonical IDs only", phase3_ids <= canonical_phase3_ids(), ",".join(sorted(phase3_ids - canonical_phase3_ids())))
    add(checks, "Phase 3 noncanonical exclusion", not any(any(marker in ",".join(row.values()) for marker in NONCANONICAL_MARKERS) for row in phase3_seed), "no removed/noncanonical markers in seed table")
    phase3_runs = {(row["batching"], row["condition"], float(row["lambda_language_alignment"]), int(row["seed"])) for row in phase3_seed}
    expected_phase3 = {("paired", condition, lam, seed) for condition in ("full_model", "no_successor") for lam in PHASE3_LAMBDA_GRID for seed in PHASE3_SEEDS}
    add(checks, "Phase 3 lambda-seed completeness", phase3_runs == expected_phase3, f"observed={len(phase3_runs)} expected={len(expected_phase3)}")
    coverage = read_csv(RESULTS / "phase3_controlled_alignment" / "phase3_pair_coverage.csv")
    add(checks, "Phase 3 pair coverage", all(float(row["min_pair_coverage"]) == 1.0 and float(row["max_pair_coverage"]) == 1.0 for row in coverage), str(len(coverage)))

    phase4_hash_path = RESULTS / "phase4_case_studies" / "candidate_manifest_pre_text.sha256"
    hash_text = phase4_hash_path.read_text(encoding="utf-8").strip().split()[0]
    add(checks, "Phase 4 hash unchanged", hash_text == PHASE4_HASH, hash_text)
    phase4_cases = {row["id"] or row["family_id"] for row in read_csv(RESULTS / "phase4_case_studies" / "candidate_manifest_pre_text.csv")}
    expected_cases = {"3.31", "5.5", "5.63", "4.31", "4.431", "2.13", "2.19", "5.524", "6.241", "5.64"}
    add(checks, "Phase 4 selected cases unchanged", phase4_cases == expected_cases, ",".join(sorted(phase4_cases)))

    reports = RESULTS / "canonical_reports"
    required = [
        "phase1_ablation_summary.csv",
        "retained_lexical_references.csv",
        "phase2_family_holdout_summary.csv",
        "phase3_paired_alignment_summary.csv",
        "phase4_case_manifest_summary.csv",
        "canonical_report_index.md",
    ]
    add(checks, "canonical reports present", all((reports / name).exists() for name in required), ",".join(name for name in required if not (reports / name).exists()))
    if (reports / "phase3_paired_alignment_summary.csv").exists():
        phase3_report = read_csv(reports / "phase3_paired_alignment_summary.csv")
        add(checks, "canonical report excludes noncanonical records", not any(any(marker in ",".join(row.values()) for marker in NONCANONICAL_MARKERS) for row in phase3_report), "phase3 paired table checked")

    with tempfile.TemporaryDirectory() as tmp:
        manifest = dry_run_export(Path(tmp), dry_run=True, quiet=True)
    excluded = manifest.get("excluded_noncanonical_artifacts", [])
    add(checks, "canonical export excludes noncanonical records", not excluded or all(any(marker in item for marker in NONCANONICAL_MARKERS) for item in excluded), ",".join(excluded))

    required_outputs = [
        RESULTS / "phase1_ablations" / "phase1_ablation_summary.csv",
        RESULTS / "phase2_family_holdout" / "phase2_summary.csv",
        RESULTS / "phase3_controlled_alignment" / "phase3_summary.csv",
        RESULTS / "phase4_case_studies" / "candidate_manifest_pre_text.csv",
    ]
    add(checks, "required output files exist", all(path.exists() for path in required_outputs), ",".join(str(path) for path in required_outputs if not path.exists()))

    ok = all(check["ok"] for check in checks)
    payload = {"ok": ok, "checks": checks}
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Canonical Verification Report", "", f"Status: {'PASS' if ok else 'FAIL'}", ""]
    lines.extend(f"- [{'x' if check['ok'] else ' '}] {check['name']}: {check['detail']}" for check in checks)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify canonical empirical evidence without retraining.")
    parser.add_argument("--json-out", type=Path, default=RESULTS / "canonical_verification.json")
    parser.add_argument("--report-out", type=Path, default=RESULTS / "CANONICAL_VERIFICATION_REPORT.md")
    args = parser.parse_args()
    raise SystemExit(verify(args.json_out.resolve(), args.report_out.resolve()))


if __name__ == "__main__":
    main()
