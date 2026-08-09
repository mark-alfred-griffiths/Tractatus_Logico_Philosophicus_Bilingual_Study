#!/usr/bin/env python3
"""Export table-ready CSV sources for the final manuscript."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPORTS = ROOT / "results" / "dsh_validation" / "canonical_reports"
TABLE_DIR = ROOT / "paper" / "tables"

TABLES = [
    {
        "table_id": "Table 1",
        "manuscript_caption": (
            "Retained-corpus ablation diagnostics. Values are mean +/- sample SD "
            "across seeds 0-9. Structure and text Top-1/MRR are same-ID "
            "German-English retrieval metrics computed after all proposition IDs "
            "have been observed during training; they are not held-out scores. "
            "The sibling contrast is unrelated-minus-sibling distance, so larger "
            "positive values indicate stronger retained family cohesion. Parent, "
            "depth, and successor accuracies are supervised formal-position "
            "diagnostics with non-comparable target granularities."
        ),
        "table_file": "paper/tables/table1_retained_ablations.csv",
        "source_data": "results/dsh_validation/canonical_reports/phase1_ablation_summary.csv",
    },
    {
        "table_id": "Table 2",
        "manuscript_caption": (
            "Retained-corpus cross-language retrieval reference levels using the "
            "exact input spans retained under the 96-token limit. Proposition "
            "identifiers are excluded. Surface baselines are deterministic means "
            "across German-to-English and English-to-German directions; the "
            "neural row reports the ten-seed retained-corpus full-model mean. "
            "These are fitted-corpus reference levels rather than evidence of "
            "held-out generalization."
        ),
        "table_file": "paper/tables/table2_lexical_references.csv",
        "source_data": "results/dsh_validation/canonical_reports/retained_lexical_references.csv",
    },
    {
        "table_id": "Table 3",
        "manuscript_caption": (
            "Results from deterministic five-fold immediate-parent-family holdout. "
            "Values are mean +/- sample SD across 15 seed-fold runs per model "
            "condition. Test-candidate retrieval ranks only candidates in the "
            "held-out fold; complete-candidate Top-1 ranks all 526 proposition "
            "IDs. The lexical row is a deterministic exact-input character "
            "3-5-gram TF-IDF reference repeated in the same seed-fold summary. "
            "Parent and successor exact unseen-class accuracies are excluded "
            "because most held-out classes are absent from training."
        ),
        "table_file": "paper/tables/table3_family_holdout.csv",
        "source_data": "results/dsh_validation/canonical_reports/phase2_family_holdout_summary.csv",
    },
    {
        "table_id": "Table 4",
        "manuscript_caption": (
            "Controlled retained-corpus alignment results. Values are mean +/- "
            "sample SD across seeds 0-9. Every paired run has verified pair "
            "coverage of 1.0000, meaning that each German-English same-ID pair "
            "contributes to the alignment objective once per epoch. These "
            "retained-corpus results test the mechanics of direct pairwise "
            "attraction and do not establish held-out generalization or semantic "
            "equivalence."
        ),
        "table_file": "paper/tables/table4_paired_alignment.csv",
        "source_data": "results/dsh_validation/canonical_reports/phase3_paired_alignment_summary.csv",
    },
    {
        "table_id": "Table 5",
        "manuscript_caption": (
            "Verified pre-specified scholarly case-study prompts. All rows come "
            "from the frozen pre-text manifest, selected before wording was "
            "joined. Values are retained-corpus diagnostics from the paired full "
            "model with lambda = 0.00 unless otherwise stated. Family distance is "
            "mean pairwise sibling distance; z is relative to actual matched "
            "families. Bilingual Jaccard is cross-direction k = 10 overlap after "
            "excluding the same-ID counterpart. Parent and successor values are "
            "mean ranks. These diagnostics select prompts for human close reading "
            "and do not establish philosophical relations, semantic equivalence, "
            "anomaly, or mistranslation."
        ),
        "table_file": "paper/tables/table5_case_study_prompts.csv",
        "source_data": "results/dsh_validation/canonical_reports/phase4_case_manifest_summary.csv",
    },
]

MANIFEST_COLUMNS = [
    "table_id",
    "manuscript_caption",
    "table_file",
    "source_data",
    "generation_script",
    "verification_command",
    "canonical_status",
    "notes",
]


def copy_table(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def write_manifest(path: Path) -> None:
    rows = []
    for table in TABLES:
        rows.append(
            {
                "table_id": table["table_id"],
                "manuscript_caption": table["manuscript_caption"],
                "table_file": table["table_file"],
                "source_data": table["source_data"],
                "generation_script": "tools/export_paper_tables.py",
                "verification_command": "python3 -m unittest tests/test_paper_tables.py",
                "canonical_status": "derived_paper_output",
                "notes": "Table-ready copy derived from canonical report CSV without manual value entry.",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export_tables() -> None:
    for table in TABLES:
        copy_table(ROOT / table["source_data"], ROOT / table["table_file"])
    write_manifest(TABLE_DIR / "table_manifest.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    export_tables()
    print(f"wrote {len(TABLES)} table CSVs and paper/tables/table_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
