"""Create a consolidated evidence bundle for DSH validation phases 1-4."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dsh_validation"
BUNDLE = RESULTS / "dsh_validation_bundle"
ZIP_PATH = RESULTS / "dsh_validation_bundle.zip"


@dataclass(frozen=True)
class Phase:
    name: str
    directory: Path
    objective: str
    conditions: str
    seeds: str
    key_result: str


PHASES = [
    Phase(
        "Phase 1",
        RESULTS / "phase1_ablations",
        "Retained-corpus formal-objective ablations for bilingual retrieval, formal prediction and family geometry.",
        "full_model; reconstruction_only; no_successor; parent_depth_only; successor_only; shuffled_joint_targets; shuffled_no_successor",
        "0-9",
        "Full-model retained-corpus structure Top-1 0.9360 +/- 0.0128; reconstruction-only structure Top-1 0.0035 +/- 0.0021; successor-only Top-1 0.9944 +/- 0.0022.",
    ),
    Phase(
        "Phase 2",
        RESULTS / "phase2_family_holdout",
        "Five-fold immediate-parent-family hold-out evaluation of generalisation to unseen texts and unseen families.",
        "full_model; no_successor; reconstruction_only",
        "0-2 across five folds",
        "Held-out test-candidate structure Top-1 0.0739 +/- 0.0236 for full_model and 0.0811 +/- 0.0188 for no_successor.",
    ),
    Phase(
        "Phase 3",
        RESULTS / "phase3_controlled_alignment",
        "Controlled paired-batch bilingual alignment sweep.",
        "paired full_model lambda 0.00, 0.03, 0.10, 0.30, 1.00; paired no_successor same grid",
        "0-9",
        "Pair coverage was 100%; paired full_model Top-1 increased from 0.9443 +/- 0.0100 at lambda 0.00 to 0.9959 +/- 0.0032 at lambda 1.00.",
    ),
    Phase(
        "Phase 4",
        RESULTS / "phase4_case_studies",
        "Text-blind selection of proposition-level scholarly case-study dossiers.",
        "paired_full_model_align000 primary; paired_full_model_align003 and paired_no_successor_align000 robustness",
        "0-9",
        "Ten selected dossiers were frozen by numerical/formal criteria before text join; pre-text manifest SHA-256 c4c6ac3c5473f47d181fdb8f1e155eab2d938a9bd674f2435c7fb48bc29e5ffc.",
    ),
]


REPORT_FILES = [
    "phase1_ablations/phase1_ablation_report.md",
    "phase1_ablations/phase1_verification_report.md",
    "phase2_family_holdout/phase2_holdout_report.md",
    "phase2_family_holdout/phase2_leakage_checks.md",
    "phase2_family_holdout/phase2_verification_report.md",
    "phase3_controlled_alignment/phase3_alignment_report.md",
    "phase3_controlled_alignment/phase3_verification_report.md",
    "phase4_case_studies/phase4_case_selection_protocol.md",
    "phase4_case_studies/phase4_case_studies_report.md",
    "phase4_case_studies/phase4_verification_report.md",
    "phase4_case_studies/dossiers",
]

SUMMARY_FILES = [
    "phase1_ablations/phase1_ablation_summary.csv",
    "phase1_ablations/phase1_seed_level_results.csv",
    "phase1_ablations/phase1_paired_differences.csv",
    "phase1_ablations/phase1_recomputed_seed_level_results.csv",
    "phase2_family_holdout/phase2_summary.csv",
    "phase2_family_holdout/phase2_seed_fold_results.csv",
    "phase2_family_holdout/phase2_fold_manifest.csv",
    "phase2_family_holdout/phase2_matched_family_results.csv",
    "phase3_controlled_alignment/phase3_summary.csv",
    "phase3_controlled_alignment/phase3_seed_results.csv",
    "phase3_controlled_alignment/phase3_pair_coverage.csv",
    "phase3_controlled_alignment/phase3_epoch_trajectories.parquet",
    "phase4_case_studies/phase4_robustness_summary.csv",
    "phase4_case_studies/candidate_manifest_pre_text.csv",
    "phase4_case_studies/candidate_manifest_pre_text.sha256",
    "phase4_case_studies/candidate_manifest_with_text.csv",
    "phase4_case_studies/data/family_candidate_metrics.csv",
    "phase4_case_studies/data/family_matched_controls.csv",
    "phase4_case_studies/data/family_member_texts.csv",
    "phase4_case_studies/data/family_seed_metrics.csv",
    "phase4_case_studies/data/hierarchy_sequence_candidate_metrics.csv",
    "phase4_case_studies/data/hierarchy_sequence_seed_metrics.csv",
    "phase4_case_studies/data/neighbourhood_candidate_metrics.csv",
    "phase4_case_studies/data/neighbourhood_seed_metrics.csv",
]

CONFIG_FILES = [
    "phase1_ablations/phase1_config_manifest.json",
    "phase1_ablations/configs",
    "phase2_family_holdout/phase2_config_manifest.json",
    "phase2_family_holdout/configs",
    "phase2_family_holdout/ids",
    "phase3_controlled_alignment/configs/paired_full_model.json",
    "phase3_controlled_alignment/configs/paired_no_successor.json",
    "phase3_controlled_alignment/configs/no_successor_expansion_decision.json",
    "phase4_case_studies/data/case_selection_config.json",
    "phase4_case_studies/data/manifest_freeze.json",
]

COMMAND_FILES = [
    "phase1_ablations/phase1_commands.sh",
    "phase2_family_holdout/phase2_commands.sh",
    "phase4_case_studies/phase4_commands.sh",
]

FIGURE_DIRS = [
    "phase1_ablations/figures",
    "phase2_family_holdout/figures",
    "phase3_controlled_alignment/figures/retrieval_across_lambda.png",
    "phase3_controlled_alignment/figures/formal_parent_accuracy_across_lambda.png",
    "phase3_controlled_alignment/figures/formal_depth_accuracy_across_lambda.png",
    "phase3_controlled_alignment/figures/formal_successor_accuracy_across_lambda.png",
    "phase3_controlled_alignment/figures/same_id_distance_across_lambda.png",
    "phase3_controlled_alignment/figures/alignment_loss_trajectories.png",
    "phase3_controlled_alignment/figures/structure_mean_norm_across_lambda.png",
    "phase3_controlled_alignment/figures/posterior_variance_across_lambda.png",
    "phase4_case_studies/figures",
]

NONCANONICAL_MARKERS = [
    "random_full_model",
    "paired_versus_random",
    "bilingual_alignment_lambda_sweep",
]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_clean_bundle_root() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)


def copy_artifact(relative: str, target_dir: Path) -> list[Path]:
    source = RESULTS / relative
    if not source.exists():
        return []
    copied: list[Path] = []
    phase_name = source.relative_to(RESULTS).parts[0]
    destination = target_dir / phase_name / source.name
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative_source = path.relative_to(RESULTS)
            if is_noncanonical_removed_artifact(relative_source):
                continue
            file_destination = destination / path.relative_to(source)
            file_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, file_destination)
            copied.append(file_destination)
    else:
        if is_noncanonical_removed_artifact(source.relative_to(RESULTS)):
            return []
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def is_noncanonical_removed_artifact(relative: Path) -> bool:
    text = relative.as_posix()
    return any(marker in text for marker in NONCANONICAL_MARKERS)


def allows_noncanonical_marker_text(relative: Path) -> bool:
    """Allow marker text only where it records ambient provenance, not evidence."""
    text = relative.as_posix()
    return text in {
        "configs/phase1_ablations/phase1_config_manifest.json",
        "configs/phase2_family_holdout/phase2_config_manifest.json",
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_publication_claims_matrix() -> None:
    rows = [
        {
            "claim": "The retained-corpus supervised structure latent gives high German-English same-ID retrieval.",
            "evidence source": "Phase 1 ablations; Phase 3 paired lambda=0.00",
            "retained-corpus or held-out": "retained-corpus",
            "ablation-supported": "yes, but strongly qualified by successor-only and shuffled-target controls",
            "seed count": "10",
            "principal metric": "Structure Top-1 0.9360 +/- 0.0128 in Phase 1; paired-batch Top-1 0.9443 +/- 0.0100 in Phase 3",
            "uncertainty": "sample SD across seeds; Phase 1 includes paired bootstrap contrasts",
            "limitations": "Retained-corpus setting; same IDs and formal labels are observed during fitting; shuffled shared targets also support high retrieval.",
            "recommended wording": "In retained-corpus evaluation, the supervised structure latent reliably retrieves German-English counterparts.",
            "keep, revise or remove": "revise",
        },
        {
            "claim": "Reconstruction alone explains little of the retained-corpus bilingual retrieval.",
            "evidence source": "Phase 1 reconstruction_only ablation",
            "retained-corpus or held-out": "retained-corpus",
            "ablation-supported": "yes",
            "seed count": "10",
            "principal metric": "Structure Top-1 0.0035 +/- 0.0021; text Top-1 0.0077 +/- 0.0034",
            "uncertainty": "sample SD across seeds; paired contrasts against full_model",
            "limitations": "Applies to current VAE configuration and preprocessing.",
            "recommended wording": "Under the current configuration, reconstruction alone does not produce measurable same-ID retrieval.",
            "keep, revise or remove": "keep",
        },
        {
            "claim": "Retrieval primarily resides in the structure latent rather than the text latent.",
            "evidence source": "Phase 1 latent-specific retrieval; Phase 2 held-out latent-specific retrieval",
            "retained-corpus or held-out": "retained-corpus and held-out",
            "ablation-supported": "yes, with smaller held-out effect sizes",
            "seed count": "10 retained; 15 seed-fold runs held-out",
            "principal metric": "Phase 1 full_model structure Top-1 0.9360 +/- 0.0128 versus text Top-1 0.0096 +/- 0.0042; Phase 2 full_model structure test Top-1 0.0739 +/- 0.0236 versus text Top-1 0.0288 +/- 0.0136",
            "uncertainty": "sample SD across seeds or seed-fold runs",
            "limitations": "Held-out retrieval remains low in absolute terms.",
            "recommended wording": "The supervised structure latent carries the dominant retrieval signal, though held-out retrieval is much weaker than retained-corpus retrieval.",
            "keep, revise or remove": "revise",
        },
        {
            "claim": "The successor objective is sufficient to reproduce most retained-corpus bilingual retrieval.",
            "evidence source": "Phase 1 successor_only ablation",
            "retained-corpus or held-out": "retained-corpus",
            "ablation-supported": "yes",
            "seed count": "10",
            "principal metric": "successor_only structure Top-1 0.9944 +/- 0.0022; MRR 0.9971 +/- 0.0011",
            "uncertainty": "sample SD across seeds",
            "limitations": "Successor labels are nearly proposition-specific; this does not establish semantic or hierarchy-general retrieval.",
            "recommended wording": "Much retained-corpus retrieval can be produced by proposition-specific sequence supervision.",
            "keep, revise or remove": "keep as limitation",
        },
        {
            "claim": "True hierarchy-derived supervision is necessary for sibling cohesion.",
            "evidence source": "Phase 1 true-target versus shuffled-target ablations",
            "retained-corpus or held-out": "retained-corpus",
            "ablation-supported": "yes",
            "seed count": "10",
            "principal metric": "Sibling-versus-unrelated contrast 4.0361 +/- 0.0632 for full_model, 3.4324 +/- 0.0362 for no_successor, 0.0330 +/- 0.0733 for shuffled_joint_targets",
            "uncertainty": "sample SD and paired bootstrap contrasts",
            "limitations": "Retained-corpus geometry; not a philosophical relation claim.",
            "recommended wording": "Sibling cohesion depends on true hierarchy-derived targets in the retained-corpus representation.",
            "keep, revise or remove": "keep",
        },
        {
            "claim": "Same-ID bilingual retrieval survives unseen-family held-out evaluation.",
            "evidence source": "Phase 2 family hold-out",
            "retained-corpus or held-out": "held-out",
            "ablation-supported": "partly; full_model and no_successor exceed reconstruction_only but remain weak",
            "seed count": "15 seed-fold runs per condition",
            "principal metric": "Test-candidate structure Top-1 0.0739 +/- 0.0236 for full_model; 0.0811 +/- 0.0188 for no_successor; 0.0158 +/- 0.0056 for reconstruction_only",
            "uncertainty": "sample SD across seed-fold runs",
            "limitations": "Performance is far below retained-corpus levels; fold family sizes and unseen classes constrain interpretation.",
            "recommended wording": "A small held-out same-ID retrieval signal remains under unseen-family splitting.",
            "keep, revise or remove": "revise",
        },
        {
            "claim": "Removing the successor objective reduces held-out performance.",
            "evidence source": "Phase 2 no_successor comparison",
            "retained-corpus or held-out": "held-out",
            "ablation-supported": "no",
            "seed count": "15 seed-fold runs per condition",
            "principal metric": "no_successor test-candidate structure Top-1 0.0811 +/- 0.0188 versus full_model 0.0739 +/- 0.0236",
            "uncertainty": "sample SD across seed-fold runs",
            "limitations": "Initial seeds 0-2 only; small differences should not be described as meaningful without stronger evidence.",
            "recommended wording": "In the current held-out run, no_successor is not worse than full_model on same-ID retrieval.",
            "keep, revise or remove": "remove",
        },
        {
            "claim": "Exact parent and successor classification can be evaluated normally under unseen-family splitting.",
            "evidence source": "Phase 2 leakage and held-out reports",
            "retained-corpus or held-out": "held-out",
            "ablation-supported": "no; task is invalid for unseen classes without separate coverage accounting",
            "seed count": "15 seed-fold runs per supervised condition",
            "principal metric": "Parent seen-class coverage about 0.0133 for full_model; unseen target classes are identified separately",
            "uncertainty": "coverage reported by seed-fold run",
            "limitations": "Many test parent and successor labels were not observed in the training fold.",
            "recommended wording": "For held-out families, parent and successor heads are retained as training losses but exact unseen-class test accuracy is not a valid primary metric.",
            "keep, revise or remove": "remove",
        },
        {
            "claim": "Controlled pairwise alignment has a direct retained-corpus attraction effect when pair coverage is fixed.",
            "evidence source": "Phase 3 controlled paired-batch sweep",
            "retained-corpus or held-out": "retained-corpus",
            "ablation-supported": "yes for paired-batch retained-corpus setting",
            "seed count": "10",
            "principal metric": "Pair coverage 1.0; full_model same-ID distance falls from 0.7473 +/- 0.0157 at lambda=0.00 to 0.6843 +/- 0.0158 at lambda=0.03 and 0.4069 +/- 0.0148 at lambda=1.00",
            "uncertainty": "sample SD across seeds",
            "limitations": "Does not by itself establish held-out generalisation or philosophical equivalence.",
            "recommended wording": "With paired batches, increasing lambda directly tightens same-ID latent distances in retained-corpus evaluation.",
            "keep, revise or remove": "keep",
        },
        {
            "claim": "High lambda increases realised same-ID distance or causes the earlier high-weight deterioration.",
            "evidence source": "Phase 3 controlled paired-batch sweep",
            "retained-corpus or held-out": "retained-corpus",
            "ablation-supported": "no under controlled pair exposure",
            "seed count": "10",
            "principal metric": "full_model Top-1 rises to 0.9959 +/- 0.0032 and same-ID distance falls to 0.4069 +/- 0.0148 at lambda=1.00",
            "uncertainty": "sample SD across seeds",
            "limitations": "Comparator addresses batch composition in retained-corpus runs; other optimisation settings were not broadly varied.",
            "recommended wording": "The earlier high-weight pattern does not remain when every pair contributes once per epoch.",
            "keep, revise or remove": "remove",
        },
        {
            "claim": "The selected case studies are rigorous computational prompts for close reading.",
            "evidence source": "Phase 4 text-blind case-study selection",
            "retained-corpus or held-out": "retained-corpus with robustness comparisons",
            "ablation-supported": "partly; lambda=0.03 and no_successor robustness are reported per case",
            "seed count": "10",
            "principal metric": "Pre-text manifest frozen with SHA-256 c4c6ac3c5473f47d181fdb8f1e155eab2d938a9bd674f2435c7fb48bc29e5ffc; ten seed-level dossiers retained",
            "uncertainty": "seed-level values shown in dossiers and robustness summary",
            "limitations": "The model does not make philosophical conclusions; selected cases require human close reading.",
            "recommended wording": "The representation supplies reproducible, text-blind case-study candidates for scholarly interpretation.",
            "keep, revise or remove": "keep",
        },
    ]
    write_csv(
        BUNDLE / "publication_claims_matrix.csv",
        [
            "claim",
            "evidence source",
            "retained-corpus or held-out",
            "ablation-supported",
            "seed count",
            "principal metric",
            "uncertainty",
            "limitations",
            "recommended wording",
            "keep, revise or remove",
        ],
        rows,
    )


def write_experiment_inventory(commit_hash: str) -> None:
    rows = []
    for phase in PHASES:
        rows.append(
            {
                "phase": phase.name,
                "output directory": str(phase.directory.relative_to(ROOT)),
                "objective": phase.objective,
                "conditions": phase.conditions,
                "seeds or folds": phase.seeds,
                "key result": phase.key_result,
                "verification artifact": str((phase.directory / f"{phase.name.lower().replace(' ', '')}_verification_report.md").relative_to(ROOT))
                if False
                else "see verification_index.md",
                "source commit": commit_hash,
            }
        )
    write_csv(
        BUNDLE / "experiment_inventory.csv",
        [
            "phase",
            "output directory",
            "objective",
            "conditions",
            "seeds or folds",
            "key result",
            "verification artifact",
            "source commit",
        ],
        rows,
    )


def write_markdown_files(commit_hash: str, branch: str, copied_count: int) -> None:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    executive_summary = f"""# DSH Validation Evidence Bundle

Generated: {generated_at}

Branch: `{branch}`

Source commit: `{commit_hash}`

This bundle consolidates validation evidence from Phases 1-4 without revising the manuscript.
It copies the phase reports, summary tables, selected figures, configurations and command logs into one reviewable directory and packages the directory as `dsh_validation_bundle.zip`.

## Headline Findings

- Retained-corpus same-ID retrieval is high in the supervised structure latent, but Phase 1 shows that successor-only and shuffled shared targets can also produce strong retrieval. The manuscript should not treat retained-corpus retrieval as pure semantic or hierarchy-general evidence.
- Reconstruction alone does not explain the retained-corpus retrieval signal.
- Held-out family evaluation leaves only a small same-ID retrieval signal, with much lower absolute performance than retained-corpus evaluation.
- True hierarchy-derived targets support sibling cohesion; shuffled targets remove that family-organisation evidence.
- Controlled paired batching gives lambda a clear interpretation: every same-ID pair contributes once per epoch, and increased lambda tightens same-ID distances in retained-corpus evaluation.
- Phase 4 case studies are reproducible, text-blind prompts for close reading; they do not license machine-generated philosophical conclusions.

## Bundle Contents

- `publication_claims_matrix.csv`: claim-level evidence, limitations and recommended manuscript disposition.
- `experiment_inventory.csv`: phase objectives, conditions, seeds and key results.
- `unresolved_questions.md`: issues that should remain open after these validations.
- `recommended_manuscript_changes.md`: revision guidance for a later manuscript phase.
- `verification_index.md`: pointers to verification and leakage reports.
- `all_phase_reports/`, `all_summary_tables/`, `selected_figures/`, `configs/`, `commands/`: copied supporting artifacts.

Copied artifact count: {copied_count}
"""
    (BUNDLE / "executive_summary.md").write_text(executive_summary, encoding="utf-8")

    unresolved = """# Unresolved Questions

- How much of retained-corpus same-ID retrieval should be attributed to proposition-specific formal labels rather than transferable cross-language structure?
- Why does held-out family retrieval remain low in absolute terms, and would more seeds or alternative regularisation change that result?
- Which parent and successor evaluations can be reformulated for unseen-family testing without treating unseen labels as ordinary classification errors?
- Which Phase 4 case-study patterns survive independent model specifications beyond lambda=0.03 and no-successor robustness checks?
- How should the manuscript distinguish corpus description from generalisation to unseen proposition families?
- Can bilingual-neighbourhood divergence cases be explained by truncation, length, or lexical-baseline effects before close reading begins?
"""
    (BUNDLE / "unresolved_questions.md").write_text(unresolved, encoding="utf-8")

    recommended_changes = """# Recommended Manuscript Changes

Do not revise the manuscript in this phase. The following changes are recommended for a separate manuscript-revision phase after the empirical bundle has been reviewed.

## Retained-Corpus Claims

- Reword high same-ID retrieval claims as retained-corpus findings from the supervised structure latent.
- Add the Phase 1 ablations prominently: reconstruction-only fails, successor-only succeeds, and shuffled shared targets retain retrieval while losing sibling cohesion.
- Avoid wording that treats retained-corpus bilingual retrieval as direct semantic equivalence or as independent evidence of philosophical relation.

## Held-Out Generalisation

- Add the family-held-out evaluation as the main evidence for generalisation beyond the fitted corpus.
- State that held-out retrieval survives only weakly and is much lower than retained-corpus retrieval.
- Separate valid held-out depth and child-count metrics from invalid exact parent/successor unseen-class accuracies.

## Alignment Experiment

- Replace any interpretation based on incidental minibatch co-occurrence with the controlled paired-batch results.
- State that lambda has a clear causal interpretation only in the paired-batch retained-corpus setting.
- Remove or revise any claim that high lambda increases realised same-ID distance under controlled pair coverage.

## Case Studies

- Present Phase 4 dossiers as text-blind computational prompts for close reading.
- Include the selection protocol and manifest hash if case studies are used.
- Do not claim that latent distances establish philosophical dependence, semantic equivalence, anomaly, or mistranslation.
"""
    (BUNDLE / "recommended_manuscript_changes.md").write_text(recommended_changes, encoding="utf-8")

    verification = """# Verification Index

This index points to the phase-level verification records copied into `all_phase_reports/`.

- Phase 1: `all_phase_reports/phase1_ablations/phase1_verification_report.md`
- Phase 2: `all_phase_reports/phase2_family_holdout/phase2_verification_report.md`
- Phase 2 leakage checks: `all_phase_reports/phase2_family_holdout/phase2_leakage_checks.md`
- Phase 3: `all_phase_reports/phase3_controlled_alignment/phase3_verification_report.md`
- Phase 4: `all_phase_reports/phase4_case_studies/phase4_verification_report.md`
- Phase 4 text-blind protocol: `all_phase_reports/phase4_case_studies/phase4_case_selection_protocol.md`

## Bundle-Level Checks

- Required top-level bundle files are present.
- Required copied artifact directories are present.
- `main.tex` and `references.bib` are not written by the bundling script.
- The zip archive is built from the consolidated directory after file generation.

Reviewers should use the original phase verification reports for independent recomputation details and leakage checks.
"""
    (BUNDLE / "verification_index.md").write_text(verification, encoding="utf-8")


def write_canonical_command_manifest() -> Path:
    path = BUNDLE / "commands" / "phase3_controlled_alignment" / "phase3_canonical_commands.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "python3 tools/phase3_controlled_alignment.py run --batching paired --conditions full_model --skip-existing",
                "python3 tools/phase3_controlled_alignment.py successor-control --skip-existing",
                "python3 tools/phase3_controlled_alignment.py verify",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def copy_all_artifacts() -> list[Path]:
    copied: list[Path] = []
    groups = [
        (REPORT_FILES, BUNDLE / "all_phase_reports"),
        (SUMMARY_FILES, BUNDLE / "all_summary_tables"),
        (CONFIG_FILES, BUNDLE / "configs"),
        (COMMAND_FILES, BUNDLE / "commands"),
        (FIGURE_DIRS, BUNDLE / "selected_figures"),
    ]
    for relative_paths, target_dir in groups:
        target_dir.mkdir(parents=True, exist_ok=True)
        for relative in relative_paths:
            copied.extend(copy_artifact(relative, target_dir))
    return copied


def write_bundle_manifest(copied: list[Path]) -> None:
    rows = []
    for path in sorted(p for p in BUNDLE.rglob("*") if p.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(BUNDLE)),
                "bytes": str(path.stat().st_size),
                "sha256": sha256(path),
            }
        )
    write_csv(BUNDLE / "bundle_file_manifest.csv", ["path", "bytes", "sha256"], rows)


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUNDLE.parent))


def verify_bundle() -> None:
    required = [
        "executive_summary.md",
        "publication_claims_matrix.csv",
        "experiment_inventory.csv",
        "unresolved_questions.md",
        "recommended_manuscript_changes.md",
        "verification_index.md",
        "all_phase_reports",
        "all_summary_tables",
        "selected_figures",
        "configs",
        "commands",
    ]
    missing = [item for item in required if not (BUNDLE / item).exists()]
    if missing:
        raise SystemExit(f"missing required bundle artifacts: {missing}")
    if not ZIP_PATH.exists():
        raise SystemExit(f"zip archive was not created: {ZIP_PATH}")
    noncanonical = [path for path in BUNDLE.rglob("*") if path.is_file() and is_noncanonical_removed_artifact(path.relative_to(BUNDLE))]
    if noncanonical:
        raise SystemExit(f"canonical bundle contains removed/noncanonical artifacts: {[str(path.relative_to(BUNDLE)) for path in noncanonical[:20]]}")
    noncanonical_content = []
    for path in BUNDLE.rglob("*"):
        relative = path.relative_to(BUNDLE)
        if allows_noncanonical_marker_text(relative):
            continue
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".sh", ".txt"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(marker in text for marker in NONCANONICAL_MARKERS):
                noncanonical_content.append(path)
    if noncanonical_content:
        raise SystemExit(f"canonical bundle contains removed/noncanonical content markers: {[str(path.relative_to(BUNDLE)) for path in noncanonical_content[:20]]}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())
    for item in required:
        expected_prefix = f"{BUNDLE.name}/{item}"
        if not any(name == expected_prefix or name.startswith(expected_prefix + "/") for name in names):
            raise SystemExit(f"zip archive is missing {item}")


def main() -> None:
    branch = run_git(["branch", "--show-current"])
    if branch != "main":
        raise SystemExit(f"refusing to create bundle on branch {branch!r}")

    commit_hash = run_git(["rev-parse", "HEAD"])
    ensure_clean_bundle_root()
    copied = copy_all_artifacts()
    copied.append(write_canonical_command_manifest())
    write_publication_claims_matrix()
    write_experiment_inventory(commit_hash)
    write_markdown_files(commit_hash, branch, len(copied))
    write_bundle_manifest(copied)
    make_zip()
    verify_bundle()
    print(json.dumps({"bundle": str(BUNDLE), "zip": str(ZIP_PATH), "copied_files": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
