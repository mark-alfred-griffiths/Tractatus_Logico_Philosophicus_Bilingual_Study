# Reproduction

Run commands from the repository root.

## Default Final-Paper Path

The current manuscript target is `paper/Tractatus_final.docx`. The default
reproduction path is non-training and works from retained evidence under
`results/dsh_validation/`.

Start with a dry run:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
```

Review the printed command list. Then run:

```bash
python3 tools/reproduce_final_paper_outputs.py
```

The wrapper prints each command before running it. It verifies or regenerates:

```text
results/dsh_validation/canonical_reports/
paper/tables/
paper/figures/figure_manifest.csv
paper/final_paper_manifest.csv
```

It does not retrain models, overwrite checkpoints, rewrite raw parquet files,
rewrite latents, or replace retained per-seed metrics.

The generated validation bundle and heavy retained empirical artifacts are
external release/archive materials, not tracked Git files. To recreate a local
bundle copy after restoring any needed external artifacts, run:

```bash
python3 tools/reproduce_final_paper_outputs.py --include-bundle
```

The resulting `results/dsh_validation/dsh_validation_bundle/` and
`results/dsh_validation/dsh_validation_bundle.zip` paths are ignored by Git.

## Narrow Checks

Use these checks when editing paper-facing manifests, figures, tables, or the
safe wrapper:

```bash
python3 tools/validate_final_paper_manifest.py
python3 tools/validate_paper_figure_manifest.py
python3 -m unittest tests/test_paper_tables.py
```

For the canonical evidence layer itself:

```bash
python3 tools/verify_canonical_evidence.py
```

## Evidence Inputs

The current canonical evidence layer is:

```text
results/dsh_validation/phase1_ablations/
results/dsh_validation/phase2_family_holdout/
results/dsh_validation/phase3_controlled_alignment/
results/dsh_validation/phase4_case_studies/
results/dsh_validation/canonical_reports/
results/dsh_validation/CANONICAL_VERIFICATION_REPORT.md
results/dsh_validation/canonical_verification.json
```

Heavy retained artifacts excluded from Git history are listed in:

```text
docs/heavy_artifacts_manifest.csv
```

Restore them from the external archive only when exact checkpoints, raw parquet
outputs, smoke outputs, or bundle copies are needed.

The retained datasets are:

```text
tractatus_structure_latents/data/tractatus.json
tractatus_structure_latents/data/tractatus_bilingual.json
```

## Deliberate Empirical Reruns

The commands below rerun empirical phases and can write checkpoints, logs, raw
outputs, per-seed metrics, reports, and figures. They are deliberate training
or phase-execution commands, not the default final-paper reproduction path.

Phase 1 retained-corpus ablations:

```bash
python3 tools/phase1_ablations.py run --skip-existing
```

Phase 2 immediate-parent-family holdout:

```bash
python3 tools/phase2_family_holdout.py run --skip-existing
```

Phase 3 controlled paired-batch bilingual alignment:

```bash
python3 tools/phase3_controlled_alignment.py run --batching paired --conditions full_model --skip-existing
python3 tools/phase3_controlled_alignment.py successor-control --skip-existing
```

Phase 4 frozen text-blind case selection:

```bash
python3 tools/phase4_case_studies.py run
```

After deliberate reruns, rebuild and verify paper-facing reports:

```bash
python3 tools/build_canonical_reports.py
python3 tools/verify_canonical_evidence.py
python3 tools/export_paper_tables.py
python3 tools/build_final_paper_manifest.py
```

## Dataset Rebuilds

Dataset rebuilds are separate from the default paper wrapper:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus.json \
  --languages en
```

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus_bilingual.json \
  --languages en,de
```
