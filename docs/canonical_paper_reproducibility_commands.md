# Codex Commands: Canonical Paper Reproducibility

Run these commands from the repository root to regenerate or verify the current
final-paper output layer from retained artefacts. This pipeline does not retrain
models and must not modify checkpoints, cached latents, raw parquet files, or
retained `seed*.metrics.json` files.

## Complete Paper-Output Pipeline

First review the command list:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
```

After review, run the wrapper:

```bash
python3 tools/reproduce_final_paper_outputs.py
```

If bundle generation is unavailable or known to fail locally, skip only that
step:

```bash
python3 tools/reproduce_final_paper_outputs.py --skip-bundle
```

The wrapper prints each command before running it.

## Expected Outputs

Paper figures:

```text
paper/figures/family_case_distance_matrix.pdf
paper/figures/family_case_distance_matrix.png
paper/figures/family_case_distance_matrix_data.csv
paper/figures/figure_manifest.csv
```

Paper tables:

```text
paper/tables/table1_retained_ablations.csv
paper/tables/table2_lexical_references.csv
paper/tables/table3_family_holdout.csv
paper/tables/table4_paired_alignment.csv
paper/tables/table5_case_study_prompts.csv
paper/tables/table_manifest.csv
```

Canonical reports and manifests:

```text
results/dsh_validation/canonical_reports/
results/dsh_validation/CANONICAL_VERIFICATION_REPORT.md
results/dsh_validation/canonical_verification.json
paper/final_paper_manifest.csv
```

## Validation Checks

```bash
python3 tools/validate_paper_figure_manifest.py
python3 -m unittest tests/test_paper_tables.py
python3 tools/validate_final_paper_manifest.py
python3 tools/verify_canonical_evidence.py
```
