# Results

The current results layer for the final paper is `results/dsh_validation/`. It
contains retained Phase 1-4 empirical outputs, canonical reports, verification
files, and the validation bundle.

## Canonical Evidence

```text
results/dsh_validation/phase1_ablations/
results/dsh_validation/phase2_family_holdout/
results/dsh_validation/phase3_controlled_alignment/
results/dsh_validation/phase4_case_studies/
```

Phase roles:

- Phase 1: retained-corpus ablations.
- Phase 2: immediate-parent-family holdout.
- Phase 3: controlled paired-batch bilingual alignment.
- Phase 4: frozen text-blind case selection.

The main verification files are:

```text
results/dsh_validation/CANONICAL_VERIFICATION_REPORT.md
results/dsh_validation/canonical_verification.json
```

Compact paper-facing reports are in:

```text
results/dsh_validation/canonical_reports/
results/dsh_validation/canonical_reports/canonical_report_index.md
```

The validation bundle is:

```text
docs/heavy_artifacts_manifest.csv
```

The validation bundle zip, raw parquet outputs, smoke outputs, and checkpoints
are external archive artifacts rather than tracked Git files. See
[heavy_artifacts.md](heavy_artifacts.md).

## Paper-Facing Derivatives

Final manuscript figure and table manifests live under `paper/`:

```text
paper/figures/figure_manifest.csv
paper/tables/table_manifest.csv
paper/final_paper_manifest.csv
```

The final figure currently marked as used by the manuscript is:

```text
paper/figures/family_case_distance_matrix.png
paper/figures/family_case_distance_matrix.pdf
paper/figures/family_case_distance_matrix_data.csv
```

Publication-quality validation figures are generated under:

```text
paper/figures/validation/
```

Expected validation artifacts are:

```text
paper/figures/validation/Figure_1_Retained_Ablation_Diagnostics_publication.pdf
paper/figures/validation/Figure_1_Retained_Ablation_Diagnostics_publication.png
paper/figures/validation/Figure_1_Retained_Ablation_Diagnostics_publication.tiff
paper/figures/validation/Figure_2_Retained_vs_Holdout_Retrieval_publication.pdf
paper/figures/validation/Figure_2_Retained_vs_Holdout_Retrieval_publication.png
paper/figures/validation/Figure_2_Retained_vs_Holdout_Retrieval_publication.tiff
paper/figures/validation/figure_1_plot_data.csv
paper/figures/validation/figure_2_plot_data.csv
```

These validation artifacts are listed in `paper/final_paper_manifest.csv` with
hashes and source provenance. The manuscript-use `figure_manifest.csv` remains
restricted to figures used directly in the manuscript.

Table-ready exports are:

```text
paper/tables/table1_retained_ablations.csv
paper/tables/table2_lexical_references.csv
paper/tables/table3_family_holdout.csv
paper/tables/table4_paired_alignment.csv
paper/tables/table5_case_study_prompts.csv
```

Regenerate or verify these derivatives with:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
```

## Interpretation Guardrail

The empirical evidence concerns proposition-number-derived structure in this
small, paired, highly structured Tractatus corpus. It should not be read as
proof of a general language-invariant logical manifold.
