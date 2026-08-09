# Documentation Index

This index is organised for readers of the final manuscript.

## Final Paper Route

1. Start with [README](../README.md) for the repository overview.
2. Read [Paper notes](../paper/PAPER.md) for manuscript-facing files.
3. Use [Reproduction](reproduction.md) for the safe non-training wrapper and
   deliberate retraining commands.
4. Use [Results](results.md) for the Phase 1-4 evidence map.
5. Use [Repository layout for paper](repository_layout_for_paper.md) for status
   labels, retained evidence boundaries, and migration policy.

## Current Manifests And Evidence

- [Final paper manifest](../paper/final_paper_manifest.csv)
- [Figure manifest](../paper/figures/figure_manifest.csv)
- [Table manifest](../paper/tables/table_manifest.csv)
- [Canonical verification report](../results/dsh_validation/CANONICAL_VERIFICATION_REPORT.md)
- [Canonical report index](../results/dsh_validation/canonical_reports/canonical_report_index.md)

The Phase 1-4 outputs under `results/dsh_validation/` are the canonical
evidence layer for the final paper.

## Method And Data Background

- [Dataset](dataset.md)
- [Model](model.md)
- [Experiments](experiments.md)
- [Run artifacts](../runs/RUN_ARTIFACTS.md)

## Command Reference

- [Canonical paper reproducibility commands](canonical_paper_reproducibility_commands.md)

The current safe entry point remains:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
```
