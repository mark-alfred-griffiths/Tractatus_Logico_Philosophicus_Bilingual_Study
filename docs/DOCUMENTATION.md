# Documentation Index

- [Dataset](dataset.md)
- [Model](model.md)
- [Experiments](experiments.md)
- [Results](results.md)
- [Reproduction](reproduction.md)
- [Canonical paper reproducibility commands](canonical_paper_reproducibility_commands.md)

## Current Paper State

The repository documentation is current for the canonical paper source. The
paper has an audit/reproduction entry point for all reported TF-IDF, Jaccard,
and Euclidean-distance values and for the paper figures, including Figure 7.
For the complete current pipeline,
run the commands in [Reproduction](reproduction.md), section
`Complete Canonical Paper-Output Pipeline`.

The short report-only reproduction command is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/reproduce_paper_metrics_and_figures.py \
  --metrics --figures --skip-checkpoints \
  --out-dir reports/paper_reproducibility/reproduced
```

This workflow uses retained artefacts and does not retrain models, overwrite
`runs/`, or modify canonical manuscript files.

The complete pipeline additionally regenerates summary artefacts under
`runs/seed_sweeps/*/summaries/`, all `paper/figures/` outputs, and the two
paper-facing summaries via:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/write_paper_results_summaries.py
```

For the deeper retained-experiment audit used to verify metric definitions,
same-ID distances, lambda comparisons, latent scale/variance, and child-count
diagnostics, use:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/empirical_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_empirical_audit.py
```

That audit writes to `reports/empirical_audit/` and is separate from the routine
paper-output reproduction command.
