# Canonical reproducibility completion check

## Verdict

The documented canonical workflow is complete for the current paper-output layer.
The deleted stage-numbered report folders are not required by the documented
commands.

## Commands Run Successfully

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep runs/seed_sweeps/monolingual_split_24_8_reg005 --out runs/seed_sweeps/monolingual_split_24_8_reg005/summaries/regenerated_comparison
PYTHONDONTWRITEBYTECODE=1 python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 -m tractatus_structure_latents.evaluation.generate_paper_figures --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 --representative-alignment align003 --representative-seed 0 --out-dir paper/figures --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json --family-distance-data paper/figures/family_case_distance_matrix_data.csv
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 tools/reproduce_paper_metrics_and_figures.py --metrics --figures --skip-checkpoints --out-dir reports/paper_reproducibility/reproduced
PYTHONDONTWRITEBYTECODE=1 python3 tools/write_paper_results_summaries.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_empirical_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/empirical_audit.py
```

## Output Completeness

- Paper-value checks: 101.
- Paper-value mismatches: 0.
- TF-IDF checks: 4.
- Jaccard checks: 16.
- Euclidean checks: 72.
- Paper figure files expected: 17.
- Missing or empty paper figure files: 0.
- Figure manifest rows: 7.
- Figure manifest failures: 0.
- Empirical-audit required output files: 10.
- Missing or empty empirical-audit output files: 0.
- Empirical-audit tests: 5 passed.

## Canonical Inputs

- Retained run metrics under `runs/seed_sweeps/**/metrics/seed*.metrics.json`.
- Retained run latents/checkpoints where the empirical audit recomputes posterior
  variance and child-count diagnostics.
- Canonical Figure 7 source data at
  `paper/figures/family_case_distance_matrix_data.csv`.
- Canonical reproduced statistics under
  `reports/paper_reproducibility/reproduced/`.
- Canonical empirical audit outputs under `reports/empirical_audit/`.

## Documentation Check

- Active documentation files checked: 9.
- Files with old stage-numbered references: 0.
- Files with bytecode-unsafe paper report commands: 0.
- Source-artifact labels with old stage references in
  `reproduced_values_vs_paper.csv`: 0.

## Important Boundary

The workflow reproduces the retained-corpus paper-output layer. It does not
retrain published models and does not require the deleted stage-numbered report
directories.
