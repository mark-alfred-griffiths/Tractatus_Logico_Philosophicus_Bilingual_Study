# Canonical paper pipeline test report

## Commands tested

The documented canonical paper-output and empirical-audit commands were run successfully.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep runs/seed_sweeps/monolingual_split_24_8_reg005 --out runs/seed_sweeps/monolingual_split_24_8_reg005/summaries/regenerated_comparison
PYTHONDONTWRITEBYTECODE=1 python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 -m tractatus_structure_latents.evaluation.generate_paper_figures --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 --representative-alignment align003 --representative-seed 0 --out-dir paper/figures --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json --family-distance-data paper/figures/family_case_distance_matrix_data.csv
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 tools/reproduce_paper_metrics_and_figures.py --metrics --figures --skip-checkpoints --out-dir reports/paper_reproducibility/reproduced
PYTHONDONTWRITEBYTECODE=1 python3 tools/write_paper_results_summaries.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_empirical_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/empirical_audit.py
```

## Results

- Reproduced paper-value checks: 101.
- Reproduced paper-value mismatches: 0.
- TF-IDF checks: 4.
- Jaccard checks: 16.
- Euclidean checks: 72.
- Source-artifact labels with stage references: 0.
- Figure manifest rows: 7.
- Figure manifest failures: 0.
- Documentation files checked for canonical wording: 9.
- Documentation files with old stage-numbered references: 0.
- Documentation files with bytecode-unsafe paper report commands: 0.
- Empirical-audit tests: 5 passed.
- Empirical-audit report: `reports/empirical_audit/report_for_chatgpt.md`.

## Supporting files

- `reports/paper_reproducibility/figure_validation.csv`
- `reports/paper_reproducibility/documentation_canonical_wording_check.csv`
- `reports/paper_reproducibility/pipeline_test_manifest.json`
