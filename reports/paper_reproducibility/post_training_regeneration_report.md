# Post-training regeneration report

## Verdict

PASS: all documented post-training regeneration commands completed successfully in the final rerun.

## Scope

This was post-training regeneration only. No model training, fine-tuning, or optimisation loop was run.

## Checks

- Monolingual manifest seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
- Required outputs missing or empty: 0.
- Paper-value checks: 101.
- Paper-value mismatches: 0.
- TF-IDF checks: 4.
- Jaccard checks: 16.
- Euclidean checks: 72.
- Source-artifact labels with stage references: 0.
- Figure manifest rows: 7.
- Figure manifest failures: 0.
- Required paper figure files: 17.
- Missing or empty paper figure files: 0.
- Documentation files checked: 9.
- Documentation files with old stage references: 0.
- Documentation files with bytecode-unsafe report commands: 0.
- Empirical-audit branch recorded: `empirical-audit`.

## Outputs

- `reports/paper_reproducibility/reproduced/reproduced_values_vs_paper.csv`
- `reports/paper_reproducibility/reproduced/reproduced_figure_manifest.csv`
- `reports/paper_reproducibility/figure_validation.csv`
- `reports/paper_reproducibility/documentation_canonical_wording_check.csv`
- `reports/empirical_audit/report_for_chatgpt.md`
