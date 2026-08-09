# Phase 3 Verification Report

- All tabular/report deliverables are present.
- Observed the expected 100 canonical paired condition/lambda/seed final metric rows.
- All required final metric columns are present.
- Summary means and sample SDs recompute exactly from seed metrics.
- Paired runs have 100% pair coverage for every seed/lambda/condition.
- Epoch trajectory parquet reconstructs from raw per-run CSV files.
- All required figures are present.
- No tracked diffs touch manuscript files or canonical run outputs.
- Git branch: hold-out-and-ablation.
- Git commit during verification: 6c274141e3da275777c689c2073319da26739b24.

## Git status

```
M README.md
 M results/dsh_validation/phase3_controlled_alignment/figures/alignment_loss_trajectories.png
 M results/dsh_validation/phase3_controlled_alignment/figures/formal_depth_accuracy_across_lambda.png
 M results/dsh_validation/phase3_controlled_alignment/figures/formal_parent_accuracy_across_lambda.png
 M results/dsh_validation/phase3_controlled_alignment/figures/formal_successor_accuracy_across_lambda.png
 M results/dsh_validation/phase3_controlled_alignment/figures/posterior_variance_across_lambda.png
 M results/dsh_validation/phase3_controlled_alignment/figures/retrieval_across_lambda.png
 M results/dsh_validation/phase3_controlled_alignment/figures/same_id_distance_across_lambda.png
 M results/dsh_validation/phase3_controlled_alignment/figures/structure_mean_norm_across_lambda.png
 M results/dsh_validation/phase3_controlled_alignment/phase3_alignment_report.md
 M results/dsh_validation/phase3_controlled_alignment/phase3_epoch_trajectories.parquet
 M results/dsh_validation/phase3_controlled_alignment/phase3_pair_coverage.csv
 M results/dsh_validation/phase3_controlled_alignment/phase3_seed_results.csv
 M results/dsh_validation/phase3_controlled_alignment/phase3_summary.csv
 M tools/create_dsh_validation_bundle.py
 M tools/phase3_controlled_alignment.py
 M tools/reproduce_paper_metrics_and_figures.py
 M tools/write_paper_results_summaries.py
 M tractatus_structure_latents/evaluation/__pycache__/analyse_seed_sweep.cpython-310.pyc
 M tractatus_structure_latents/evaluation/__pycache__/generate_paper_figures.cpython-310.pyc
 M tractatus_structure_latents/evaluation/__pycache__/plot_bilingual_alignment_sweep.cpython-310.pyc
 M tractatus_structure_latents/evaluation/__pycache__/visualise_latents.cpython-310.pyc
 M tractatus_structure_latents/evaluation/generate_paper_figures.py
 M tractatus_structure_latents/evaluation/plot_bilingual_alignment_sweep.py
 M tractatus_structure_latents/scripts/__pycache__/run_bilingual_alignment_seed_sweep.cpython-310.pyc
 M tractatus_structure_latents/scripts/run_bilingual_alignment_seed_sweep.py
 M tractatus_structure_latents/training/__pycache__/data.cpython-310.pyc
 M tractatus_structure_latents/training/__pycache__/train_vae.cpython-310.pyc
?? .idea/
?? EMPIRICAL_PIPELINE_AUDIT.md
?? legacy/
?? results/dsh_validation/CANONICAL_VERIFICATION_REPORT.md
?? results/dsh_validation/canonical_reports/
?? results/dsh_validation/canonical_verification.json
?? results/dsh_validation/phase1_ablations/checkpoints/
?? results/dsh_validation/phase1_ablations/smoke/
?? results/dsh_validation/phase2_family_holdout/checkpoints/
?? results/dsh_validation/phase2_family_holdout/smoke/
?? results/dsh_validation/phase3_controlled_alignment/checkpoints/
?? results/dsh_validation/phase3_controlled_alignment/raw/
?? results/dsh_validation/phase3_controlled_alignment/smoke/
?? tests/__pycache__/
?? tests/test_canonical_pipeline.py
?? tools/__pycache__/
?? tools/build_canonical_reports.py
?? tools/canonical_experiments.py
?? tools/export_canonical_evidence.py
?? tools/verify_canonical_evidence.py
```

## Git diff stat

```
README.md                                          | 183 +++-------------
 .../figures/alignment_loss_trajectories.png        | Bin 212634 -> 176891 bytes
 .../formal_depth_accuracy_across_lambda.png        | Bin 76504 -> 70661 bytes
 .../formal_parent_accuracy_across_lambda.png       | Bin 74295 -> 68518 bytes
 .../formal_successor_accuracy_across_lambda.png    | Bin 52978 -> 46628 bytes
 .../figures/posterior_variance_across_lambda.png   | Bin 63324 -> 67472 bytes
 .../figures/retrieval_across_lambda.png            | Bin 58988 -> 53622 bytes
 .../figures/same_id_distance_across_lambda.png     | Bin 74791 -> 66456 bytes
 .../figures/structure_mean_norm_across_lambda.png  | Bin 62063 -> 55871 bytes
 .../phase3_alignment_report.md                     |  22 +-
 .../phase3_epoch_trajectories.parquet              | Bin 1202428 -> 986082 bytes
 .../phase3_pair_coverage.csv                       |  20 --
 .../phase3_seed_results.csv                        | 222 +++++++++----------
 .../phase3_controlled_alignment/phase3_summary.csv | 194 -----------------
 tools/create_dsh_validation_bundle.py              |  46 +++-
 tools/phase3_controlled_alignment.py               | 238 +++++++++++++++------
 tools/reproduce_paper_metrics_and_figures.py       |  13 +-
 tools/write_paper_results_summaries.py             |  19 +-
 .../__pycache__/analyse_seed_sweep.cpython-310.pyc | Bin 10025 -> 10063 bytes
 .../generate_paper_figures.cpython-310.pyc         | Bin 4573 -> 8148 bytes
 .../plot_bilingual_alignment_sweep.cpython-310.pyc | Bin 9530 -> 10952 bytes
 .../__pycache__/visualise_latents.cpython-310.pyc  | Bin 6829 -> 6881 bytes
 .../evaluation/generate_paper_figures.py           |  13 +-
 .../evaluation/plot_bilingual_alignment_sweep.py   |  22 +-
 ..._bilingual_alignment_seed_sweep.cpython-310.pyc | Bin 6384 -> 7491 bytes
 .../scripts/run_bilingual_alignment_seed_sweep.py  |  39 +++-
 .../training/__pycache__/data.cpython-310.pyc      | Bin 7870 -> 9921 bytes
 .../training/__pycache__/train_vae.cpython-310.pyc | Bin 11040 -> 17879 bytes
 28 files changed, 438 insertions(+), 593 deletions(-)
```
