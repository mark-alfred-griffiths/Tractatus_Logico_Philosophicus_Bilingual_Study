# Empirical Audit Inventory

Commit: `633da4d77d374ef1bb404019497c1787ebe0d9c6`
Branch: `empirical-audit`
Dirty state:
- `M README.md`
- ` M docs/DOCUMENTATION.md`
- ` M docs/experiments.md`
- ` M docs/reproduction.md`
- ` M docs/results.md`
- ` M paper/PAPER.md`
- ` D paper/archive/tractatus_bilingual_latent_structure_paper.txt`
- ` D paper/archive/tractatus_latent_logic_paper.txt`
- ` M paper/bilingual_results_summary.txt`
- `D  paper/bilingual_structure_latents_in_the_tractatus.pdf`
- ` M paper/figures/bilingual_alignment_retrieval_sweep.pdf`
- ` M paper/figures/bilingual_latent_pca_depth_align003_seed000.pdf`
- ` M paper/figures/bilingual_latent_pca_language_align003_seed000.pdf`
- ` M paper/figures/bilingual_reconstruction_sweep.pdf`
- ` M paper/figures/bilingual_retrieval_structure_tradeoff.pdf`
- ` M paper/figures/bilingual_structure_accuracy_sweep.pdf`
- ` M paper/figures/monolingual_latent_pca_depth_reg005_seed000.pdf`
- ` M paper/main.tex`
- ` M paper/monolingual_results_summary.txt`
- ` M paper/references.bib`
- ` M runs/RUN_ARTIFACTS.md`
- ` M runs/seed_sweeps/SEED_SWEEPS.md`
- ` M runs/seed_sweeps/monolingual_split_24_8_reg005/manifest.json`
- ` M tractatus_structure_latents/evaluation/generate_paper_figures.py`
- `?? .idea/`
- `?? docs/canonical_paper_reproducibility_commands.md`
- `?? paper/bilingual_structure_latents_in_the_tractatus.pdf`
- `?? paper/figures/family_case_distance_matrix.pdf`
- `?? paper/figures/family_case_distance_matrix.png`
- `?? paper/figures/family_case_distance_matrix_data.csv`
- `?? reports/`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/per_seed_metrics.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/seed_metric_trends.png`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/summary.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/summary.json`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/summary.md`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/per_seed_metrics.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/seed_metric_trends.png`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/summary.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/summary.json`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/summary.md`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/per_seed_metrics.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/seed_metric_trends.png`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/summary.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/summary.json`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/summary.md`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/per_seed_metrics.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/seed_metric_trends.png`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/summary.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/summary.json`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/summary.md`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/per_seed_metrics.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/seed_metric_trends.png`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/summary.csv`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/summary.json`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/summary.md`
- `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison/`
- `?? tests/`
- `?? tools/`

## Canonical Artifacts
| artifact | path | status |
| --- | --- | --- |
| package | tractatus_structure_latents | found |
| paper/main.tex | paper/main.tex | found |
| paper bilingual summary | paper/bilingual_results_summary.txt | found |
| bilingual summary JSON | runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json | found |
| bilingual sweep root | runs/seed_sweeps/bilingual_alignment_lambda_sweep | found |
| monolingual sweep root | runs/seed_sweeps/monolingual_split_24_8_reg005 | found |
| align000 per-seed files | runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 | metrics=10, checkpoints=10, latents=10 |
| align003 per-seed files | runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 | metrics=10, checkpoints=10, latents=10 |
| align010 per-seed files | runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 | metrics=10, checkpoints=10, latents=10 |
| align030 per-seed files | runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 | metrics=10, checkpoints=10, latents=10 |
| align100 per-seed files | runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 | metrics=10, checkpoints=10, latents=10 |
| monolingual per-seed files | runs/seed_sweeps/monolingual_split_24_8_reg005 | metrics=10, checkpoints=10, latents=10 |

## Implementation Evidence
- Training language alignment loss: `tractatus_structure_latents/training/train_vae.py:119`. It builds within-batch German-English pairs sharing the same proposition index and returns `F.mse_loss` over `structure_mu` at `tractatus_structure_latents/training/train_vae.py:140`; the weighted term is added at `tractatus_structure_latents/training/train_vae.py:271`.
- Same-ID and retrieval metrics: `tractatus_structure_latents/evaluation/evaluate_structure.py:50`. They operate on exported evaluation latents; export uses `outputs['structure_mu']` when `--latent-part structure` is selected at `tractatus_structure_latents/evaluation/evaluate_structure.py:212`.
- Child-count target: `tractatus_structure_latents/training/data.py:124`. Head and loss: `tractatus_structure_latents/models/vae.py:106`, `tractatus_structure_latents/models/vae.py:200`.
- Lambda label construction: `tractatus_structure_latents/scripts/run_bilingual_alignment_seed_sweep.py:9`; default sweep values are at `tractatus_structure_latents/scripts/run_bilingual_alignment_seed_sweep.py:89`.

## Metric Keys Found
| file | metric_keys |
| --- | --- |
| runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/metrics/seed000.metrics.json | cross_language_mrr, cross_language_mrr_de_to_en, cross_language_mrr_en_to_de, cross_language_top1_id_accuracy, cross_language_top1_id_accuracy_de_to_en, cross_language_top1_id_accuracy_en_to_de, depth_accuracy, depth_accuracy_by_language.de, depth_accuracy_by_language.en, kl, kl_structure, kl_text, loss, mean_cross_language_parent_child_distance, mean_parent_child_distance, mean_same_id_cross_language_distance, mean_sibling_distance, mean_unrelated_distance, next_accuracy, next_accuracy_by_language.de, next_accuracy_by_language.en, parent_accuracy, parent_accuracy_by_language.de, parent_accuracy_by_language.en, perplexity, reconstruction_loss, reconstruction_loss_by_language.de, reconstruction_loss_by_language.en |
| runs/seed_sweeps/monolingual_split_24_8_reg005/metrics/seed000.metrics.json | depth_accuracy, depth_accuracy_by_language.en, kl, kl_structure, kl_text, loss, mean_parent_child_distance, mean_sibling_distance, mean_unrelated_distance, next_accuracy, next_accuracy_by_language.en, parent_accuracy, parent_accuracy_by_language.en, perplexity, reconstruction_loss, reconstruction_loss_by_language.en |

## Recommended Audit Implementation Plan
1. Recompute published cross-language and relation-distance metrics from cached `*_structure.pt` latents, because those files are the exported posterior means used by the canonical evaluator.
2. Load checkpoints in `eval()` mode only for quantities not saved in metrics: posterior variance/std and child-count predictions.
3. Compare recomputed metrics with every per-seed JSON and report maximum absolute differences.
4. Use seed-matched align000/align003 metrics for exploratory paired summaries with bootstrap CIs and exact sign-flip p-values.

## Unresolved Questions From Inventory
- No existing saved child-count predictions or metrics were found in per-seed metric JSON files.
- The evaluator does not save posterior log-variance, so posterior variance must be recomputed from checkpoints.
- Principal metrics are retained-corpus evaluations unless another held-out split artifact is identified.
