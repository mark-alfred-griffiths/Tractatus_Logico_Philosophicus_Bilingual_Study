# Empirical Audit Report for ChatGPT

## 1. Executive Findings
- The training alignment loss, reported same-ID diagnostic, and cross-language retrieval all use posterior means of the structure latent, not sampled latents, for the relevant published evaluation path.
- The same-ID diagnostic is directional over both German-to-English and English-to-German sources and is a mean Euclidean distance, while the training alignment loss is mean squared error over within-batch pairs.
- Recomputed cached-latent metrics match the per-seed JSON metrics to numerical precision; the maximum absolute discrepancy is shown below.
- Child-count is implemented as a regression auxiliary head trained with MSE. No saved child-count metrics or decoding rule were present, so it should be described as an auxiliary objective unless the newly computed regression diagnostics are explicitly reported as audit results.
- The lambda 0.03 versus 0.00 comparison is seed-matched and exploratory. It supports descriptive wording such as numerically higher for selected retrieval metrics, not formal model-selection significance.

## 2. Repository Version, Environment, and Canonical Artifacts
- Commit: `633da4d77d374ef1bb404019497c1787ebe0d9c6`
- Branch: `empirical-audit`
- Dirty state: `M README.md`, ` M docs/DOCUMENTATION.md`, ` M docs/experiments.md`, ` M docs/reproduction.md`, ` M docs/results.md`, ` M paper/PAPER.md`, ` D paper/archive/tractatus_bilingual_latent_structure_paper.txt`, ` D paper/archive/tractatus_latent_logic_paper.txt`, ` M paper/bilingual_results_summary.txt`, `D  paper/bilingual_structure_latents_in_the_tractatus.pdf`, ` M paper/figures/bilingual_alignment_retrieval_sweep.pdf`, ` M paper/figures/bilingual_latent_pca_depth_align003_seed000.pdf`, ` M paper/figures/bilingual_latent_pca_language_align003_seed000.pdf`, ` M paper/figures/bilingual_reconstruction_sweep.pdf`, ` M paper/figures/bilingual_retrieval_structure_tradeoff.pdf`, ` M paper/figures/bilingual_structure_accuracy_sweep.pdf`, ` M paper/figures/monolingual_latent_pca_depth_reg005_seed000.pdf`, ` M paper/main.tex`, ` M paper/monolingual_results_summary.txt`, ` M paper/references.bib`, ` M runs/RUN_ARTIFACTS.md`, ` M runs/seed_sweeps/SEED_SWEEPS.md`, ` M runs/seed_sweeps/monolingual_split_24_8_reg005/manifest.json`, ` M tractatus_structure_latents/evaluation/generate_paper_figures.py`, `?? .idea/`, `?? docs/canonical_paper_reproducibility_commands.md`, `?? paper/bilingual_structure_latents_in_the_tractatus.pdf`, `?? paper/figures/family_case_distance_matrix.pdf`, `?? paper/figures/family_case_distance_matrix.png`, `?? paper/figures/family_case_distance_matrix_data.csv`, `?? reports/`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/per_seed_metrics.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/seed_metric_trends.png`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/summary.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/summary.json`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/summaries/summary.md`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/per_seed_metrics.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/seed_metric_trends.png`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/summary.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/summary.json`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/summaries/summary.md`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/per_seed_metrics.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/seed_metric_trends.png`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/summary.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/summary.json`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/summary.md`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/per_seed_metrics.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/seed_metric_trends.png`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/summary.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/summary.json`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/summaries/summary.md`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/per_seed_metrics.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/seed_metric_trends.png`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/summary.csv`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/summary.json`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/summaries/summary.md`, `?? runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison/`, `?? tests/`, `?? tools/`
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

## 3. Exact Alignment-Loss Definition
`language_alignment_loss` is implemented at `tractatus_structure_latents/training/train_vae.py:119` and is added to the training objective at `tractatus_structure_latents/training/train_vae.py:271`. It receives `outputs['structure_mu']`, groups batch items by proposition index, forms pairs only when language ids differ, and returns `F.mse_loss(z[left], z[right])` at `tractatus_structure_latents/training/train_vae.py:140`. This is a mean squared error over all paired elements and latent dimensions in the current shuffled training batch. Training randomness comes from the training seed and shuffled `DataLoader` seeded at `tractatus_structure_latents/training/train_vae.py:25`.

## 4. Exact Same-ID-Distance Definition
Same-ID distance is implemented inside `_cross_language_metrics` at `tractatus_structure_latents/evaluation/evaluate_structure.py:50`. Evaluation calls `model.eval()` and exports `outputs['structure_mu']` for `--latent-part structure` at `tractatus_structure_latents/evaluation/evaluate_structure.py:212`; the split model returns `mu` in eval mode at `tractatus_structure_latents/models/vae.py:111` and splits structure means at `tractatus_structure_latents/models/vae.py:127`. The metric appends `torch.dist(z[source_i], z[target_i]).item()` at `tractatus_structure_latents/evaluation/evaluate_structure.py:74` for each source sample and each other language with the same proposition id, then averages directional distances.

## 5. Exact Retrieval Definition
Retrieval is in the same `_cross_language_metrics` function at `tractatus_structure_latents/evaluation/evaluate_structure.py:50`. For each source item and target language, candidates are all items in the target language. Distances are Euclidean norms at `tractatus_structure_latents/evaluation/evaluate_structure.py:82`, sorted ascending. Top-1 is 1 when the nearest candidate id matches the source id, and MRR is `1 / rank` of the same id at `tractatus_structure_latents/evaluation/evaluate_structure.py:85`.

## 6. Definition Comparison
| quantity | latent | formula | pairing | mode | random |
| --- | --- | --- | --- | --- | --- |
| training alignment loss | structure_mu | F.mse_loss over paired tensors | same dataset index, different language, within batch | train | batch shuffle and model training |
| same-ID diagnostic | exported structure_mu | mean torch.dist Euclidean | same proposition id, other language, directional | eval | none in evaluation |
| Top-1/MRR retrieval | exported structure_mu | Euclidean nearest-neighbour rank | source item to all target-language candidates | eval | none in evaluation |

## 7. Recomputed Same-ID and Latent Scale/Variance
Maximum absolute difference between recomputed cached-latent metrics and per-seed JSON values: `0`.

Mean same-ID posterior-mean Euclidean distance by condition:
| condition | lambda | n | mean | sd | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| align000 | 0 | 10 | 0.7598 | 0.01955 | 0.7203 | 0.787 |
| align003 | 0.03 | 10 | 0.7486 | 0.01729 | 0.7128 | 0.7714 |
| align010 | 0.1 | 10 | 0.7756 | 0.01396 | 0.7435 | 0.7935 |
| align030 | 0.3 | 10 | 0.9502 | 0.03976 | 0.8821 | 0.9993 |
| align100 | 1 | 10 | 1.197 | 0.03978 | 1.142 | 1.275 |

Mean structure-mu norm by condition:
| condition | lambda | n | mean | sd | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| align000 | 0 | 10 | 4.202 | 0.02218 | 4.173 | 4.245 |
| align003 | 0.03 | 10 | 4.199 | 0.0243 | 4.168 | 4.242 |
| align010 | 0.1 | 10 | 4.175 | 0.03081 | 4.13 | 4.222 |
| align030 | 0.3 | 10 | 4.061 | 0.03658 | 4.004 | 4.106 |
| align100 | 1 | 10 | 3.528 | 0.1083 | 3.348 | 3.679 |

Mean posterior variance by condition:
| condition | lambda | n | mean | sd | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| align000 | 0 | 10 | 0.4298 | 0.01152 | 0.4025 | 0.4388 |
| align003 | 0.03 | 10 | 0.4295 | 0.01072 | 0.4035 | 0.4385 |
| align010 | 0.1 | 10 | 0.4317 | 0.01109 | 0.4071 | 0.4446 |
| align030 | 0.3 | 10 | 0.4419 | 0.01508 | 0.4159 | 0.4681 |
| align100 | 1 | 10 | 0.4947 | 0.01899 | 0.4665 | 0.5348 |

## 8. Evidence-Based Explanation of the High-Lambda Pattern
The high-lambda rise in same-ID distance is not explained by a mean-versus-sample mismatch: both cached evaluation latents and recomputed checkpoint latents are posterior means, and cached means match checkpoint means as reported in `latent_scale_variance_by_seed.csv`. It is also not a lambda-labelling error: manifests map align000/003/010/030/100 to 0.00/0.03/0.10/0.30/1.00 and non-lambda settings match align000.
The supported pattern is that strong alignment coincides with reduced structure-mean norm and reduced relational separation: parent-child, unrelated, and cross-language parent-child distances contract at high lambda while same-ID distance rises in the published Euclidean diagnostic. Posterior variance does not contract; it increases at high lambda in the recomputed checkpoint diagnostics. The exact optimization mechanism remains unresolved because no per-epoch latent trajectories or batch-level alignment-loss traces are saved.

## 9. Child-Count Implementation, Distribution, Metrics, and Recommendation
The target is `row['child_count']` converted to a float tensor at `tractatus_structure_latents/training/data.py:124`. The split-latent head is `Linear(structure_latent_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)` at `tractatus_structure_latents/models/vae.py:106`. The loss is `F.mse_loss(outputs['child_count'], child_count_targets.float())` at `tractatus_structure_latents/models/vae.py:200`, weighted by lambda_child. This is regression, not classification.
Target distribution from the data files:
| dataset | language | child_count | support | proportion |
| --- | --- | --- | --- | --- |
| monolingual | all | 0 | 358 | 0.6806 |
| monolingual | all | 1 | 64 | 0.1217 |
| monolingual | all | 2 | 27 | 0.05133 |
| monolingual | all | 3 | 20 | 0.03802 |
| monolingual | all | 4 | 21 | 0.03992 |
| monolingual | all | 5 | 8 | 0.01521 |
| monolingual | all | 6 | 9 | 0.01711 |
| monolingual | all | 7 | 4 | 0.007605 |
| monolingual | all | 8 | 9 | 0.01711 |
| monolingual | all | 9 | 2 | 0.003802 |
| monolingual | all | 10 | 2 | 0.003802 |
| monolingual | all | 11 | 1 | 0.001901 |
| monolingual | all | 14 | 1 | 0.001901 |
| bilingual | all | 0 | 716 | 0.6806 |
| bilingual | all | 1 | 128 | 0.1217 |
| bilingual | all | 2 | 54 | 0.05133 |
| bilingual | all | 3 | 40 | 0.03802 |
| bilingual | all | 4 | 42 | 0.03992 |
| bilingual | all | 5 | 16 | 0.01521 |
| bilingual | all | 6 | 18 | 0.01711 |
| bilingual | all | 7 | 8 | 0.007605 |
| bilingual | all | 8 | 18 | 0.01711 |
| bilingual | all | 9 | 4 | 0.003802 |
| bilingual | all | 10 | 4 | 0.003802 |
| bilingual | all | 11 | 2 | 0.001901 |
| bilingual | all | 14 | 2 | 0.001901 |

Child-count regression metrics, all languages/samples, averaged across seeds:
| condition | lambda | n | mean | sd | rmse_mean | mean_baseline_mae |
| --- | --- | --- | --- | --- | --- | --- |
| align000 | 0 | 10 | 0.49614 | 0.026249 | 0.72207 | 1.3431 |
| align003 | 0.03 | 10 | 0.49646 | 0.026887 | 0.72308 | 1.3431 |
| align010 | 0.1 | 10 | 0.50322 | 0.025692 | 0.73631 | 1.3431 |
| align030 | 0.3 | 10 | 0.53511 | 0.030225 | 0.79327 | 1.3431 |
| align100 | 1 | 10 | 0.62773 | 0.038985 | 0.94566 | 1.3431 |
| monolingual_split_24_8_reg005 |  | 10 | 0.51808 | 0.031936 | 0.76015 | 1.3431 |
Recommendation: describe child-count as an auxiliary objective unless the paper adds the audit-computed regression metrics with the regression task definition and baseline. The original per-seed JSON files did not contain child-count performance.

## 10. Paired Lambda 0.00 Versus 0.03 Results
Paired differences are defined as lambda 0.03 minus lambda 0.00. Bootstrap CIs use seed `20260717` and `10000` resamples. Exact sign-flip p-values enumerate all 2^10 sign assignments. Effect size is `mean(diff) / sample_sd(diff)`.
| metric | n | mean_difference_003_minus_000 | sd_difference | bootstrap_ci_95_low | bootstrap_ci_95_high | sign_flip_p_value_two_sided | seeds_favouring_003 | seeds_favouring_000 | standardised_paired_effect_size_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross_language_top1_id_accuracy | 10 | 0.001711 | 0.0054293 | -0.0014259 | 0.0048479 | 0.35938 | 6 | 3 | 0.31515 |
| cross_language_mrr | 10 | 0.0012484 | 0.0028311 | -0.00041508 | 0.0029024 | 0.19531 | 7 | 3 | 0.44097 |
| parent_accuracy | 10 | 0.00076046 | 0.0032251 | -0.0010456 | 0.0026616 | 0.55469 | 6 | 3 | 0.23579 |
| depth_accuracy | 10 | -0.00038023 | 0.0010218 | -0.00095057 | 0.00019011 | 0.39844 | 3 | 6 | -0.3721 |
| next_accuracy | 10 | -0.0012357 | 0.0050905 | -0.0043726 | 0.0015209 | 0.51172 | 5 | 4 | -0.24276 |
| reconstruction_loss | 10 | 0.0017475 | 0.01176 | -0.0053328 | 0.0084867 | 0.64062 | 3 | 7 | 0.1486 |
| perplexity | 10 | 0.0060266 | 0.038271 | -0.016965 | 0.028009 | 0.625 | 3 | 7 | 0.15747 |
| kl_text | 10 | 0.019022 | 0.014296 | 0.011419 | 0.028126 | 0.0019531 | 0 | 10 | 1.3306 |
| kl_structure | 10 | -0.010681 | 0.018501 | -0.022122 | -0.00061224 | 0.087891 | 7 | 3 | -0.57729 |
| mean_same_id_cross_language_distance | 10 | -0.011207 | 0.0047477 | -0.013829 | -0.0083466 | 0.0019531 | 10 | 0 | -2.3605 |

## 11. Empirical Statements Supported by the Audit
- Alignment loss is MSE on posterior structure means within shuffled training batches.
- Same-ID distance and retrieval are deterministic retained-corpus diagnostics over evaluation posterior structure means.
- Lambda 0.03 is numerically slightly higher than lambda 0.00 for aggregate Top-1 and MRR means in the retained seed sweep, with small paired differences relative to seed variation.
- Directional retrieval and by-language structure metrics are reciprocal in the sense that both de-to-en and en-to-de are reported and similar in magnitude.
- Strong alignment coincides with reduced structure-mean norm and reduced relational distances, while posterior variance and same-ID Euclidean distance rise at high lambda.

## 12. Statements Not Supported by the Audit
- The audit does not support claims of formal model-selection significance for lambda 0.03.
- The audit does not support held-out generalisation claims for the principal reported metrics.
- The audit does not support semantic equivalence, philosophical understanding, or language-invariant logic claims.
- The audit does not support treating child-count as an originally reported structural performance metric.

## 13. Exact Unresolved Questions and Missing Artifacts
- No saved child-count predictions or metrics were present in canonical per-seed JSON files.
- No per-epoch latent-scale, posterior-variance, or alignment-loss traces were found, limiting causal explanation of the high-lambda pattern.
- PCA figures are representative diagnostics for `align003` seed000, not multi-seed summaries.
- No held-out split artifact was identified for the bilingual principal metrics.

## 14. Reproduction Commands
```bash
python3 tools/empirical_audit.py
python3 -m unittest tests/test_empirical_audit.py
```

Machine-readable outputs are in `reports/empirical_audit/`.
