# Phase 3 Controlled Alignment Report

Git commit analysed: `6c274141e3da275777c689c2073319da26739b24`.
Seed-level runs: 100. Paired-batch minimum pair coverage: 1.0000.

## Design

The canonical controlled sweep uses an ID-level paired sampler. Each epoch shuffles proposition IDs, places the German and English rows for each ID in the same minibatch, computes same-ID structure-latent MSE over every observed pair, and asserts complete pair coverage. Historical random-row alignment runs are retained only as legacy provenance and are excluded from this report.

## Final Metrics

### paired full_model

| lambda | Top-1 | Top-5 | Top-10 | MRR | rank | same-ID distance |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.9443 +/- 0.0100 (n=10) | 0.9997 +/- 0.0005 (n=10) | 1.0000 +/- 0.0000 (n=10) | 0.9707 +/- 0.0054 (n=10) | 1.0661 +/- 0.0131 (n=10) | 0.7473 +/- 0.0157 (n=10) |
| 0.03 | 0.9615 +/- 0.0067 (n=10) | 0.9999 +/- 0.0003 (n=10) | 1.0000 +/- 0.0000 (n=10) | 0.9799 +/- 0.0038 (n=10) | 1.0441 +/- 0.0096 (n=10) | 0.6843 +/- 0.0158 (n=10) |
| 0.10 | 0.9802 +/- 0.0040 (n=10) | 1.0000 +/- 0.0000 (n=10) | 1.0000 +/- 0.0000 (n=10) | 0.9899 +/- 0.0021 (n=10) | 1.0213 +/- 0.0045 (n=10) | 0.5929 +/- 0.0146 (n=10) |
| 0.30 | 0.9908 +/- 0.0037 (n=10) | 1.0000 +/- 0.0000 (n=10) | 1.0000 +/- 0.0000 (n=10) | 0.9953 +/- 0.0018 (n=10) | 1.0097 +/- 0.0037 (n=10) | 0.4883 +/- 0.0101 (n=10) |
| 1.00 | 0.9959 +/- 0.0032 (n=10) | 1.0000 +/- 0.0000 (n=10) | 1.0000 +/- 0.0000 (n=10) | 0.9979 +/- 0.0016 (n=10) | 1.0043 +/- 0.0034 (n=10) | 0.4069 +/- 0.0148 (n=10) |

### paired no_successor

| lambda | Top-1 | MRR | same-ID distance | parent acc. | depth acc. | successor Top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.4829 +/- 0.0191 (n=10) | 0.6778 +/- 0.0128 (n=10) | 0.6578 +/- 0.0183 (n=10) | 0.7567 +/- 0.0152 (n=10) | 0.9802 +/- 0.0050 (n=10) | 0.0020 +/- 0.0012 (n=10) |
| 0.03 | 0.5230 +/- 0.0154 (n=10) | 0.7057 +/- 0.0116 (n=10) | 0.5976 +/- 0.0162 (n=10) | 0.7569 +/- 0.0154 (n=10) | 0.9801 +/- 0.0044 (n=10) | 0.0023 +/- 0.0012 (n=10) |
| 0.10 | 0.5893 +/- 0.0149 (n=10) | 0.7519 +/- 0.0094 (n=10) | 0.5119 +/- 0.0112 (n=10) | 0.7560 +/- 0.0140 (n=10) | 0.9809 +/- 0.0048 (n=10) | 0.0023 +/- 0.0014 (n=10) |
| 0.30 | 0.6651 +/- 0.0196 (n=10) | 0.7988 +/- 0.0111 (n=10) | 0.4129 +/- 0.0083 (n=10) | 0.7493 +/- 0.0187 (n=10) | 0.9795 +/- 0.0055 (n=10) | 0.0021 +/- 0.0010 (n=10) |
| 1.00 | 0.7113 +/- 0.0152 (n=10) | 0.8252 +/- 0.0100 (n=10) | 0.3319 +/- 0.0094 (n=10) | 0.7214 +/- 0.0200 (n=10) | 0.9656 +/- 0.0105 (n=10) | 0.0026 +/- 0.0013 (n=10) |

## Metric Coverage

The final seed table includes parent, depth and successor accuracy; child-count MAE/RMSE; directional and combined Top-1, Top-5, Top-10, MRR and rank; same-ID distance; wider-neighbourhood Jaccard at k=5, 10 and 20; reconstruction and perplexity; KL terms; sibling, parent-child, cross-language parent-child and unrelated distances; structure-mean norm; and posterior variance.

Epoch trajectories are stored in `phase3_epoch_trajectories.parquet`, with pair coverage in `phase3_pair_coverage.csv`.

## Analysis

Effect of lambda: In the paired full model, lambda=0.03 changes structure Top-1 by +0.0172 and same-ID distance by -0.0630 relative to lambda=0.00. From lambda=0.00 to 1.00, Top-1 changes by +0.0516 and same-ID distance changes by -0.3404.

Effect of the successor objective: The no-successor pilot met the expansion rule and was expanded to seeds 0-9. Recorded reasons: Top-1 differs by 0.4503 at lambda=0.00, Top-1 differs by 0.4309 at lambda=0.03, Top-1 differs by 0.3878 at lambda=0.10, Top-1 differs by 0.3216 at lambda=0.30, Top-1 differs by 0.2915 at lambda=1.00. At lambda=1.00, no-successor minus full-model Top-1 is -0.2846.

Optimisation failure: The epoch trajectories expose raw alignment MSE, weighted contribution, gradient norm, reconstruction loss and formal losses per epoch. High-lambda behaviour should be interpreted jointly with reconstruction/perplexity and formal-head metrics rather than from retrieval alone.

Latent-scale changes: The report separates same-ID distance from structure-mean norm and posterior variance. The full-model same-ID distance decreases over the tested grid; the no-successor same-ID distance decreases over the tested grid.

Relational-geometry changes: The final seed table reports sibling, parent-child, cross-language parent-child and unrelated distances separately, so a change in same-ID distance is not treated as a uniform contraction of all distances.

## Required Questions

1. Does lambda=0.03 still produce only a small local tightening? The paired full-model change at lambda=0.03 is Top-1 +0.0172 and same-ID distance -0.0630; this should be described as small only relative to the seed-level SDs in `phase3_summary.csv`.

2. Does high lambda still increase realised same-ID distance? In the paired full model, same-ID distance change from lambda=0.00 to 1.00 is -0.3404.

3. Does high-weight deterioration remain after pair coverage is controlled? Pair coverage is 1.0000 for paired runs. Retrieval change from lambda=0.00 to 1.00 is +0.0516; use this paired result rather than legacy random-row alignment runs for the causal statement.

4. Is the failure regime caused or amplified by the successor objective? The no-successor expansion completed. At lambda=1.00, its Top-1 differs from the full model by -0.2846; compare the full grid in `phase3_summary.csv` before attributing the regime solely to successor supervision.

5. Does paired batching itself change lambda=0 performance? This canonical report does not promote the historical random-row comparator. Any such comparison must be requested from the legacy archive explicitly and cannot be used as manuscript evidence.

6. Can the alignment sweep now support a clear causal statement about direct pairwise attraction? Yes for the paired-batch conditions: every German-English ID pair contributes exactly once per epoch, including lambda=0.00 controls. The statement should still be limited to the measured retained-corpus setting and reported alongside optimisation and latent-scale diagnostics.

## Figures

- `figures/retrieval_across_lambda.png`
- `figures/formal_parent_accuracy_across_lambda.png`
- `figures/formal_depth_accuracy_across_lambda.png`
- `figures/formal_successor_accuracy_across_lambda.png`
- `figures/same_id_distance_across_lambda.png`
- `figures/alignment_loss_trajectories.png`
- `figures/structure_mean_norm_across_lambda.png`
- `figures/posterior_variance_across_lambda.png`

## Recommended interpretation of the alignment experiment

The controlled paired-batch sweep gives lambda a direct operational interpretation: it is the weight on an observed same-ID German-English structure-latent MSE term for every proposition pair in every epoch. Publication claims should use this paired analysis as the authoritative bilingual alignment evidence and should report retrieval, same-ID distance, formal-head performance, latent scale and relational distances together.
