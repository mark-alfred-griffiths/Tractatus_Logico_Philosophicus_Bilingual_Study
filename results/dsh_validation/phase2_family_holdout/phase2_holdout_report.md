# Phase 2 Family-Held-Out Generalisation Report

This evaluation uses deterministic five-fold immediate-parent-family holdout. German and English samples for each proposition ID remain in the same fold. Vocabularies are built from training-fold texts only; held-out-only tokens use the normal `<unk>` path.

## Summary

| condition | structure test Top-1 | structure complete Top-1 | structure MRR | text test Top-1 | depth acc. | child MAE | sibling contrast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_model` | 0.0739 +/- 0.0236 | 0.0577 +/- 0.0222 | 0.1644 +/- 0.0333 | 0.0288 +/- 0.0136 | 0.3489 +/- 0.0252 | 1.1808 +/- 0.2582 | 0.1142 +/- 0.0612 |
| `no_successor` | 0.0811 +/- 0.0188 | 0.0599 +/- 0.0149 | 0.1661 +/- 0.0245 | 0.0282 +/- 0.0142 | 0.3632 +/- 0.0288 | 1.1745 +/- 0.2495 | 0.0977 +/- 0.0421 |
| `reconstruction_only` | 0.0158 +/- 0.0056 | 0.0041 +/- 0.0035 | 0.0665 +/- 0.0086 | 0.0444 +/- 0.0138 | 0.0746 +/- 0.1380 | 1.0198 +/- 0.2577 | 0.0003 +/- 0.0009 |

## Required questions

1. Same-ID bilingual retrieval under unseen-family splitting is estimated by full_model structure test-candidate Top-1 = 0.0739.
2. Removing successor gives no_successor structure test-candidate Top-1 = 0.0811; compare with full_model before treating any difference as meaningful.
3. The full_model text-latent Top-1 is 0.0288; structure-vs-text generalisation should be read from the paired fold/seed table.
4. Sibling cohesion in unseen families is summarised by full_model structure sibling-versus-matched-unrelated contrast = 0.1142; matched family tail probabilities are in `phase2_matched_family_results.csv`.
5. Retained-corpus performance is overstated where Phase 1 full_model retrieval exceeds held-out full_model retrieval; Phase 2 tests unseen texts and immediate-parent families.
6. Exact parent and successor classification cannot be validly interpreted for held-out targets whose class IDs were absent from training; the report separates seen-class coverage and seen-class accuracy.

## Implications for publication claims

Retained-corpus findings: Phase 1 remains the fitted-corpus analysis of objective contributions.

Demonstrated held-out generalisation: claims should be limited to metrics that survive this unseen-family split, especially same-ID retrieval, depth, child-count, and sibling-cohesion contrasts.

Unresolved generalisation questions: exact parent/successor prediction for unseen class IDs is not a valid held-out classification task without a different label formulation; larger seed sweeps can refine uncertainty without changing the fixed folds.
