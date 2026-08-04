# Phase 1 Ablation Report

All conditions used the shared bilingual vocabulary/parameters, 24-dimensional text latent, 8-dimensional structure latent, GRU encoder/decoder, manuscript KL/reconstruction settings, lambda_language_alignment=0.00, and seeds 0-9.

## Seed-level summary

| condition | structure Top-1 | structure MRR | text Top-1 | text MRR | k=10 Jaccard | sibling contrast | successor Top-1 | parent acc. | depth acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_model` | 0.9360 +/- 0.0128 | 0.9657 +/- 0.0071 | 0.0096 +/- 0.0042 | 0.0342 +/- 0.0075 | 0.6263 +/- 0.0055 | 4.0361 +/- 0.0632 | 0.6161 +/- 0.0226 | 0.8596 +/- 0.0090 | 0.9633 +/- 0.0120 |
| `reconstruction_only` | 0.0035 +/- 0.0021 | 0.0182 +/- 0.0037 | 0.0077 +/- 0.0034 | 0.0310 +/- 0.0066 | 0.0126 +/- 0.0017 | 0.0000 +/- 0.0002 | 0.0016 +/- 0.0010 | 0.0041 +/- 0.0028 | 0.1462 +/- 0.1537 |
| `no_successor` | 0.4593 +/- 0.0121 | 0.6615 +/- 0.0059 | 0.0115 +/- 0.0040 | 0.0341 +/- 0.0064 | 0.5950 +/- 0.0095 | 3.4324 +/- 0.0362 | 0.0023 +/- 0.0016 | 0.7967 +/- 0.0124 | 0.9832 +/- 0.0081 |
| `parent_depth_only` | 0.3854 +/- 0.0186 | 0.5982 +/- 0.0130 | 0.0089 +/- 0.0040 | 0.0340 +/- 0.0070 | 0.6066 +/- 0.0134 | 3.5252 +/- 0.0453 | 0.0021 +/- 0.0017 | 0.8019 +/- 0.0148 | 0.9870 +/- 0.0059 |
| `successor_only` | 0.9944 +/- 0.0022 | 0.9972 +/- 0.0011 | 0.0054 +/- 0.0028 | 0.0279 +/- 0.0066 | 0.5228 +/- 0.0085 | 0.0120 +/- 0.0502 | 0.7960 +/- 0.0216 | 0.0019 +/- 0.0020 | 0.1518 +/- 0.0402 |
| `shuffled_joint_targets` | 0.9345 +/- 0.0095 | 0.9652 +/- 0.0052 | 0.0096 +/- 0.0048 | 0.0332 +/- 0.0075 | 0.6139 +/- 0.0096 | 0.0330 +/- 0.0733 | 0.0008 +/- 0.0010 | 0.0041 +/- 0.0010 | 0.3330 +/- 0.0051 |
| `shuffled_no_successor` | 0.4694 +/- 0.0111 | 0.6688 +/- 0.0072 | 0.0106 +/- 0.0041 | 0.0340 +/- 0.0058 | 0.5953 +/- 0.0115 | 0.0234 +/- 0.0565 | 0.0027 +/- 0.0015 | 0.0169 +/- 0.0015 | 0.3263 +/- 0.0029 |

## Paired differences versus full_model

| condition | metric | mean diff | 95% bootstrap CI |
| --- | --- | ---: | ---: |
| `no_successor` | `structure_cross_language_top1` | -0.4767 | [-0.4873, -0.4660] |
| `no_successor` | `structure_cross_language_mrr` | -0.3042 | [-0.3096, -0.2986] |
| `no_successor` | `structure_cross_language_same_id_distance` | -0.0903 | [-0.0950, -0.0860] |
| `no_successor` | `structure_wider_neighbourhood_jaccard_k10` | -0.0313 | [-0.0383, -0.0239] |
| `no_successor` | `structure_sibling_vs_unrelated_contrast` | -0.6036 | [-0.6286, -0.5785] |
| `parent_depth_only` | `structure_cross_language_top1` | -0.5507 | [-0.5629, -0.5392] |
| `parent_depth_only` | `structure_cross_language_mrr` | -0.3675 | [-0.3756, -0.3592] |
| `parent_depth_only` | `structure_cross_language_same_id_distance` | -0.0938 | [-0.1017, -0.0865] |
| `parent_depth_only` | `structure_wider_neighbourhood_jaccard_k10` | -0.0197 | [-0.0286, -0.0096] |
| `parent_depth_only` | `structure_sibling_vs_unrelated_contrast` | -0.5108 | [-0.5290, -0.4916] |
| `reconstruction_only` | `structure_cross_language_top1` | -0.9325 | [-0.9394, -0.9257] |
| `reconstruction_only` | `structure_cross_language_mrr` | -0.9475 | [-0.9519, -0.9427] |
| `reconstruction_only` | `structure_cross_language_same_id_distance` | -0.7562 | [-0.7639, -0.7487] |
| `reconstruction_only` | `structure_wider_neighbourhood_jaccard_k10` | -0.6137 | [-0.6176, -0.6101] |
| `reconstruction_only` | `structure_sibling_vs_unrelated_contrast` | -4.0360 | [-4.0745, -3.9994] |
| `shuffled_joint_targets` | `structure_cross_language_top1` | -0.0015 | [-0.0080, 0.0050] |
| `shuffled_joint_targets` | `structure_cross_language_mrr` | -0.0004 | [-0.0039, 0.0030] |
| `shuffled_joint_targets` | `structure_cross_language_same_id_distance` | 0.0147 | [0.0059, 0.0225] |
| `shuffled_joint_targets` | `structure_wider_neighbourhood_jaccard_k10` | -0.0124 | [-0.0184, -0.0062] |
| `shuffled_joint_targets` | `structure_sibling_vs_unrelated_contrast` | -4.0031 | [-4.0543, -3.9543] |
| `shuffled_no_successor` | `structure_cross_language_top1` | -0.4666 | [-0.4754, -0.4584] |
| `shuffled_no_successor` | `structure_cross_language_mrr` | -0.2969 | [-0.3023, -0.2920] |
| `shuffled_no_successor` | `structure_cross_language_same_id_distance` | -0.0908 | [-0.0966, -0.0849] |
| `shuffled_no_successor` | `structure_wider_neighbourhood_jaccard_k10` | -0.0310 | [-0.0380, -0.0240] |
| `shuffled_no_successor` | `structure_sibling_vs_unrelated_contrast` | -4.0127 | [-4.0417, -3.9799] |
| `successor_only` | `structure_cross_language_top1` | 0.0584 | [0.0512, 0.0656] |
| `successor_only` | `structure_cross_language_mrr` | 0.0315 | [0.0276, 0.0355] |
| `successor_only` | `structure_cross_language_same_id_distance` | -0.1850 | [-0.1944, -0.1764] |
| `successor_only` | `structure_wider_neighbourhood_jaccard_k10` | -0.1035 | [-0.1091, -0.0971] |
| `successor_only` | `structure_sibling_vs_unrelated_contrast` | -4.0241 | [-4.0697, -3.9752] |

## Required questions

1. Reconstruction alone produced structure-latent Top-1 0.0035; text-latent Top-1 was 0.0077. This is the controlled estimate for retrieval available without formal supervision.
2. Retrieval primarily resides in the latent with higher Top-1/MRR within each condition; in reconstruction_only the text-vs-structure split is 0.0077 versus 0.0035.
3. Removing successor left structure-latent Top-1 0.4593. Interpret this against full_model using the paired CI table, not by a mean comparison alone.
4. Successor alone reached structure-latent Top-1 0.9944, directly testing whether near proposition-specific labels can reproduce the retrieval effect.
5. Shuffled shared targets reached structure-latent Top-1 0.9345; because German and English share each shuffled target tuple, this tests target-sharing without true formal position.
6. Sibling cohesion is supported only if true-target conditions retain a positive sibling-versus-unrelated contrast beyond shuffled controls. The full_model contrast is 4.0361; shuffled_joint_targets is 0.0330.
7. Parent/depth without successor reached Top-1 0.3854; successor_only and no_successor identify which formal objectives add evidence beyond proposition-specific identity.

Small numerical differences should be treated as descriptive unless their paired bootstrap intervals and the task design make the direction stable and interpretable.

## Implications for the current manuscript

Claims about same-ID bilingual retrieval remain supported only to the extent that the full_model exceeds reconstruction_only and shuffled shared-target controls. Claims that retrieval reflects hierarchy rather than proposition-specific formal labels require qualification if successor_only or shuffled_joint_targets approach the full_model. Sibling/family-organisation claims remain supported when true hierarchy-supervised conditions show stronger sibling-versus-unrelated contrast than shuffled controls; otherwise they should be narrowed. No manuscript text was edited in this phase.
