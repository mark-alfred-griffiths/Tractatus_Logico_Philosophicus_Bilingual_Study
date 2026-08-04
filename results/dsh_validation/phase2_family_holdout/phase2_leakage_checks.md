# Phase 2 Leakage Checks

- Fold 0: no proposition ID occurs in both train and test.
- Fold 0: no immediate-parent family is divided across folds.
- Fold 0: 396 held-out-only tokens map through `<unk>`.
- Fold 1: no proposition ID occurs in both train and test.
- Fold 1: no immediate-parent family is divided across folds.
- Fold 1: 434 held-out-only tokens map through `<unk>`.
- Fold 2: no proposition ID occurs in both train and test.
- Fold 2: no immediate-parent family is divided across folds.
- Fold 2: 455 held-out-only tokens map through `<unk>`.
- Fold 3: no proposition ID occurs in both train and test.
- Fold 3: no immediate-parent family is divided across folds.
- Fold 3: 436 held-out-only tokens map through `<unk>`.
- Fold 4: no proposition ID occurs in both train and test.
- Fold 4: no immediate-parent family is divided across folds.
- Fold 4: 345 held-out-only tokens map through `<unk>`.
- German and English counterparts are represented by one proposition ID and therefore share the same fold.
- Parent/successor heads use a fixed global proposition-ID class space; seen/unseen target coverage is reported separately.
- No normalisation/calibration objects are fitted outside the training vocabulary builders; model selection is not performed from test metrics.
