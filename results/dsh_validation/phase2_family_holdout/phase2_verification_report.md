# Phase 2 Verification Report

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
- Verified 45 condition-fold-seed rows from raw parquet files.
- Verified 408 reported summary rows where raw/seed metrics apply.
- Verified 869 protected canonical file hashes unchanged.
- Fold manifest rows: 526.
- Raw per-proposition rows: 9468.
- Git branch: hold-out-and-ablation.
- Git commit during verification: 4d46c626c57c51b7b7a7049e8ba288d73f6bade8.

## Git status

```
M tractatus_structure_latents/training/__pycache__/train_vae.cpython-310.pyc
 M tractatus_structure_latents/training/data.py
 M tractatus_structure_latents/training/train_vae.py
?? .idea/
?? results/dsh_validation/phase1_ablations/checkpoints/
?? results/dsh_validation/phase1_ablations/smoke/
?? results/dsh_validation/phase2_family_holdout/
?? tests/__pycache__/
?? tools/__pycache__/
?? tools/phase2_family_holdout.py
```

## Git diff stat

```
.../training/__pycache__/train_vae.cpython-310.pyc  | Bin 11040 -> 11893 bytes
 tractatus_structure_latents/training/data.py        |   4 ++++
 tractatus_structure_latents/training/train_vae.py   |  11 +++++++++++
 3 files changed, 15 insertions(+)
```
