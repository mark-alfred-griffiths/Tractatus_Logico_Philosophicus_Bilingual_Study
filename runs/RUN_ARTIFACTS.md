# runs

This folder contains generated experiment artifacts and run documentation.

The canonical generated artifacts should live under:

```text
runs/seed_sweeps/
```

Avoid adding loose checkpoints, metrics, latent tensors, or plots directly to the root of `runs/`. Older root-level retained artifacts have been superseded by repeated-seed sweeps.

## Canonical Contents

```text
runs/
  RUN_ARTIFACTS.md
  seed_sweeps/
    SEED_SWEEPS.md
    monolingual_split_24_8_reg005/
    bilingual_alignment_lambda_sweep/
```

## Regeneration

Use the root [README.md](../README.md) for the full regeneration workflow:

```text
1. Build datasets.
2. Run monolingual seeds 0..9.
3. Run bilingual lambda values 0.00, 0.03, 0.10, 0.30, 1.00 across seeds 0..9.
4. Aggregate metrics.
5. Generate paper figures.
```

For the current paper-output layer, the complete canonical non-training pipeline
also refreshes generated summary artefacts under `runs/seed_sweeps/*/summaries/`
from retained `seed*.metrics.json` files, regenerates all paper figures
including Figure 7, reproduces TF-IDF/Jaccard/Euclidean statistics, and rewrites
the two paper-facing result summaries:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --out runs/seed_sweeps/monolingual_split_24_8_reg005/summaries/regenerated_comparison

PYTHONDONTWRITEBYTECODE=1 python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 \
  --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison

PYTHONDONTWRITEBYTECODE=1 python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
  --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --representative-alignment align003 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json \
  --family-distance-data paper/figures/family_case_distance_matrix_data.csv

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 tools/reproduce_paper_metrics_and_figures.py \
  --metrics --figures --skip-checkpoints \
  --out-dir reports/paper_reproducibility/reproduced

PYTHONDONTWRITEBYTECODE=1 python3 tools/write_paper_results_summaries.py
```

This pipeline may create or update files under `runs/seed_sweeps/*/summaries/`.
It must not alter checkpoints, cached latents, or retained `seed*.metrics.json`
files.

## Derived-Value Reproduction

The canonical paper does not require retraining the retained runs. Reported TF-IDF,
Jaccard, same-ID, relation-distance, and Figure 7 Euclidean-distance values are
reproduced from retained CSV/JSON artefacts, cached latents, and prior verified
audit outputs with this report-only command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/reproduce_paper_metrics_and_figures.py \
  --metrics --figures --skip-checkpoints \
  --out-dir reports/paper_reproducibility/reproduced
```

The command writes under `reports/paper_reproducibility/` and
must not write to `runs/`.

## Artifact Policy

Keep:

```text
runs/RUN_ARTIFACTS.md
runs/seed_sweeps/SEED_SWEEPS.md
runs/seed_sweeps/monolingual_split_24_8_reg005/
runs/seed_sweeps/bilingual_alignment_lambda_sweep/
```

Do not keep old one-off root artifacts unless they are intentionally archived and documented.
