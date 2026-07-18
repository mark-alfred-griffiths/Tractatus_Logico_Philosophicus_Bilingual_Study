# Experiments

The canonical experiments are repeated-seed sweeps. Generated artifacts live under `runs/seed_sweeps/`.

## Monolingual Study

Balanced English split-latent model:

```text
study:           runs/seed_sweeps/monolingual_split_24_8_reg005/
languages:       en
seeds:           0..9
z_text:          24
z_structure:      8
beta_text:        0.01
beta_structure:   0.05
```

The monolingual study tests whether `z_structure` recovers Tractatus hierarchy under English-only text reconstruction.

## Bilingual Alignment Lambda Sweep

German/English split-latent model across alignment strengths:

```text
study:           runs/seed_sweeps/bilingual_alignment_lambda_sweep/
languages:       en,de
seeds:           0..9
lambda values:   0.00, 0.03, 0.10, 0.30, 1.00
z_text:          24
z_structure:      8
beta_text:        0.01
beta_structure:   0.05
```

The bilingual study tests the tradeoff between structural prediction, reconstruction, and same-id cross-language retrieval.

## Canonical Artifacts

```text
runs/seed_sweeps/monolingual_split_24_8_reg005/
runs/seed_sweeps/bilingual_alignment_lambda_sweep/
paper/figures/
```

Metric sweep figures are generated as means across seeds `0..9` with +/- one-standard-deviation error bars. PCA figures use a representative model and include the seed in title and filename.

## Generate Figures

```bash
python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
  --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --representative-alignment align003 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json \
  --family-distance-data paper/figures/family_case_distance_matrix_data.csv
```

## Canonical Reproducibility Audit

The canonical paper keeps the retained trained models fixed. Regenerating paper
outputs means aggregating retained `seed*.metrics.json` files, regenerating
figures, refreshing derived TF-IDF/Jaccard/Euclidean statistics, and rebuilding
the human-readable summaries. It does not run optimisation or retrain.

Complete canonical paper-output regeneration:

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

This regenerates summary artefacts under `runs/seed_sweeps/*/summaries/`, all
paper figures including the Figure 7 family 2.2 Euclidean distance matrix,
paper audit CSV/JSON statistics under `reports/`, and
`paper/monolingual_results_summary.txt` plus
`paper/bilingual_results_summary.txt`.

## Non-Canonical Historical Directions

Earlier one-off retained checkpoints and baseline plots have been superseded by repeated-seed sweeps. Do not add new generated artifacts directly to the root of `runs/` unless they are explicitly archived and documented.
