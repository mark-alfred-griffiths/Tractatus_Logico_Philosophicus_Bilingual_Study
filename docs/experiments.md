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
  --representative-alignment align010 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
```

## Non-Canonical Historical Directions

Earlier one-off retained checkpoints and baseline plots have been superseded by repeated-seed sweeps. Do not add new generated artifacts directly to the root of `runs/` unless they are explicitly archived and documented.
