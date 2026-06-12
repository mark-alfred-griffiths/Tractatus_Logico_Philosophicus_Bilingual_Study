# bilingual_alignment_lambda_sweep

German/English split-latent alignment-strength sweep across multiple seeds.

## Configuration

```text
languages:                   en,de
seeds:                       0..9
lambda_language_alignment:   0.00, 0.03, 0.10, 0.30, 1.00
z_text:                      24
z_structure:                  8
beta:                         0.01
beta_text:                    0.01
beta_structure:               0.05
lambda_parent:                0.2
lambda_depth:                 0.1
lambda_next:                  0.2
lambda_child:                 0.02
lr:                           0.001
epochs:                      80
batch_size:                  32
```

## Layout

```text
bilingual_alignment_lambda_sweep/
  align000/
  align003/
  align010/
  align030/
  align100/
  summaries/
```

Each `alignXYZ/` folder contains:

```text
checkpoints/
logs/
metrics/
latents/
summaries/
manifest.json
```

## Run Full Sweep

```bash
python3 -m tractatus_structure_latents.scripts.run_bilingual_alignment_seed_sweep \
  --out-root runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --lambdas 0.00,0.03,0.10,0.30,1.00 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --skip-existing
```

This runs `50` bilingual trainings.

## Generate Per-Lambda Summaries

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 \
  --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison
```

## Generate Paper Figures

Metric sweep figures use means across seeds `0..9` with +/- one-standard-deviation error bars. PCA figures use a representative trained model and include the seed label.

```bash
python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
  --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --representative-alignment align003 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
```

This recreates mean/std versions of:

```text
paper/figures/bilingual_alignment_retrieval_sweep.png
paper/figures/bilingual_structure_accuracy_sweep.png
paper/figures/bilingual_retrieval_structure_tradeoff.png
paper/figures/bilingual_reconstruction_sweep.png
```
