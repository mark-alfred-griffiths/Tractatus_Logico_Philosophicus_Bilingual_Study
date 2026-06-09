# Reproduction

Run commands from the repository root.

## 1. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

Required core packages include PyTorch, Matplotlib, scikit-learn, and UMAP.

## 2. Build Datasets

Build the English-only dataset:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus.json \
  --languages en
```

Build the bilingual German/English dataset:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus_bilingual.json \
  --languages en,de
```

## 3. Run Monolingual Seed Sweep

```bash
for seed in {0..9}; do
  seed_name=$(printf 'seed%03d' "$seed")

  python3 -m tractatus_structure_latents.training.train_vae \
    --data tractatus_structure_latents/data/tractatus.json \
    --split-latent \
    --text-latent-dim 24 \
    --structure-latent-dim 8 \
    --epochs 80 \
    --batch-size 32 \
    --beta 0.01 \
    --beta-text 0.01 \
    --beta-structure 0.05 \
    --lambda-parent 0.2 \
    --lambda-depth 0.1 \
    --lambda-next 0.2 \
    --lambda-child 0.02 \
    --lr 0.001 \
    --seed "$seed" \
    --out "runs/seed_sweeps/monolingual_split_24_8_reg005/checkpoints/${seed_name}.pt" \
    2>&1 | tee "runs/seed_sweeps/monolingual_split_24_8_reg005/logs/${seed_name}.train.log"

  python3 -m tractatus_structure_latents.evaluation.evaluate_structure \
    --data tractatus_structure_latents/data/tractatus.json \
    --checkpoint "runs/seed_sweeps/monolingual_split_24_8_reg005/checkpoints/${seed_name}.pt" \
    --batch-size 64 \
    --latent-part structure \
    --export-latents "runs/seed_sweeps/monolingual_split_24_8_reg005/latents/${seed_name}_structure.pt" \
    > "runs/seed_sweeps/monolingual_split_24_8_reg005/metrics/${seed_name}.metrics.json"
done
```

## 4. Run Bilingual Alignment Lambda Sweep

```bash
python3 -m tractatus_structure_latents.scripts.run_bilingual_alignment_seed_sweep \
  --out-root runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --lambdas 0.00,0.03,0.10,0.30,1.00 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --skip-existing
```

This runs `50` bilingual trainings.

## 5. Generate Summaries

Monolingual summary:

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --out runs/seed_sweeps/summary
```

Per-lambda bilingual summaries:

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 \
  --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison
```

## 6. Generate Paper Figures

```bash
python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
  --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --representative-alignment align010 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
```

Metric sweep figures show means across seeds `0..9` with +/- one-standard-deviation error bars. PCA figures use representative seed `0` and include seed labels in titles and filenames.
