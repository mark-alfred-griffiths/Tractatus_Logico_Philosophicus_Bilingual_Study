# monolingual_split_24_8_reg005

English-only balanced split-latent seed sweep.

## Configuration

```text
languages:              en
seeds:                  0..9
z_text:                 24
z_structure:             8
beta:                    0.01
beta_text:               0.01
beta_structure:          0.05
lambda_parent:           0.2
lambda_depth:            0.1
lambda_next:             0.2
lambda_child:            0.02
lr:                      0.001
epochs:                 80
batch_size:             32
```

## Layout

```text
checkpoints/seed000.pt
checkpoints/seed000.vocab.json
logs/seed000.train.log
metrics/seed000.metrics.json
latents/seed000_structure.pt
latents/seed000_structure.ids.json
summaries/
manifest.json
```

## Run All Seeds

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

## Generate Summary

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --out runs/seed_sweeps/summary
```
