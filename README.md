# Tractatus Structure Latents

This repository explores whether the hierarchical numbering of Wittgenstein's *Tractatus Logico-Philosophicus* can supervise a model toward proposition-structure latents rather than simple text prediction.

The project does not claim to process logic or to learn general philosophical logic. The implemented supervision comes from formal relations derivable from proposition numbers: parent, depth, next-proposition, child count, and bilingual same-id alignment.

The current model family is a split-latent RNN-VAE:

```text
z_text      -> language-conditioned text reconstruction
z_structure -> hierarchy prediction and cross-language structure alignment
```

The canonical experiments are now repeated-seed sweeps. Generated run artifacts should live under `runs/seed_sweeps/`, not as loose root-level files in `runs/`.

## Canonical Experiments

```text
runs/seed_sweeps/monolingual_split_24_8_reg005/
  English-only balanced split-latent sweep over seeds 0..9.

runs/seed_sweeps/bilingual_alignment_lambda_sweep/
  German/English alignment-strength sweep over lambda values and seeds 0..9.
```

The bilingual lambda sweep covers:

```text
lambda_language_alignment = 0.00, 0.03, 0.10, 0.30, 1.00
```

Each generated study folder uses the same artifact layout:

```text
checkpoints/   model checkpoints and vocab files
logs/          training logs
metrics/       per-seed evaluation JSON
latents/       exported structure latent tensors and ids
summaries/     aggregate CSV, JSON, Markdown, and plots
manifest.json  fixed study configuration
```

## Documentation

- [Dataset](docs/dataset.md)
- [Model](docs/model.md)
- [Experiments](docs/experiments.md)
- [Results](docs/results.md)
- [Reproduction](docs/reproduction.md)
- [Documentation index](docs/DOCUMENTATION.md)
- [Paper notes](paper/PAPER.md)
- [Run artifacts](runs/RUN_ARTIFACTS.md)
- [Seed sweeps](runs/seed_sweeps/SEED_SWEEPS.md)

## Repository Layout

```text
tractatus_structure_latents/
  data/                  dataset builder and Tractatus JSON datasets
  models/                encoder, decoder, VAE, split-latent VAE, dynamics module
  training/              VAE training CLI
  evaluation/            structure metrics, sweep analysis, and latent visualisation
  active_inference/      exploratory prototype, not part of canonical sweeps
  scripts/               dataset construction and sweep runner CLIs
runs/                    run artifact documentation and generated experiment outputs
runs/seed_sweeps/        canonical repeated-seed study artifacts
paper/                   plain-text paper and generated figures
docs/                    project documentation
```

The canonical Python package path is `tractatus_structure_latents/`.

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Build or refresh the English-only dataset:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus.json \
  --languages en
```

Build or refresh the bilingual German/English dataset:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus_bilingual.json \
  --languages en,de
```

## Run All Canonical Sweeps

Run the full monolingual seed sweep:

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

Run the full bilingual alignment lambda sweep:

```bash
python3 -m tractatus_structure_latents.scripts.run_bilingual_alignment_seed_sweep \
  --out-root runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --lambdas 0.00,0.03,0.10,0.30,1.00 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --skip-existing
```

This runs `50` bilingual trainings: `5` alignment strengths times `10` seeds.

## Generate Results

Aggregate the monolingual sweep:

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --out runs/seed_sweeps/summary
```

Aggregate each bilingual alignment-strength folder, if you want per-lambda CSV/Markdown summaries:

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 \
  --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison
```

Generate paper figures. Metric sweep figures use means across seeds `0..9` with +/- one-standard-deviation error bars. PCA figures use a representative trained model and include the seed in the title and filename.

```bash
python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
  --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --representative-alignment align010 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
```

This recreates metric sweep figures:

```text
paper/figures/bilingual_alignment_retrieval_sweep.png
paper/figures/bilingual_structure_accuracy_sweep.png
paper/figures/bilingual_retrieval_structure_tradeoff.png
paper/figures/bilingual_reconstruction_sweep.png
```

It also creates representative PCA figures such as:

```text
paper/figures/bilingual_latent_pca_language_align010_seed000.png
paper/figures/bilingual_latent_pca_depth_align010_seed000.png
paper/figures/monolingual_latent_pca_depth_reg005_seed000.png
```

## Visualise Representative Latents

Monolingual depth PCA for seed `000`:

```bash
python3 -m tractatus_structure_latents.evaluation.visualise_latents \
  --latents runs/seed_sweeps/monolingual_split_24_8_reg005/latents/seed000_structure.pt \
  --ids runs/seed_sweeps/monolingual_split_24_8_reg005/latents/seed000_structure.ids.json \
  --data tractatus_structure_latents/data/tractatus.json \
  --method pca \
  --colour-by depth \
  --out runs/seed_sweeps/monolingual_split_24_8_reg005/summaries/seed000_pca_depth.png
```

Bilingual language PCA for lambda `0.10`, seed `000`:

```bash
python3 -m tractatus_structure_latents.evaluation.visualise_latents \
  --latents runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/latents/seed000_structure.pt \
  --ids runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/latents/seed000_structure.ids.json \
  --data tractatus_structure_latents/data/tractatus_bilingual.json \
  --method pca \
  --colour-by language \
  --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/summaries/seed000_pca_language.png
```

## Cleanup Policy

The root of `runs/` should not accumulate loose generated checkpoints or metrics. Keep canonical generated outputs under:

```text
runs/seed_sweeps/monolingual_split_24_8_reg005/
runs/seed_sweeps/bilingual_alignment_lambda_sweep/
```

## License

This repository uses a mixed-license structure:

- Code is licensed under [Apache-2.0](LICENSE).
- Documentation and paper text are licensed under [CC BY 4.0](LICENSE-DOCS.md).
- Tractatus-derived text/data are covered by the source-text notice in [DATA_NOTICE.md](DATA_NOTICE.md).

The repository does not claim copyright over the underlying public-domain Tractatus text.
