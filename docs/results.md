# Results

Canonical results are generated from repeated-seed sweeps under `runs/seed_sweeps/`.

## Canonical Result Locations

```text
runs/seed_sweeps/monolingual_split_24_8_reg005/summaries/
runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/
runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
paper/figures/
```

## Monolingual Study

The monolingual study is:

```text
runs/seed_sweeps/monolingual_split_24_8_reg005/
```

It uses the balanced English split-latent setting with `beta_structure=0.05` across seeds `0..9`.

Generate its summary with:

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --out runs/seed_sweeps/summary
```

## Bilingual Alignment Lambda Sweep

The bilingual study is:

```text
runs/seed_sweeps/bilingual_alignment_lambda_sweep/
```

It evaluates:

```text
lambda_language_alignment = 0.00, 0.03, 0.10, 0.30, 1.00
seeds = 0..9
```

Generate per-lambda summaries with:

```bash
python3 -m tractatus_structure_latents.evaluation.analyse_seed_sweep \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030 \
  runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100 \
  --out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/per_lambda_comparison
```

## Figures

Canonical paper figures are in `paper/figures/`.

Metric sweep figures use means across seeds with +/- one-standard-deviation error bars:

```text
paper/figures/bilingual_alignment_retrieval_sweep.png
paper/figures/bilingual_structure_accuracy_sweep.png
paper/figures/bilingual_retrieval_structure_tradeoff.png
paper/figures/bilingual_reconstruction_sweep.png
```

Representative PCA figures are seed-labeled:

```text
paper/figures/bilingual_latent_pca_language_align010_seed000.png
paper/figures/bilingual_latent_pca_depth_align010_seed000.png
paper/figures/monolingual_latent_pca_depth_reg005_seed000.png
```

Regenerate all canonical figures with:

```bash
python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
  --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --representative-alignment align010 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
```

## Interpretation Guardrail

The bilingual result is evidence for proposition-level alignment in this small, paired, highly structured Tractatus corpus. It should not be read as proof of a general language-invariant logical manifold.
