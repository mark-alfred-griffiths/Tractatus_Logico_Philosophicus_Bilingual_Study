# Paper Figures

This folder contains the plain-text paper files and generated figures for the Tractatus structure-latents work.

Main text files:

```text
bilingual_structure_latents_in_the_tractatus.txt
main.tex
archive/tractatus_latent_logic_paper.txt
archive/tractatus_bilingual_latent_structure_paper.txt
```

Figure outputs are written to:

```text
paper/figures/
```

## Canonical Figure Generation

Generate paper figures from the canonical seed-sweep outputs:

```bash
python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
  --seed-sweep-dir runs/seed_sweeps/bilingual_alignment_lambda_sweep \
  --monolingual-dir runs/seed_sweeps/monolingual_split_24_8_reg005 \
  --representative-alignment align003 \
  --representative-seed 0 \
  --out-dir paper/figures \
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
```

Metric sweep figures show means across seeds `0..9` with +/- one-standard-deviation error bars.

Representative PCA figures use one trained model and include the seed label in both the title and filename. The default representative model is:

```text
alignment: align003
seed:      seed000
```

## Generated Metric Figures

```text
paper/figures/bilingual_alignment_retrieval_sweep.png
paper/figures/bilingual_structure_accuracy_sweep.png
paper/figures/bilingual_retrieval_structure_tradeoff.png
paper/figures/bilingual_reconstruction_sweep.png
```

## Generated Representative PCA Figures

```text
paper/figures/bilingual_latent_pca_language_align003_seed000.png
paper/figures/bilingual_latent_pca_depth_align003_seed000.png
paper/figures/monolingual_latent_pca_depth_reg005_seed000.png
```

Use the markers beginning with `[INSERT GRAPHIC: ...]` in the paper text when laying out the PDF in a word processor or page-layout tool.
