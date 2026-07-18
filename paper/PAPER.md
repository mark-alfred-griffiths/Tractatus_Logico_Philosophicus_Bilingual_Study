# Paper Figures

This folder contains the plain-text paper files and generated figures for the Tractatus structure-latents work.

## Current Paper Source

The current paper source is `paper/main.tex` with bibliography
`paper/references.bib`. The paper uses the retained empirical outputs. Use the
complete canonical pipeline to regenerate
paper-facing summaries, all figures, and the TF-IDF/Jaccard/Euclidean audit
statistics without retraining:

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
  --summary-out runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json \
  --family-distance-data paper/figures/family_case_distance_matrix_data.csv
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

## Generated Family Figure

```text
paper/figures/family_case_distance_matrix.png
paper/figures/family_case_distance_matrix.pdf
paper/figures/family_case_distance_matrix_data.csv
```

## Generated Representative PCA Figures

```text
paper/figures/bilingual_latent_pca_language_align003_seed000.png
paper/figures/bilingual_latent_pca_depth_align003_seed000.png
paper/figures/monolingual_latent_pca_depth_reg005_seed000.png
```

Use the markers beginning with `[INSERT GRAPHIC: ...]` in the paper text when laying out the PDF in a word processor or page-layout tool.
