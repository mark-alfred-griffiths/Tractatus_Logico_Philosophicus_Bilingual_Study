# Codex Commands: Canonical Paper Reproducibility

Run these commands from the repository root to regenerate the current
paper-output layer from retained artefacts. This pipeline does not retrain
models and must not modify checkpoints, cached latents, or retained
`seed*.metrics.json` files.

## Complete Paper-Output Pipeline

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

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 -m tractatus_structure_latents.evaluation.generate_paper_figures \
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

## Expected Outputs

Paper figures:

```text
paper/figures/bilingual_alignment_retrieval_sweep.pdf
paper/figures/bilingual_alignment_retrieval_sweep.png
paper/figures/bilingual_structure_accuracy_sweep.pdf
paper/figures/bilingual_structure_accuracy_sweep.png
paper/figures/bilingual_retrieval_structure_tradeoff.pdf
paper/figures/bilingual_retrieval_structure_tradeoff.png
paper/figures/bilingual_reconstruction_sweep.pdf
paper/figures/bilingual_reconstruction_sweep.png
paper/figures/bilingual_latent_pca_language_align003_seed000.pdf
paper/figures/bilingual_latent_pca_language_align003_seed000.png
paper/figures/bilingual_latent_pca_depth_align003_seed000.pdf
paper/figures/bilingual_latent_pca_depth_align003_seed000.png
paper/figures/monolingual_latent_pca_depth_reg005_seed000.pdf
paper/figures/monolingual_latent_pca_depth_reg005_seed000.png
paper/figures/family_case_distance_matrix.pdf
paper/figures/family_case_distance_matrix.png
paper/figures/family_case_distance_matrix_data.csv
```

Metric/statistic reproduction outputs:

```text
reports/paper_reproducibility/reproduced/reproduced_tfidf_values.csv
reports/paper_reproducibility/reproduced/reproduced_jaccard_values.csv
reports/paper_reproducibility/reproduced/reproduced_euclidean_distances.csv
reports/paper_reproducibility/reproduced/reproduced_same_id_and_relation_distances.csv
reports/paper_reproducibility/reproduced/reproduced_values_vs_paper.csv
reports/paper_reproducibility/reproduced/reproduced_figure_manifest.csv
reports/paper_reproducibility/reproduced/reproduction_manifest.json
```

Paper-facing summaries:

```text
paper/monolingual_results_summary.txt
paper/bilingual_results_summary.txt
```

## Optional Empirical Audit

The canonical empirical audit is not required for routine figure/statistic
regeneration, but it verifies retained-experiment definitions and secondary
diagnostics behind the paper:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/empirical_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_empirical_audit.py
```

Outputs are written to:

```text
reports/empirical_audit/
```

## Validation Checks

```bash
python3 - <<'PY'
import csv
from pathlib import Path

base = Path("reports/paper_reproducibility/reproduced")
rows = list(csv.DictReader((base / "reproduced_values_vs_paper.csv").open()))
print("value checks:", len(rows))
print("mismatches:", sum(row["status"] != "match" for row in rows))
print("TF-IDF:", sum(row["metric_family"] == "TF-IDF" for row in rows))
print("Jaccard:", sum(row["metric_family"] == "Jaccard" for row in rows))
print("Euclidean:", sum(row["metric_family"] == "Euclidean" for row in rows))

figures = sorted(path.name for path in Path("paper/figures").glob("*") if path.is_file())
print("paper figure files:", len(figures))
for name in figures:
    print(name)
PY
```
