#!/usr/bin/env python3
"""Write canonical monolingual and bilingual paper result summaries.

The summaries are derived from retained seed-sweep metrics and generated audit
CSVs. This script does not train models and does not write under `runs/` except
for reading the existing regenerated summary artefacts.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RUNS = ROOT / "runs" / "seed_sweeps"
EMPIRICAL = ROOT / "reports" / "empirical_audit"
REPRODUCED = ROOT / "reports" / "paper_reproducibility" / "reproduced"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def fmt_pm(mean: float, sd: float) -> str:
    return f"{fmt(mean)}+/-{fmt(sd)}"


def metric(summary: dict, name: str) -> tuple[float, float]:
    stats = summary["metrics"][name]
    return float(stats["mean"]), float(stats["std"])


def aggregate(rows: Iterable[dict[str, str]], field: str) -> tuple[float, float]:
    vals = [float(row[field]) for row in rows]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def row_for(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def structural_references() -> dict[str, dict[str, float | int]]:
    rows = load_json(ROOT / "tractatus_structure_latents" / "data" / "tractatus.json")
    n = len(rows)
    parent_labels = [row["parent_id"] if row["parent_id"] is not None else "__ROOT__" for row in rows]
    depth_labels = [row["depth"] for row in rows]
    child_counts = [row["child_count"] for row in rows]
    return {
        "parent": {
            "observed_class_count": len(set(parent_labels)),
            "majority_accuracy": Counter(parent_labels).most_common(1)[0][1] / n,
        },
        "depth": {
            "observed_class_count": len(set(depth_labels)),
            "majority_accuracy": Counter(depth_labels).most_common(1)[0][1] / n,
        },
        "next": {
            "observed_class_count": n,
            "uniform_random_accuracy_observed_classes": 1 / n,
        },
        "child_count": {
            "zero_proportion": child_counts.count(0) / n,
            "constant_zero_mae": sum(abs(value) for value in child_counts) / n,
        },
    }


def write_monolingual() -> None:
    summary = load_json(RUNS / "monolingual_split_24_8_reg005" / "summaries" / "summary.json")
    child_rows = [row for row in read_csv(EMPIRICAL / "child_count_seed_metrics.csv") if row["condition"] == "monolingual_split_24_8_reg005" and row["language"] == "all"]
    euclidean_rows = read_csv(REPRODUCED / "reproduced_euclidean_distances.csv")
    family_metrics = {row["metric"]: row for row in euclidean_rows if row.get("metric", "").startswith("family 2.2")}
    child_mae, child_mae_sd = aggregate(child_rows, "mae")
    child_rmse, child_rmse_sd = aggregate(child_rows, "rmse")
    baseline_by_task = structural_references()

    parent_mu, parent_sd = metric(summary, "parent_accuracy")
    depth_mu, depth_sd = metric(summary, "depth_accuracy")
    next_mu, next_sd = metric(summary, "next_accuracy")
    rec_mu, rec_sd = metric(summary, "reconstruction_loss")
    ppl_mu, ppl_sd = metric(summary, "perplexity")
    kl_mu, kl_sd = metric(summary, "kl_structure")
    sibling_mu, sibling_sd = metric(summary, "mean_sibling_distance")
    pc_mu, pc_sd = metric(summary, "mean_parent_child_distance")
    unrelated_mu, unrelated_sd = metric(summary, "mean_unrelated_distance")

    text = f"""Monolingual Results Summary: English Split-Latent Tractatus Study

Canonical paper state
---------------------
This summary was regenerated for the canonical paper from retained artefacts. It
summarises the English-only split-latent configuration and proposition-family
diagnostics.

Canonical inputs
----------------
- runs/seed_sweeps/monolingual_split_24_8_reg005/metrics/seed*.metrics.json
- runs/seed_sweeps/monolingual_split_24_8_reg005/summaries/summary.json
- reports/empirical_audit/child_count_seed_metrics.csv
- reports/empirical_audit/child_count_distribution.csv
- reports/paper_reproducibility/reproduced/reproduced_euclidean_distances.csv
- paper/figures/family_case_distance_matrix_data.csv

Model and retained sweep
------------------------
- z_text: 24 dimensions
- z_structure: 8 dimensions
- beta_text: 0.01
- beta_structure: 0.05
- seeds: 0..9
- dataset: tractatus_structure_latents/data/tractatus.json

Core ten-seed summary
---------------------
parent accuracy:       {fmt_pm(parent_mu, parent_sd)}
depth accuracy:        {fmt_pm(depth_mu, depth_sd)}
next accuracy:         {fmt_pm(next_mu, next_sd)}
child-count MAE:       {fmt_pm(child_mae, child_mae_sd)}
child-count RMSE:      {fmt_pm(child_rmse, child_rmse_sd)}
reconstruction loss:   {fmt_pm(rec_mu, rec_sd)}
perplexity:            {fmt_pm(ppl_mu, ppl_sd)}
KL structure:          {fmt_pm(kl_mu, kl_sd)}
sibling distance:      {fmt_pm(sibling_mu, sibling_sd)}
parent-child distance: {fmt_pm(pc_mu, pc_sd)}
unrelated distance:    {fmt_pm(unrelated_mu, unrelated_sd)}

Task reference levels
---------------------
parent classes:        {int(baseline_by_task['parent']['observed_class_count'])} observed, majority reference {fmt(float(baseline_by_task['parent']['majority_accuracy']))}
depth classes:         {int(baseline_by_task['depth']['observed_class_count'])} observed, majority reference {fmt(float(baseline_by_task['depth']['majority_accuracy']))}
successor classes:     {int(baseline_by_task['next']['observed_class_count'])} observed, random Top-1 reference {fmt(float(baseline_by_task['next']['uniform_random_accuracy_observed_classes']))}
child-count zero rate: {fmt(float(baseline_by_task['child_count']['zero_proportion']))}, constant-zero MAE {fmt(float(baseline_by_task['child_count']['constant_zero_mae']))}

Proposition-family diagnostics
------------------------------
family 2.2 mean pairwise distance: {fmt_pm(float(family_metrics['family 2.2 mean pairwise distance']['value']), float(family_metrics['family 2.2 mean pairwise distance']['sample_sd']))}
matched-random mean distance:      5.2159
matched-null percentile mean:      {fmt(float(family_metrics['family 2.2 matched-null percentile mean']['value']))}
family 2.2 outlier seed count:     {int(float(family_metrics['family 2.2 outlier seed count']['value']))}/10
Figure 7 source data:              paper/figures/family_case_distance_matrix_data.csv

Canonical figure artifacts
--------------------------
- paper/figures/monolingual_latent_pca_depth_reg005_seed000.png
- paper/figures/monolingual_latent_pca_depth_reg005_seed000.pdf
- paper/figures/family_case_distance_matrix.png
- paper/figures/family_case_distance_matrix.pdf
- paper/figures/family_case_distance_matrix_data.csv

Interpretation guardrail
------------------------
These are retained-corpus diagnostics. The model uses numbering-derived targets
as supervision, so the result is not unsupervised discovery of the Tractatus
hierarchy and not a model interpretation of Wittgenstein.
"""
    (PAPER / "monolingual_results_summary.txt").write_text(text, encoding="utf-8")


def write_bilingual() -> None:
    rows = load_json(RUNS / "bilingual_alignment_lambda_sweep" / "summaries" / "summary.json")
    rows = sorted(rows, key=lambda row: float(row["lambda_language_alignment"]))
    tfidf = read_csv(REPRODUCED / "reproduced_tfidf_values.csv")
    jaccard = read_csv(REPRODUCED / "reproduced_jaccard_values.csv")
    relation = read_csv(REPRODUCED / "reproduced_same_id_and_relation_distances.csv")
    paired = read_csv(EMPIRICAL / "lambda_000_vs_003_summary.csv")

    labels = [("0.00", "000"), ("0.03", "003"), ("0.10", "010"), ("0.30", "030"), ("1.00", "100")]
    by_tag = {str(row["tag"]).zfill(3): row for row in rows}

    def table_line(tag: str, metrics: list[str]) -> str:
        row = by_tag[tag]
        return " ".join(fmt_pm(row[m], row[f"{m}_std"]).rjust(15) for m in metrics)

    exact_j = row_for(jaccard, method="Exact-token Jaccard")
    word_t = row_for(tfidf, method="Word TF-IDF")
    char_t = row_for(tfidf, method="Character 3-5 TF-IDF")
    random_j = row_for(jaccard, condition="Random", comparison="Independent sets")
    cross_000 = row_for(jaccard, condition="0.00", comparison="Cross-direction")
    within_000 = row_for(jaccard, condition="0.00", comparison="Within-language")
    cross_003 = row_for(jaccard, condition="0.03", comparison="Cross-direction")
    within_003 = row_for(jaccard, condition="0.03", comparison="Within-language")
    same_003 = row_for(paired, metric="mean_same_id_cross_language_distance")
    top1_003 = row_for(paired, metric="cross_language_top1_id_accuracy")
    mrr_003 = row_for(paired, metric="cross_language_mrr")

    relation_lines = []
    for lam, _tag in labels:
        subset = [row for row in relation if row["condition"] == lam]
        values = {row["metric"]: fmt_pm(float(row["mean"]), float(row["sample_sd"])) for row in subset}
        relation_lines.append(
            f"{lam:<7} {values['same-ID distance']:<15} {values['sibling']:<15} {values['parent-child']:<15} {values['cross-language parent-child']:<30} {values['unrelated']:<15}"
        )

    text = f"""Bilingual Companion Paper Results Summary
==========================================

Canonical source files
----------------------
- runs/seed_sweeps/bilingual_alignment_lambda_sweep/summaries/summary.json
- runs/seed_sweeps/bilingual_alignment_lambda_sweep/align000/metrics/seed*.metrics.json
- runs/seed_sweeps/bilingual_alignment_lambda_sweep/align003/metrics/seed*.metrics.json
- runs/seed_sweeps/bilingual_alignment_lambda_sweep/align010/metrics/seed*.metrics.json
- runs/seed_sweeps/bilingual_alignment_lambda_sweep/align030/metrics/seed*.metrics.json
- runs/seed_sweeps/bilingual_alignment_lambda_sweep/align100/metrics/seed*.metrics.json
- reports/paper_reproducibility/reproduced/reproduced_tfidf_values.csv
- reports/paper_reproducibility/reproduced/reproduced_jaccard_values.csv
- reports/paper_reproducibility/reproduced/reproduced_same_id_and_relation_distances.csv
- tractatus_structure_latents/data/tractatus_bilingual.json

Canonical sweep
---------------
- Alignment values: 0.00, 0.03, 0.10, 0.30, 1.00
- Seeds per alignment value: 10, seeds 0..9
- Metric figures report means with +/- one-standard-deviation error bars.
- Tables report mean +/- sample SD across seeds.
- Retrieval uses 526 candidates in each direction.

Core bilingual seed-sweep metrics
---------------------------------
lambda  seeds  parent          depth           next            top1            mrr
{chr(10).join(f"{lam:<7} 10     {table_line(tag, ['parent_accuracy', 'depth_accuracy', 'next_accuracy', 'cross_language_top1_id_accuracy', 'cross_language_mrr'])}" for lam, tag in labels)}

Directional cross-language retrieval
------------------------------------
lambda  de->en top1     en->de top1     de->en mrr      en->de mrr
{chr(10).join(f"{lam:<7} {table_line(tag, ['cross_language_top1_id_accuracy_de_to_en', 'cross_language_top1_id_accuracy_en_to_de', 'cross_language_mrr_de_to_en', 'cross_language_mrr_en_to_de'])}" for lam, tag in labels)}

By-language structure prediction
--------------------------------
lambda  parent_de       parent_en       depth_de        depth_en        next_de         next_en
{chr(10).join(f"{lam:<7} {table_line(tag, ['parent_accuracy_by_language.de', 'parent_accuracy_by_language.en', 'depth_accuracy_by_language.de', 'depth_accuracy_by_language.en', 'next_accuracy_by_language.de', 'next_accuracy_by_language.en'])}" for lam, tag in labels)}

Reconstruction and latent regularization diagnostics
----------------------------------------------------
lambda  reconstruction  perplexity      kl_text         kl_structure     same_id_distance
{chr(10).join(f"{lam:<7} {table_line(tag, ['reconstruction_loss', 'perplexity', 'kl_text', 'kl_structure', 'mean_same_id_cross_language_distance'])}" for lam, tag in labels)}

Structure-latent distance diagnostics
-------------------------------------
lambda  same_id         sibling         parent_child    cross_language_parent_child    unrelated
{chr(10).join(relation_lines)}

Exact-input lexical retrieval references
----------------------------------------
method                  top1            mrr             notes
Exact-token Jaccard     {fmt(float(exact_j['top1_mean']))}          {fmt(float(exact_j['mrr_mean']))}          exact model input spans, IDs excluded
Word TF-IDF             {fmt(float(word_t['top1_mean']))}          {fmt(float(word_t['mrr_mean']))}          shared word feature space
Character 3-5 TF-IDF    {fmt(float(char_t['top1_mean']))}          {fmt(float(char_t['mrr_mean']))}          character n-grams

Wider-neighbourhood Jaccard at k=10
-----------------------------------
condition comparison       jaccard
0.00      cross-direction  {fmt_pm(float(cross_000['mean_jaccard']), float(cross_000['sample_sd']))}
0.00      within-language  {fmt_pm(float(within_000['mean_jaccard']), float(within_000['sample_sd']))}
0.03      cross-direction  {fmt_pm(float(cross_003['mean_jaccard']), float(cross_003['sample_sd']))}
0.03      within-language  {fmt_pm(float(within_003['mean_jaccard']), float(within_003['sample_sd']))}
random    independent sets {fmt(float(random_j['mean_jaccard']))}

Paired lambda 0.03 minus 0.00 diagnostics
-----------------------------------------
top1 difference:          {fmt(float(top1_003['mean_difference_003_minus_000']))} [{fmt(float(top1_003['bootstrap_ci_95_low']))}, {fmt(float(top1_003['bootstrap_ci_95_high']))}]
MRR difference:           {fmt(float(mrr_003['mean_difference_003_minus_000']))} [{fmt(float(mrr_003['bootstrap_ci_95_low']))}, {fmt(float(mrr_003['bootstrap_ci_95_high']))}]
same-ID distance change:  {fmt(float(same_003['mean_difference_003_minus_000']))} [{fmt(float(same_003['bootstrap_ci_95_low']))}, {fmt(float(same_003['bootstrap_ci_95_high']))}]

Canonical figure artifacts
--------------------------
- paper/figures/bilingual_alignment_retrieval_sweep.png
- paper/figures/bilingual_alignment_retrieval_sweep.pdf
- paper/figures/bilingual_structure_accuracy_sweep.png
- paper/figures/bilingual_structure_accuracy_sweep.pdf
- paper/figures/bilingual_retrieval_structure_tradeoff.png
- paper/figures/bilingual_retrieval_structure_tradeoff.pdf
- paper/figures/bilingual_reconstruction_sweep.png
- paper/figures/bilingual_reconstruction_sweep.pdf
- paper/figures/bilingual_latent_pca_language_align003_seed000.png
- paper/figures/bilingual_latent_pca_language_align003_seed000.pdf
- paper/figures/bilingual_latent_pca_depth_align003_seed000.png
- paper/figures/bilingual_latent_pca_depth_align003_seed000.pdf

Interpretation guardrail
------------------------
These results are retained-corpus diagnostics, not held-out generalisation. The
lambda=0.00 condition has no direct pairwise same-ID loss, but German and
English still share model parameters, target label spaces, mixed-corpus
training, and possible surface cues. Retrieval and neighbourhood overlap do not
establish semantic equivalence or language-invariant logic.
"""
    (PAPER / "bilingual_results_summary.txt").write_text(text, encoding="utf-8")


def main() -> None:
    write_monolingual()
    write_bilingual()
    print("wrote paper/monolingual_results_summary.txt")
    print("wrote paper/bilingual_results_summary.txt")


if __name__ == "__main__":
    main()
