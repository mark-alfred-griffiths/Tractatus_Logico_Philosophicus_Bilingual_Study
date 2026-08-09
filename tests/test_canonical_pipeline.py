from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.canonical_experiments import canonical_phase3_ids
from tools.export_canonical_evidence import export
from tools.phase1_ablations import lexical_reference_metrics
from tools.phase2_family_holdout import lexical_references, text_len
from tractatus_structure_latents.training.data import TractatusDataset
from tractatus_structure_latents.training.train_vae import PairedLanguageBatchSampler, set_seed


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dsh_validation"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_value(path: Path, condition: str, metric: str, field: str = "mean") -> float:
    for row in read_csv(path):
        if row["condition"] == condition and row["metric"] == metric:
            return float(row[field])
    raise KeyError((condition, metric))


class CanonicalPipelineTests(unittest.TestCase):
    def test_tractatus_dataset_loads_legacy_text_rows(self) -> None:
        rows = [
            {"id": "1", "parent_id": None, "next_id": "2", "depth": 0, "child_count": 1, "text": "The world is all that is the case."},
            {"id": "2", "parent_id": "1", "next_id": None, "depth": 1, "child_count": 0, "text": "What is the case is the existence of atomic facts."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            dataset = TractatusDataset(path)

        self.assertEqual(dataset.languages, ["en"])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.samples[0]["language"], "en")
        self.assertEqual(dataset.samples[0]["text"], rows[0]["text"])

    def test_tractatus_dataset_loads_multilingual_text_rows(self) -> None:
        rows = [
            {"id": "1", "parent_id": None, "next_id": None, "depth": 0, "child_count": 0, "texts": {"de": "Die Welt ist alles.", "en": "The world is everything."}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multilingual.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            dataset = TractatusDataset(path)

        self.assertEqual(dataset.languages, ["de", "en"])
        self.assertEqual(len(dataset), 2)
        self.assertEqual([(sample["language"], sample["text"]) for sample in dataset.samples], [("de", "Die Welt ist alles."), ("en", "The world is everything.")])

    def test_phase_lexical_helpers_accept_legacy_english_rows(self) -> None:
        rows = [
            {"id": "1", "parent_id": None, "next_id": "2", "depth": 0, "child_count": 1, "text": "The world is all that is the case."},
            {"id": "2", "parent_id": "1", "next_id": None, "depth": 1, "child_count": 0, "text": "Atomic facts are independent of one another."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            dataset = TractatusDataset(path)

        self.assertGreater(text_len(rows[0]), 0.0)
        self.assertEqual(lexical_reference_metrics(dataset), {})
        self.assertEqual(lexical_references(rows, {"1"}, {"2"}), {})

    def test_paired_sampler_coverage_and_language_integrity(self) -> None:
        rows = [
            {"id": "1", "parent_id": None, "next_id": "2", "depth": 0, "child_count": 1, "texts": {"de": "eins", "en": "one"}},
            {"id": "2", "parent_id": "1", "next_id": None, "depth": 1, "child_count": 0, "texts": {"de": "zwei", "en": "two"}},
            {"id": "3", "parent_id": "1", "next_id": None, "depth": 1, "child_count": 0, "texts": {"de": "drei", "en": "three"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            dataset = TractatusDataset(path, languages=["de", "en"])
            sampler = PairedLanguageBatchSampler(dataset, batch_size=4, generator=set_seed(123))
            observed: set[str] = set()
            for batch in sampler:
                by_id: dict[str, set[str]] = {}
                for sample_i in batch:
                    sample = dataset.samples[sample_i]
                    by_id.setdefault(str(sample["id"]), set()).add(str(sample["language"]))
                for prop_id, languages in by_id.items():
                    self.assertEqual(languages, {"de", "en"})
                    observed.add(prop_id)
            self.assertEqual(observed, {"1", "2", "3"})
            self.assertEqual(sampler.corpus_pair_count, 3)

    def test_canonical_registry_contains_only_paired_alignment(self) -> None:
        self.assertTrue(all(label.startswith("paired_") for label in canonical_phase3_ids()))
        self.assertFalse(any(label.startswith("random_") for label in canonical_phase3_ids()))

    def test_canonical_phase3_report_contains_only_paired_ids(self) -> None:
        rows = read_csv(RESULTS / "phase3_controlled_alignment" / "phase3_seed_results.csv")
        labels = {row.get("experiment_id", row.get("label", "")) for row in rows}
        self.assertTrue(labels <= canonical_phase3_ids())
        self.assertFalse(any(label.startswith("random_") for label in labels))

    def test_canonical_export_excludes_removed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export(Path(tmp), dry_run=True, quiet=True)
        joined = "\n".join(row["path"] for row in manifest["files"])
        self.assertNotIn("random_full_model", joined)
        self.assertNotIn("paired_versus_random", joined)
        self.assertNotIn("bilingual_alignment_lambda_sweep", joined)

    def test_phase1_regression_values(self) -> None:
        path = RESULTS / "phase1_ablations" / "phase1_ablation_summary.csv"
        self.assertTrue(math.isclose(summary_value(path, "full_model", "structure_cross_language_top1"), 0.9360, abs_tol=5e-5))
        self.assertTrue(math.isclose(summary_value(path, "full_model", "structure_cross_language_mrr"), 0.9657, abs_tol=5e-5))
        self.assertTrue(math.isclose(summary_value(path, "successor_only", "structure_cross_language_top1"), 0.9944, abs_tol=5e-5))
        self.assertTrue(math.isclose(summary_value(path, "shuffled_joint_targets", "structure_sibling_vs_unrelated_contrast"), 0.0330, abs_tol=5e-5))

    def test_phase2_regression_values(self) -> None:
        path = RESULTS / "phase2_family_holdout" / "phase2_summary.csv"
        self.assertTrue(math.isclose(summary_value(path, "full_model", "structure_test_candidates_top1"), 0.0739, abs_tol=5e-5))
        self.assertTrue(math.isclose(summary_value(path, "no_successor", "structure_test_candidates_top1"), 0.0811, abs_tol=5e-5))
        self.assertTrue(math.isclose(summary_value(path, "reconstruction_only", "structure_test_candidates_mrr"), 0.0665, abs_tol=5e-5))
        self.assertTrue(math.isclose(summary_value(path, "full_model", "reference_char_3_5_tfidf_test_top1"), 0.5619, abs_tol=5e-5))

    def test_phase3_regression_values_and_coverage(self) -> None:
        path = RESULTS / "phase3_controlled_alignment" / "phase3_summary.csv"
        self.assertTrue(math.isclose(summary_value(path, "full_model", "structure_cross_language_top1"), 0.9443, abs_tol=5e-5))
        rows = read_csv(path)
        value = next(float(row["mean"]) for row in rows if row["condition"] == "full_model" and row["lambda_language_alignment"] == "1.0" and row["metric"] == "structure_cross_language_top1")
        self.assertTrue(math.isclose(value, 0.9959, abs_tol=5e-5))
        coverage = read_csv(RESULTS / "phase3_controlled_alignment" / "phase3_pair_coverage.csv")
        self.assertTrue(all(float(row["min_pair_coverage"]) == 1.0 and float(row["max_pair_coverage"]) == 1.0 for row in coverage))

    def test_frozen_case_manifest_hash(self) -> None:
        text = (RESULTS / "phase4_case_studies" / "candidate_manifest_pre_text.sha256").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("c4c6ac3c5473f47d181fdb8f1e155eab2d938a9bd674f2435c7fb48bc29e5ffc"))

    def test_family_fold_integrity(self) -> None:
        rows = read_csv(RESULTS / "phase2_family_holdout" / "phase2_fold_manifest.csv")
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        family_to_folds: dict[str, set[str]] = {}
        for row in rows:
            family_to_folds.setdefault(row["family_id"], set()).add(row["fold"])
        self.assertTrue(all(len(folds) == 1 for folds in family_to_folds.values()))


if __name__ == "__main__":
    unittest.main()
