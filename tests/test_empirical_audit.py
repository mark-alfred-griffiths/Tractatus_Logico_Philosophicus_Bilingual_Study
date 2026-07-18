from __future__ import annotations

import math
import unittest

import torch

from tools.empirical_audit import (
    BILINGUAL_ROOT,
    Study,
    compare_configs,
    cross_language_metrics,
    lambda_tag,
    load_cached_latents,
    paired_bootstrap_ci,
    paired_effect_size,
    sign_flip_p_value,
)


class EmpiricalAuditTests(unittest.TestCase):
    def test_lambda_tag_mapping(self) -> None:
        self.assertEqual(lambda_tag(0.00), "align000")
        self.assertEqual(lambda_tag(0.03), "align003")
        self.assertEqual(lambda_tag(0.10), "align010")
        self.assertEqual(lambda_tag(0.30), "align030")
        self.assertEqual(lambda_tag(1.00), "align100")

    def test_cross_language_metric_definitions(self) -> None:
        z = torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [1.0, 0.0],
                [1.0, 2.0],
            ]
        )
        metadata = [
            {"id": "a", "language": "en", "index": 1},
            {"id": "a", "language": "de", "index": 1},
            {"id": "b", "language": "en", "index": 2},
            {"id": "b", "language": "de", "index": 2},
        ]
        rows = [
            {"id": "a", "parent_id": None},
            {"id": "b", "parent_id": "a"},
        ]
        metrics = cross_language_metrics(z, metadata, rows)
        self.assertEqual(metrics["same_id_pair_count_directional"], 4.0)
        self.assertAlmostEqual(metrics["mean_same_id_cross_language_distance"], 3.5)
        self.assertAlmostEqual(metrics["mean_same_id_cross_language_mse"], 7.25)
        self.assertAlmostEqual(metrics["cross_language_top1_id_accuracy"], 0.5)
        self.assertGreater(metrics["cross_language_mrr"], 0.5)

    def test_deterministic_cached_latent_loading(self) -> None:
        study = Study("align003", 0.03, BILINGUAL_ROOT / "align003")
        z1, metadata1 = load_cached_latents(study, 0)
        z2, metadata2 = load_cached_latents(study, 0)
        self.assertTrue(torch.equal(z1, z2))
        self.assertEqual(metadata1, metadata2)
        self.assertEqual(tuple(z1.shape), (1052, 8))

    def test_pair_config_diff_is_only_lambda(self) -> None:
        studies = [
            Study("align000", 0.0, BILINGUAL_ROOT / "align000"),
            Study("align003", 0.03, BILINGUAL_ROOT / "align003"),
        ]
        comparison = compare_configs(studies)
        self.assertEqual(comparison["align003"]["non_lambda_differences_from_align000"], [])

    def test_exact_sign_flip_and_effect_size(self) -> None:
        diffs = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(sign_flip_p_value(diffs), 0.25)
        self.assertAlmostEqual(paired_effect_size(diffs), 2.0)
        low1, high1 = paired_bootstrap_ci(diffs, resamples=100, seed=123)
        low2, high2 = paired_bootstrap_ci(diffs, resamples=100, seed=123)
        self.assertTrue(math.isfinite(low1))
        self.assertEqual((low1, high1), (low2, high2))


if __name__ == "__main__":
    unittest.main()
