"""Focused tests for the isolated temporal v4 controls runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_temporal_models_v4 as temporal


def small_cohort():
    return {
        "name": "selection",
        "user_ids": np.asarray([10, 20, 30, 40, 50], dtype=np.uint64),
        "target_ids": np.asarray([100, 200, 300, 400, 500], dtype=np.int64),
        "target_candidates": np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        "n_total": np.asarray([0, 2, 4, 5, 6], dtype=np.int64),
        "n_positive": np.asarray([0, 1, 0, 1, 2], dtype=np.int64),
        "hashes": {"temporal_protocol_sha256": "protocol", "temporal_manifest_sha256": "manifest"},
    }


class TemporalModelsV4Tests(unittest.TestCase):
    def test_training_batch_accepts_arrow_boolean_and_rejects_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.parquet'
            pq.write_table(pa.table({'user_key':pa.array([1,2],type=pa.uint64()),
                'app_id':pa.array([100,200],type=pa.int64()),'positive':[True,True]}),path)
            batches=list(temporal._positive_training_batches(path))
            self.assertEqual(sum(b.num_rows for b in batches),2)
            pq.write_table(pa.table({'user_key':pa.array([1,2],type=pa.uint64()),
                'app_id':pa.array([100,200],type=pa.int64()),'positive':[True,False]}),path)
            with self.assertRaises(ValueError):list(temporal._positive_training_batches(path))

    def test_temporal_ndcg_uses_float64_for_saved_int8_ranks(self):
        cohort = small_cohort()
        ranks = np.asarray([0, 0, 0, 3, 10], dtype=np.int8)
        result = temporal._rank_metrics(ranks, np.full(5, 10, dtype=np.int8),
                                       cohort, np.full(5, 100))
        expected = (.5 + 1 / np.log2(11.)) / 2
        self.assertAlmostEqual(result['primary']['ndcg'], expected, places=15)
        paired = temporal.paired_comparison(np.zeros(5, dtype=np.int8), ranks, cohort, reps=20)
        self.assertAlmostEqual(paired['game_macro']['NDCG@10']['delta'], expected, places=15)

    def test_primary_requires_both_total_and_positive_history(self):
        cohort = small_cohort()
        np.testing.assert_array_equal(temporal.primary_mask(cohort), [False, False, False, True, True])
        # Cohort stays intact: all five target eligible users and their short/empty histories remain.
        self.assertEqual(len(cohort["user_ids"]), 5)

    def test_activity_inverse_degree_scale_matches_definition(self):
        scale = temporal.activity_inverse_degree_row_scale(np.asarray([1, 4, 16]))
        np.testing.assert_allclose(scale, [1.0, 0.5, 0.25], rtol=0, atol=1e-7)
        with self.assertRaises(ValueError):
            temporal.activity_inverse_degree_row_scale(np.asarray([2, 0]))

    def test_knn_builder_keeps_full_candidate_by_input_shape_and_no_self_neighbors(self):
        apps = np.asarray([100, 200, 300], dtype=np.int64)
        x = sparse.csr_matrix(np.asarray([
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.uint8))
        cols = np.asarray([0, 1], dtype=np.int32)
        binary = temporal._build_similarity(x, apps, cols, "binary", neighbors=100, block_size=1)
        inverse = temporal._build_similarity(x, apps, cols, "activity_inverse_degree", neighbors=100, block_size=2)
        self.assertEqual(binary.shape, (2, 3))
        self.assertEqual(inverse.shape, (2, 3))
        for matrix, own in ((binary, cols[0]), (inverse, cols[0])):
            self.assertFalse(np.any(matrix.indices[matrix.indptr[0]:matrix.indptr[1]] == own))
            self.assertTrue(np.all(np.isfinite(matrix.data)))
        # Candidate 100 co-occurs most with 200; App IDs resolve equal ties.
        first = binary.indices[binary.indptr[0]:binary.indptr[1]]
        self.assertEqual(first.tolist(), [1, 2])

    def test_score_sanitizer_uses_t0_availability_and_all_visible_exclusions(self):
        raw = np.asarray([[2.0, 3.0, -1.0, 4.0]])
        exclusions = sparse.csr_matrix([[1, 0, 0, 1]], dtype=np.uint8)
        got = temporal._sanitize_scores(raw, exclusions, np.asarray([True, False, True, True]))
        np.testing.assert_array_equal(got, [[0.0, 0.0, 0.0, 0.0]])
        raw = np.asarray([[2.0, 3.0, 5.0, 4.0]])
        got = temporal._sanitize_scores(raw, exclusions, np.asarray([True, False, True, True]))
        np.testing.assert_array_equal(got, [[0.0, 0.0, 5.0, 0.0]])

    def test_metrics_report_short_and_zero_positive_strata_alongside_primary(self):
        cohort = small_cohort()
        summary = temporal._rank_metrics(
            np.asarray([0, 1, 0, 0, 2], dtype=np.int8),
            np.asarray([0, 3, 1, 10, 10], dtype=np.int8),
            cohort,
            np.asarray([0, 4, 10, 100, 1_000], dtype=np.int64),
        )
        self.assertEqual(summary["all_eligible_users"], 5)
        self.assertEqual(summary["primary"]["users"], 2)
        self.assertEqual(summary["primary"]["recall"], 0.5)
        self.assertEqual(summary["by_total_history"]["zero-input"]["users"], 1)
        self.assertEqual(summary["by_total_history"]["1-2"]["users"], 1)
        self.assertEqual(summary["by_total_history"]["3-4"]["users"], 1)
        self.assertEqual(summary["zero-positive-input"]["users"], 2)

    def test_paired_ci_uses_only_primary_users_and_retains_pairing(self):
        cohort = small_cohort()
        result = temporal.paired_comparison(
            np.asarray([0, 1, 0, 0, 3], dtype=np.int8),
            np.asarray([0, 0, 1, 2, 3], dtype=np.int8),
            cohort,
            reps=50,
        )
        self.assertEqual(result["primary_users"], 2)
        self.assertEqual(sum(result["hit_overlap"].values()), 2)
        self.assertIn("ci95", result["game_macro"]["NDCG@10"])

    def test_evaluator_refuses_confirmation_before_creating_output(self):
        cohort = small_cohort()
        cohort["name"] = "confirmation"
        class NeverScore:
            def score(self, q):
                raise AssertionError("confirmation cohort must be rejected before scoring")
        catalog = {"candidate_ids": np.arange(5), "available": np.ones(5, dtype=bool), "hashes": {}}
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "confirmation"
            with self.assertRaisesRegex(ValueError, "selection only"):
                temporal._evaluate_model("binary_item_knn_k100", NeverScore(), cohort, catalog,
                                         np.zeros(5, dtype=np.int64), out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
