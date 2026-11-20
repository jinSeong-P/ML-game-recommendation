"""Unit checks for the reporting-only v4 supplemental evaluator."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

import evaluate_supplemental_v4 as supplemental
from build_experiments_v4_split import uniform_target


def make_toy_cohort():
    # Catalog column order is ascending by App ID; 50 is a non-Linux input.
    apps = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    linux = np.array([True, True, True, True, False])
    pools = ([10, 20, 30], [20, 40])
    user_ids = np.array([101, 202], dtype=np.uint64)
    target_ids = np.array([
        uniform_target(int(user_ids[i]), list(pools[i]), 20261108, "selection")
        for i in range(2)
    ], dtype=np.int64)
    visible = [sorted(set(pool) - {int(target_ids[i])}) for i, pool in enumerate(pools)]
    visible_apps = visible[0] + visible[1]
    visible_rows = [0] * len(visible[0]) + [1] * len(visible[1])
    positive = sparse.csr_matrix(
        (np.ones(len(visible_apps), dtype=np.float32),
         (visible_rows, [int(np.searchsorted(apps, x)) for x in visible_apps])), shape=(2, 5)
    )
    negative = sparse.csr_matrix(
        (np.ones(2, dtype=np.float32), ([0, 1], [4, 4])), shape=(2, 5)
    )
    # Original selection query/exclusion inputs, including both positives and negatives.
    exclusion_cols = [int(np.searchsorted(apps[linux], x)) for x in visible_apps]
    exclusions = sparse.csr_matrix(
        (np.ones(len(visible_apps), dtype=np.float32), (visible_rows, exclusion_cols)), shape=(2, 4)
    )
    return apps, linux, dict(
        name="selection",
        user_ids=user_ids,
        positive=positive,
        negative=negative,
        exclusions=exclusions,
        target_ids=target_ids,
        target_candidates=np.searchsorted(apps[linux], target_ids),
        n_total=np.array([3, 2], dtype=np.int64),
        n_positive=np.array([2, 1], dtype=np.int64),
        hashes={"users.parquet": "users", "targets.parquet": "targets", "query.parquet": "query"},
    )


class SupplementalV4Tests(unittest.TestCase):
    def test_seed_20261108_recreates_original_target_and_query(self):
        apps, linux, cohort = make_toy_cohort()
        with patch.object(supplemental, "catalog", return_value=(apps, linux)):
            rebuilt = supplemental.build_seed_cohort(cohort, 20261108)
        np.testing.assert_array_equal(rebuilt["target_ids"], cohort["target_ids"])
        np.testing.assert_array_equal(rebuilt["positive"].toarray(), cohort["positive"].toarray())
        np.testing.assert_array_equal(rebuilt["exclusions"].toarray(), cohort["exclusions"].toarray())
        np.testing.assert_array_equal(rebuilt["n_positive"], [2, 1])
        np.testing.assert_array_equal(rebuilt["n_total"], [3, 2])
        np.testing.assert_array_equal(rebuilt["target_pool_size"], [3, 2])

    def test_target_swap_moves_original_target_back_into_query_and_removes_new_target(self):
        apps, linux, cohort = make_toy_cohort()
        with patch.object(supplemental, "catalog", return_value=(apps, linux)):
            sampled, _ = supplemental.sample_target_ids(cohort, 20261109)
            rebuilt = supplemental.build_seed_cohort(cohort, 20261109)
        np.testing.assert_array_equal(rebuilt["target_ids"], sampled)
        for row, target in enumerate(sampled):
            target_col = int(np.searchsorted(apps, target))
            candidate_col = int(np.searchsorted(apps[linux], target))
            self.assertEqual(float(rebuilt["positive"][row, target_col]), 0.0)
            self.assertEqual(float(rebuilt["exclusions"][row, candidate_col]), 0.0)
            self.assertEqual(int(rebuilt["positive"].getrow(row).sum()), int(cohort["n_positive"][row]))
            self.assertEqual(int(rebuilt["exclusions"].getrow(row).sum()), int(cohort["exclusions"].getrow(row).sum()))

    def test_hidden_negative_sampling_is_stable_and_retains_four_input_users(self):
        apps = np.array([10, 20, 30, 40], dtype=np.int64)
        linux = np.ones(4, dtype=bool)
        # Primary users are rows 0, 2, 3. Row 2 has no negative and is omitted;
        # row 0 and 3 each retain their sampled negative in no input mask.
        negative = sparse.csr_matrix(
            (np.ones(3, dtype=np.float32), ([0, 0, 3], [0, 1, 2])), shape=(4, 4)
        )
        exclusions = sparse.csr_matrix(
            (np.ones(6, dtype=np.float32),
             ([0, 0, 0, 2, 3, 3], [0, 1, 3, 2, 1, 2])), shape=(4, 4)
        )
        cohort = dict(
            name="selection",
            user_ids=np.array([100, 200, 300, 400], dtype=np.uint64),
            positive=sparse.csr_matrix((4, 4), dtype=np.float32),
            negative=negative,
            exclusions=exclusions,
            target_ids=np.array([40, 10, 20, 30], dtype=np.int64),
            target_candidates=np.array([3, 0, 1, 2], dtype=np.int64),
            n_total=np.array([5, 4, 5, 5], dtype=np.int64),
            n_positive=np.array([2, 1, 3, 2], dtype=np.int64),
            hashes={},
        )
        self.assertEqual(supplemental.hidden_negative_index(100, 7), supplemental.hidden_negative_index(100, 7))
        with patch.object(supplemental, "catalog", return_value=(apps, linux)):
            selected = supplemental.build_hidden_negative_cohort(cohort)
        np.testing.assert_array_equal(selected["user_ids"], [100, 400])
        np.testing.assert_array_equal(selected["n_total"], [5, 5])
        np.testing.assert_array_equal(selected["n_total_after_negative_hide"], [4, 4])
        self.assertEqual(len(selected["hidden_negative_ids"]), 2)
        for row, app_id in enumerate(selected["hidden_negative_ids"]):
            col = int(np.searchsorted(apps, app_id))
            self.assertEqual(float(selected["negative"][row, col]), 0.0)
            self.assertEqual(float(selected["exclusions"][row, col]), 0.0)

    def test_paired_bootstrap_uses_aligned_users(self):
        left = np.array([0.0, 1.0, 0.5, 0.0])
        right = np.array([1.0, 1.0, 0.0, 0.0])
        result = supplemental.paired_user_bootstrap(left, right, reps=100, seed=19)
        self.assertAlmostEqual(result["delta"], 0.125)
        self.assertEqual(len(result["ci95"]), 2)
        self.assertLessEqual(result["ci95"][0], result["ci95"][1])

    def test_prediction_writer_saves_ranked_recommendations_and_hashes(self):
        apps = np.array([10, 20, 30, 40], dtype=np.int64)
        linux = np.ones(4, dtype=bool)
        cohort = dict(
            name="selection",
            user_ids=np.array([100, 200], dtype=np.uint64),
            positive=sparse.csr_matrix((2, 4), dtype=np.float32),
            exclusions=sparse.csr_matrix((np.ones(2, dtype=np.float32), ([0, 1], [0, 1])), shape=(2, 4)),
            target_ids=np.array([30, 10], dtype=np.int64),
            n_total=np.array([5, 4], dtype=np.int64),
            n_positive=np.array([2, 1], dtype=np.int64),
        )
        raw = np.array([[1., 4., 3., 2.], [4., 3., 2., 1.]])

        def fake_scores(_model, current, begin, end):
            scores = raw[begin:end].copy()
            rows, cols = current["exclusions"][begin:end].nonzero()
            scores[rows, cols] = 0
            return scores

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(supplemental, "catalog", return_value=(apps, linux)), \
             patch.object(supplemental, "masked_scores", side_effect=fake_scores):
            folder = Path(tmp) / "binary_item_knn_k100" / "20261108"
            manifest = supplemental.evaluate_predictions(
                "binary_item_knn_k100", object(), cohort, cohort["target_ids"], folder,
                {"protocol": "test", "model": "binary_item_knn_k100"},
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["files"]["per_user.parquet"]["rows"], 2)
            self.assertEqual(manifest["files"]["recommendations.parquet"]["rows"], 6)
            per_user = pq.read_table(folder / "per_user.parquet").to_pydict()
            np.testing.assert_array_equal(per_user["target_rank"], [2, 1])
            recs = pq.read_table(folder / "recommendations.parquet").to_pydict()
            self.assertEqual(recs["app_id"][:3], [20, 30, 40])

            hidden_cohort = dict(
                cohort,
                n_total=np.array([5, 5], dtype=np.int64),
                hidden_negative_ids=np.array([20, 30], dtype=np.int64),
                n_total_after_negative_hide=np.array([4, 4], dtype=np.int64),
            )
            hidden_folder = Path(tmp) / "hidden_negative" / "binary_item_knn_k100"
            hidden_manifest = supplemental.evaluate_predictions(
                "binary_item_knn_k100", object(), hidden_cohort, cohort["target_ids"], hidden_folder,
                {"protocol": "test", "model": "binary_item_knn_k100", "hidden_negative": True},
                hidden_negative=True,
            )
            metric = hidden_manifest["metrics"]["hidden_negative"]
            self.assertEqual(metric["users"], 2)
            self.assertEqual(metric["hidden_negative_top10_users"], 2)
            self.assertEqual(metric["hidden_negative_top10_rate"], 1.0)
            hidden_per_user = pq.read_table(hidden_folder / "per_user.parquet").to_pydict()
            np.testing.assert_array_equal(hidden_per_user["hidden_negative_rank"], [1, 2])

    def test_averaged_seed_summary_resamples_users_after_seed_averaging(self):
        cohort = dict(
            n_total=np.array([5, 5, 2, 4]),
            target_ids=np.array([10, 10, 20, 20]),
        )
        outcomes = {}
        for model in supplemental.CONTROLS:
            outcomes[model] = {}
            for seed in supplemental.SEEDS:
                if model == supplemental.CONTROLS[0]:
                    ranks = np.array([1, 0, 5, 0], dtype=np.int8)
                elif model == supplemental.CONTROLS[1]:
                    ranks = np.array([0, 1, 0, 5], dtype=np.int8)
                else:
                    ranks = np.array([1, 1, 0, 0], dtype=np.int8)
                outcomes[model][seed] = (ranks, np.full(4, 10, dtype=np.int8))
        model_summary, paired, transformed = supplemental.averaged_seed_summary(outcomes, cohort)
        self.assertAlmostEqual(model_summary[supplemental.CONTROLS[0]]["primary_5_plus"]["mean_user_recall_at_10"], 0.5)
        comparison = paired[f"{supplemental.CONTROLS[1]}_minus_{supplemental.CONTROLS[0]}"]["primary_5_plus"]["recall_at_10"]
        self.assertAlmostEqual(comparison["delta"], 0.0)
        np.testing.assert_allclose(transformed[supplemental.CONTROLS[0]]["recall"], [1.0, 0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
