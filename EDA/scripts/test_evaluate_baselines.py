"""Hand-computed holdout, scoring, tie, zero-list and saved-output checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from eda_large_review_frank import digest
from evaluate_baselines import (checked_positions, diagnostic_controls, evaluate_model,
                                load_cohort, metrics, pair_matrix, rank_positive,
                                score_batch, support_buckets)


class EvaluationTests(unittest.TestCase):
    def test_topk_boundary_ties_use_app_id_and_reject_nonpositive(self):
        apps = np.arange(15, 0, -1)
        scores = np.ones(15)
        chosen = rank_positive(scores, apps)
        self.assertEqual(apps[chosen].tolist(), list(range(1, 11)))
        scores[:4] = [0, -1, 2, 3]
        self.assertEqual(apps[rank_positive(scores, apps)][:2].tolist(), [12, 13])
        self.assertEqual(rank_positive(np.zeros(15), apps).tolist(), [])
        self.assertEqual(rank_positive(scores, apps, 0).tolist(), [])
        with self.assertRaises(ValueError):
            rank_positive([float("nan")], [10])

    def test_exact_uint64_user_mapping_and_unknown_failfast(self):
        ids = np.array([76561197960265890, 76561197960265891], dtype=np.uint64)
        np.testing.assert_array_equal(checked_positions(ids[::-1], ids, "user"), [1, 0])
        with self.assertRaises(ValueError):
            checked_positions([90], np.array([10, 20]), "game")

    def test_duplicate_pairs_rejected(self):
        with self.assertRaises(ValueError):
            pair_matrix([0, 0], [1, 1], (1, 3), "query")

    def test_knn_divides_by_all_positive_inputs_including_unsupported(self):
        model = SimpleNamespace(similarities=sparse.csr_matrix([[0.8, 0, 0]]))
        query = sparse.csr_matrix([[1, 1, 1]])
        np.testing.assert_allclose(score_batch(model, query, np.array([3]), "item_knn"), [[0.8 / 3]])

    def test_tfidf_mean_profile_is_cosine_not_mean_candidate_cosines(self):
        model = SimpleNamespace(matrix=sparse.csr_matrix([[1, 0], [0, 1], [0, 0], [1, 0]], dtype=np.float32),
                                linux=np.array([False, False, False, True]))
        query = sparse.csr_matrix([[1, 1, 1, 0]], dtype=np.float32)
        np.testing.assert_allclose(score_batch(model, query, np.array([3]), "tfidf"), [[1 / np.sqrt(2)]], rtol=1e-6)

    def test_empty_lists_remain_in_denominator_and_single_target_ndcg(self):
        summary = metrics(np.array([1, 2, 0, 0]), np.array([10, 10, 0, 2]))
        self.assertEqual(summary["users"], 4)
        self.assertEqual(summary["Recall@10"], 0.5)
        self.assertAlmostEqual(summary["NDCG@10"], (1 + 1 / np.log2(3)) / 4)
        self.assertEqual(summary["empty_recommendation_users"], 1)
        self.assertEqual(summary["short_list_users"], 2)

    def test_support_zero_bucket_separate_and_train_threshold_ties(self):
        np.testing.assert_array_equal(support_buckets([0, 1, 5, 10, 20], [5, 10, 20]), [0, 1, 2, 3, 4])

    def make_saved_cohort(self, root):
        folder = root / "validation"
        folder.mkdir()
        ids = [76561197960265890, 76561197960265891]
        apps = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        candidates = np.array([10, 20, 60, 70, 80])
        pq.write_table(pa.table({"user_key": pa.array(ids, pa.uint64()), "n_total_inputs": [5, 5],
            "n_positive_inputs": [2, 1]}), folder / "users.parquet")
        qapps = [10, 20, 30, 40, 50] * 2
        query = pa.table({"user_key": pa.array([ids[0]] * 5 + [ids[1]] * 5, pa.uint64()),
            "app_id": qapps, "positive": [True, False, True, False, False, False, True, False, False, False],
            "linux": [app in candidates for app in qapps]})
        pq.write_table(query, folder / "query.parquet")
        pq.write_table(query.select(["user_key", "app_id"]), folder / "exclusions.parquet")
        pq.write_table(pa.table({"user_key": pa.array(ids, pa.uint64()), "app_id": [60, 70],
            "positive": [True, True], "linux": [True, True]}), folder / "targets.parquet")
        manifest = {"status": "complete", "validation": {"passed": True},
            "files": {f"validation/{name}.parquet": {"sha256": digest(folder / f"{name}.parquet")}
                      for name in ("users", "query", "targets", "exclusions")}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        protocol = {"split_manifest_sha256": digest(root / "manifest.json"), "cohort_users": {"validation": 2}}
        return apps, candidates, protocol

    def test_saved_cohort_masks_negatives_and_keeps_hidden_targets_available(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            apps, candidates, protocol = self.make_saved_cohort(root)
            cohort = load_cohort(root, "validation", apps, candidates, protocol)
            self.assertEqual(cohort["positive"].indices.tolist(), [0, 2, 1])
            self.assertEqual(cohort["exclusions"].indices.tolist(), [0, 1, 0, 1])
            self.assertEqual(cohort["target_candidates"].tolist(), [2, 3])
            # Candidate 20 is a negative observation for user one and must be masked.
            similarities = sparse.csr_matrix([[0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0], [0.8, 0, 0.2, 0, 0, 0, 0, 0],
                [0.8, 0, 0, 0, 0, 0, 0, 0], [0.8, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
            def recommend(positives, excludes, limit):
                cols = [int(np.flatnonzero(apps == app)[0]) for app in positives]
                scores = similarities[:, cols].toarray().mean(axis=1)
                valid = [i for i, app in enumerate(candidates) if app not in excludes and app not in positives and scores[i] > 0]
                chosen = sorted(valid, key=lambda i: (-scores[i], candidates[i]))[:limit]
                return [{"app_id": int(candidates[i]), "score": float(scores[i])} for i in chosen]
            model = SimpleNamespace(app_ids=apps, similarities=similarities, recommend=recommend)
            summary = evaluate_model(model, "item_knn", cohort, candidates, np.array([100, 90, 10, 5, 1]),
                                     [5, 10, 90], root / "out", 1)
            self.assertEqual(summary["Recall@10"], 0.5)
            self.assertEqual(summary["NDCG@10"], 0.5)
            self.assertEqual(summary["empty_recommendation_users"], 1)
            self.assertTrue(all(v == 0 for v in summary["saved_file_violations"].values()))
            saved = pq.read_table(root / "out/per_user.parquet").to_pydict()
            self.assertEqual(saved["user_key"], cohort["user_ids"].tolist())
            self.assertEqual(saved["recommendation_count"], [3, 0])

    def test_saved_cohort_rejects_changed_fingerprint_and_target_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            apps, candidates, protocol = self.make_saved_cohort(root)
            target_path = root / "validation/targets.parquet"
            table = pq.read_table(target_path).set_column(1, "app_id", pa.array([10, 70]))
            pq.write_table(table, target_path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_cohort(root, "validation", apps, candidates, protocol)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"]["validation/targets.parquet"]["sha256"] = digest(target_path)
            (root / "manifest.json").write_text(json.dumps(manifest))
            protocol["split_manifest_sha256"] = digest(root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "absent"):
                load_cohort(root, "validation", apps, candidates, protocol)

    def test_diagnostic_popularity_uses_exclusions_and_exact_random_expectation(self):
        cohort = {"user_ids": np.array([1, 2], dtype=np.uint64),
            "exclusions": sparse.csr_matrix([[1, 0, 0], [1, 1, 0]]),
            "target_candidates": np.array([2, 2])}
        result = diagnostic_controls(cohort, np.array([10, 20, 30]), np.array([100, 90, 80]), [1, 10, 100])
        pop = result["train_popularity_diagnostic_only_not_personalized_fallback"]
        self.assertEqual(pop["Recall@10"], 1)
        self.assertAlmostEqual(pop["NDCG@10"], (1 / np.log2(3) + 1) / 2)
        random = result["uniform_random_exact_expectation_not_sampled"]
        self.assertEqual(random["Recall@10"], 1)
        self.assertAlmostEqual(random["NDCG@10"], ((1 + 1 / np.log2(3)) / 2 + 1) / 2)


if __name__ == "__main__":
    unittest.main()
