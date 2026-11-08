"""Focused unit tests for deterministic train-only collaborative artifacts."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_collaborative_v3 as model


class CollaborativeV3Tests(unittest.TestCase):
    def test_roster_pins_standard_collaborative_variants(self):
        roster = model._variant_roster()
        self.assertEqual(roster["min_cooccurrence_3_cosine_k100"]["parameters"]["minimum_cooccurrence"], 3)
        self.assertEqual(roster["asymmetric_cosine_alpha025_locality2_k100"]["parameters"]["locality_exponent"], 2)
        rp3 = roster["rp3beta_alpha1_beta03_k100"]["parameters"]
        self.assertEqual(rp3["algorithm"], "RP3beta")
        self.assertEqual((rp3["length"], rp3["alpha"], rp3["beta"]), (3, 1.0, .3))
        self.assertEqual(roster["ease_lambda1000_exact"]["parameters"]["catalog_items"], 37610)
        self.assertEqual(roster["ials_rank64_reg0_1_conf20_iter10_seed20261108"]["parameters"]["seed"], 20261108)
        self.assertEqual(roster["positive_negative_relation_beta30"]["parameters"]["negative_negative"], "excluded")
        self.assertEqual(roster["time_decay_3year_cosine_k100"]["parameters"]["half_life_days"], 1095)

    def test_select_one_row_minimum_cosine_tie_breaks_by_app_id(self):
        cols = np.array([2, 0, 1, 3], dtype=np.int32)
        counts = np.array([4, 5, 5, 1], dtype=np.int64)
        cosines = np.array([.7, .7, .7, .99], dtype=np.float64)
        apps = np.array([10, 20, 30, 40], dtype=np.int64)
        selected, values = model._select_one_row(
            cols, counts, cosines, apps, limit=2, min_count=2, selection="cosine")
        np.testing.assert_array_equal(selected, [0, 1])
        np.testing.assert_allclose(values, [.7, .7])

    def test_cooccurrence_selection_uses_count_then_cosine(self):
        cols = np.array([0, 1, 2], dtype=np.int32)
        counts = np.array([10, 11, 11], dtype=np.int64)
        cosines = np.array([.99, .2, .3], dtype=np.float64)
        apps = np.array([10, 20, 30], dtype=np.int64)
        selected, values = model._select_one_row(
            cols, counts, cosines, apps, limit=2, selection="cooccurrence")
        np.testing.assert_array_equal(selected, [1, 2])
        np.testing.assert_allclose(values, [.2, .3])

    def test_input_orientation_keeps_top_candidates_per_input(self):
        apps = np.array([10, 20, 30, 40], dtype=np.int64)
        candidates = np.array([10, 30], dtype=np.int64)
        candidate_rows = np.array([0, 0, 1, 1], dtype=np.int32)
        input_cols = np.array([1, 3, 1, 3], dtype=np.int32)
        cosine = np.array([.4, .2, .8, .7])
        got = model._input_oriented_matrix(candidate_rows, input_cols, cosine,
                                           apps, candidates, (2, 4), neighbors=1)
        self.assertEqual(got[1, 1], np.float32(.8))
        self.assertEqual(got[1, 3], np.float32(.7))
        self.assertEqual(got.nnz, 2)

    def test_weighted_cosine_and_rp3_are_candidate_by_input(self):
        x = sparse.csr_matrix(np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=np.uint8)).tocsc()
        apps = np.array([10, 20, 30], dtype=np.int64)
        candidates = np.array([10, 20])
        outputs = model._weighted_model_blocked(
            x, np.array([1., .5, .25]), apps, candidates,
            "activity_inverse_degree", betas=(.3,))
        activity = outputs["activity_inverse_degree_cosine_k100"]
        rp3 = outputs["rp3beta_alpha1_beta03_k100"]
        self.assertEqual(activity.shape, (2, 3))
        self.assertEqual(rp3.shape, (2, 3))
        self.assertEqual(activity[0, 0], 0)
        self.assertEqual(activity[1, 1], 0)
        self.assertGreater(activity[0, 1], 0)
        self.assertGreater(rp3[0, 1], 0)
        self.assertAlmostEqual(float(activity[0, 1]), 1.5 / np.sqrt(2.625), places=6)
        self.assertAlmostEqual(float(rp3[0, 1]), 1.5 / (3.0 * 2.0**.3), places=6)

    def test_signed_relation_excludes_negative_negative_pairs(self):
        positive = sparse.csr_matrix(np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32))
        negative = sparse.csr_matrix(np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float32))
        apps = np.array([10, 20, 30, 40], dtype=np.int64)
        candidates = np.array([10, 30], dtype=np.int64)
        observed_support = np.array([3, 5, 2, 2], dtype=np.int64)
        got = model._signed_item_matrix(
            positive, negative, apps, candidates, observed_support, neighbors=3)
        self.assertAlmostEqual(float(got[0, 1]), 1.0 / np.sqrt(15), places=6)
        self.assertAlmostEqual(float(got[1, 1]), -2.0 / np.sqrt(10), places=6)
        self.assertEqual(got[0, 0], 0)
        self.assertEqual(got[1, 2], 0)

    def test_artifact_writer_persists_hashes_and_train_only_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = model.MODEL_ROOT
            try:
                model.MODEL_ROOT = Path(tmp)
                matrix = sparse.csr_matrix(np.array([[0, .5], [.25, 0]], dtype=np.float32))
                folder = model._write_model(
                    "tiny", matrix, {"neighbors": 1}, {"train": {"sha256": "abc"}})
                manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertFalse(manifest["ranking_metrics_computed"])
                self.assertTrue(manifest["fit_scope"].startswith("split_v1 train only"))
                self.assertTrue(manifest["files"]["similarities_csr.npz"]["sha256"])
                self.assertEqual(sparse.load_npz(folder / "similarities_csr.npz").shape, (2, 2))
            finally:
                model.MODEL_ROOT = old_root


if __name__ == "__main__":
    unittest.main()

