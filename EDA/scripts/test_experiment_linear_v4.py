"""Focused tests for v4 collaborative and linear model builders."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_linear_v4 as model


class LinearV4Tests(unittest.TestCase):
    def test_roster_freezes_requested_model_grids_and_explicit_modes(self):
        roster = model._variant_roster()
        self.assertIn("binary_cosine_k100", roster)
        self.assertIn("activity_inverse_degree_cosine_k100", roster)
        self.assertIn("account_pair_rho250_fixed_k100", roster)
        self.assertIn("account_pair_rho250_reselected_k100", roster)
        self.assertIn("dan_ease_lambda100_a100_b500_p300_scale_matched", roster)
        self.assertIn("rlae_lambda100_xi250", roster)
        self.assertIn("rdlae_lambda100_p100_xi500", roster)
        self.assertIn("fslim_m500_l21000_l1r010", roster)
        self.assertEqual(len([k for k in roster if k.startswith("dan_ease_")]), 96)
        self.assertEqual(roster["account_pair_rho250_fixed_k100"]["parameters"]["neighbor_policy"], "fixed")
        self.assertEqual(roster["account_pair_rho250_reselected_k100"]["parameters"]["neighbor_policy"], "reselected")
        self.assertEqual(roster["rdlae_lambda100_p100_xi500"]["parameters"]["xi"], .5)

    def test_pair_contribution_weights_keep_singletons_and_interpolate_pair_totals(self):
        degree = np.array([1, 2, 5], dtype=np.int64)
        w0 = model._pair_contribution_weights(degree, 0.0)
        w1 = model._pair_contribution_weights(degree, 1.0)
        w25 = model._pair_contribution_weights(degree, .25)
        self.assertEqual(float(w0[0]), 1.0)
        self.assertEqual(float(w1[0]), 1.0)
        self.assertAlmostEqual(float(w0[1]), .5)
        self.assertAlmostEqual(float(w1[1]), .5)
        self.assertAlmostEqual(float(w0[2]), .2)
        self.assertAlmostEqual(float(w1[2]), .05)
        np.testing.assert_allclose(w25, .75 * w0 + .25 * w1, rtol=1e-7)
        # Per-user total pair mass is (d-1)/2 at rho=0 and 1/2 at rho=1.
        for d, inverse_degree, equal_pair in zip(degree, w0, w1):
            pair_count = d * (d - 1) / 2
            if d > 1:
                self.assertAlmostEqual(pair_count * float(inverse_degree), (d - 1) / 2)
                self.assertAlmostEqual(pair_count * float(equal_pair), .5)

    def test_dan_popularity_factors_handle_zero_support(self):
        support = np.array([100, 0, 25], dtype=np.int64)
        inputs, candidates = model._dan_popularity_factors(support, .2)
        self.assertAlmostEqual(inputs[0], 100 ** .2)
        self.assertEqual(inputs[1], 0.0)
        self.assertEqual(candidates[1], 0.0)
        self.assertAlmostEqual(candidates[2], 25 ** -.2)

    def test_dan_candidate_row_uses_source_input_over_candidate_orientation(self):
        support = np.array([4, 100, 0], dtype=np.int64)
        base = np.array([.2, 99.0, .8], dtype=np.float64)
        got = model._dan_candidate_coefficients(base, support, candidate_col=1, alpha=.5)
        np.testing.assert_allclose(got, [.04, 0.0, 0.0], rtol=1e-6)

    def test_rlae_multiplier_matches_author_lagrangian_equation(self):
        gamma = np.array([100.0, 100.0, 100.0])
        inverse_diagonal = np.array([.002, .006, .009])
        for xi in (0.0, .25, .5, .75, 1.0):
            condition = (1.0 - gamma * inverse_diagonal) > xi
            lagrangian = np.where(condition, (1.0 - xi) / inverse_diagonal - gamma, 0.0)
            author_target_scale = (gamma + lagrangian) * inverse_diagonal
            observed = model._rlae_multiplier(gamma, inverse_diagonal, xi)
            np.testing.assert_allclose(observed, author_target_scale, rtol=1e-12, atol=1e-12)

    def test_rlae_candidate_weights_match_author_column_formula(self):
        inverse_column = np.array([-.0007, .004, -.0002], dtype=np.float64)
        candidate_col = 1
        p_jj = inverse_column[candidate_col]
        self.assertGreater(p_jj, 0)
        gamma = 100.0
        xi = .5
        condition = (1.0 - gamma * p_jj) > xi
        lagrangian = ((1.0 - xi) / p_jj - gamma) if condition else 0.0
        author_weights = -inverse_column * (gamma + lagrangian)
        ease_weights = -inverse_column / p_jj
        derived_weights = ease_weights * model._rlae_multiplier(
            np.array([gamma]), np.array([p_jj]), xi)[0]
        author_weights[candidate_col] = 0.0
        derived_weights[candidate_col] = 0.0
        np.testing.assert_allclose(derived_weights, author_weights, rtol=1e-12, atol=1e-12)

    def test_rlae_reuses_ease_candidate_rows_with_candidate_diagonal_multiplier(self):
        ease = np.array([[0.0, .2, -.1], [.3, 0.0, .4]], dtype=np.float32)
        diagonal = np.array([.004, .002], dtype=np.float32)
        got = model._rlae_weights_from_ease(ease, diagonal, 100, .5, [0, 1])
        np.testing.assert_allclose(got[0], [0.0, .2 * .5, -.1 * .5])
        np.testing.assert_allclose(got[1], [.3 * .5, 0.0, .4 * .5])

    def test_sampled_linear_solve_residual_checks_original_gram_equation(self):
        from scipy.linalg.lapack import get_lapack_funcs
        gram = np.array([[2.0, .5, .1], [.5, 3.0, .25], [.1, .25, 1.5]], dtype=np.float32)
        gamma = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        system = gram.copy(order="F")
        system[np.diag_indices(3)] += gamma.astype(np.float32)
        factor = np.linalg.cholesky(system).astype(np.float32, order="F")
        potrs = get_lapack_funcs("potrs", (factor,))
        receipt = model._sample_solve_residual(gram, gamma, factor, potrs, np.array([0, 2]))
        self.assertEqual(receipt["sampled_rhs_count"], 2)
        self.assertLess(receipt["max_backward_error"], receipt["tolerance"])

    def test_rdlae_diagonal_penalty_matches_author_support_adjustment(self):
        support = np.array([0, 1, 1000], dtype=np.int64)
        actual = model._rdlae_diagonal_penalty(support, 100, .3)
        expected = 100 + support * (.3 / .7)
        np.testing.assert_allclose(actual, expected)

    def test_coordinate_descent_matches_diagonal_elastic_net_solution(self):
        gram = np.diag([2.0, 5.0, 0.0])
        cross = np.array([6.0, 3.0, 0.0])
        coefficients, iterations, converged = model._coordinate_descent_nonnegative(
            gram, cross, l1=1.0, l2=2.0, tolerance=1e-10, max_iterations=20)
        np.testing.assert_allclose(coefficients, [1.25, 2.0 / 7.0, 0.0], rtol=1e-6)
        self.assertTrue(converged)
        self.assertGreaterEqual(iterations, 1)

    def test_topk_breaks_score_ties_by_app_id_and_stores_sorted_columns(self):
        cols = np.array([2, 0, 1], dtype=np.int32)
        scores = np.array([.5, .5, .5])
        apps = np.array([10, 20, 30], dtype=np.int64)
        got_cols, got_values = model._topk_by_score(cols, scores, apps, 2)
        np.testing.assert_array_equal(got_cols, [0, 1])
        np.testing.assert_allclose(got_values, [.5, .5])

    def test_weighted_cosine_fixed_neighbors_rescore_without_reselection(self):
        x = sparse.csr_matrix(np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
        ], dtype=np.uint8))
        apps = np.array([10, 20, 30, 40], dtype=np.int64)
        candidates = np.array([10, 30], dtype=np.int64)
        degrees = np.diff(x.indptr)
        rho0 = model._pair_contribution_weights(degrees, 0.0)
        base, _, base_rows = model._weighted_cosine_rows(x, apps, candidates, rho0, 2)
        rho1 = model._pair_contribution_weights(degrees, 1.0)
        reselected, fixed, _ = model._weighted_cosine_rows(
            x, apps, candidates, rho1, 2, fixed_neighbor_rows=base_rows)
        for row in range(len(candidates)):
            np.testing.assert_array_equal(fixed.indices[fixed.indptr[row]:fixed.indptr[row+1]],
                                          base.indices[base.indptr[row]:base.indptr[row+1]])
        self.assertEqual(reselected.shape, fixed.shape)
        self.assertTrue(reselected.has_sorted_indices)
        self.assertTrue(fixed.has_sorted_indices)

    def test_sparse_artifact_writer_saves_sorted_matrix_support_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = model.MODEL_ROOT
            try:
                model.MODEL_ROOT = Path(tmp)
                apps = np.array([10, 20, 30], dtype=np.int64)
                candidates = np.array([10, 30], dtype=np.int64)
                support = np.array([2, 1, 2], dtype=np.int64)
                matrix = sparse.csr_matrix(np.array([[0, .5, 0], [0, .25, 0]], dtype=np.float32))
                sources = {"train": {"sha256": "train-hash"}}
                folder = model._save_sparse_artifact(
                    "test_sparse", matrix, "item_knn", {"neighbors": 1}, sources,
                    support, apps, candidates, {"train_positive_users": 2})
                manifest = __import__("json").loads((folder / "manifest.json").read_text("utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertFalse(manifest["validation_test_interactions_read"])
                self.assertFalse(manifest["ranking_metrics_computed"])
                self.assertEqual(manifest["source_hashes"], {"train": "train-hash"})
                self.assertTrue(manifest["files"]["similarities_csr.npz"]["sha256"])
                self.assertTrue(sparse.load_npz(folder / "similarities_csr.npz").has_sorted_indices)
                np.testing.assert_array_equal(np.load(folder / "support.npy"), support)
            finally:
                model.MODEL_ROOT = old_root


if __name__ == "__main__":
    unittest.main()
