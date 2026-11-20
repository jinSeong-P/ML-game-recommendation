"""Ranking preservation and masked component ranks for fresh fusion controls."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse

from compare_fusion_v4 import MaskedRrf, positive_group_scores, CachedRanker


class Fixed:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def score(self, q):
        return self.values.copy()


class FusionComparisonV4Tests(unittest.TestCase):
    def test_monotone_positive_transform_is_within_user(self):
        values = positive_group_scores(np.array([-10., -9., 900., 899.]), np.array([0, 0, 2, 2]))
        np.testing.assert_allclose(values, [np.exp(-1), 1, 1, np.exp(-1)])
        self.assertTrue(np.all(values > 0))
        self.assertEqual(len(positive_group_scores(np.array([]), np.array([]))), 0)

    def test_rrf_excludes_observed_game_before_component_rank_assignment(self):
        with patch('compare_fusion_v4.RrfScorer.__init__', return_value=None):
            model = MaskedRrf(dict(exclusions=sparse.csr_matrix([[1, 0, 0]])))
        model.cf = Fixed([[100, 2, 1]]); model.content = Fixed([[100, 1, 2]])
        model.ids = np.array([10, 20, 30]); model.sup = np.array([100, 1000, 0]); model.adaptive = False
        got = model.score(sparse.csr_matrix([[1]]))
        self.assertEqual(got[0, 0], 0)
        np.testing.assert_allclose(got[0, 1:], [.5/61+.5/62]*2)

    def test_cached_reranker_keeps_empty_users_and_candidate_rows(self):
        model = CachedRanker(np.array([0, 2]), np.array([4, 5]), np.array([1., .5]), {})
        a = model.score(sparse.csr_matrix((2, 1)))
        b = model.score(sparse.csr_matrix((1, 1)))
        self.assertEqual(a[0, 4], 1); self.assertEqual(a[1].sum(), 0)
        self.assertEqual(b[0, 5], .5)


if __name__ == '__main__':
    unittest.main()
