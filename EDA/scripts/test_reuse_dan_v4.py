import unittest
import numpy as np
from reuse_dan_v4 import rescale_rows
from experiment_linear_v4 import _dan_candidate_coefficients


class ReuseDanTests(unittest.TestCase):
    def test_cached_base_matches_direct_author_popularity_formula(self):
        support = np.array([1, 100, 0, 2000])
        base = np.array([[0, .2, -.1, .5], [-.2, .8, 0, 0]], dtype=np.float32)
        cols = np.array([0, 3])
        for alpha in [0., .1, .2, .3]:
            actual = rescale_rows(base, support, cols, alpha)
            expected = np.stack([_dan_candidate_coefficients(row, support, col, alpha)
                                 for row, col in zip(base, cols)])
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.all(actual[:, 2] == 0))


if __name__ == '__main__':
    unittest.main()
