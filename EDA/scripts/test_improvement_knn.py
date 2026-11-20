"""Synthetic invariants for v2 sparse item-KNN construction and diagnostics."""

import sys
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from improvement_knn import diagnostic_category, rank_neighbors, _refresh_build_manifest, _categorize_diagnostic_rows, MODEL_VERSION, KS, SHRINKAGES


class NeighborRankingTests(unittest.TestCase):
    def test_count_math_does_not_overflow_signed_int32(self):
        app_ids = np.array([10, 20], dtype=np.int64)
        support = np.array([3_000_000_000, 3_000_000_000], dtype=np.int64)
        columns = np.array([1], dtype=np.int32)
        counts = np.array([2_500_000_000], dtype=np.int64)

        selected, values = rank_neighbors(0, columns, counts, app_ids, support,
                                          neighbors=1, shrinkage=0)

        np.testing.assert_array_equal(selected, [1])
        self.assertAlmostEqual(values[0], 5 / 6)

    def test_singleton_support_is_retained_and_zero_support_is_ignored(self):
        app_ids = np.array([100, 200, 300, 400], dtype=np.int64)
        support = np.array([5, 0, 1, 5], dtype=np.int64)
        columns = np.array([0, 1, 2, 3], dtype=np.int32)
        counts = np.array([5, 1, 1, 2], dtype=np.int64)

        selected, values = rank_neighbors(0, columns, counts, app_ids, support,
                                          neighbors=10, shrinkage=0)

        # Self and the unsupported item are excluded, while a singleton remains.
        np.testing.assert_array_equal(selected, [2, 3])
        np.testing.assert_allclose(values, [1.0 / np.sqrt(5), 2.0 / 5.0])

    def test_shrinkage_reselects_from_all_raw_neighbors(self):
        app_ids = np.array([100, 200, 300], dtype=np.int64)
        support = np.array([100, 100, 1], dtype=np.int64)
        columns = np.array([1, 2], dtype=np.int32)
        counts = np.array([9, 1], dtype=np.int64)

        raw_cols, raw_values = rank_neighbors(0, columns, counts, app_ids, support,
                                              neighbors=1, shrinkage=0)
        shrunk_cols, shrunk_values = rank_neighbors(0, columns, counts, app_ids, support,
                                                    neighbors=1, shrinkage=10)

        np.testing.assert_array_equal(raw_cols, [2])
        np.testing.assert_array_equal(shrunk_cols, [1])
        self.assertGreater(raw_values[0], shrunk_values[0])

    def test_rank_ties_use_app_id_ascending_and_storage_is_column_sorted(self):
        app_ids = np.array([90, 20, 10, 30], dtype=np.int64)
        support = np.array([10, 10, 10, 10], dtype=np.int64)
        columns = np.array([3, 2, 1], dtype=np.int32)
        counts = np.array([5, 5, 5], dtype=np.int64)

        selected, values = rank_neighbors(0, columns, counts, app_ids, support,
                                          neighbors=2, shrinkage=0)

        # IDs 10 and 20 win the score tie; CSR output stores column order 1,2.
        np.testing.assert_array_equal(selected, [1, 2])
        np.testing.assert_allclose(values, [0.5, 0.5])

    def test_zero_support_candidate_and_zero_k_return_empty_arrays(self):
        app_ids = np.array([10, 20], dtype=np.int64)
        support = np.array([0, 1], dtype=np.int64)
        columns = np.array([1], dtype=np.int32)
        counts = np.array([1], dtype=np.int64)

        selected, values = rank_neighbors(0, columns, counts, app_ids, support, 1, 0)
        self.assertEqual(selected.size, 0)
        self.assertEqual(values.size, 0)
        selected, values = rank_neighbors(1, columns, counts, app_ids, support, 0, 0)
        self.assertEqual(selected.size, 0)
        self.assertEqual(values.size, 0)


class DiagnosticCategoryTests(unittest.TestCase):
    def test_sql_uses_train_support_when_no_input_has_a_retained_v1_edge(self):
        con=duckdb.connect()
        try:
            con.execute('''CREATE TEMP TABLE diag_user_metrics AS SELECT * FROM
                (VALUES (1,10,0,1,3,0,0),(2,10,0,1,0,0,0),(3,10,0,0,0,0,0))
                t(user_id,target_support,supported_positive_inputs,train_supported_positive_inputs,
                  raw_connection_count,retained_connection_count,target_rank_v1)''')
            _categorize_diagnostic_rows(con)
            self.assertEqual(con.execute('SELECT category FROM diag_categorized ORDER BY user_id').fetchall(),
                             [('pruned',),('raw_connection0',),('all_positive_inputs_support0',)])
        finally:con.close()

    def test_diagnostic_refresh_preserves_published_model_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'manifest.json').write_text(json.dumps({'version':MODEL_VERSION,'status':'complete'}),encoding='utf-8')
            original={}
            for k in KS:
                for lam in SHRINKAGES:
                    folder=root/f'k{k}_l{lam}';folder.mkdir()
                    path=folder/'manifest.json'
                    path.write_text(json.dumps({'code_sha256':{'builder':'original-fit-code'},'status':'complete'}),encoding='utf-8')
                    original[path]=path.read_bytes()
            report=root/'diagnosis.json'
            report.write_text(json.dumps({'training_support_summary':{},'cohorts':{}}),encoding='utf-8')
            _refresh_build_manifest(root,report)
            for path,content in original.items():self.assertEqual(path.read_bytes(),content)

    def test_all_six_k0_categories_are_ordered_and_exclusive(self):
        examples = [
            ((0, 2, 1, 1, 0), "target_support0"),
            ((2, 0, 0, 0, 0), "all_positive_inputs_support0"),
            ((2, 1, 0, 0, 0), "raw_connection0"),
            ((2, 1, 3, 0, 0), "pruned"),
            ((2, 1, 2, 1, 0), "rank_failure"),
            ((2, 1, 2, 1, 7), "hit"),
        ]
        for values, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(diagnostic_category(*values), expected)


if __name__ == "__main__":
    unittest.main()
