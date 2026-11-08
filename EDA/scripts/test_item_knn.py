"""Focused mathematical, retrieval, ID-width and artifact checks for Item-KNN."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from build_item_knn import _sort_train_pairs, file_inventory
from item_knn import (ItemKnnCatalog, _build_user_item_csr,
                      _top_cosine_neighbors)


class CooccurrenceMathTests(unittest.TestCase):
    def test_exact_binary_cosine_keeps_singleton_support_and_breaks_ties_by_app_id(self):
        app_ids = np.array([10, 20, 30, 40], dtype=np.int64)
        # Candidate 10 has three users, including one singleton. Its support
        # denominator is 3; dropping the singleton would incorrectly yield 1/2.
        support = np.array([3, 2, 2, 99], dtype=np.int64)
        cooc_cols = np.array([0, 1, 2, 3], dtype=np.int32)
        cooc_counts = np.array([3, 1, 1, 1], dtype=np.int32)
        cols, values = _top_cosine_neighbors(
            0, cooc_cols, cooc_counts, app_ids, support, neighbors=2)
        np.testing.assert_array_equal(app_ids[cols], [20, 30])
        np.testing.assert_allclose(values, [1 / np.sqrt(6), 1 / np.sqrt(6)], rtol=0, atol=1e-15)
        self.assertNotIn(10, app_ids[cols])  # self-similarity is zero

    def test_singleton_item_and_zero_support_candidates_are_supported(self):
        app_ids = np.array([10, 20, 30], dtype=np.int64)
        support = np.array([1, 1, 0], dtype=np.int64)
        cols, values = _top_cosine_neighbors(
            0, np.array([0, 1]), np.array([1, 1]), app_ids, support, neighbors=10)
        np.testing.assert_array_equal(app_ids[cols], [20])
        np.testing.assert_array_equal(values, [1.0])
        empty_cols, empty_values = _top_cosine_neighbors(
            2, np.array([], dtype=np.int32), np.array([], dtype=np.int32),
            app_ids, support, neighbors=10)
        self.assertEqual(len(empty_cols), 0)
        self.assertEqual(len(empty_values), 0)

    def test_streamed_csr_preserves_uint64_user_ids_above_float_exact_range(self):
        ids = [2**53 + 1, 2**53 + 1, 2**53 + 2, 2**64 - 2]
        table = pa.table({
            "user_key": pa.array(ids, type=pa.uint64()),
            "app_id": pa.array([10, 30, 20, 10], type=pa.int64()),
        })
        # One row per batch exercises user groups split across batch edges.
        matrix, support, user_count = _build_user_item_csr(
            table.to_batches(max_chunksize=1),
            np.array([10, 20, 30, 40], dtype=np.int64), expected_pairs=4, expected_users=3)
        self.assertEqual(user_count, 3)
        self.assertEqual(matrix.shape, (3, 4))
        np.testing.assert_array_equal(np.diff(matrix.indptr), [2, 1, 1])
        np.testing.assert_array_equal(support, [2, 1, 1, 0])
        np.testing.assert_array_equal(matrix.indices, [0, 2, 1, 0])

    def test_small_external_sort_and_candidate_block_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "train.parquet"
            sorted_path = root / "sorted.parquet"
            temp_dir = root / "duckdb_temp"
            # Deliberately unsorted. The negative pair must not enter support.
            pq.write_table(pa.table({
                "user_key": pa.array([2**64 - 2, 2**53 + 1, 2**53 + 2,
                                      2**53 + 1, 2**53 + 1], type=pa.uint64()),
                "app_id": pa.array([30, 30, 30, 10, 20], type=pa.int64()),
                "positive": pa.array([True, False, True, True, True]),
            }), source)
            _sort_train_pairs(source, sorted_path, temp_dir)
            sorted_file = pq.ParquetFile(sorted_path)
            matrix, support, users = _build_user_item_csr(
                sorted_file.iter_batches(batch_size=2),
                np.array([10, 20, 30], dtype=np.int64), expected_pairs=4, expected_users=3)
            sorted_file.close()
            self.assertEqual(users, 3)
            np.testing.assert_array_equal(support, [1, 1, 2])

            user_item_csc = matrix.tocsc()
            candidate_by_user = user_item_csc[:, [0, 1]].T.tocsr().astype(np.int32)
            cooccurrence = candidate_by_user @ user_item_csc
            self.assertEqual(cooccurrence.dtype, np.dtype("int32"))
            np.testing.assert_array_equal(cooccurrence.toarray(), [[1, 1, 0], [1, 1, 0]])
            cols, values = _top_cosine_neighbors(
                0, cooccurrence.indices[:cooccurrence.indptr[1]],
                cooccurrence.data[:cooccurrence.indptr[1]],
                np.array([10, 20, 30]), support, neighbors=10)
            np.testing.assert_array_equal(cols, [1])
            np.testing.assert_allclose(values, [1.0])


class QueryScoringTests(unittest.TestCase):
    def setUp(self):
        self.app_ids = np.array([1, 2, 10, 20, 30], dtype=np.int64)
        self.candidates = np.array([10, 20, 30], dtype=np.int64)
        # Candidate 10 links only to query 1; candidate 20 only to query 2.
        # Candidate 30 ties candidate 10. A query over [1,2] divides every row
        # by two even when one of its links is absent.
        self.matrix = sparse.csr_matrix(np.array([
            [.8, 0, 0, 0, 0],
            [0, .6, 0, 0, 0],
            [.8, 0, 0, 0, 0],
        ], dtype=np.float64))
        self.index = ItemKnnCatalog(self.app_ids, self.candidates, self.matrix)

    def test_score_uses_all_positive_queries_and_ties_use_app_id(self):
        results = self.index.recommend([1, 2])
        self.assertEqual([row["app_id"] for row in results], [10, 30, 20])
        np.testing.assert_allclose([row["score"] for row in results], [.4, .4, .3])
        self.assertEqual(set(results[0]), {"app_id", "score"})

    def test_positive_and_all_observed_exclusions_are_applied(self):
        results = self.index.recommend([1, 2], exclude_app_ids=[10, 30])
        self.assertEqual([row["app_id"] for row in results], [20])
        self.assertAlmostEqual(results[0]["score"], .3)

    def test_positive_candidate_is_masked_and_no_scores_are_filled(self):
        results = self.index.recommend([10, 1, 2])
        self.assertEqual([row["app_id"] for row in results], [30, 20])
        self.assertNotIn(10, [row["app_id"] for row in results])
        self.assertEqual(self.index.recommend([10]), [])
        self.assertEqual(self.index.recommend([]), [])
        self.assertEqual(self.index.recommend([1, 2], limit=0), [])

    def test_duplicate_unknown_queries_and_invalid_limits_fail(self):
        with self.assertRaises(ValueError):
            self.index.recommend([1, 1])
        with self.assertRaises(ValueError):
            self.index.recommend([999])
        with self.assertRaises(ValueError):
            self.index.recommend([1], limit=11)


class ArtifactTests(unittest.TestCase):
    def _write_artifact(self, root, matrix=None):
        app_ids = np.array([1, 2, 10, 20], dtype=np.int64)
        candidates = np.array([10, 20], dtype=np.int64)
        support = np.array([4, 2, 3, 1], dtype=np.int64)
        if matrix is None:
            matrix = sparse.csr_matrix(np.array([
                [0, .5, 0, 0],
                [.25, 0, 0, 0],
            ], dtype=np.float64))
        pq.write_table(pa.table({
            "row_id": pa.array([0, 1, 2, 3], type=pa.int32()),
            "app_id": pa.array(app_ids),
            "linux": pa.array([False, False, True, True]),
            "positive_user_support": pa.array(support),
        }), root / "game_rows.parquet")
        pq.write_table(pa.table({
            "row_id": pa.array([0, 1], type=pa.int32()),
            "app_id": pa.array(candidates),
            "input_row_id": pa.array([2, 3], type=pa.int32()),
        }), root / "candidate_rows.parquet")
        (root / "parameters.json").write_text(json.dumps({"neighbors": 2}), encoding="utf-8")
        sparse.save_npz(root / "similarities_csr.npz", matrix)
        files = file_inventory(root)
        (root / "manifest.json").write_text(json.dumps({
            "version": "binary-item-knn-v1", "status": "complete", "files": files,
            "parameters": {"neighbors": 2},
            "index": {"shape": [2, 4], "nnz": matrix.nnz},
        }), encoding="utf-8")

    def test_saved_index_round_trip_hashes_support_and_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_artifact(root)
            loaded = ItemKnnCatalog.load(root)
            np.testing.assert_array_equal(loaded.app_ids, [1, 2, 10, 20])
            np.testing.assert_array_equal(loaded.candidate_app_ids, [10, 20])
            np.testing.assert_array_equal(loaded.support, [4, 2, 3, 1])
            self.assertEqual(loaded.recommend([1])[0]["app_id"], 20)
            with (root / "similarities_csr.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                ItemKnnCatalog.load(root)

    def test_loader_rejects_self_similarity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = sparse.csr_matrix(np.array([
                [0, 0, 1, 0],  # self-link at candidate 10 / input row 10
                [0, 0, 0, 0],
            ], dtype=np.float64))
            self._write_artifact(root, matrix)
            with self.assertRaisesRegex(ValueError, "Self-similarity"):
                ItemKnnCatalog.load(root)


if __name__ == "__main__":
    unittest.main()
