"""Focused unit coverage for V4 content artifact preparation helpers."""
from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from experiment_content_v4 import (
    _candidate_row_positions,
    _manifest_for_embedding,
    _sample_teacher_pairs,
    _tokenization_audit,
    sha_json,
)


class V4ContentHelpersTest(unittest.TestCase):
    def test_embedding_manifest_freezes_channel_and_token_limit(self):
        hashes = {"descriptions_parquet_sha256": "a" * 64,
                  "metadata_extension_audit_sha256": "b" * 64}
        description = _manifest_for_embedding("description", 512, hashes, 1024)
        metadata = _manifest_for_embedding("metadata", 1024, hashes, 1024)
        self.assertEqual(description["channel"], "description")
        self.assertIn("description_clean", description["source_policy"])
        self.assertEqual(description["model"]["revision"], "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
        self.assertEqual(metadata["embedding"]["shape"], [37_610, 1024])
        self.assertIn("metadata_features", metadata["source_policy"])
        self.assertNotEqual(sha_json(description), sha_json(metadata))

    def test_teacher_sampling_uses_top_neighbors_and_weak_non_edges(self):
        # Rows are candidate games; columns are possible input games.
        graph = sparse.csr_matrix(np.asarray([[0.8, 0.95, 0.0, 0.0],
                                              [0.0, 0.9, 0.8, 0.99]], dtype=np.float32))
        support = np.ones(4, dtype=np.int64)
        features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        pos_cand, pos_input, pos_value, neg_cand, neg_input = _sample_teacher_pairs(
            graph, np.asarray([1, 3]), support, features, top_k=1, seed=7)

        self.assertEqual(len(pos_cand), 2)
        self.assertEqual(len(neg_cand), 2 * len(pos_cand))
        np.testing.assert_allclose(pos_value, [0.8, 0.9], rtol=1e-6)
        self.assertEqual(list(zip(pos_cand.tolist(), pos_input.tolist())), [(0, 0), (1, 1)])
        self.assertTrue(np.all(graph[neg_cand, neg_input].A1 == 0))
        candidate_rows = np.asarray([1, 3])
        self.assertFalse(np.any(candidate_rows[pos_cand] == pos_input))
        self.assertFalse(np.any(candidate_rows[neg_cand] == neg_input))

    def test_candidate_positions_do_not_use_large_app_ids_as_indices(self):
        app_ids = np.asarray([10, 768_200, 1_000_000, 2_200_000], dtype=np.int64)
        linux = np.asarray([False, True, False, True])
        np.testing.assert_array_equal(_candidate_row_positions(app_ids, linux), [1, 3])

    def test_tokenization_audit_counts_truncation_thresholds(self):
        class Tokenizer:
            def __call__(self, batch, **kwargs):
                # Test rows contain 2, 513, and 1025 tokens including specials.
                return type("Encoding", (), {"input_ids": [[1] * count for count in (2, 513, 1025)]})()

        audit = _tokenization_audit(Tokenizer(), ["", "description", "metadata"])
        self.assertEqual(audit["rows"], 3)
        self.assertEqual(audit["empty_rows"], 1)
        self.assertEqual(audit["max_tokens_including_special_tokens"], 1025)
        self.assertEqual(audit["rows_over_512_tokens"], 2)
        self.assertEqual(audit["rows_over_1024_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
