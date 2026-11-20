"""Focused tests for v2 TF-IDF feature changes and safe metadata parsing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from improvement_tfidf import (
    _write_model_artifact,
    choose_tag_source,
    contributions_for_pair,
    make_text_vectorizer,
    normalize_concept,
    parse_metadata_labels,
    _quantile_group,
)
from tfidf_content import TfidfCatalog


class TextFeatureVariantTests(unittest.TestCase):
    def test_percentile_boundary_keeps_all_ties_in_lower_group(self):
        threshold, mask, details = _quantile_group(np.array([1, 2, 2, 2, 3]), 0.5)
        self.assertEqual(threshold, 2.0)
        self.assertEqual(mask.tolist(), [False, False, False, False, True])
        self.assertEqual(details["games_at_threshold"], 3)
        self.assertTrue(details["threshold_exclusive"])

    def test_t1_changes_repeated_term_weight_but_not_other_settings(self):
        documents = ["echo echo unique", "echo unique", "other topic"]
        baseline = make_text_vectorizer()
        t1 = make_text_vectorizer(sublinear_tf=True)
        baseline_matrix = baseline.fit_transform(documents).tocsr()
        t1_matrix = t1.fit_transform(documents).tocsr()

        self.assertEqual(baseline.get_params()["ngram_range"], (1, 1))
        self.assertEqual(t1.get_params()["ngram_range"], (1, 1))
        self.assertFalse(baseline.get_params()["sublinear_tf"])
        self.assertTrue(t1.get_params()["sublinear_tf"])
        self.assertEqual(baseline.get_params()["max_df"], t1.get_params()["max_df"])
        self.assertEqual(baseline.get_params()["stop_words"], t1.get_params()["stop_words"])
        np.testing.assert_allclose(np.linalg.norm(baseline_matrix.toarray(), axis=1), [1, 1, 1], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(t1_matrix.toarray(), axis=1), [1, 1, 1], atol=1e-6)
        self.assertFalse(np.allclose(baseline_matrix.toarray()[0], t1_matrix.toarray()[0]))
        terms = baseline.get_feature_names_out().tolist()
        echo_index = terms.index("echo")
        self.assertGreater(float(baseline_matrix[0, echo_index]), float(t1_matrix[0, echo_index]))

    def test_t2_adds_phrase_feature_while_preserving_unigrams(self):
        documents = ["space ship", "space rock", "rock garden"]
        baseline = make_text_vectorizer()
        bigram = make_text_vectorizer(ngram_range=(1, 2))
        baseline.fit_transform(documents)
        bigram_matrix = bigram.fit_transform(documents)
        terms = bigram.get_feature_names_out().tolist()
        self.assertIn("space", terms)
        self.assertIn("ship", terms)
        self.assertIn("space ship", terms)
        self.assertEqual(bigram.get_params()["sublinear_tf"], baseline.get_params()["sublinear_tf"])
        self.assertGreater(bigram_matrix.shape[1], len(baseline.get_feature_names_out()))

    def test_real_profile_candidate_term_products_sum_to_cosine(self):
        profile = sparse.csr_matrix([[0.6, 0.8, 0]], dtype=np.float32)
        candidate = sparse.csr_matrix([[0.5, 0.5, 0.70710677]], dtype=np.float32)
        rows = contributions_for_pair(profile, candidate, ["alpha", "beta", "gamma"],
                                       np.array([1.2, 1.4, 1.7]), np.array([2, 3, 1]), 4)
        expected = float(profile.multiply(candidate).sum())
        self.assertAlmostEqual(sum(row["contribution"] for row in rows), expected, places=7)
        self.assertEqual([row["term"] for row in rows], ["beta", "alpha"])
        self.assertTrue(all("document_frequency" in row and "idf" in row for row in rows))

    def test_saved_core_artifact_loads_as_tfidf_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "control"
            matrix = sparse.csr_matrix(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32))
            _write_model_artifact(
                folder,
                model_id="test_control",
                app_ids=np.array([10, 20, 30]),
                linux=np.array([False, True, True]),
                matrix=matrix,
                terms=["alpha", "beta"],
                idf=np.array([1.0, 1.0], dtype=np.float32),
                parameters={"test": True},
                source_hashes={"source": "test"},
                code_hashes={"code": "test"},
            )
            model = TfidfCatalog.load(folder)
            self.assertEqual(model.recommend([10])[0]["app_id"], 30)
            self.assertEqual(pq.read_table(folder / "game_rows.parquet")["app_id"].to_pylist(), [10, 20, 30])
            self.assertIn("document_frequency", pq.read_table(folder / "vocabulary_idf.parquet").column_names)


class MetadataParsingTests(unittest.TestCase):
    def test_json_and_python_literal_metadata_parsers_keep_phrase_labels(self):
        tags, tag_status, tag_parser = parse_metadata_labels('{"Action RPG": 13, "Café": 4}', "tags")
        genres, genre_status, genre_parser = parse_metadata_labels("['Massively Multiplayer', 'Turn-Based']", "genres")
        self.assertEqual(tags, ["Action RPG", "Café"])
        self.assertEqual(tag_status, "valid")
        self.assertEqual(tag_parser, "json")
        self.assertEqual(genres, ["Massively Multiplayer", "Turn-Based"])
        self.assertEqual(genre_status, "valid")
        self.assertEqual(genre_parser, "python_literal")
        self.assertEqual(normalize_concept("  ACTION\u00a0  RPG  "), "action rpg")

    def test_a_valid_tags_win_and_c_tags_are_used_only_as_fallback(self):
        self.assertEqual(choose_tag_source(["A Tag"], "valid", ["C Tag"], "valid"), (["A Tag"], "A"))
        self.assertEqual(choose_tag_source([], "missing", ["C Tag"], "valid"), (["C Tag"], "C"))
        self.assertEqual(choose_tag_source([], "invalid", [], "empty"), ([], "missing"))

    def test_malicious_python_expression_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "should_not_exist.txt"
            expression = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('executed')"
            labels, status, parser = parse_metadata_labels(expression, "genres")
            self.assertEqual(labels, [])
            self.assertEqual(status, "invalid")
            self.assertEqual(parser, "invalid")
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
