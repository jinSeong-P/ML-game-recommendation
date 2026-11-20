"""Behavioral checks for metadata precedence, text cleanup and exact retrieval."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from build_tfidf_content import file_inventory, prepare_rows
from tfidf_content import (TfidfCatalog, clean_description, make_vectorizer,
                           script_profile, select_description, tokenize_description)


class DescriptionTests(unittest.TestCase):
    def test_a_priority_and_unicode_blank_fallback(self):
        self.assertEqual(select_description("A text", "C text"), ("A text", "A.description"))
        self.assertEqual(select_description(" \u00a0\t", "C text"), ("C text", "C.short_description"))
        self.assertEqual(select_description(None, " "), (None, "missing"))

    def test_html_entities_hidden_content_and_block_boundaries(self):
        text = "<p>Space &amp; Sea</p><p>Build</p><script>secret</script><style>hidden</style>"
        self.assertEqual(clean_description(text), "Space & Sea Build")
        self.assertEqual(clean_description("&lt;b&gt;play&lt;/b&gt; &amp;amp; explore"), "play & explore")

    def test_known_bbcode_urls_and_unicode_controls(self):
        text = "[b]Space[/b]\u00a0[url=https://example.org]Build[/url]\u2028[img]photo[/img] https://example.org\u200b"
        self.assertEqual(clean_description(text), "Space Build")

    def test_meaningful_prose_and_brackets_are_preserved(self):
        text = "Free-to-play with in-game purchases. Minimum attention. Kick A** [space] <3"
        self.assertEqual(clean_description(text), text)

    def test_cleanup_does_not_trigger_unapproved_second_fallback(self):
        rows, _ = prepare_rows([(1, True, "<script>noise</script>", "Usable C text")])
        self.assertEqual(rows[0]["text_source"], "A.description")
        self.assertEqual(rows[0]["text_status"], "empty_after_cleaning")
        self.assertTrue(rows[0]["clean_empty_with_unused_c_alternative"])

    def test_missing_tokenless_and_duplicate_games_remain_distinct(self):
        rows, groups = prepare_rows([(1, True, "", ""), (2, True, "a !", "Unused"),
                                     (3, False, "Space build", None), (4, True, "Space build", None)])
        self.assertEqual([r["text_status"] for r in rows], ["raw_missing", "zero_after_tokenization", "supported", "supported"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(groups[0]["app_ids"], [3, 4])

    def test_unicode_tokens_and_script_are_not_english_only(self):
        self.assertEqual(tokenize_description("SPACE café A 7 42 中文遊戲"), ["space", "café", "42", "中文遊戲"])
        self.assertEqual(script_profile("中文 игра"), "CJK+CYRILLIC")


class CosineRetrievalTests(unittest.TestCase):
    def setUp(self):
        # Query app 1 is non-Linux, app 4 deliberately has no features.
        self.index = TfidfCatalog([1, 2, 3, 4, 5], [False, True, True, True, True],
            sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 0], [1, 0]], dtype=np.float32)))

    def test_non_linux_input_linux_candidates_positive_only_and_stable_ties(self):
        result = self.index.recommend([1])
        self.assertEqual([r["app_id"] for r in result], [2, 5])
        self.assertTrue(all(r["cosine"] == 1.0 for r in result))
        self.assertEqual(self.index.matrix.shape[0], 5)

    def test_equal_mean_cosine_is_not_max_similarity(self):
        result = self.index.recommend([1, 3])
        self.assertEqual([r["app_id"] for r in result], [2, 5])
        self.assertAlmostEqual(result[0]["cosine"], 1 / np.sqrt(2), places=6)

    def test_negative_observed_games_are_excluded_but_hidden_target_is_available(self):
        # App 2 is observed-negative; app 5 is a hidden target, absent from mask.
        self.assertEqual([r["app_id"] for r in self.index.recommend([1], [1, 2])], [5])

    def test_zero_profile_and_short_lists_do_not_get_filled(self):
        self.assertEqual(self.index.recommend([4]), [])
        self.assertEqual(self.index.recommend([]), [])
        self.assertEqual(len(self.index.recommend([1], [2])), 1)

    def test_duplicate_queries_and_unknown_games_fail_fast(self):
        with self.assertRaises(ValueError):
            self.index.recommend([1, 1])
        with self.assertRaises(ValueError):
            self.index.recommend([999])

    def test_tfidf_fit_has_l2_rows_and_preserves_empty_document(self):
        vectors = make_vectorizer().fit_transform(["red red blue", "blue", ""])
        norms = np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
        np.testing.assert_allclose(norms, [1, 1, 0], atol=1e-6)
        self.assertEqual(vectors.dtype, np.float32)

    def test_saved_index_round_trip_and_fingerprint_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pq.write_table(pa.table({"row_id": list(range(5)), "app_id": self.index.app_ids,
                                     "linux": self.index.linux}), root / "game_rows.parquet")
            pq.write_table(pa.table({"term": ["space", "build"]}), root / "vocabulary_idf.parquet")
            sparse.save_npz(root / "tfidf_csr.npz", self.index.matrix)
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "files": file_inventory(root)}), encoding="utf-8")
            loaded = TfidfCatalog.load(root)
            self.assertEqual(loaded.recommend([1]), self.index.recommend([1]))
            with (root / "tfidf_csr.npz").open("ab") as file:
                file.write(b"changed")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                TfidfCatalog.load(root)


if __name__ == "__main__":
    unittest.main()
