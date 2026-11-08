"""Unit tests for v3 content model feature construction and artifact contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from experiment_content_v3 import (
    DEFAULT_V2_METADATA,
    MINILM_REVISION,
    MISSING_DESCRIPTION_TARGET_APPS,
    MODEL_CONFIGS,
    STEAM_BASE,
    _accepted_key_mask,
    _extra_metadata_features,
    _require_child,
    _save_sparse_matrix,
    _sorted_train_review_keys,
    _verify_sampled_train_lineage,
    _verify_inventory,
    digest,
    file_inventory,
    make_char_vectorizer,
    make_word_vectorizer,
    select_frank_game_description,
    select_long_description,
    verify_model_artifact,
    write_json,
)


class ContentV3FeatureTests(unittest.TestCase):
    def test_long_description_prefers_about_then_detail_then_v1(self):
        value, source = select_long_description(
            {"about_the_game": "  ", "detailed_description": "Long C detail"}, "short v1"
        )
        self.assertEqual((value, source), ("Long C detail", "C.detailed_description"))
        value, source = select_long_description(
            {"about_the_game": "Long C about", "detailed_description": "Long C detail"}, "short v1"
        )
        self.assertEqual((value, source), ("Long C about", "C.about_the_game"))
        value, source = select_long_description({}, "short v1")
        self.assertEqual((value, source), ("short v1", "frozen v1 A-first/C.short description"))

    def test_fill_priority_prefers_pinned_frank_long_fields(self):
        self.assertEqual(
            select_frank_game_description({"about_the_game": "about", "detailed_description": "detail"}),
            ("about", "about_the_game"),
        )
        self.assertEqual(
            select_frank_game_description({"about_the_game": "", "detailed_description": "detail"}),
            ("detail", "detailed_description"),
        )
        self.assertEqual(
            select_frank_game_description({"short_description": "short"}),
            ("short", "short_description"),
        )
        self.assertEqual(select_frank_game_description(None), (None, None))

    def test_word_tfidf_settings_include_unigrams_and_bigrams(self):
        vectorizer = make_word_vectorizer((1, 2))
        matrix = vectorizer.fit_transform(["space ship game", "space ship story", "other game"])
        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertFalse(vectorizer.sublinear_tf)
        self.assertEqual(vectorizer.min_df, 1)
        self.assertEqual(vectorizer.max_df, 1.0)
        self.assertIn("space ship", vectorizer.get_feature_names_out())
        np.testing.assert_allclose(np.linalg.norm(matrix.toarray(), axis=1), [1, 1, 1], atol=1e-6)

    def test_character_tfidf_settings_are_multilingual_and_bounded(self):
        vectorizer = make_char_vectorizer()
        docs = ["space ship 遊戲", "space ship 游戏", "another game"]
        matrix = vectorizer.fit_transform(docs)
        self.assertEqual(vectorizer.analyzer, "char_wb")
        self.assertEqual(vectorizer.ngram_range, (3, 5))
        self.assertEqual(vectorizer.min_df, 2)
        self.assertTrue(vectorizer.sublinear_tf)
        self.assertEqual(vectorizer.max_features, 100_000)
        self.assertLessEqual(matrix.shape[1], 100_000)
        self.assertEqual(matrix.dtype, np.float32)

    def test_metadata_extension_keeps_field_prefix_and_phrase(self):
        features, audit = _extra_metadata_features({
            "categories": "['Single-player', 'Online Co-op']",
            "developers": "['Studio One']",
            "publishers": ["Publisher Two"],
        })
        self.assertEqual(features, [
            "category::online co-op", "category::single-player", "developer::studio one",
            "publisher::publisher two",
        ])
        self.assertEqual(audit["categories"]["status"], "valid")
        self.assertEqual(audit["developers"]["status"], "valid")
        self.assertEqual(audit["publishers"]["status"], "valid")

    def test_accepted_train_review_id_mask(self):
        train_keys = np.asarray([10, 20, 30, 50], dtype=np.uint64)
        ids = np.asarray([9, 10, 19, 20, 30, 31, 50, 51], dtype=np.uint64)
        self.assertEqual(_accepted_key_mask(ids, train_keys).tolist(),
                         [False, True, False, True, True, False, True, False])

    def test_predeclared_content_names_have_fixed_embedding_revision_and_fill_scope(self):
        self.assertEqual(len(MISSING_DESCRIPTION_TARGET_APPS), 10)
        self.assertEqual(len(set(MISSING_DESCRIPTION_TARGET_APPS)), 10)
        self.assertEqual(MODEL_CONFIGS["description_sentence_embedding"]["model_revision"], MINILM_REVISION)
        self.assertEqual(MODEL_CONFIGS["description_char_wb_tfidf"]["family"], "content")
        self.assertEqual(MODEL_CONFIGS["train_review_body_tfidf"]["family"], "content")
        self.assertEqual(MODEL_CONFIGS["description_missing_fill_bigram"]["vectorizer"],
                         "v2_T2_word_unigram_bigram_tfidf")
        self.assertTrue(DEFAULT_V2_METADATA.as_posix().endswith("improvement_v2/tfidf/metadata_channel"))


class ContentV3ArtifactTests(unittest.TestCase):
    def test_output_inventory_ignores_incomplete_building_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "registry.json").write_text("{}", encoding="utf-8")
            (output / "complete.txt").write_text("ok", encoding="utf-8")
            stage = output / ".train_review_body_tfidf.building-deadbeef"
            stage.mkdir()
            pq.write_table(pa.table({"x": [1]}), stage / "partial.parquet")
            inventory = file_inventory(output)
            self.assertEqual(set(inventory), {"complete.txt"})

    def test_manifest_inventory_checks_all_matrix_and_row_mapping_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            pq.write_table(pa.table({
                "row_id": [0, 1, 2], "app_id": [10, 20, 30], "linux": [True, False, True],
            }), folder / "game_rows.parquet")
            matrix = sparse.csr_matrix(np.asarray([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32))
            info = _save_sparse_matrix(folder, matrix)
            manifest = {
                "status": "complete", "family": "content", "matrix": info,
                "files": file_inventory(folder),
            }
            write_json(folder / "manifest.json", manifest)
            checked = verify_model_artifact(folder, expected_rows=3)
            self.assertEqual(checked["status"], "passed")
            self.assertEqual(checked["matrix_shape"], [3, 2])

    def test_inventory_detects_changed_file_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "piece.txt").write_text("before", encoding="utf-8")
            inventory = file_inventory(folder)
            (folder / "piece.txt").write_text("after", encoding="utf-8")
            with self.assertRaises(ValueError):
                _verify_inventory(folder, inventory)

    def test_artifact_paths_are_confined_to_owned_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "content"
            output.mkdir()
            self.assertEqual(_require_child(output, output / "model"), (output / "model").resolve())
            with self.assertRaises(ValueError):
                _require_child(output, root / "outside")

    def test_train_review_key_sort_stream_deduplicates_without_global_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "split"
            (split / "train").mkdir(parents=True)
            lineage = split / "train/review_lineage.parquet"
            pq.write_table(pa.table({
                "review_key": pa.array([30, 10, 20, 20], type=pa.uint64()),
                "user_key": pa.array([STEAM_BASE + 3, STEAM_BASE + 1, STEAM_BASE + 2, STEAM_BASE + 2], type=pa.uint64()),
                "app_id": pa.array([300, 100, 200, 200], type=pa.int64()),
            }), lineage)
            write_json(split / "manifest.json", {
                "files": {"train/review_lineage.parquet": {"rows": 4, "sha256": digest(lineage)}},
            })
            work = root / "work"
            work.mkdir()
            keys, keys_path, count = _sorted_train_review_keys(split, work)
            self.assertEqual(count, 3)
            self.assertEqual(np.asarray(keys).tolist(), [10, 20, 30])
            self.assertTrue(keys_path.is_file())
            keys._mmap.close()
            del keys
            reused, reused_path, reused_count = _sorted_train_review_keys(split, work)
            self.assertEqual(reused_count, 3)
            self.assertEqual(reused_path, keys_path)
            self.assertEqual(np.asarray(reused).tolist(), [10, 20, 30])
            reused._mmap.close()

    def test_train_review_row_count_uses_ingestion_manifest_schema(self):
        from experiment_content_v3 import _expected_train_review_row_count

        self.assertEqual(_expected_train_review_row_count({"raw_data_rows_parsed": 113_883_717}), 113_883_717)
        self.assertEqual(_expected_train_review_row_count({"canonical_rows_written": 10}), 10)
        with self.assertRaisesRegex(ValueError, "raw source row count"):
            _expected_train_review_row_count({})

    def test_complete_review_scan_can_be_reused_without_rescanning_archive(self):
        from experiment_content_v3 import _completed_review_scan_info

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = root / "sample.parquet"
            sample_lineage = root / "sample_lineage.parquet"
            marker = root / "scan_complete.json"
            pq.write_table(pa.table({"review_text": ["a", "b"]}), sample)
            pq.write_table(pa.table({"review_key": pa.array([1, 2], type=pa.uint64())}), sample_lineage)
            self.assertEqual(
                _completed_review_scan_info(sample, sample_lineage, marker, 100),
                {"raw_csv_rows_scanned": 100,
                 "accepted_train_catalog_review_rows_seen": None,
                 "sampled_unique_review_keys": 2},
            )
            write_json(marker, {"raw_csv_rows_scanned": 100,
                                "accepted_train_catalog_review_rows_seen": 7,
                                "sampled_unique_review_keys": 2,
                                "sample_parquet_sha256": digest(sample)})
            self.assertEqual(
                _completed_review_scan_info(sample, sample_lineage, marker, 100)
                ["accepted_train_catalog_review_rows_seen"], 7
            )

    def test_sampled_review_tuple_must_match_accepted_train_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "split"
            (split / "train").mkdir(parents=True)
            lineage = split / "train/review_lineage.parquet"
            pq.write_table(pa.table({
                "review_key": pa.array([10, 20, 20], type=pa.uint64()),
                "user_key": pa.array([STEAM_BASE + 1, STEAM_BASE + 2, STEAM_BASE + 2], type=pa.uint64()),
                "app_id": pa.array([100, 200, 200], type=pa.int64()),
            }), lineage)
            sample = root / "sample.parquet"
            pq.write_table(pa.table({
                "review_key": pa.array([10, 20], type=pa.uint64()),
                "user_key": pa.array([STEAM_BASE + 1, STEAM_BASE + 2], type=pa.uint64()),
                "app_id": pa.array([100, 200], type=pa.int64()),
            }), sample)
            work = root / "work"
            work.mkdir()
            self.assertEqual(_verify_sampled_train_lineage(sample, lineage, work, 2), 2)
            pq.write_table(pa.table({
                "review_key": pa.array([10, 20], type=pa.uint64()),
                "user_key": pa.array([STEAM_BASE + 1, STEAM_BASE + 2], type=pa.uint64()),
                "app_id": pa.array([100, 201], type=pa.int64()),
            }), sample)
            with self.assertRaisesRegex(ValueError, "review_key/user_key/app_id"):
                _verify_sampled_train_lineage(sample, lineage, work, 2)


if __name__ == "__main__":
    unittest.main()
