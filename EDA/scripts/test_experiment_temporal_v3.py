"""Focused checks for the cutoff protocol, cohort ranking, and K100 refit."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from unittest import mock

import experiment_temporal_v3 as temporal

from experiment_temporal_v3 import (
    CUTOFF_UNIX,
    TEMPORAL_PROTOCOL,
    _asof_history_predicate,
    _fit_knn,
    choose_target_app,
    freeze_protocol,
    iter_query_groups,
    rank_score_row,
    uniform_index,
)


class TemporalProtocolTests(unittest.TestCase):
    def test_protocol_is_frozen_without_reading_data_and_refuses_revisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol_path, first_sha = freeze_protocol(Path(temporary))
            self.assertEqual(json.loads(protocol_path.read_text("utf-8")), TEMPORAL_PROTOCOL)
            second_path, second_sha = freeze_protocol(Path(temporary))
            self.assertEqual(protocol_path, second_path)
            self.assertEqual(first_sha, second_sha)
            self.assertIn("timestamp_updated", TEMPORAL_PROTOCOL["training"]["eligibility"])
            self.assertIn("full-snapshot split_v1", " ".join(TEMPORAL_PROTOCOL["known_scope_limits"]))
            payload = json.loads(protocol_path.read_text("utf-8"))
            payload["cutoff"]["unix_seconds"] += 1
            protocol_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze_protocol(Path(temporary))

    def test_shared_asof_predicate_requires_both_timestamps_and_includes_cutoff(self):
        predicate = _asof_history_predicate("r", CUTOFF_UNIX)
        self.assertIn("r.timestamp_created IS NOT NULL", predicate)
        self.assertIn("r.timestamp_updated IS NOT NULL", predicate)
        self.assertIn(f"r.timestamp_created <= {CUTOFF_UNIX}", predicate)
        self.assertIn(f"r.timestamp_updated <= {CUTOFF_UNIX}", predicate)

    def test_uniform_target_selection_is_stable_and_ascending_list_is_required(self):
        self.assertEqual(uniform_index(2**53 + 7, 17), uniform_index(2**53 + 7, 17))
        apps = [101, 202, 303, 404]
        selected = choose_target_app(2**53 + 7, apps)
        self.assertIn(selected, apps)
        self.assertEqual(selected, choose_target_app(2**53 + 7, apps))
        with self.assertRaises(ValueError):
            choose_target_app(7, [101, 303, 202])

    def test_target_rank_excludes_all_visible_games_and_breaks_ties_by_app_id(self):
        candidates = np.array([100, 200, 300, 400], dtype=np.int64)
        indices = np.array([0, 1, 2, 3], dtype=np.int32)
        scores = np.array([0.9, 0.8, 0.8, 0.7], dtype=np.float64)
        # App 200 represents a visible query game; all query rows enter the mask,
        # regardless of whether the row is positive or negative.
        rank, available, target_score = rank_score_row(
            candidates, indices, scores, 300, [200, 999])
        self.assertEqual(rank, 2)
        self.assertEqual(available, 3)
        self.assertEqual(target_score, 0.8)
        masked_rank, masked_available, _ = rank_score_row(
            candidates, indices, scores, 300, [200, 300])
        self.assertIsNone(masked_rank)
        self.assertEqual(masked_available, 2)
        missing_rank, _, _ = rank_score_row(
            candidates, np.array([0, 1]), np.array([0.9, 0.8]), 300, [])
        self.assertIsNone(missing_rank)

    def test_query_groups_remain_intact_across_parquet_batch_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "query.parquet"
            users = [2**53 + 1, 2**53 + 1, 2**53 + 2,
                     2**53 + 2, 2**53 + 2, 2**53 + 4]
            pq.write_table(pa.table({
                "user_key": pa.array(users, type=pa.uint64()),
                "app_id": pa.array([10, 20, 30, 40, 50, 60], type=pa.int64()),
                "positive": pa.array([True, False, True, True, False, True]),
            }), path, row_group_size=2)
            groups = list(iter_query_groups(path, scan_batch_size=2))
            self.assertEqual([group[0] for group in groups], [2**53 + 1, 2**53 + 2, 2**53 + 4])
            np.testing.assert_array_equal(groups[0][1], [10, 20])
            np.testing.assert_array_equal(groups[0][2], [True, False])
            np.testing.assert_array_equal(groups[1][1], [30, 40, 50])
            np.testing.assert_array_equal(groups[1][2], [True, True, False])

    def test_validation_materialization_uses_both_timestamps_and_keeps_target_out_of_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lineage_path = root / "validation_lineage.parquet"
            input_path = root / "input_apps.parquet"
            candidates_path = root / "linux_candidates.parquet"
            users_path = root / "validation_users.parquet"
            cohort_path = root / "cohort"
            users = [2**53 + 1, 2**53 + 2, 2**53 + 3]
            lineages = []
            raw_rows = []
            review_key = 1

            def add_event(user, app, positive, created, updated):
                nonlocal review_key
                lineages.append((user, app, review_key, positive))
                raw_rows.append((user, app, review_key, positive, created, updated))
                review_key += 1

            # One fully eligible history with two later, unseen Linux targets.
            for app, positive in zip([10, 11, 12, 13, 14], [True, True, False, False, False]):
                add_event(users[0], app, positive, CUTOFF_UNIX, CUTOFF_UNIX)
            add_event(users[0], 30, True, CUTOFF_UNIX + 1, CUTOFF_UNIX + 1)
            add_event(users[0], 31, True, CUTOFF_UNIX + 2, CUTOFF_UNIX + 2)

            # Four rows are visible because the fifth was updated after cutoff.
            for app, positive in zip([10, 11, 12, 13, 14], [True, True, False, False, False]):
                updated = CUTOFF_UNIX + 1 if app == 14 else CUTOFF_UNIX
                add_event(users[1], app, positive, CUTOFF_UNIX, updated)
            add_event(users[1], 31, True, CUTOFF_UNIX + 3, CUTOFF_UNIX + 3)

            # A previously visible Linux app cannot also be selected as target.
            for app, positive in zip([10, 11, 12, 13, 30], [True, True, False, False, True]):
                add_event(users[2], app, positive, CUTOFF_UNIX, CUTOFF_UNIX)
            add_event(users[2], 30, True, CUTOFF_UNIX + 4, CUTOFF_UNIX + 4)
            add_event(users[2], 31, True, CUTOFF_UNIX + 5, CUTOFF_UNIX + 5)

            pq.write_table(pa.table({
                "user_key": pa.array([row[0] for row in lineages], type=pa.uint64()),
                "app_id": pa.array([row[1] for row in lineages], type=pa.int64()),
                "review_key": pa.array([row[2] for row in lineages], type=pa.uint64()),
                "positive": pa.array([row[3] for row in lineages], type=pa.bool_()),
            }), lineage_path)
            pq.write_table(pa.table({
                "app_id": pa.array([10, 11, 12, 13, 14, 30, 31], type=pa.int64()),
                "linux": pa.array([False, False, False, False, False, True, True]),
            }), input_path)
            pq.write_table(pa.table({"app_id": pa.array([30, 31], type=pa.int64())}), candidates_path)
            pq.write_table(pa.table({"user_key": pa.array(users, type=pa.uint64())}), users_path)
            raw_schema = pa.schema([
                ("user_key", pa.uint64()), ("app_id", pa.int64()),
                ("review_key", pa.uint64()), ("positive", pa.bool_()),
                ("timestamp_created", pa.int64()), ("timestamp_updated", pa.int64()),
            ])

            def relation(con):
                con.register("_test_raw", pa.Table.from_pylist([
                    {"user_key": row[0], "app_id": row[1], "review_key": row[2],
                     "positive": row[3], "timestamp_created": row[4],
                     "timestamp_updated": row[5]} for row in raw_rows
                ], schema=raw_schema))
                con.execute("CREATE TABLE valid_raw AS SELECT * FROM _test_raw")
                con.unregister("_test_raw")

            con = duckdb.connect(":memory:")
            try:
                relation(con)
                with mock.patch.multiple(
                    temporal,
                    VALIDATION_LINEAGE=lineage_path,
                    INPUT_APPS=input_path,
                    LINUX_CANDIDATES=candidates_path,
                    VALIDATION_USERS=users_path,
                ):
                    counts = temporal._materialize_validation(con, cohort_path)
            finally:
                con.close()

            self.assertEqual(counts["cohort_users"], 2)
            query = pq.read_table(cohort_path / "query.parquet").to_pydict()
            targets = pq.read_table(cohort_path / "targets.parquet").to_pydict()
            exclusions = pq.read_table(cohort_path / "exclusions.parquet").to_pydict()
            by_user = {int(user): set() for user in users}
            for user, app in zip(query["user_key"], query["app_id"]):
                by_user[int(user)].add(int(app))
            target_by_user = dict(zip(targets["user_key"], targets["app_id"]))
            self.assertEqual(set(target_by_user), {users[0], users[2]})
            self.assertEqual(target_by_user[users[2]], 31)
            self.assertNotIn(target_by_user[users[0]], by_user[users[0]])
            self.assertNotIn(target_by_user[users[2]], by_user[users[2]])
            self.assertNotIn(31, by_user[users[0]])
            self.assertNotIn(31, by_user[users[2]])
            self.assertEqual(len(exclusions["app_id"]), len(query["app_id"]))


class TemporalKnnFitTests(unittest.TestCase):
    def test_training_materialization_buckets_accepted_asof_lineage_before_global_sort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / "train"
            train_dir.mkdir()
            lineage_path = root / "train_lineage.parquet"
            input_path = root / "input_apps.parquet"
            destination = train_dir / "positive_catalog.parquet"
            raw_schema = pa.schema([
                ("user_key", pa.uint64()), ("app_id", pa.int64()),
                ("review_key", pa.uint64()), ("positive", pa.bool_()),
                ("timestamp_created", pa.int64()), ("timestamp_updated", pa.int64()),
            ])
            raw_rows = [
                (2**53 + 1, 10, 1001, True, CUTOFF_UNIX, CUTOFF_UNIX),
                (2**53 + 1, 10, 1002, True, CUTOFF_UNIX + 1, CUTOFF_UNIX + 1),
                (2**53 + 2, 20, 2001, True, CUTOFF_UNIX - 1, CUTOFF_UNIX + 1),
                (2**53 + 2, 30, 2002, True, CUTOFF_UNIX - 1, CUTOFF_UNIX - 1),
            ]
            pq.write_table(pa.table({
                "user_key": pa.array([row[0] for row in raw_rows], type=pa.uint64()),
                "app_id": pa.array([row[1] for row in raw_rows], type=pa.int64()),
                "review_key": pa.array([row[2] for row in raw_rows], type=pa.uint64()),
                "positive": pa.array([row[3] for row in raw_rows], type=pa.bool_()),
            }), lineage_path)
            pq.write_table(pa.table({
                "app_id": pa.array([10, 20, 30], type=pa.int64()),
                "linux": pa.array([True, False, True]),
            }), input_path)
            con = duckdb.connect(":memory:")
            try:
                con.register("_raw", pa.Table.from_pylist([
                    {"user_key": row[0], "app_id": row[1], "review_key": row[2],
                     "positive": row[3], "timestamp_created": row[4],
                     "timestamp_updated": row[5]} for row in raw_rows
                ], schema=raw_schema))
                con.execute("CREATE TABLE valid_raw AS SELECT * FROM _raw")
                con.unregister("_raw")
                with mock.patch.multiple(
                    temporal,
                    TRAIN_LINEAGE=lineage_path,
                    INPUT_APPS=input_path,
                    TRAIN_PARTITION_BUCKETS=4,
                ):
                    counts = temporal._materialize_train(con, destination)
            finally:
                con.close()
            rows = pq.read_table(destination).to_pydict()
            self.assertEqual(counts["eligible_review_events"], 2)
            self.assertEqual(counts["positive_pairs"], 2)
            self.assertEqual(counts["positive_users"], 2)
            self.assertEqual(rows["user_key"], [2**53 + 1, 2**53 + 2])
            self.assertEqual(rows["app_id"], [10, 30])
            self.assertEqual(rows["positive"], [True, True])

    def test_cutoff_k100_refit_uses_full_catalog_and_singleton_support(self):
        apps = np.array([10, 20, 30, 40], dtype=np.int64)
        linux = np.array([True, True, False, True], dtype=bool)
        candidates = np.array([10, 20, 40], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.parquet"
            pq.write_table(pa.table({
                "user_key": pa.array([2**53 + 1, 2**53 + 1, 2**53 + 2, 2**53 + 2],
                                      type=pa.uint64()),
                "app_id": pa.array([10, 20, 20, 30], type=pa.int64()),
                "positive": pa.array([True, True, True, True]),
            }), train)
            model_dir = root / "model"
            model_dir.mkdir()
            counts = _fit_knn(train, model_dir, apps, linux, candidates,
                              expected_pairs=4, expected_users=2, block_size=1)
            matrix = sparse.load_npz(model_dir / "similarities_csr.npz").tocsr()
            self.assertEqual(matrix.shape, (3, 4))
            self.assertEqual(counts["positive_train_singleton_users"], 0)
            self.assertEqual(counts["input_games_with_positive_support"], 3)
            self.assertEqual(counts["linux_candidates_without_positive_support"], 1)
            self.assertAlmostEqual(matrix[0, 1], 1 / np.sqrt(2))
            self.assertAlmostEqual(matrix[1, 0], 1 / np.sqrt(2))
            self.assertAlmostEqual(matrix[1, 2], 1 / np.sqrt(2))
            self.assertEqual(matrix[2].nnz, 0)
            rows = pq.read_table(model_dir / "game_rows.parquet").to_pydict()
            self.assertEqual(rows["positive_user_support"], [1, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()
