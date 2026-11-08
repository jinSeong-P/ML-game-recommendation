"""Focused tests for the v4 time-window and temporal-split rules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from experiment_temporal_v4 import (
    PAIR_SCHEMA,
    T0,
    T1,
    T2,
    TEMPORAL_PROTOCOL,
    choose_target_app,
    create_temporal_cohort_tables,
    digest,
    filter_reserved_training_pairs,
    freeze_protocol,
    history_visible,
    materialize_t0_training_from_lineage,
    release_available,
    target_in_horizon,
    validate_old_temporal_training_artifact,
)


def event(user, app, positive, linux, role, created, updated, review):
    return (user, app, review, positive, role, created, updated, linux)


def temporal_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("""CREATE TEMP TABLE role_events(
        user_key UBIGINT, app_id BIGINT, review_key UBIGINT,
        positive BOOLEAN, role VARCHAR, timestamp_created BIGINT,
        timestamp_updated BIGINT, linux BOOLEAN
    )""")
    con.execute("""CREATE TEMP TABLE candidate_availability(
        app_id BIGINT, available_at_t0 BOOLEAN
    )""")
    con.executemany("INSERT INTO candidate_availability VALUES (?, ?)", [
        (1, True), (2, True), (3, True), (4, True), (5, True),
        (6, False), (7, True), (10, True), (11, True), (12, True),
        (13, True), (20, True),
    ])
    con.executemany("INSERT INTO role_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        # Selection history at T0; later-created or later-updated items are excluded.
        event(101, 1, True, True, "selection", T0, T0, 1),
        event(101, 2, False, False, "selection", T0 - 10, T0, 2),
        event(101, 7, True, True, "selection", T0 + 1, T1 + 1, 3),
        event(101, 8, True, True, "selection", T0 - 1, T0 + 1, 4),
        # Only app 3 is a valid, available, unseen target in (T0, T1].
        event(101, 3, True, True, "selection", T0 + 100, T1, 5),
        event(101, 4, True, True, "selection", T0 + 200, T1 + 1, 6),
        event(101, 5, True, True, "selection", T1 + 1, T1 + 1, 7),
        event(101, 6, True, True, "selection", T0 + 300, T1, 8),
        event(101, 1, True, True, "selection", T0 + 400, T1, 9),
        # Confirmation's lower bound is open and upper bound is closed.
        event(202, 10, True, True, "confirmation", T0, T0, 10),
        event(202, 11, True, True, "confirmation", T1 + 1, T2, 11),
        event(202, 12, True, True, "confirmation", T2 + 1, T2 + 1, 12),
        event(202, 13, True, True, "confirmation", T1, T1, 13),
        # A valid horizon target remains cohort-eligible even with no T0 inputs.
        event(303, 20, True, True, "fusion_fit", T0 + 1, T1, 20),
    ])
    return con


class TemporalV4Tests(unittest.TestCase):
    def test_time_boundary_rules_are_inclusive_only_where_declared(self):
        self.assertTrue(history_visible(T0, T0))
        self.assertFalse(history_visible(None, T0))
        self.assertFalse(history_visible(T0, None))
        self.assertFalse(history_visible(T0 - 1, T0 + 1))
        self.assertFalse(history_visible(T0 + 1, T0))
        self.assertTrue(target_in_horizon(T0 + 1, T1, T0, T1))
        self.assertTrue(target_in_horizon(T1, T1, T0, T1))
        self.assertFalse(target_in_horizon(T0, T1, T0, T1))
        self.assertFalse(target_in_horizon(T1 + 1, T1 + 1, T0, T1))
        self.assertTrue(target_in_horizon(T1 + 1, T2, T1, T2))
        self.assertFalse(target_in_horizon(T1, T1, T1, T2))
        self.assertFalse(target_in_horizon(T2 + 1, T2 + 1, T1, T2))

    def test_unknown_or_post_t0_release_dates_are_unavailable(self):
        self.assertTrue(release_available("2022-12-31"))
        self.assertTrue(release_available("2022-12-30T08:00:00Z"))
        self.assertFalse(release_available("2023-01-01"))
        self.assertFalse(release_available(None))
        self.assertFalse(release_available("coming soon"))

    def test_target_sampler_is_deterministic_and_requires_sorted_unique_apps(self):
        apps = [11, 29, 47]
        chosen = choose_target_app(76561197960265728, apps, role="selection")
        self.assertIn(chosen, apps)
        self.assertEqual(chosen, choose_target_app(76561197960265728, apps, role="selection"))
        with self.assertRaises(ValueError):
            choose_target_app(3, [29, 11], role="selection")
        with self.assertRaises(ValueError):
            choose_target_app(3, [11, 11], role="selection")

    def test_role_cohorts_mask_future_inputs_unavailable_targets_and_confirm_horizon(self):
        con = temporal_fixture()
        selection = create_temporal_cohort_tables(con, "selection")
        target = con.execute("SELECT app_id,timestamp_created,timestamp_updated FROM temporal_selection_selected_targets").fetchall()
        query = con.execute("SELECT app_id,positive FROM temporal_selection_query ORDER BY app_id").fetchall()
        users = con.execute("SELECT n_total_inputs,n_positive_inputs,n_eligible_targets FROM temporal_selection_users").fetchall()
        self.assertEqual(selection["users"], 1)
        self.assertEqual(selection["eligible_target_pairs"], 1)
        self.assertEqual(target, [(3, T0 + 100, T1)])
        self.assertEqual(query, [(1, True), (2, False)])
        self.assertEqual(users, [(2, 1, 1)])
        self.assertEqual(selection["target_query_overlap"], 0)

        confirmation = create_temporal_cohort_tables(con, "confirmation")
        conf_target = con.execute("SELECT app_id,timestamp_created,timestamp_updated FROM temporal_confirmation_selected_targets").fetchall()
        self.assertEqual(confirmation["users"], 1)
        self.assertEqual(conf_target, [(11, T1 + 1, T2)])
        self.assertEqual(confirmation["query_pairs"], 1)

        fusion = create_temporal_cohort_tables(con, "fusion_fit")
        fit_user = con.execute("SELECT n_total_inputs,n_positive_inputs FROM temporal_fusion_fit_users").fetchall()
        self.assertEqual(fusion["users"], 1)
        self.assertEqual(fusion["zero_input_users"], 1)
        self.assertEqual(fit_user, [(0, 0)])
        self.assertEqual(con.execute("SELECT count(*) FROM temporal_fusion_fit_query").fetchone()[0], 0)
        con.close()

    def test_temporal_training_filter_excludes_all_reserved_role_accounts(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TEMP TABLE v4_role_accounts(user_key UBIGINT, role VARCHAR)")
        con.execute("INSERT INTO v4_role_accounts VALUES (20, 'confirmation'), (40, 'selection')")
        source = pa.Table.from_pylist([
            {"user_key": 10, "app_id": 1, "positive": True},
            {"user_key": 20, "app_id": 2, "positive": True},
            {"user_key": 30, "app_id": 3, "positive": True},
            {"user_key": 40, "app_id": 4, "positive": True},
        ], schema=PAIR_SCHEMA)
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.parquet"
            output_path = Path(temp_dir) / "t0.parquet"
            pq.write_table(source, source_path, compression="zstd")
            stats = filter_reserved_training_pairs(con, source_path, output_path)
            rows = pq.read_table(output_path).to_pylist()
        self.assertEqual(stats, {"positive_pairs": 2, "positive_users": 2, "reserved_pair_overlap": 0})
        self.assertEqual([row["user_key"] for row in rows], [10, 30])
        con.close()

    def test_lineage_fallback_uses_both_timestamps_and_excludes_all_roles(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TEMP TABLE input_apps(app_id BIGINT, linux BOOLEAN)")
        con.execute("INSERT INTO input_apps VALUES (1, true), (2, false), (3, true)")
        con.execute("CREATE TEMP TABLE v4_role_accounts(user_key UBIGINT, role VARCHAR)")
        con.execute("INSERT INTO v4_role_accounts VALUES (20, 'confirmation')")
        con.execute("""CREATE TEMP TABLE valid_raw(
            user_key UBIGINT, app_id BIGINT, review_key UBIGINT, positive BOOLEAN,
            timestamp_created BIGINT, timestamp_updated BIGINT
        )""")
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            split = base / "split"
            lineage = split / "train/review_lineage.parquet"
            lineage.parent.mkdir(parents=True)
            rows = [
                (10, 1, 100, True),
                (20, 1, 200, True),
                (30, 1, 300, True),
                (40, 1, 400, True),
                (50, 2, 500, False),
            ]
            schema = pa.schema([
                ("user_key", pa.uint64()), ("app_id", pa.int64()),
                ("review_key", pa.uint64()), ("positive", pa.bool_()),
            ])
            pq.write_table(pa.Table.from_pylist([
                dict(user_key=u, app_id=a, review_key=r, positive=p) for u, a, r, p in rows
            ], schema=schema), lineage, compression="zstd")
            con.executemany("INSERT INTO valid_raw VALUES (?, ?, ?, ?, ?, ?)", [
                (10, 1, 100, True, T0, T0),
                (20, 1, 200, True, T0, T0),
                (30, 1, 300, True, T0 + 1, T0 + 1),
                (40, 1, 400, True, T0 - 1, T0 + 1),
                (50, 2, 500, False, T0, T0),
            ])
            stage = base / "stage"
            stage.mkdir()
            destination = stage / "train/positive_catalog.parquet"
            stats = materialize_t0_training_from_lineage(con, stage, destination, split)
            result = pq.read_table(destination).to_pylist()
        self.assertEqual(stats["positive_pairs"], 1)
        self.assertEqual(result, [{"user_key": 10, "app_id": 1, "positive": True}])
        con.close()

    def test_old_temporal_training_artifact_hash_failure_routes_to_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            temporal = root / "temporal"
            result = temporal / "result"
            builder = root / "experiment_temporal_v3.py"
            snapshot = temporal / "prediction_replay/temporal_builder_snapshot.py"
            protocol = temporal / "protocol.json"
            train = result / "training/positive_catalog.parquet"
            builder.parent.mkdir(parents=True, exist_ok=True)
            builder.write_text("# frozen old builder\n", encoding="utf-8")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(builder.read_bytes())
            protocol.write_text(json.dumps({"cutoff": {"unix_seconds": T0}}), encoding="utf-8")
            train.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([
                {"user_key": 10, "app_id": 1, "positive": True},
                {"user_key": 20, "app_id": 2, "positive": True},
            ], schema=PAIR_SCHEMA), train, compression="zstd")
            train_receipt = {"bytes": train.stat().st_size, "sha256": digest(train), "rows": 2}
            model = {
                "status": "complete", "cutoff_unix": T0, "protocol_sha256": digest(protocol),
                "source_fingerprints": {}, "train_support": {"positive_pairs": 2},
                "builder_sha256": digest(builder),
            }
            (result / "model").mkdir(parents=True, exist_ok=True)
            (result / "model/manifest.json").write_text(json.dumps(model), encoding="utf-8")
            report = {
                "status": "complete", "protocol": {"sha256": digest(protocol)},
                "source_fingerprints": {}, "builder_sha256": digest(builder),
                "files": {"training/positive_catalog.parquet": train_receipt},
                "training": {"positive_pairs": 2},
            }
            (result / "report.json").write_text(json.dumps(report), encoding="utf-8")
            valid, reasons, path = validate_old_temporal_training_artifact(
                temporal, root, builder, check_source_hashes=False)
            self.assertTrue(valid, reasons)
            self.assertEqual(path, train)
            with train.open("ab") as stream:
                stream.write(b"changed")
            valid, reasons, path = validate_old_temporal_training_artifact(
                temporal, root, builder, check_source_hashes=False)
            self.assertFalse(valid)
            self.assertIsNone(path)
            self.assertTrue(any("old_output_hash_or_size_mismatch" in reason for reason in reasons))

    def test_temporal_protocol_is_immutable_once_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _ = freeze_protocol(Path(temp_dir) / "temporal_protocol.json")
            self.assertEqual(json.loads(path.read_text("utf-8")), TEMPORAL_PROTOCOL)
            altered = json.loads(path.read_text("utf-8"))
            altered["cutoffs"]["T0"]["unix_seconds"] += 1
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze_protocol(path)


if __name__ == "__main__":
    unittest.main()
