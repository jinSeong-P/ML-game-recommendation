"""Behavioral tests for held-out targets and persisted split isolation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from build_modeling_splits import create_holdout, inventory, uniform_index, validate_holdout
from eda_large_review_frank import BASE
from verify_modeling_splits import verify_split


def validation_account(user):
    return int(hashlib.md5(f"steam-eda-validation-v1:{user}".encode()).hexdigest()[:15], 16) % 10 == 0


class TargetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("CREATE TABLE hist(user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN)")
        data = []
        for u in range(1, 6):
            n = 5 if u == 4 else 6
            for app in range(1, n + 1):
                positive = app <= (1 if u == 3 else 2)
                linux = app == 1 and u != 5
                data.append((BASE + u, app, positive, linux))
        self.con.executemany("INSERT INTO hist VALUES (?,?,?,?)", data)

    def tearDown(self):
        self.con.close()

    def test_after_holdout_thresholds_and_both_label_masking(self):
        create_holdout(self.con, "hist", "test", 20261107)
        self.assertEqual(self.con.execute("SELECT count(*) FROM test_cohort").fetchone()[0], 2)
        self.assertEqual(self.con.execute("SELECT DISTINCT n_total_inputs,n_positive_inputs FROM test_cohort").fetchall(), [(5, 1)])
        self.assertEqual(self.con.execute("SELECT count(*) FROM test_exclusions").fetchone()[0], 10)
        self.assertEqual(self.con.execute("SELECT count(*) FROM test_query WHERE NOT positive").fetchone()[0], 8)
        self.assertEqual(self.con.execute("SELECT DISTINCT app_id FROM test_targets").fetchall(), [(1,)])
        self.assertTrue(all(v == 0 for v in validate_holdout(self.con, "test").values()))

    def test_reservation_table_is_never_shadowed(self):
        self.con.execute(f"CREATE TABLE test_users AS SELECT {BASE+3}::UBIGINT user_key")
        create_holdout(self.con, "hist", "test", 7)
        self.assertEqual(self.con.execute("SELECT * FROM test_users").fetchall(), [(BASE+3,)])

    def test_selection_is_invariant_to_row_order_and_unrelated_users(self):
        self.con.execute(f"UPDATE hist SET linux=true WHERE user_key={BASE+1} AND app_id=2")
        create_holdout(self.con, "hist", "test", 9)
        first = self.con.execute("SELECT * FROM test_targets ORDER BY user_key").fetchall()
        self.con.execute("CREATE TABLE reversed_hist AS SELECT * FROM hist ORDER BY app_id DESC,user_key DESC")
        con2 = duckdb.connect()
        try:
            con2.register("source", self.con.execute("SELECT * FROM reversed_hist").fetch_arrow_table())
            con2.execute("CREATE TABLE hist AS SELECT * FROM source")
            create_holdout(con2, "hist", "test", 9)
            self.assertEqual(first, con2.execute("SELECT * FROM test_targets ORDER BY user_key").fetchall())
        finally:
            con2.close()

    def test_rejection_sampling_rejects_upper_tail(self):
        fake = Mock()
        fake.side_effect = [Mock(digest=lambda: ((1 << 256) - 1).to_bytes(32, "big")),
                            Mock(digest=lambda: (12).to_bytes(32, "big"))]
        with patch("build_modeling_splits.hashlib.sha256", fake):
            self.assertEqual(uniform_index(BASE+1, 10, 5, "test"), 2)
        self.assertEqual(fake.call_count, 2)
        self.assertTrue(fake.call_args_list[0].args[0].endswith(b":0"))
        self.assertTrue(fake.call_args_list[1].args[0].endswith(b":1"))

    def test_no_target_candidates_is_rejected(self):
        with self.assertRaises(ValueError):
            uniform_index(BASE, 0, 1, "test")

    def test_validation_fails_if_target_is_masked(self):
        create_holdout(self.con, "hist", "test", 1)
        self.con.execute("INSERT INTO test_exclusions SELECT user_key,app_id FROM test_targets")
        with self.assertRaisesRegex(ValueError, "masked_targets"):
            validate_holdout(self.con, "test")


class PersistedSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train_user = next(u for u in range(BASE+10, BASE+100) if not validation_account(u))
        self.val_user = next(u for u in range(BASE+100, BASE+200) if validation_account(u))
        self.test_user = BASE+1000
        self.pairs = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64()), ("positive", pa.bool_())])
        self.hist = self.pairs.append(pa.field("linux", pa.bool_()))
        self.lineage = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64()), ("review_key", pa.uint64()), ("positive", pa.bool_())])
        self.write("train/interactions.parquet", [(self.train_user, 2, True), (self.train_user, 9, False)], self.pairs)
        self.write("train/positive_catalog.parquet", [(self.train_user, 2, True)], self.pairs)
        self.write("train/review_lineage.parquet", [(self.train_user, 2, 1, True), (self.train_user, 9, 2, False)], self.lineage)
        self.write("catalog/input_apps.parquet", [(a, a in {1, 2}) for a in range(1, 7)], pa.schema([("app_id", pa.int64()), ("linux", pa.bool_())]))
        self.write("catalog/linux_candidates.parquet", [(1,), (2,)], pa.schema([("app_id", pa.int64())]))
        self.write("reserved_test_accounts.parquet", [(self.test_user,), (self.test_user+1,)], pa.schema([("user_key", pa.uint64())]))
        for split, user, offset in [("validation", self.val_user, 10), ("test", self.test_user, 30)]:
            history = [(user, a, a <= 2, a <= 2) for a in range(1, 7)]
            query = history[1:]
            self.write(f"{split}/query.parquet", query, self.hist)
            self.write(f"{split}/targets.parquet", history[:1], self.hist)
            self.write(f"{split}/users.parquet", [(user, 5, 1, 2)], pa.schema([("user_key", pa.uint64()), ("n_total_inputs", pa.int64()), ("n_positive_inputs", pa.int64()), ("n_eligible_targets", pa.int64())]))
            self.write(f"{split}/exclusions.parquet", [(row[0], row[1]) for row in query], pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64())]))
            self.write(f"{split}/review_lineage.parquet", [(u, a, offset+a, p) for u, a, p, _ in history], self.lineage)
            if split == "validation":
                self.write("validation/interactions.parquet", [row[:3] for row in history], self.pairs)
        self.refresh_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, rows, schema):
        path = self.root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        table = pa.Table.from_arrays([pa.array([r[i] for r in rows], type=f.type) for i, f in enumerate(schema)], schema=schema)
        pq.write_table(table, path)

    def refresh_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps({"files": inventory(self.root)}), encoding="utf-8")

    def test_valid_split_retains_singleton_and_negative_archives(self):
        result = verify_split(self.root)
        self.assertTrue(result["passed"])
        self.assertEqual(result["counts"]["positive_singleton_accounts"], 1)
        self.assertEqual(result["counts"]["reserved_test_accounts"], 2)

    def test_lineage_review_id_leak_is_detected(self):
        rows = [(self.test_user, a, 1 if a == 1 else 30+a, a <= 2) for a in range(1, 7)]
        self.write("test/review_lineage.parquet", rows, self.lineage)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "train_test_original_review_intersections"):
            verify_split(self.root)

    def test_changed_parquet_fingerprint_is_detected(self):
        path = self.root / "test/targets.parquet"
        with path.open("ab") as f:
            f.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "resized"):
            verify_split(self.root)


if __name__ == "__main__":
    unittest.main()
