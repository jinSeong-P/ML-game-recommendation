"""Focused tests for the v4 split reservation and no-leakage rules."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_experiments_v4_split import (
    DEFAULT_QUOTAS,
    PAIR_SCHEMA,
    ROLE_NAMES,
    STRATA,
    build_role_assignments,
    create_role_cohort_tables,
    filter_training_positive,
    role_for_rank,
    uniform_index,
    uniform_target,
)


def assignment_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("""CREATE TEMP TABLE user_stats(
        user_key UBIGINT,
        n_total_before_hide BIGINT,
        n_positive_before_hide BIGINT,
        n_eligible_targets BIGINT
    )""")
    con.execute("CREATE TEMP TABLE prior_evaluation_accounts(user_key UBIGINT)")
    rows = []
    user_key = 100
    for stratum, original_total in (("1-2", 2), ("3-4", 4), ("5+", 6)):
        # Four available users per stratum leaves three slots after one is excluded.
        for _ in range(4):
            rows.append((user_key, original_total, 2, 1))
            user_key += 1
    con.executemany("INSERT INTO user_stats VALUES (?, ?, ?, ?)", rows)
    con.execute("INSERT INTO prior_evaluation_accounts VALUES (100), (104), (108)")
    return con


class ExperimentV4SplitTests(unittest.TestCase):
    def test_role_rank_boundaries_are_disjoint(self):
        quotas = {
            stratum: {"fusion_fit": 2, "selection": 3, "confirmation": 4}
            for stratum in STRATA
        }
        self.assertEqual(role_for_rank("1-2", 1, quotas), "fusion_fit")
        self.assertEqual(role_for_rank("1-2", 2, quotas), "fusion_fit")
        self.assertEqual(role_for_rank("1-2", 3, quotas), "selection")
        self.assertEqual(role_for_rank("1-2", 5, quotas), "selection")
        self.assertEqual(role_for_rank("1-2", 6, quotas), "confirmation")
        self.assertEqual(role_for_rank("1-2", 9, quotas), "confirmation")
        self.assertIsNone(role_for_rank("1-2", 10, quotas))
        self.assertIsNone(role_for_rank("unknown", 1, quotas))

    def test_role_assignment_excludes_known_evaluations_and_is_deterministic(self):
        quotas = {stratum: {role: 1 for role in ROLE_NAMES} for stratum in STRATA}
        con = assignment_fixture()
        build_role_assignments(con, seed=17, quotas=quotas)
        first = con.execute(
            "SELECT user_key, role, input_stratum FROM role_assignments ORDER BY user_key"
        ).fetchall()
        self.assertEqual(len(first), 9)
        self.assertEqual(
            con.execute("SELECT count(*) FROM role_assignments a JOIN prior_evaluation_accounts p USING(user_key)").fetchone()[0],
            0,
        )
        self.assertEqual(
            con.execute("SELECT count(*) - count(DISTINCT user_key) FROM role_assignments").fetchone()[0],
            0,
        )
        by_role_stratum = con.execute(
            "SELECT role, input_stratum, count(*) FROM role_assignments GROUP BY 1,2"
        ).fetchall()
        self.assertEqual(len(by_role_stratum), 9)
        con.close()

        con2 = assignment_fixture()
        build_role_assignments(con2, seed=17, quotas=quotas)
        second = con2.execute(
            "SELECT user_key, role, input_stratum FROM role_assignments ORDER BY user_key"
        ).fetchall()
        self.assertEqual(first, second)
        con2.close()

    def test_single_target_is_uniformly_indexed_and_hidden_from_query(self):
        ids = [101, 205, 999]
        chosen = uniform_target(76561197960265728, ids, seed=17, role="confirmation")
        self.assertIn(chosen, ids)
        self.assertEqual(chosen, uniform_target(76561197960265728, ids, seed=17, role="confirmation"))
        self.assertGreaterEqual(uniform_index(123, 3, 17, "selection"), 0)
        self.assertLess(uniform_index(123, 3, 17, "selection"), 3)

        con = duckdb.connect(":memory:")
        con.execute("""CREATE TEMP TABLE user_stats(
            user_key UBIGINT, n_total_before_hide BIGINT,
            n_positive_before_hide BIGINT, n_eligible_targets BIGINT
        )""")
        con.execute("INSERT INTO user_stats VALUES (1001, 3, 2, 1)")
        con.execute("CREATE TEMP TABLE prior_evaluation_accounts(user_key UBIGINT)")
        con.execute("INSERT INTO prior_evaluation_accounts VALUES (9000)")
        con.execute("""CREATE TEMP TABLE role_assignments(
            user_key UBIGINT, role VARCHAR, input_stratum VARCHAR, sample_rank BIGINT
        )""")
        con.execute("INSERT INTO role_assignments VALUES (1001, 'fusion_fit', '1-2', 1)")
        con.execute("""CREATE TEMP TABLE selected_history(
            user_key UBIGINT, app_id BIGINT, positive BOOLEAN, linux BOOLEAN, role VARCHAR
        )""")
        con.execute("""INSERT INTO selected_history VALUES
            (1001, 10, true, true, 'fusion_fit'),
            (1001, 20, true, false, 'fusion_fit'),
            (1001, 30, false, true, 'fusion_fit')""")
        create_role_cohort_tables(con, "fusion_fit", seed=17)
        target = con.execute("SELECT user_key, app_id, positive, linux, target_order FROM fusion_fit_targets").fetchall()
        query = con.execute("SELECT user_key, app_id, positive, linux FROM fusion_fit_query ORDER BY app_id").fetchall()
        users = con.execute("SELECT * FROM fusion_fit_users").fetchall()
        exclusions = con.execute("SELECT * FROM fusion_fit_exclusions ORDER BY app_id").fetchall()
        self.assertEqual(target, [(1001, 10, True, True, 1)])
        self.assertEqual(query, [(1001, 20, True, False), (1001, 30, False, True)])
        self.assertEqual(users, [(1001, 2, 1, 1)])
        self.assertEqual(exclusions, [(1001, 20), (1001, 30)])
        self.assertFalse(any(row[1] == target[0][1] for row in query))
        con.close()

    def test_filtered_training_pairs_exclude_every_reserved_account_and_stay_sorted(self):
        source_rows = [
            (10, 1, True), (10, 4, True),
            (20, 2, True), (20, 3, True),
            (30, 5, True),
        ]
        source = pa.Table.from_pylist(
            [{"user_key": u, "app_id": a, "positive": p} for u, a, p in source_rows],
            schema=PAIR_SCHEMA,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "train.parquet"
            output_path = Path(temp_dir) / "filtered.parquet"
            pq.write_table(source, source_path, compression="zstd")
            rows, users = filter_training_positive(
                source_path,
                output_path,
                np.asarray([20], dtype=np.uint64),
            )
            result = pq.read_table(output_path).to_pylist()
            result_schema = pq.ParquetFile(output_path).schema_arrow
        self.assertEqual(rows, 3)
        self.assertEqual(users, 2)
        self.assertEqual([(r["user_key"], r["app_id"]) for r in result], [(10, 1), (10, 4), (30, 5)])
        self.assertEqual(result_schema, PAIR_SCHEMA)

    def test_default_quota_is_fixed_to_requested_short_medium_primary_counts(self):
        self.assertEqual(DEFAULT_QUOTAS["1-2"], {"fusion_fit": 5_000, "selection": 5_000, "confirmation": 5_000})
        self.assertEqual(DEFAULT_QUOTAS["3-4"], {"fusion_fit": 5_000, "selection": 5_000, "confirmation": 5_000})
        self.assertEqual(DEFAULT_QUOTAS["5+"], {"fusion_fit": 10_000, "selection": 10_000, "confirmation": 10_000})


if __name__ == "__main__":
    unittest.main()
