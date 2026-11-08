"""Small synthetic tests for the supplemental v3 cohort builder."""

import unittest

import duckdb

from backlog_cohorts_v3 import (
    APP_REVIEW_CAP,
    _create_frank_augmentation,
    _create_kieran_only_mask,
    create_capped_history,
    create_multi_holdout,
    create_single_holdout,
    create_temporal_holdout,
    uniform_index,
    validate_cohort,
)
from eda_large_review_frank import BASE


class BacklogCohortTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("""CREATE TABLE history(
            user_key UBIGINT, app_id BIGINT, positive BOOLEAN, linux BOOLEAN,
            timestamp_created BIGINT, playtime_forever DOUBLE)""")
        rows = []
        # User 1 qualifies for minimum-three single-target and two-target tests.
        for app, positive, linux, ts in [
            (1, True, True, 10), (2, True, True, 12), (3, True, False, 14),
            (4, False, False, 16), (5, False, False, 18), (6, False, False, 20),
            (7, False, False, 22), (8, False, False, 24),
        ]:
            rows.append((BASE + 1, app, positive, linux, ts, float(ts)))
        # User 2 has a target but no positive left after hiding it.
        for app, positive, linux, ts in [(1, True, True, 10), (2, False, False, 12),
                                         (3, False, False, 14), (4, False, False, 16)]:
            rows.append((BASE + 2, app, positive, linux, ts, float(ts)))
        self.con.executemany("INSERT INTO history VALUES (?,?,?,?,?,?)", rows)
        self.con.execute("CREATE TABLE _v3_input_apps AS SELECT app_id,linux FROM history")
        self.con.execute("CREATE TABLE _v3_linux_candidates AS SELECT DISTINCT app_id FROM history WHERE linux")

    def tearDown(self):
        self.con.close()

    def test_single_target_minimum_inputs_and_mask(self):
        create_single_holdout(self.con, "history", "min3", 123, 3, 1, "validation")
        users = self.con.execute("SELECT * FROM min3_users ORDER BY user_key").fetchall()
        self.assertEqual([row[0] for row in users], [BASE + 1])
        self.assertEqual(users[0][1:3], (7, 2))
        self.assertEqual(self.con.execute("SELECT count(*) FROM min3_targets").fetchone()[0], 1)
        self.assertEqual(self.con.execute("SELECT count(*) FROM min3_query").fetchone()[0], 7)
        checks = validate_cohort(self.con, "min3", 3, 1)
        self.assertTrue(all(value == 0 for value in checks.values()))
        self.assertEqual(uniform_index(BASE + 1, 9, 123, "validation"),
                         uniform_index(BASE + 1, 9, 123, "validation"))

    def test_multi_target_selection_is_without_replacement_and_audited(self):
        create_multi_holdout(self.con, "history", "multi2", seed=20261107,
                             min_total=5, min_positive=1, target_count=2)
        targets = self.con.execute("SELECT app_id,target_order FROM multi2_targets ORDER BY target_order").fetchall()
        self.assertEqual(len(targets), 2)
        self.assertEqual({row[1] for row in targets}, {1, 2})
        self.assertEqual(len({row[0] for row in targets}), 2)
        self.assertEqual(self.con.execute("SELECT count(*) FROM multi2_query").fetchone()[0], 6)
        self.assertTrue(all(v == 0 for v in validate_cohort(self.con, "multi2", 5, 1, 2).values()))

    def test_app_review_cap_is_deterministic_and_respected(self):
        self.con.execute("CREATE TABLE cap_source(user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN)")
        self.con.executemany("INSERT INTO cap_source VALUES (?,?,?,?)",
                             [(BASE + i, 99, i % 2 == 0, True) for i in range(APP_REVIEW_CAP + 5)])
        create_capped_history(self.con, "cap_source", "cap_one", cap=APP_REVIEW_CAP, seed=5)
        first = self.con.execute("SELECT user_key FROM cap_one ORDER BY user_key").fetchall()
        self.assertEqual(len(first), APP_REVIEW_CAP)
        self.con.execute("CREATE TABLE cap_source_reversed AS SELECT * FROM cap_source ORDER BY user_key DESC")
        create_capped_history(self.con, "cap_source_reversed", "cap_two", cap=APP_REVIEW_CAP, seed=5)
        second = self.con.execute("SELECT user_key FROM cap_two ORDER BY user_key").fetchall()
        self.assertEqual(first, second)
        self.assertLessEqual(self.con.execute("SELECT max(n) FROM (SELECT count(*) n FROM cap_one GROUP BY app_id)").fetchone()[0], APP_REVIEW_CAP)

    def test_temporal_holdout_uses_only_strictly_earlier_events(self):
        self.con.execute("CREATE TABLE temporal_history AS SELECT * FROM history WHERE false")
        events = [
            (BASE + 10, 1, True, True, 10, 1.0),
            (BASE + 10, 2, True, False, 12, 1.0),
            (BASE + 10, 3, False, False, 14, 1.0),
            (BASE + 10, 4, False, False, 16, 1.0),
            (BASE + 10, 5, False, False, 18, 1.0),
            (BASE + 10, 6, True, True, 30, 1.0),
            (BASE + 10, 7, True, False, 32, 1.0),  # same-time row is excluded
            (BASE + 10, 8, True, True, 32, 1.0),  # latest positive Linux target
            (BASE + 10, 9, False, False, 35, 1.0),  # future non-target
            (BASE + 11, 1, True, True, 10, 1.0),
            (BASE + 11, 2, False, False, 11, 1.0),
            (BASE + 11, 3, False, False, 12, 1.0),
            (BASE + 11, 4, False, False, 13, 1.0),
            (BASE + 11, 5, False, False, 14, 1.0),
        ]
        self.con.executemany("INSERT INTO temporal_history VALUES (?,?,?,?,?,?)", events)
        create_temporal_holdout(self.con, "temporal_history", 20, "temporal", 5, 1)
        self.assertEqual(self.con.execute("SELECT user_key,app_id FROM temporal_targets").fetchall(), [(BASE + 10, 8)])
        query_apps = {row[0] for row in self.con.execute("SELECT app_id FROM temporal_query").fetchall()}
        self.assertEqual(query_apps, {1, 2, 3, 4, 5, 6})
        self.assertNotIn(7, query_apps)
        self.assertNotIn(8, query_apps)
        self.assertNotIn(9, query_apps)
        self.assertEqual(self.con.execute("SELECT n_total_inputs,n_positive_inputs FROM temporal_users").fetchall(), [(6, 3)])

    def test_frank_augmentation_removes_overlap_target_and_future_rows(self):
        self.con.execute("""CREATE TABLE _v3_frank_base_users AS
            SELECT ?::UBIGINT user_key,5::BIGINT n_total_inputs,1::BIGINT n_positive_inputs,2::BIGINT n_eligible_targets""", [BASE+20])
        self.con.execute("CREATE TABLE _v3_frank_base_query(user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN)")
        self.con.executemany("INSERT INTO _v3_frank_base_query VALUES (?,?,?,?)",
            [(BASE+20, 1, True, True), (BASE+20, 2, False, False), (BASE+20, 4, True, True),
             (BASE+20, 5, False, False), (BASE+20, 6, False, False)])
        self.con.execute("CREATE TABLE _v3_frank_base_targets(user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN)")
        self.con.execute("INSERT INTO _v3_frank_base_targets VALUES (?,?,true,true)", [BASE+20, 3])
        self.con.execute("CREATE TABLE _v3_frank_base_target_times AS SELECT user_key,app_id,positive,linux,1::BIGINT target_order,10::BIGINT timestamp_created FROM _v3_frank_base_targets")
        self.con.execute("CREATE TABLE _v3_frank_base_exclusions AS SELECT user_key,app_id FROM _v3_frank_base_query")
        self.con.execute("""CREATE TABLE _v3_reserved_kieran_hist(
            user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN,timestamp_created BIGINT,playtime_forever DOUBLE)""")
        self.con.executemany("INSERT INTO _v3_reserved_kieran_hist VALUES (?,?,?,?,?,?)", [
            (BASE+20, 2, True, False, 5, 1.0),  # overlaps frank; frank row wins
            (BASE+20, 3, True, True, 6, 1.0),  # hidden target must stay hidden
            (BASE+20, 7, True, True, 8, 4.0),  # earlier novel input
            (BASE+20, 8, False, False, 12, 6.0),  # future novel input
        ])
        _create_frank_augmentation(self.con, "before_target_time", True)
        _create_frank_augmentation(self.con, "all", False)
        before = self.con.execute("SELECT app_id FROM frank_aug_before_target_time_added ORDER BY app_id").fetchall()
        all_rows = self.con.execute("SELECT app_id FROM frank_aug_all_added ORDER BY app_id").fetchall()
        self.assertEqual(before, [(7,)])
        self.assertEqual(all_rows, [(7,), (8,)])
        augmented_targets = self.con.execute("SELECT app_id FROM frank_aug_before_target_time_targets").fetchall()
        self.assertEqual(augmented_targets, [(3,)])

    def test_kieran_only_mask_excludes_target_and_known_frank_inputs(self):
        # Reuse the augmentation fixture shape with a baseline frank cohort.
        self.con.execute("""CREATE TABLE _v3_frank_base_users AS
            SELECT ?::UBIGINT user_key,5::BIGINT n_total_inputs,1::BIGINT n_positive_inputs,2::BIGINT n_eligible_targets""", [BASE+30])
        self.con.execute("CREATE TABLE _v3_frank_base_query(user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN)")
        self.con.executemany("INSERT INTO _v3_frank_base_query VALUES (?,?,?,?)",
                             [(BASE+30, 1, True, True), (BASE+30, 2, False, False)])
        self.con.execute("CREATE TABLE _v3_frank_base_targets(user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN)")
        self.con.execute("INSERT INTO _v3_frank_base_targets VALUES (?,?,true,true)", [BASE+30, 3])
        self.con.execute("CREATE TABLE _v3_frank_base_target_times AS SELECT user_key,app_id,positive,linux,1::BIGINT target_order,10::BIGINT timestamp_created FROM _v3_frank_base_targets")
        self.con.execute("CREATE TABLE _v3_frank_base_exclusions AS SELECT user_key,app_id FROM _v3_frank_base_query")
        self.con.execute("""CREATE TABLE _v3_reserved_kieran_hist(
            user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN,timestamp_created BIGINT,playtime_forever DOUBLE)""")
        self.con.executemany("INSERT INTO _v3_reserved_kieran_hist VALUES (?,?,?,?,?,?)", [
            (BASE+30, 1, True, True, 5, 1.0), (BASE+30, 3, True, True, 6, 2.0),
            (BASE+30, 4, True, True, 7, 3.0), (BASE+30, 5, False, True, 8, 4.0),
        ])
        _create_kieran_only_mask(self.con)
        extra = self.con.execute("SELECT app_id FROM _v3_kieran_only_positive_mask ORDER BY app_id").fetchall()
        self.assertEqual(extra, [(4,)])


if __name__ == "__main__":
    unittest.main()
