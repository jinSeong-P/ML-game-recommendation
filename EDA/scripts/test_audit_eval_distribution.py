"""Check target restoration, user weighting, and candidate masks analytically."""

import unittest

import duckdb

from audit_eval_distribution import user_mask_sensitivity


class DistributionAuditTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("CREATE TABLE linux_rank AS SELECT i app_id,i rank_ FROM range(1,21) t(i)")
        self.con.execute("CREATE TABLE history(user_key INTEGER,app_id INTEGER,positive BOOLEAN,linux BOOLEAN)")
        self.con.execute("INSERT INTO history SELECT 1,i,true,true FROM range(1,7) t(i)")
        self.con.execute("INSERT INTO history SELECT 2,i,false,true FROM range(1,6) t(i)")
        self.con.execute("INSERT INTO history VALUES (2,20,true,true)")
        self.con.execute("CREATE TABLE cohort(user_key INTEGER,n_targets INTEGER)")
        self.con.execute("INSERT INTO cohort VALUES (1,6),(2,1)")

    def tearDown(self):
        self.con.close()

    def test_each_user_has_equal_weight_and_target_is_restored(self):
        result = user_mask_sensitivity(self.con, "cohort", "history")
        # For user 1, whichever of six targets is hidden becomes popularity #1
        # after the other five games are masked. User 2's sole target remains
        # rank 15. Seven possible targets must still weight the users equally.
        self.assertAlmostEqual(result["masked_popularity_recall_at_10"], 0.5)
        self.assertAlmostEqual(result["masked_popularity_ndcg_at_10"], 0.5)
        self.assertAlmostEqual(result["target_weight_sum"], 2)
        self.assertAlmostEqual(result["random_recall_at_10"], 10 / 15)

    def test_extra_mask_cannot_remove_selected_target(self):
        self.con.execute("CREATE TABLE extra(user_key INTEGER,app_id INTEGER)")
        self.con.execute("INSERT INTO extra VALUES (2,20),(2,7)")
        result = user_mask_sensitivity(self.con, "cohort", "history", ["extra"])
        # The target is already in the history mask. Adding it again must not
        # remove it or reduce the pool twice. Only app 7 is newly masked.
        self.assertEqual(result["minimum_candidate_count"], 14)
        self.assertAlmostEqual(result["random_recall_at_10"], (10 / 15 + 10 / 14) / 2)
        self.assertAlmostEqual(result["masked_popularity_recall_at_10"], 0.5)


if __name__ == "__main__":
    unittest.main()
