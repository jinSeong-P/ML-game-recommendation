"""Regression checks for evaluation-design diagnostics on small synthetic tables."""

import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from eda_large_review_frank import BASE
from diagnose_eval_design import prepare_cohorts, target_concentration, unobserved_positives


class EvalDesignDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.con = duckdb.connect()
        con = self.con
        con.execute('CREATE TABLE catalog(app_id BIGINT,linux BOOLEAN,reviewed BOOLEAN)')
        con.execute('INSERT INTO catalog VALUES (1,true,true),(2,true,true),(3,false,true),(4,false,true),(9,true,false)')
        con.execute('CREATE TABLE app_stats(app_id BIGINT,final_train_positive BIGINT)')
        con.execute('INSERT INTO app_stats VALUES (1,90),(2,10),(3,50),(4,50),(9,500)')
        # User 1: test account, reviewed apps 1(+),2(+),3(-) in frank, and 1,4(+) in Kieran.
        con.execute('CREATE TABLE frank_raw(user_key UBIGINT,app_id BIGINT,review_key UBIGINT,positive BOOLEAN)')
        con.execute(f'INSERT INTO frank_raw VALUES ({BASE+1},1,11,true),({BASE+1},2,12,true),({BASE+1},3,13,false),({BASE+1},4,14,true)')
        con.execute('CREATE TABLE frank AS SELECT user_key,app_id,positive FROM frank_raw')
        con.execute(f'CREATE TABLE test_users AS SELECT {BASE+1}::UBIGINT user_key')
        con.execute('CREATE TABLE clean(user_key UBIGINT,app_id BIGINT,positive BOOLEAN)')
        con.execute(f'''INSERT INTO clean VALUES ({BASE+1},1,true),({BASE+1},3,false),({BASE+1},4,true),
            ({BASE+2},1,true),({BASE+2},3,true),({BASE+2},4,true),({BASE+2},9,true)''')
        con.execute('CREATE TABLE user_stats(user_key UBIGINT,is_test BOOLEAN,is_validation BOOLEAN)')
        con.execute(f'INSERT INTO user_stats VALUES ({BASE+1},true,false),({BASE+2},false,true)')
        path = Path(self.temp.name)/'quarantine.parquet'
        pq.write_table(pa.table({'review_key': pa.array([14], pa.uint64())}), path)
        prepare_cohorts(con, path, min_games=3)

    def tearDown(self):
        self.con.close()
        self.temp.cleanup()

    def test_quarantined_pair_leaves_the_frank_history(self):
        row = self.con.execute('SELECT n,n_positive,n_targets FROM test_cohort').fetchone()
        self.assertEqual(row, (3, 2, 2))

    def test_concentration_weights_each_user_once_and_ranks_only_linux_catalog(self):
        result = target_concentration(self.con)
        # App 9 is outside the input catalog, so app 1 is the most popular Linux game.
        self.assertEqual(self.con.execute('SELECT app_id FROM linux_rank WHERE rank_=1').fetchone()[0], 1)
        self.assertAlmostEqual(result['train_positive_reviews']['top_10'], 1.0)
        self.assertAlmostEqual(result['frank_targets']['top_10'], 1.0)
        self.assertAlmostEqual(result['uniform_over_linux_catalog']['top_10'], 1.0)

    def test_unobserved_positive_counts_only_kieran_side_linux_positive(self):
        self.con.execute(f'INSERT INTO clean VALUES ({BASE+1},2,false)')
        # A positive Linux game seen only in Kieran; the catalog gains app 5.
        self.con.execute('INSERT INTO catalog VALUES (5,true,true)')
        self.con.execute(f'INSERT INTO clean VALUES ({BASE+1},5,true)')
        result = unobserved_positives(self.con)
        self.assertEqual(result['users'], 1)
        self.assertEqual(result['kieran_only_linux_positive_pairs'], 1)
        self.assertEqual(result['frank_only_pairs'], 0)
        self.assertAlmostEqual(result['users_with_kieran_only_linux_positive_share'], 1.0)


if __name__ == '__main__':
    unittest.main()
