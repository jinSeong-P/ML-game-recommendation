"""Regression checks for strict source cleaning and whole-user EDA previews."""

import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from eda_large_review_frank import BASE, analyze, build_source


class LargeOriginalEdaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.con = duckdb.connect()
        self.con.execute("SET threads=2")

    def tearDown(self):
        self.con.close()
        self.temp.cleanup()

    def source(self, rows):
        schema = pa.schema([
            ("user_id",pa.string()),("app_id",pa.int64()),("review_id",pa.string()),
            ("positive",pa.bool_()),("timestamp_created",pa.int64()),
            ("timestamp_updated",pa.int64()),("playtime_forever",pa.float64()),("language",pa.string())])
        complete = [dict(zip(schema.names, list(row)+[1600000000,1600000001,60.,"english"])) for row in rows]
        path = Path(self.temp.name)/"reviews_000.parquet"
        pq.write_table(pa.Table.from_pylist(complete,schema=schema),path)
        return build_source(self.con,str(path))

    def test_quarantine_identity_and_label_conflicts_without_float_repair(self):
        facts=self.source([
            (str(BASE+1),10,"1",True),(str(BASE+1),10,"1",True),
            (str(BASE+2),20,"2",True),(str(BASE+2),20,"3",False),
            (str(BASE+3),30,"4",True),(str(BASE+4),30,"4",True),
            ("7.656119796026573e16",40,"5",True),
            (str(BASE+5),40,"6",True)])
        self.assertEqual(facts['raw_rows'],8)
        self.assertEqual(facts['quarantined_rows'],1)
        self.assertEqual(facts['review_ids_with_conflicting_owner_or_app'],1)
        self.assertEqual(facts['conflicting_label_pairs'],1)
        self.assertEqual(self.con.execute('SELECT count(*) FROM clean').fetchone()[0],2)

    def test_exclusion_removes_other_games_and_retains_shared_item_catalog(self):
        self.source([
            (str(BASE+1),10,"1",True),(str(BASE+1),20,"2",True),
            (str(BASE+1),30,"3",False),
            (str(BASE+2),10,"4",True),(str(BASE+2),20,"5",True),
            (str(BASE+3),10,"6",False),(str(BASE+3),20,"7",True)])
        self.con.execute(f"CREATE TABLE frank_raw AS SELECT {BASE+1}::UBIGINT user_key,10::BIGINT app_id,1::UBIGINT review_key,true positive,1600000000::BIGINT timestamp_created,1600000001::BIGINT timestamp_updated UNION ALL SELECT {BASE+1},20,2,true,1600000000,1600000001")
        self.con.execute('CREATE TABLE frank AS SELECT user_key,app_id,positive FROM frank_raw')
        self.con.execute(f'CREATE TABLE test_users AS SELECT {BASE+1}::UBIGINT user_key')
        self.con.execute('CREATE TABLE catalog(app_id BIGINT,linux BOOLEAN,reviewed BOOLEAN)')
        self.con.execute('INSERT INTO catalog VALUES (10,false,true),(20,true,true),(30,false,true)')
        self.con.execute('UPDATE valid_raw SET timestamp_created=NULL WHERE review_key=1')
        result=analyze(self.con)
        self.assertEqual(result['after_test_exclusion']['removed_test_pairs'],3)
        self.assertEqual(result['after_test_exclusion']['pairs'],4)
        self.assertEqual(result['after_test_exclusion']['apps'],2)
        self.assertTrue(all(v==0 for v in result['post_exclusion_checks'].values()))
        self.assertEqual(result['heldout_target_item_support']['targets_without_training_reviews'],0)
        self.assertEqual(result['supported_cohort_sizes']['any_review_support']['after_test_exclusion'][0]['potential_users'],1)
        self.assertEqual(result['cross_source_overlap']['common_review_ids_with_changed_created_time'],1)
        self.assertEqual(result['after_validation_reservation']['pairs']+result['validation_preview']['pairs'],4)

    def test_multiple_positive_targets_hide_exactly_one_not_all(self):
        # Both positives may be alternative hidden targets. Hiding either one
        # leaves exactly two inputs; asking for three inputs cannot qualify.
        self.con.execute('CREATE TABLE frank(user_key UBIGINT,app_id BIGINT,positive BOOLEAN)')
        self.con.execute(f'INSERT INTO frank VALUES ({BASE+1},10,true),({BASE+1},20,true),({BASE+1},30,false)')
        self.con.execute(f'CREATE TABLE test_users AS SELECT {BASE+1}::UBIGINT user_key')
        self.con.execute('CREATE TABLE catalog(app_id BIGINT,linux BOOLEAN,reviewed BOOLEAN)')
        self.con.execute('INSERT INTO catalog VALUES (10,true,true),(20,true,true),(30,false,true)')
        self.con.execute('CREATE TABLE app_stats AS SELECT app_id,10 post_test_reviews,10 post_test_positive,9 final_train_reviews,9 final_train_positive FROM catalog')
        from eda_large_review_frank import cohort_coverage
        rows=cohort_coverage(self.con,1,False)['after_test_exclusion']
        self.assertEqual(rows[0]['potential_users'],1)
        self.assertEqual(rows[0]['possible_target_pairs'],2)
        self.assertEqual(rows[1]['potential_users'],0)
        self.assertEqual(rows[0]['potential_users_with_positive_query'],1)
        self.con.execute('UPDATE frank SET positive=false WHERE app_id=10')
        rows=cohort_coverage(self.con,1,False)['after_test_exclusion']
        self.assertEqual(rows[0]['potential_users'],1)
        self.assertEqual(rows[0]['potential_users_with_positive_query'],0)
        self.assertEqual(rows[0]['potential_users_with_only_negative_query'],1)

    def test_cross_source_app_ambiguity_is_quarantined_without_guessing(self):
        self.con.execute(f'CREATE TABLE frank_raw AS SELECT {BASE+1}::UBIGINT user_key,20::BIGINT app_id,1::UBIGINT review_key')
        q=self.source([(str(BASE+1),10,'1',True),(str(BASE+2),10,'2',True)])
        self.assertEqual(q['cross_source_ambiguous_review_ids'],1)
        self.assertEqual(q['additional_cross_source_review_ids_quarantined'],1)
        self.assertEqual(self.con.execute('SELECT count(*) FROM clean').fetchone()[0],1)

    def test_integer_looking_rounded_owner_is_not_identity_evidence(self):
        self.con.execute(f'CREATE TABLE frank_raw AS SELECT {BASE+15}::UBIGINT user_key,10::BIGINT app_id,1::UBIGINT review_key')
        q=self.source([(str(BASE+16),10,'1',True),(str(BASE+2),20,'2',True)])
        # A valid-looking decimal SteamID can still be the wrong owner.
        self.assertEqual(q['quarantined_rows'],0)
        self.assertEqual(q['cross_source_ambiguous_review_ids'],1)
        self.assertEqual(self.con.execute('SELECT count(*) FROM clean').fetchone()[0],1)


if __name__=='__main__':
    unittest.main()
