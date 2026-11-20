"""Hand-computed checks for new v2 guardrails and two-channel scoring."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from tfidf_content import TfidfCatalog
from improvement_common import CombinedCatalog,log_groups,paired_rank_ci,protective_summary

class ImprovementTests(unittest.TestCase):
    def test_support_log_edges_and_zero(self):
        np.testing.assert_array_equal(log_groups([0,1,9,10,99,100,999,1000,9999,10000,100000,1000000]),[0,1,1,2,2,3,3,4,4,5,6,7])

    def test_bootstrap_preserves_pairing_and_no_change(self):
        ci=paired_rank_ci([1,2,0,10],[1,2,0,10],reps=100)
        self.assertEqual(ci['NDCG@10'],{'delta':0.,'ci95':[0.,0.]})
        ci=paired_rank_ci([0,0,0],[1,1,1],reps=100)
        self.assertEqual(ci['Recall@10']['ci95'],[1.,1.])
        self.assertEqual(ci['NDCG@10']['ci95'],[1.,1.])

    def test_macro_is_game_equal_not_user_equal(self):
        d={'target_rank':[1,1,1,0],'target_app_id':[10,10,10,20],
           'target_train_support':[1000,1000,1000,10],'recommendation_count':[10,10,10,0]}
        s=protective_summary(d)
        self.assertEqual(s['Recall@10'],.75)
        self.assertEqual(s['game_macro_Recall@10'],.5)
        self.assertEqual(s['tail']['Recall@10'],0.)
        self.assertEqual(s['empty_users'],1)

    def test_combined_missing_channel_has_zero_fixed_contribution(self):
        # Input 10 has only text; input20 has only metadata; candidates30/40 differ.
        ids=[10,20,30,40];linux=[False,False,True,True]
        t=TfidfCatalog(ids,linux,sparse.csr_matrix([[1,0],[0,0],[1,0],[0,1]],dtype=np.float32))
        m=TfidfCatalog(ids,linux,sparse.csr_matrix([[0,0],[1,0],[0,0],[1,0]],dtype=np.float32))
        q=sparse.csr_matrix([[1,1,0,0]],dtype=np.float32)
        model=CombinedCatalog(t,m,.75)
        np.testing.assert_allclose(model.batch(q,np.array([2])),[[.75,.25]])
        self.assertEqual([r['app_id'] for r in model.recommend([10,20])],[30,40])
        self.assertEqual(model.matrix.getnnz(axis=1).tolist(),[1,1,1,2])
        np.testing.assert_allclose(CombinedCatalog(t,m,1).batch(q,np.array([2])),[[1,0]])
        np.testing.assert_allclose(CombinedCatalog(t,m,0).batch(q,np.array([2])),[[0,1]])

    def test_whole_score_gain_cannot_hide_tail_loss(self):
        old={'user_key':list(range(100)),'target_app_id':[10]*99+[20],
             'n_positive_inputs':[1]*100,'n_total_inputs':[5]*100,
             'target_train_support':[1000]*99+[1],
             'target_rank':[0]*99+[1],'recommendation_count':[10]*100}
        candidate=dict(old);candidate['target_rank']=[1]*99+[0]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'baseline.parquet';pq.write_table(pa.table(old),path)
            result=protective_summary(candidate,baseline=path)
        self.assertTrue(result['guard_checks']['ndcg_ci_lower_positive'])
        self.assertTrue(result['guard_checks']['whole_recall_noninferior'])
        self.assertFalse(result['guard_checks']['tail_ndcg_noninferior'])
        self.assertFalse(result['eligible_for_automatic_adoption'])

if __name__=='__main__':unittest.main()
