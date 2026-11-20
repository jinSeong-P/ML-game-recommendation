"""Meaningful v4 invariants: target-free retrieval and paired denominators."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from experiment_v4_common import paired_comparison,MatrixScorer,metrics,rank_values
from run_candidates_v4 import candidate_sets,normalized,features,CachedResidual


class V4Tests(unittest.TestCase):
    def test_int8_rank_weights_and_averages_use_float64(self):
        r,n=rank_values(np.array([0,1,3,10],dtype=np.int8))
        self.assertEqual(n.dtype,np.dtype('float64'))
        np.testing.assert_allclose(n,[0.,1.,.5,1/np.log2(11.)],rtol=0,atol=1e-15)
    def test_budget_keeps_cf_top10_deduplicates_positive_only(self):
        ids=np.arange(300);a=np.arange(300,0,-1,dtype=float);b=np.arange(300,dtype=float)
        sup=np.r_[np.full(100,10),np.full(100,200),np.full(100,5000)]
        a[290:]=0;b[290:]=0
        sets=candidate_sets(a,b,ids,sup)
        self.assertTrue(set(range(10))<=set(sets[2]))
        self.assertEqual(len(sets[2]),200)
        for s in sets:
            self.assertEqual(len(s),len(set(s)));self.assertTrue(np.all(s<290))
        # No target argument exists: a zero-score target is never injected.
        self.assertNotIn(299,sets[2])

    def test_insufficient_candidates_remain_short(self):
        sets=candidate_sets(np.array([1.,0,0]),np.array([0.,1,0]),np.array([1,2,3]),np.array([10,500,0]))
        self.assertEqual(sets[2].tolist(),[0,1])

    def test_normalization_and_mean_query(self):
        a=normalized(np.array([[0.,0.],[2.,4.]]))
        np.testing.assert_array_equal(a,[[0,0],[.5,1]])
        score=MatrixScorer(sparse.csr_matrix([[1.,.5,0],[0,0,1.]]))
        q=sparse.csr_matrix([[1.,1,0],[0,0,0]])
        np.testing.assert_allclose(score.score(q),[[.75,0],[0,0]])

    def test_paired_bootstrap_identical_and_empty_users(self):
        c=dict(n_total=np.array([5,5,5,1]),target_ids=np.array([1,1,2,3]))
        r=np.array([1,0,10,0])
        result=paired_comparison(r,r,c,reps=100)
        self.assertEqual(result['Recall@10']['ci95'],[0.,0.])
        self.assertEqual(result['game_macro']['NDCG@10']['ci95'],[0.,0.])
        self.assertEqual(sum(result['hit_overlap'].values()),3)

    def test_primary_and_short_groups_have_separate_denominators(self):
        c=dict(n_total=np.array([5,5,2,3]),n_positive=np.array([1,2,1,2]),target_ids=np.array([1,2,1,2]))
        with patch('experiment_v4_common.catalog',return_value=(np.array([1,2]),np.array([True,True]))):
            result=metrics(np.array([1,0,1,0]),np.array([10,0,10,0]),c,np.array([100,200]))
        self.assertEqual(result['users'],2);self.assertEqual(result['recall'],.5)
        self.assertEqual(result['empty_users'],1)
        self.assertEqual(result['by_total_history']['1-2']['users'],1)


if __name__=='__main__':unittest.main()
