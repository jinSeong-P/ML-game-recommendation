"""Behavioral tests for new ranking methods, not copies of implementation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from experiment_v3_common import Scorer,CombinedScorer,NegativeScorer,RankFusionScorer,subset_cohort
from experiment_v3_query import max_similarity,NegativeFilterScorer


def small_catalog():
    return np.array([10,20,30]),np.array([True,True,True]),np.array([10,100,1000])


class RankingMethodTests(unittest.TestCase):
    def setUp(self):
        self.patcher=patch('experiment_v3_common.catalog',small_catalog)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_missing_content_and_empty_negative_remain_zero(self):
        model=Scorer(sparse.csr_matrix([[1.,0.],[0.,1.],[0.,0.]]),'content')
        q=sparse.csr_matrix([[0.,0.,1.]])
        n=sparse.csr_matrix(q.shape)
        np.testing.assert_array_equal(model.score(q),np.zeros((1,3)))
        np.testing.assert_array_equal(NegativeScorer(model,.3).score(q,n),model.score(q))

    def test_negative_feedback_changes_relative_candidate_preference(self):
        model=Scorer(sparse.eye(3,format='csr'),'content')
        q=sparse.csr_matrix([[1.,1.,0.]])
        n=sparse.csr_matrix([[0.,1.,0.]])
        original=model.score(q)
        changed=NegativeScorer(model,.3).score(q,n)
        self.assertAlmostEqual(original[0,0],original[0,1])
        self.assertGreater(changed[0,0],changed[0,1])

    def test_independent_missing_channel_not_renormalized(self):
        a=Scorer(sparse.eye(3,format='csr'),'content')
        missing=Scorer(sparse.csr_matrix((3,3)),'content')
        q=sparse.csr_matrix([[1.,0.,0.]])
        np.testing.assert_allclose(CombinedScorer([a,missing],[.5,.5]).score(q),.5*a.score(q))

    def test_rank_fusion_excludes_observed_before_assigning_ranks(self):
        class Fixed:
            def score(self,q,negative=None):return np.array([[100.,2.,1.]])
        model=RankFusionScorer(Fixed(),Fixed(),.5)
        mask=sparse.csr_matrix([[1.,0.,0.]])
        result=model.fuse(Fixed().score(None),Fixed().score(None),mask)
        self.assertEqual(result[0,0],0.)
        self.assertAlmostEqual(result[0,1],1/61)
        self.assertAlmostEqual(result[0,2],1/62)

    def test_zero_score_channel_does_not_get_rank_fusion_credit(self):
        model=RankFusionScorer(None,None,.5)
        result=model.fuse(np.array([[1.,0.,0.]]),np.array([[0.,1.,0.]]))
        self.assertEqual(result[0,2],0.)
        self.assertAlmostEqual(result[0,0],result[0,1])

    def test_max_aggregation_preserves_narrow_interest(self):
        kernel=np.array([[.9,.1,0.],[0.,.8,0.],[0.,0.,1.]])
        q=sparse.csr_matrix([[1.,1.,0.],[0.,0.,0.]])
        result=max_similarity(kernel,q)
        np.testing.assert_allclose(result[0],[.9,.8,0.])
        np.testing.assert_array_equal(result[1],[0.,0.,0.])

    def test_negative_filter_only_applies_above_threshold(self):
        class Fixed:
            spec={}
            def score(self,q):return np.array([[.8,.7,.6]])
        model=NegativeFilterScorer(Fixed(),np.eye(3),.8)
        result=model.score(sparse.csr_matrix([[1.,0.,0.]]),sparse.csr_matrix([[0.,1.,0.]]))
        np.testing.assert_allclose(result,[[.8,0.,.6]])

    def test_subsetting_keeps_queries_and_targets_paired(self):
        cohort={'user_ids':np.array([10,20,30],dtype=np.uint64),'positive':sparse.eye(3,format='csr'),
                'target_ids':np.array([1,2,3]),'name':'validation','hashes':{'a':'b'}}
        subset=subset_cohort(cohort,np.array([2,0]))
        np.testing.assert_array_equal(subset['user_ids'],[30,10])
        np.testing.assert_array_equal(subset['target_ids'],[3,1])
        self.assertEqual(subset['hashes'],cohort['hashes'])


if __name__=='__main__':unittest.main()
