"""Check inference semantics and interrupted selection, without production data."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy import sparse
import select_experiments_v3 as selection
from run_experiments_v3 import SignedScorer,FoldInScorer
from diagnose_backlog_v3 import pair_diagnostics
from evaluate_selected_supplements_v3 import CohortMaskedFusion
from replay_temporal_v3 import rank_row
from experiment_review_kernel_v3 import KernelContentScorer
from experiment_v3_common import Scorer
from verify_ease_solution_v3 import sampled_residual
from cached_knn_v3 import CachedKNNScorer
from evaluate_fusion_grid_cached_v3 import CachedRankFusion


class InferenceTests(unittest.TestCase):
    def test_cached_fusion_matches_masked_ranks_with_adaptive_weights(self):
        with patch('experiment_v3_common.catalog',return_value=(np.arange(3),np.ones(3,dtype=bool),np.array([10,1000,10000]))):
            original=selection.RankFusionScorer(None,None,.25,60,True,{'method':'test'})
        mask=sparse.csr_matrix([[1,0,0],[0,1,0]])
        expected=original.fuse(np.array([[100.,2.,1.],[100.,2.,1.]]),np.array([[1.,2.,100.],[1.,2.,100.]]),mask)
        model=CachedRankFusion(np.array([[0,1,2],[1,0,2]],dtype=np.uint16),np.array([[0,2,1],[2,0,1]],dtype=np.uint16),original)
        q=sparse.csr_matrix([[1.,0.,0.]])
        actual=np.vstack([model.score(q),model.score(q)])
        np.testing.assert_array_equal(actual,expected)

    def test_knn_cached_transpose_is_bitwise_equal_with_signed_scores(self):
        matrix=sparse.csr_matrix([[0.,-.25,1.],[.3,0.,-.1]],dtype=np.float32)
        q=sparse.csr_matrix([[1.,1.,0.],[0.,0.,0.]],dtype=np.float32)
        with patch('experiment_v3_common.catalog',return_value=(np.arange(3),np.array([True,True,False]),np.ones(3))):
            base=Scorer(matrix,'knn')
        np.testing.assert_array_equal(CachedKNNScorer(base).score(q),base.score(q))

    def test_ease_candidate_row_satisfies_excluded_self_normal_equations(self):
        gram=np.array([[5.,2.,1.],[2.,3.,1.],[1.,1.,2.]])
        inverse=np.linalg.inv(gram+2*np.eye(3));target=1
        b=-inverse[:,target]/inverse[target,target];b[target]=0
        correct=sampled_residual(gram,b,target,2,np.arange(3))
        self.assertLess(correct['relative_l2_residual'],1e-12)
        b[0]*=-1
        self.assertGreater(sampled_residual(gram,b,target,2,np.arange(3))['relative_l2_residual'],.1)

    def test_item_cosine_cache_preserves_profile_normalization(self):
        matrix=sparse.csr_matrix([[1.,0.,0.],[.6,.8,0.],[0.,0.,0.]],dtype=np.float32)
        q=sparse.csr_matrix([[1.,1.,0.],[0.,0.,1.]],dtype=np.float32)
        with patch('experiment_v3_common.catalog',return_value=(np.arange(3),np.ones(3,dtype=bool),np.ones(3))):
            base=Scorer(matrix,'content')
        cache=(matrix@matrix.T).toarray()
        actual=KernelContentScorer(base,cache,{}).score(q)
        np.testing.assert_allclose(actual,base.score(q),atol=1e-6)
        np.testing.assert_array_equal(actual[1],np.zeros(3))

    def test_temporal_replay_ranks_only_unseen_finite_positive_candidates(self):
        ids,scores=rank_row(np.arange(6),np.array([.2,.3,.3,0.,np.inf,-1.]),
            np.array([10,20,30,40,50,60]),np.array([20]))
        np.testing.assert_array_equal(ids,[30,10])
        np.testing.assert_allclose(scores,[.3,.2])
        ids,_=rank_row(np.array([],dtype=int),np.array([]),np.arange(3),[])
        self.assertEqual(len(ids),0)

    def test_supplemental_fusion_masks_each_batch_before_rank_assignment(self):
        class Fixed:
            def score(self,q,negative=None):return np.tile([[100.,2.,1.]],(q.shape[0],1))
        with patch('experiment_v3_common.catalog',return_value=(np.arange(3),np.ones(3,dtype=bool),np.ones(3))):
            fusion=selection.RankFusionScorer(Fixed(),Fixed(),.5)
        c={'exclusions':sparse.csr_matrix([[1.,0.,0.],[0.,1.,0.]])}
        model=CohortMaskedFusion(fusion,c)
        q=sparse.csr_matrix([[1.,0.,0.]])
        first=model.score(q);second=model.score(q)
        np.testing.assert_allclose(first,[[0.,1/61,1/62]])
        np.testing.assert_allclose(second,[[1/61,0.,1/62]])

    def test_signed_negative_query_subtracts_relation_without_nn_term(self):
        class Base:
            matrix=sparse.csr_matrix([[1.,-.5,0.],[-.5,1.,0.]])
            spec={}
        q=sparse.csr_matrix([[1.,0.,0.]])
        n=sparse.csr_matrix([[0.,1.,0.]])
        np.testing.assert_allclose(SignedScorer(Base(),.2).score(q,n),[[1.1,-.7]])

    def test_als_foldin_solves_confidence_system_and_empty_input(self):
        factors=np.array([[1.,0.],[0.,1.],[1.,1.]],dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);np.save(folder/'item_factors.npy',factors)
            with patch('run_experiments_v3.catalog',return_value=(np.arange(3),np.ones(3,dtype=bool),np.ones(3))),patch('run_experiments_v3.model_files',return_value={}):
                model=FoldInScorer(folder,{'parameters':{'alpha':20,'regularization':.1}})
            q=sparse.csr_matrix([[1.,0.,0.],[0.,0.,0.]])
            actual=model.score(q)
            y=factors.astype(float)
            expected=np.linalg.solve(y.T@y+19*np.outer(y[0],y[0])+.1*np.eye(2),20*y[0])@y.T
            np.testing.assert_allclose(actual[0],expected,rtol=1e-6)
            np.testing.assert_array_equal(actual[1],np.zeros(3))

    def test_case_input_attains_reported_maximum(self):
        text=sparse.csr_matrix([[1.,0.],[.99,.1],[.85,.1]])
        c={'user_ids':np.array([1]),'target_ids':np.array([10]),
           'positive':sparse.csr_matrix([[0.,1.,1.]])}
        maximum,_,_,best=pair_diagnostics(c,text,np.array(['a','b','c']),np.array(['a','b','c']),np.array([10,20,30]))
        self.assertEqual(best[0],1)
        self.assertAlmostEqual(maximum[0],.99)

    def test_family_selection_reuses_base_evidence_after_partial_fusion(self):
        values=[{'variant':'item_knn_control','family':'collaborative','provenance':{'spec':{}},'protection':{'NDCG@10':.2,'Recall@10':.3}},
                {'variant':'content_control','family':'content','provenance':{'spec':{}},'protection':{'NDCG@10':.1,'Recall@10':.2}},
                {'variant':'better','family':'collaborative','provenance':{'spec':{'x':1}},'protection':{'NDCG@10':.3,'Recall@10':.4,'eligible_for_automatic_adoption':True}}]
        with tempfile.TemporaryDirectory() as tmp,patch.object(selection,'ROOT',Path(tmp)),patch.object(selection,'expected_variants',return_value={'item_knn_control','content_control','better'}),patch.object(selection,'digest',return_value='hash'),patch.object(selection,'verify_evidence',side_effect=lambda xs:{m['variant']:'hash' for m in xs}),patch.object(selection,'manifests',return_value=values):
            first=selection.family_selection()
            values.append({'variant':first['followup_fusion_variants'][0]})
            self.assertEqual(first,selection.family_selection())


if __name__=='__main__':unittest.main()
