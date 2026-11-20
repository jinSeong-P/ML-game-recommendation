"""Numerical and saved-output tests for the new cap experiment, with tiny data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

import forgemaster_model_study as study


class CapModelStudyTests(unittest.TestCase):
    def test_retraining_preserves_singleton_support_and_weighted_cosine(self):
        apps=np.array([10,20,30,40],dtype=np.int64);linux=np.ones(4,dtype=bool)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            table=pa.table({'user_key':pa.array([1,1,2,3,3,3],type=pa.uint64()),
                'app_id':pa.array([10,20,10,20,30,40],type=pa.int64())})
            pq.write_table(table,root / 'train_positive_catalog.parquet')
            with patch.object(study,'ROOT',root),patch.object(study,'catalog_arrays',return_value=(apps,linux)):
                study.save_json(root / 'training_source.json',{'status':'complete','protocol_sha256':study.digest(study.PROTOCOL),
                    'training':{'pairs':6,'users':3},'files':study.inventory([root / 'train_positive_catalog.parquet'])})
                study.fit()
                np.testing.assert_array_equal(np.load(root / 'models/positive_support.npy'),[2,2,1,1])
                binary=sparse.load_npz(root / 'models/binary_item_knn_k100.npz').toarray()
                weighted=sparse.load_npz(root / 'models/activity_inverse_degree_cosine_k100.npz').toarray()
                self.assertAlmostEqual(binary[0,1],.5)
                self.assertAlmostEqual(weighted[0,1],.5/np.sqrt(1.5*(.5+1/3)),places=5)
                self.assertAlmostEqual(weighted[1,2],(1/3)/np.sqrt((.5+1/3)*(1/3)),places=5)
                np.testing.assert_array_equal(np.diag(binary),np.zeros(4))
                np.testing.assert_array_equal(np.diag(weighted),np.zeros(4))

    def test_neighbors_remove_self_and_break_ties_by_app_id(self):
        apps=np.array([10,20,30,40])
        cols,values=study.top_neighbors(np.array([3,2,1,0]),np.array([.2,.5,.5,1.]),0,apps)
        np.testing.assert_array_equal(cols,[1,2,3])
        np.testing.assert_allclose(values,[.5,.5,.2])

    def test_unsupported_positive_input_stays_in_score_denominator(self):
        query=sparse.csr_matrix([[1.,0.,1.]])
        weights=sparse.csr_matrix([[.8,0,0],[0,.7,0]])
        result=study.score_model('binary_item_knn_k100',query,[weights],np.ones(3),np.array([True,True,False]))
        np.testing.assert_allclose(result,[[.4,0]])

    def test_content_channels_have_independent_profile_normalization(self):
        query=sparse.csr_matrix([[1.,1.,0.,0.]])
        description=sparse.csr_matrix([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
        metadata=sparse.csr_matrix([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
        result=study.score_model('metadata_extended_tfidf',query,[description,metadata],np.ones(4),np.array([False,False,True,True]))
        np.testing.assert_allclose(result,[[.5/np.sqrt(2)+.5,.5/np.sqrt(2)]])

    def test_saved_ranking_keeps_empty_users_and_exact_steam_ids(self):
        apps=np.array([10,20,30,40,50,60,70,80],dtype=np.int64)
        linux=np.ones(8,dtype=bool)
        users=np.array([76561198000000001,76561198000000002],dtype=np.uint64)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root / 'cohorts/full';folder.mkdir(parents=True)
            targets=pa.table({'user_key':pa.array(users,type=pa.uint64()),'app_id':pa.array([70,80],type=pa.int64()),
                'positive':[True,True],'linux':[True,True]})
            pq.write_table(targets,folder / 'targets.parquet')
            records=[]
            for user,games in zip(users,[[10,20,30,40,50],[20,30,40,50,60]]):
                for i,app in enumerate(games):records.append({'user_key':int(user),'app_id':app,'positive':i==0,'linux':True})
            table=pa.Table.from_pylist(records,schema=pa.schema([('user_key',pa.uint64()),('app_id',pa.int64()),('positive',pa.bool_()),('linux',pa.bool_())]))
            pq.write_table(table,folder / 'query.parquet')
            (root / 'model_manifest.json').write_text(json.dumps({'tiny_test':True}),encoding='utf-8')
            with patch.object(study,'ROOT',root),patch.object(study,'catalog_arrays',return_value=(apps,linux)):
                cohort=study.load_cohort(folder)
                weights=sparse.lil_matrix((8,8),dtype=float)
                weights[6,0]=.8;weights[7,0]=.9
                result=study.evaluate_case('full','binary_item_knn_k100',cohort,[weights.tocsr()],np.ones(8,dtype=np.int64),apps,linux)
                self.assertEqual(result['users'],2)
                self.assertEqual(result['Recall@10'],.5)
                self.assertEqual(result['empty_recommendation_rate'],.5)
                self.assertAlmostEqual(result['NDCG@10'],.5/np.log2(3))
                saved=pq.read_table(root / 'evaluations/full/binary_item_knn_k100/per_user.parquet').to_pandas()
                np.testing.assert_array_equal(saved.user_key.to_numpy(),users)
                self.assertEqual(saved.hit_rank.tolist(),[2,0])
                self.assertTrue(all(not value for key,value in json.loads((root / 'evaluations/full/binary_item_knn_k100/manifest.json').read_text())['saved_output_checks'].items() if key!='saved_users'))


if __name__=='__main__':
    unittest.main()
