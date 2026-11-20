"""Reuse the CSR transpose that SciPy otherwise reconstructs each score batch."""
import numpy as np
from eda_large_review_frank import digest
from experiment_v3_common import ROOT,write_json


class CachedKNNScorer:
    def __init__(self,base):
        self.base,self.matrix,self.mode,self.spec=base,base.matrix,'knn',base.spec
        self.by_input=base.matrix.T.tocsr()
    def score(self,q,negative=None):
        n=np.maximum(np.asarray(q.sum(axis=1)).ravel(),1)
        return (q@self.by_input).toarray()/n[:,None]


def verified_cache(base,q,name):
    model=base if isinstance(base,CachedKNNScorer) else CachedKNNScorer(base)
    base=model.base
    # This explicitly caches the identical conversion done by csr @ csc.
    idx=np.unique(np.linspace(0,q.shape[0]-1,min(256,q.shape[0]),dtype=int))
    a,b=base.score(q[idx]),model.score(q[idx])
    if not np.array_equal(a,b):raise ValueError('Cached KNN layout changes scores: '+name)
    write_json(ROOT/'knn_layout_equivalence'/f'{name}.json',{'status':'bitwise_equal',
        'sample_users':len(idx),'candidate_scores_compared':int(a.size),
        'source':base.spec,'scorer_code_sha256':digest(__file__),
        'scope':'Explicitly cache the same CSR transpose used by SciPy for each original multiplication; no feature or formula change.'})
    return model
