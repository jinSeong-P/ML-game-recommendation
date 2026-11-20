"""Cache the fixed candidate transpose; preserve the frozen content score formula."""
import argparse
import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest
from experiment_v3_common import ROOT,V2,Scorer,CombinedScorer,load_matrix,model_files,frozen_cohort,evaluate,write_json
from experiment_content_combinations_v3 import freeze,VARIANTS
from run_experiments_v3 import baseline


class CachedContentScorer:
    def __init__(self,base):
        if not isinstance(base,Scorer) or base.mode!='content':raise ValueError('Expected a content representation')
        self.base,self.spec=base,base.spec
        values=base.matrix[base.linux].T
        self.candidates=values.tocsr() if sparse.issparse(values) else np.asarray(values)
    def score(self,q,negative=None):
        n=np.maximum(np.asarray(q.sum(axis=1)).ravel(),1)
        weights=q.multiply((1./n.astype(np.float32))[:,None]).tocsr()
        profile=normalize(weights@self.base.matrix,norm='l2',copy=False)
        result=profile@self.candidates
        return result.toarray() if sparse.issparse(result) else np.asarray(result)


def verified_cache(base,query,name):
    model=CachedContentScorer(base)
    selected=np.unique(np.linspace(0,query.shape[0]-1,min(128,query.shape[0]),dtype=int))
    q=query[selected]
    start=time.perf_counter();original=base.score(q);old_seconds=time.perf_counter()-start
    start=time.perf_counter();cached=model.score(q);cached_seconds=time.perf_counter()-start
    if not np.array_equal(original,cached):raise ValueError('Cached sparse layout changed content scores: '+name)
    path=ROOT/'content_layout_equivalence'/f'{name}.json'
    write_json(path,{'status':'bitwise_equal','sample_users':len(selected),'candidate_scores_compared':int(original.size),
        'source':base.spec,'scorer_code_sha256':digest(__file__),
        'original_seconds':old_seconds,'cached_seconds':cached_seconds,
        'explanation':'Same candidate transpose and CSR multiplication; avoid rebuilding the fixed transpose in every query batch.'})
    return model


def run(verify_only=False):
    plan=freeze();c=frozen_cohort()
    meta=verified_cache(load_matrix(V2/'tfidf/metadata_channel','content'),c['positive'],'metadata')
    for name in VARIANTS:
        folder=ROOT/'content'/name
        if not (folder/'manifest.json').exists():continue
        if name=='train_review_body_tfidf' and (ROOT/'review_item_cosine_cache/numerical_equivalence.json').exists():
            from experiment_review_kernel_v3 import prepare
            _,text=prepare()
        else:text=verified_cache(load_matrix(folder,'content'),c['positive'],name)
        if verify_only:continue
        spec={'method':'two_channel_content','text':text.spec,
            'metadata':model_files(V2/'tfidf/metadata_channel'),'weights':[.5,.5],
            'combination_protocol_sha256':digest(plan)}
        evaluate(name+'_metadata_0_5',CombinedScorer([text,meta],[.5,.5],spec),c,baseline('content'),family='content')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):run(a.verify_only)
