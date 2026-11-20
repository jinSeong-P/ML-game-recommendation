"""Evaluate review TF-IDF with a reusable item-cosine cache.

This changes the order of sparse multiplication, not the features or scoring
formula. A frozen execution receipt and numerical/ranking comparison precede
aggregate validation metrics. Failed/interrupted original outputs are retained.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import row_norms
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest,log
from experiment_v3_common import (ROOT,PROTOCOL,V2,catalog,load_matrix,model_files,
    write_json,frozen_cohort,evaluate,CombinedScorer)
from run_experiments_v3 import baseline
from experiment_content_combinations_v3 import freeze as freeze_combinations


CACHE=ROOT/'review_item_cosine_cache'


class KernelContentScorer:
    def __init__(self,base,kernel,spec):
        self.base,self.kernel,self.spec=base,kernel,spec
    def score(self,q,negative=None):
        n=np.maximum(np.asarray(q.sum(axis=1)).ravel(),1)
        weights=q.multiply((1./n.astype(np.float32))[:,None]).tocsr()
        # Retain the original profile norm and channel normalization. The
        # reusable cache replaces only (profile @ candidate_vectors.T).
        profile=weights@self.base.matrix
        norms=row_norms(profile)
        norms[norms==0]=1
        return np.asarray(weights@self.kernel)/norms[:,None]


def prepare():
    CACHE.mkdir(exist_ok=True)
    source=ROOT/'content/train_review_body_tfidf'
    receipt={'status':'fixed_before_review_aggregate_metrics','parent_sha256':digest(PROTOCOL),
        'source':model_files(source),'code_sha256':digest(__file__),
        'formula':'cosine(mean(positive input vectors),candidate); each channel normalized separately',
        'execution':'Precompute float32 item dot products in full fixed-catalog candidate blocks; preserve original float32 query profile norm.',
        'numerical_check':'At least 1024 spread validation users; all scores close atol1e-6 rtol1e-4; top10 candidate IDs and target ranks must agree on checked users.',
        'reason':'30.4 million review features make repeated candidate-vector products expensive; no feature, cohort or hyperparameter changes.'}
    path=CACHE/'execution_protocol.json'
    if path.exists() and json.loads(path.read_text('utf-8'))!=receipt:raise ValueError('Review execution protocol changed')
    write_json(path,receipt)
    base=load_matrix(source,'content');apps,linux,_=catalog();candidates=apps[linux]
    kernelpath=CACHE/'item_candidate_cosine.npy';complete=CACHE/'kernel_complete.json'
    if complete.exists():
        info=json.loads(complete.read_text('utf-8'))
        if info['execution_protocol_sha256']!=digest(path) or info['sha256']!=digest(kernelpath):raise ValueError('Review kernel cache changed')
    else:
        if kernelpath.exists():raise FileExistsError('Inspect partial review cosine cache')
        kernel=np.lib.format.open_memmap(kernelpath,mode='w+',dtype=np.float32,shape=(len(apps),len(candidates)))
        transpose=base.matrix.T.tocsr()
        rows=np.flatnonzero(linux);start=time.perf_counter()
        for begin in range(0,len(rows),128):
            end=min(begin+128,len(rows))
            values=(base.matrix[rows[begin:end]]@transpose).toarray()
            kernel[:,begin:end]=values.T
            if begin%1024==0 or end==len(rows):log(f'Review cosine cache: {end:,}/{len(rows):,} candidates')
        kernel.flush();del kernel,transpose
        write_json(complete,{'status':'complete','execution_protocol_sha256':digest(path),
            'sha256':digest(kernelpath),'shape':[len(apps),len(candidates)],'elapsed_seconds':time.perf_counter()-start})
    spec={'method':'precomputed_item_content_cosine','source':base.spec,
        'cache_file':str(kernelpath.relative_to(ROOT)),'cache_sha256':digest(kernelpath),
        'execution_protocol_sha256':digest(path)}
    model=KernelContentScorer(base,np.load(kernelpath,mmap_mode='r'),spec)
    return base,model


def run():
    base,model=prepare();c=frozen_cohort();apps,linux,_=catalog();ids=apps[linux]
    indices=np.unique(np.linspace(0,len(c['user_ids'])-1,min(1024,len(c['user_ids'])),dtype=int))
    max_error=0.;top_errors=0
    for begin in range(0,len(indices),128):
        q=c['positive'][indices[begin:begin+128]];mask=c['exclusions'][indices[begin:begin+128]]
        original=base.score(q);cached=model.score(q)
        if not np.allclose(original,cached,atol=1e-6,rtol=1e-4):raise ValueError('Review kernel numerical error exceeds fixed tolerance')
        max_error=max(max_error,float(np.max(np.abs(original-cached))))
        r,col=mask.nonzero();original[r,col]=0;cached[r,col]=0
        from evaluate_baselines import rank_positive
        for a,b in zip(original,cached):
            if not np.array_equal(rank_positive(a,ids),rank_positive(b,ids)):top_errors+=1
    audit={'status':'passed' if not top_errors else 'failed','sample_users':len(indices),
        'candidate_scores_compared':len(indices)*len(ids),'max_absolute_score_error':max_error,
        'top10_candidate_list_differences':top_errors,'execution_protocol_sha256':digest(CACHE/'execution_protocol.json')}
    write_json(CACHE/'numerical_equivalence.json',audit)
    if top_errors:raise ValueError('Review cache changes checked top10 rankings; preserve original scorer')
    evaluate('train_review_body_tfidf',model,c,baseline('content'),family='content')
    meta=load_matrix(V2/'tfidf/metadata_channel','content');plan=freeze_combinations()
    spec={'method':'two_channel_content','text':model.spec,'metadata':meta.spec,'weights':[.5,.5],
        'combination_protocol_sha256':digest(plan)}
    evaluate('train_review_body_tfidf_metadata_0_5',CombinedScorer([model,meta],[.5,.5],spec),c,baseline('content'),family='content')


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
