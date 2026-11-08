"""Recommend with validation-frozen v3 models and visible user input only."""
import argparse
import json
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiment_v3_common import ROOT,catalog,RankFusionScorer
from select_experiments_v3 import restore
from evaluate_baselines import rank_positive


def recommend(family,positive,negative,observed,top_k):
    selection=json.loads((ROOT/'selection_frozen.json').read_text('utf-8'))
    item=selection['models'][family]
    if item['query_weighted']:raise ValueError('Chosen model requires query metadata; use the evaluation replay API')
    model=restore(item['spec']);apps,linux,support=catalog();candidates=apps[linux]
    lookup={int(a):i for i,a in enumerate(apps)}
    positives=sorted(set(positive));negatives=sorted(set(negative))
    if set(positives)&set(negatives):raise ValueError('A game cannot be both positive and negative')
    unknown=sorted((set(positives)|set(negatives))-set(lookup))
    if unknown:raise ValueError('Input game outside the frozen catalog: '+str(unknown))
    if not positives:raise ValueError('At least one positive input is required')
    def vector(ids):
        columns=[lookup[a] for a in ids]
        return sparse.csr_matrix((np.ones(len(columns),dtype=np.float32),([0]*len(columns),columns)),shape=(1,len(apps)))
    q,n=vector(positives),vector(negatives)
    seen=set(positives)|set(negatives)|set(observed)
    mask=sparse.csr_matrix([[int(int(a) in seen) for a in candidates]])
    if isinstance(model,RankFusionScorer):scores=model.fuse(model.left.score(q,n),model.right.score(q,n),mask)[0]
    else:scores=model.score(q,n)[0]
    scores[mask.indices]=0
    chosen=rank_positive(scores,candidates)
    if item['popularity_fill'] and len(chosen)<10:
        excluded=set(mask.indices)|set(chosen)
        added=[j for j in np.lexsort((candidates,-support[linux])) if j not in excluded][:10-len(chosen)]
        chosen=np.r_[chosen,added]
    return {'family':family,'variant':item['variant'],
        'recommendations':[{'app_id':int(candidates[j]),'rank':rank,'score':float(scores[j]),
            'popularity_fill':bool(item['popularity_fill'] and scores[j]<=0)} for rank,j in enumerate(chosen[:top_k],1)]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['collaborative','content','hybrid'],default='collaborative')
    p.add_argument('--positive-apps',nargs='+',type=int,required=True)
    p.add_argument('--negative-apps',nargs='*',type=int,default=[])
    p.add_argument('--exclude-apps',nargs='*',type=int,default=[])
    p.add_argument('--top-k',type=int,choices=range(1,11),default=10)
    a=p.parse_args()
    with threadpool_limits(limits=2):print(json.dumps(recommend(a.family,a.positive_apps,a.negative_apps,a.exclude_apps,a.top_k),ensure_ascii=False,indent=2))
