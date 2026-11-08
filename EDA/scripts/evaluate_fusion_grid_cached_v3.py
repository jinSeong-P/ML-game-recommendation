"""Reuse fixed component ranks across the six predeclared RRF settings."""
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest,log
from experiment_v3_common import ROOT,RankFusionScorer,evaluate,frozen_cohort,write_json
from select_experiments_v3 import family_selection,restore
from run_experiments_v3 import baseline


class CachedRankFusion:
    def __init__(self,left_ranks,right_ranks,original):
        self.left,self.right,self.original=left_ranks,right_ranks,original
        self.spec=original.spec;self.cursor=0
    def score(self,q,negative=None):
        start,end=self.cursor,self.cursor+q.shape[0];self.cursor=end
        weight=self.original.weight
        if self.original.adaptive:weight=weight/(1+self.original.support[self.original.linux]/1000.)
        out=np.zeros((q.shape[0],self.left.shape[1]),dtype=np.float64)
        for ranks,w in [(self.left[start:end],1-weight),(self.right[start:end],weight)]:
            out+=(ranks>0)*w/(self.original.constant+ranks)
        return out


def run():
    stage=family_selection();c=frozen_cohort()
    if not stage['followup_fusion_variants']:return
    left=restore(stage['families']['collaborative']['spec']);right=restore(stage['families']['content']['spec'])
    cache=ROOT/'selected_fusion_rank_cache';cache.mkdir(exist_ok=True)
    plan={'status':'fixed_before_cached_grid_metrics','family_selection_sha256':digest(ROOT/'family_selection_before_fusion.json'),
        'cohort_hashes':c['hashes'],'code_sha256':digest(__file__),
        'method':'Same component scores, observed games masked before stable descending argsort; rank0 for nonpositive scores. Reuse ranks across fixed weights, no model or grid changes.'}
    planpath=cache/'execution_protocol.json'
    if planpath.exists() and json.loads(planpath.read_text('utf-8'))!=plan:raise ValueError('RRF execution protocol changed')
    write_json(planpath,plan)
    shape=(len(c['user_ids']),int(c['exclusions'].shape[1]))
    complete=cache/'complete.json';paths=[cache/'collaborative_ranks.npy',cache/'content_ranks.npy']
    if not complete.exists():
        if any(p.exists() for p in paths):raise FileExistsError('Inspect partial RRF rank cache')
        arrays=[np.lib.format.open_memmap(p,mode='w+',dtype=np.uint16,shape=shape) for p in paths]
        for start in range(0,shape[0],256):
            end=min(start+256,shape[0]);q=c['positive'][start:end];neg=c['negative'][start:end]
            r,col=c['exclusions'][start:end].nonzero()
            for model,arr in zip([left,right],arrays):
                values=model.score(q,neg);values[r,col]=0
                order=np.argsort(-values,axis=1,kind='stable');ranks=np.empty(order.shape,dtype=np.uint16)
                np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,shape[1]+1,dtype=np.uint16),order.shape),axis=1)
                ranks[values<=0]=0;arr[start:end]=ranks
            if start==0 or end==shape[0] or start%(256*250)==0:log(f'RRF shared ranks: {end:,}/{shape[0]:,}')
        for a in arrays:a.flush()
        del arrays
        write_json(complete,{'status':'complete','protocol_sha256':digest(planpath),'shape':shape,
            'files':{p.name:digest(p) for p in paths}})
    receipt=json.loads(complete.read_text('utf-8'))
    if receipt['protocol_sha256']!=digest(planpath):raise ValueError('RRF rank cache changed')
    for p in paths:
        if digest(p)!=receipt['files'][p.name]:raise ValueError('RRF rank file changed')
    ranks=[np.load(p,mmap_mode='r') for p in paths]
    checks=[]
    for w in [.25,.5,.75]:
        for adaptive in [False,True]:
            name='selected_rrf_'+('adaptive_' if adaptive else '')+str(w).replace('.','_')
            spec={'method':'reciprocal_rank_fusion','rank_constant':60,'content_weight':w,'adaptive':adaptive,
                'left':left.spec,'right':right.spec,'family_selection_sha256':digest(ROOT/'family_selection_before_fusion.json')}
            original=RankFusionScorer(left,right,w,60,adaptive,spec)
            checked=0
            for start in [0,shape[0]//2,shape[0]-128]:
                end=min(start+128,shape[0]);q=c['positive'][start:end];n=c['negative'][start:end]
                direct=original.fuse(left.score(q,n),right.score(q,n),c['exclusions'][start:end])
                cached=CachedRankFusion(*ranks,original);cached.cursor=start;actual=cached.score(q,n)
                if not np.array_equal(direct,actual):raise ValueError('RRF rank cache changes scores: '+name)
                checked+=actual.size
            checks.append({'variant':name,'candidate_scores_compared':checked,'status':'bitwise_equal'})
            evaluate(name,CachedRankFusion(*ranks,original),c,baseline('hybrid'),family='hybrid')
    write_json(cache/'score_equivalence.json',{'status':'bitwise_equal','execution_protocol_sha256':digest(planpath),'checks':checks})


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
