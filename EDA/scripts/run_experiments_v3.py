"""Execute the predeclared backlog experiments without changing prior runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import EDA,digest,quote,log
from experiment_v3_common import (ROOT,SPLIT,PROTOCOL,V2,catalog,current_models,
    evaluate,frozen_cohort,load_matrix,CombinedScorer,NegativeScorer,RankFusionScorer,
    model_files,Scorer,write_json)
from experiment_v3_query import weighted_query,build_item_kernel,MaxScorer,NegativeFilterScorer


def baseline(family,cohort='validation'):
    name='content_control' if family=='content' else 'item_knn_control'
    return ROOT/'evaluations'/cohort/name/'per_user.parquet'


def controls():
    c=frozen_cohort()
    for name,model in current_models().items():
        family='content' if name.startswith('content') else 'collaborative'
        evaluate(name,model,c,family=family)
        old=(V2/'evaluations/validation/metadata_alpha_0_5/per_user.parquet' if family=='content'
             else EDA/'data/processed/modeling/baseline_v1/validation/item_knn/per_user.parquet')
        a=pq.read_table(ROOT/'evaluations/validation'/name/'per_user.parquet').to_pydict()
        b=pq.read_table(old).to_pydict()
        for column in ['user_key','target_app_id','target_rank','recommendation_count']:
            if a[column]!=b[column]:raise ValueError('Control failed exact prior result: '+column)
        log('Exact previous control reproduced: '+name)


def model_registry(family):
    directories=[ROOT/'models',ROOT/'content',ROOT/'content_models',ROOT/'collaborative_models']
    models=[]
    for directory in directories:
        for p in sorted(directory.glob('*/manifest.json')):
            m=json.loads(p.read_text('utf-8'))
            if m.get('status')!='complete':continue
            if directory==ROOT/'content' and p.parent.name=='description_missing_fill_bigram':
                # A worker independently built the same frozen ablation. Keep
                # both artifacts, evaluate the root protocol's canonical one.
                canonical=ROOT/'content_models'/p.parent.name
                if not (canonical/'manifest.json').exists():raise ValueError('Missing canonical filled bigram')
                if digest(p.parent/'tfidf_csr.npz')!=digest(canonical/'tfidf_csr.npz'):
                    raise ValueError('Independent filled-bigram builds disagree')
                a=pq.read_table(p.parent/'game_rows.parquet',columns=['app_id','linux'])
                b=pq.read_table(canonical/'game_rows.parquet',columns=['app_id','linux'])
                if not a.equals(b):raise ValueError('Filled-bigram mappings disagree')
                write_json(ROOT/'filled_bigram_equivalence.json',{
                    'status':'verified_byte_identical_tfidf_and_equal_app_linux_mapping',
                    'canonical':model_files(canonical),'independent':model_files(p.parent),
                    'matrix_sha256':digest(canonical/'tfidf_csr.npz')})
                continue
            actual=m.get('family','content' if 'content' in directory.name else 'collaborative')
            if actual!=family:continue
            if p.parent.name=='train_review_body_tfidf' and (ROOT/'review_item_cosine_cache/numerical_equivalence.json').exists():
                from select_experiments_v3 import restore
                cache=ROOT/'review_item_cosine_cache'
                info=json.loads((cache/'kernel_complete.json').read_text('utf-8'))
                spec={'method':'precomputed_item_content_cosine','source':model_files(p.parent),
                    'cache_file':str((cache/'item_candidate_cosine.npy').relative_to(ROOT)),
                    'cache_sha256':info['sha256'],
                    'execution_protocol_sha256':digest(cache/'execution_protocol.json')}
                models.append((p.parent.name,restore(spec)))
            elif p.parent.name.startswith('ease_lambda'):
                from cached_knn_v3 import CachedKNNScorer
                models.append((p.parent.name,CachedKNNScorer(load_matrix(p.parent))))
            elif m.get('scorer_mode')=='signed_knn':
                model=load_matrix(p.parent)
                models.append((p.parent.name,SignedScorer(model,m['parameters']['negative_query_coefficient'])))
            elif (p.parent/'item_factors.npy').exists():
                models.append((p.parent.name,load_factor_model(p.parent,m)))
            elif m.get('components'):
                load_matrix(p.parent,'content') # Verify full listed inventory.
                children=[Scorer(sparse.load_npz(p.parent/item['path']),'content') for item in m['components']]
                models.append((p.parent.name,CombinedScorer(children,[v['weight'] for v in m['components']],model_files(p.parent))))
            else:models.append((p.parent.name,load_matrix(p.parent,'content' if family=='content' else 'knn')))
    return models


class SignedScorer:
    def __init__(self,base,beta):
        self.base,self.beta=base,float(beta)
        self.spec={'method':'signed_item_relations','beta':self.beta,'source':base.spec}
    def score(self,q,negative=None):
        n=np.maximum(np.asarray(q.sum(axis=1)).ravel(),1)
        inputs=q if negative is None else q-self.beta*negative
        return (inputs@self.base.matrix.T).toarray()/n[:,None]


class FoldInScorer:
    def __init__(self,folder,manifest):
        self.factors=np.load(folder/'item_factors.npy')
        self.apps,self.linux,_=catalog()
        params=manifest.get('parameters',{})
        self.reg=float(params.get('regularization',.1))
        self.alpha=float(params.get('alpha',params.get('confidence_scale',20)))
        self.gram=self.factors.T@self.factors + self.reg*np.eye(self.factors.shape[1])
        self.spec=model_files(folder)

    def score(self,q,negative=None):
        # implicit ALS C=alpha for observed binary positives, C=1 otherwise.
        # Fold-in solves (Y'Y + Y'(C-I)Y + lambda I)x = Y'C p.
        profiles=np.zeros((q.shape[0],self.factors.shape[1]))
        for row in range(q.shape[0]):
            ids=q.indices[q.indptr[row]:q.indptr[row+1]]
            values=q.data[q.indptr[row]:q.indptr[row+1]]
            if not len(ids):continue
            y=self.factors[ids].astype(np.float64)
            confidence=self.alpha*values
            a=self.gram+y.T@((confidence-1)[:,None]*y)
            b=y.T@confidence
            profiles[row]=np.linalg.solve(a,b)
        return profiles@self.factors[self.linux].T


def load_factor_model(folder,manifest):
    for rel,info in manifest.get('files',{}).items():
        if (folder/rel).is_file() and digest(folder/rel)!=info['sha256']:raise ValueError('Factor artifact changed')
    return FoldInScorer(folder,manifest)


def run_family(family):
    c=frozen_cohort()
    names=set()
    for name,model in model_registry(family):
        if name in names:raise ValueError('Duplicate model name '+name)
        names.add(name)
        if name.startswith('ease_lambda'):
            from cached_knn_v3 import verified_cache
            model=verified_cache(model,c['positive'],name)
        evaluate(name,model,c,baseline(family),family=family)
    if not names:raise ValueError('No completed '+family+' models')


def train_positive_rate():
    path=ROOT/'train_positive_rate.parquet'
    if not path.exists():
        con=duckdb.connect()
        try:
            con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
            con.execute(f'''COPY (SELECT app_id,count(*) n,sum(positive::INTEGER) n_positive,
                (sum(positive::INTEGER)+1.)/(count(*)+2.) positive_rate
                FROM read_parquet({quote(SPLIT/'train/interactions.parquet')}) GROUP BY app_id ORDER BY app_id)
                TO {quote(path)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        finally:con.close()
    d=pq.read_table(path).to_pydict()
    apps,linux,_=catalog()
    out=np.full(len(apps),.5)
    positions=np.searchsorted(apps,d['app_id'])
    ok=positions<len(apps)
    ok[ok]&=apps[positions[ok]]==np.asarray(d['app_id'])[ok]
    out[positions[ok]]=np.asarray(d['positive_rate'])[ok]
    return out[linux]


def run_query():
    c=frozen_cohort();models=current_models()
    # Query weights are shared by both models and never alter game visibility.
    weights={method:weighted_query(c,method) for method in ['recency','playtime']}
    prior=train_positive_rate()
    for control,base in models.items():
        family='content' if control=='content_control' else 'collaborative'
        prefix='content' if family=='content' else 'knn'
        b=baseline(family)
        for method,q in weights.items():
            weighted=CombinedScorer([base],[1.],{'method':method,'base':base.spec})
            evaluate(prefix+'_'+method,weighted,c,b,query_weights=q,family=family)
        for coefficient in [.1,.3]:
            model=NegativeScorer(base,coefficient,{'method':'negative_feedback_penalty','coefficient':coefficient,'base':base.spec})
            evaluate(prefix+'_negative_'+str(coefficient).replace('.','_'),model,c,b,family=family)
        evaluate(prefix+'_popularity_fill',base,c,b,popularity_fill=True,family=family)
        for exponent in [.5,1.]:
            model=CombinedScorer([base],[1.],{'method':'train_positive_rate','exponent':exponent,'base':base.spec,
                'rate_file_sha256':digest(ROOT/'train_positive_rate.parquet')})
            evaluate(prefix+'_positive_rate_'+str(exponent).replace('.','_'),model,c,b,
                score_transform=lambda s,q,n,i,j,e=exponent:s*(prior**e)[None,:],family=family)
        kernel=build_item_kernel(base,prefix)
        evaluate(prefix+'_max_similarity',MaxScorer(base,kernel),c,b,family=family)
        for threshold in [.5,.8]:
            evaluate(prefix+'_negative_filter_'+str(threshold).replace('.','_'),
                NegativeFilterScorer(base,kernel,threshold),c,b,family=family)
        # Negative review reliability by playtime is only valid where measured
        # query playtime exists. The experiment records the exact visible rows.
        from experiment_v3_query import context
        d,r,cols=context(c)
        neg=~np.asarray(d['positive'])
        minutes=np.array([v if v is not None and v>=0 else 0 for v in d['playtime_forever']])
        reliability=1/(1+np.log1p(minutes)/np.log1p(6000))
        negweights=sparse.csr_matrix((reliability[neg].astype(np.float32),(r[neg],cols[neg])),shape=c['negative'].shape)
        # Normalize by NUMBER of negative observations, not sum of reliability;
        # otherwise uniformly downweighting unreliable negatives would cancel.
        class ReliableNegative:
            spec={'method':'playtime_reliability_negative_penalty','coefficient':.3,'base':base.spec,
                  'reliability':'1/(1+log1p(minutes)/log1p(6000)); divide by original negative count'}
            def score(self,q,negative=None):return base.score(q)
        def transform(s,q,n,begin,end):
            weighted=negweights[begin:end]
            original=np.maximum(np.diff(c['negative'].indptr)[begin:end],1)
            mass=np.asarray(weighted.sum(axis=1)).ravel()
            # base.score normalizes profiles; scale by retained reliability mass
            # to preserve the intended confidence attenuation.
            return s-.3*base.score(weighted)*(mass/original)[:,None]
        evaluate(prefix+'_negative_playtime_reliability',ReliableNegative(),c,b,
            score_transform=transform,family=family)


def run_fusion():
    c=frozen_cohort();models=current_models()
    for weight in [.25,.5,.75]:
        for adaptive in [False,True]:
            name='rrf_'+('adaptive_' if adaptive else '')+str(weight).replace('.','_')
            spec={'method':'reciprocal_rank_fusion','rank_constant':60,'content_weight':weight,
                  'adaptive':adaptive,'left':models['item_knn_control'].spec,'right':models['content_control'].spec}
            model=RankFusionScorer(models['item_knn_control'],models['content_control'],weight,60,adaptive,spec)
            evaluate(name,model,c,baseline('hybrid'),family='hybrid')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['controls','collaborative','content','query','fusion'])
    a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.phase=='controls':controls()
        elif a.phase in ['collaborative','content']:run_family(a.phase)
        elif a.phase=='query':run_query()
        else:run_fusion()


if __name__=='__main__':main()
