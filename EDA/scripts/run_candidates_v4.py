"""V4-4 candidate diagnostics and V4-3 bounded content residual experiments.

Target labels enter only fusion_fit fitting and evaluation. Retrieval never
injects a target, and features are derived exclusively from visible inputs.
"""
from __future__ import annotations
import argparse
import gc
import json
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from eda_large_review_frank import digest,log
from improvement_common import write_json
from experiment_v4_common import (ROOT,SPLIT,PROTOCOL,catalog,support,load_cohort,MatrixScorer,
    ContentScorer,masked_scores,evaluate,paired_comparison,metrics)

POLICIES=['top50_union','top100_union','support_budget200']
FEATURES=['cf_score','content_score','log_support','log_positive_inputs',
    'connected_inputs','effective_contributions','max_contribution_share','rank_agreement',
    'cf_gap10_11','content_missing']


def top(values,apps,limit):
    valid=np.flatnonzero(np.isfinite(values)&(values>0))
    if len(valid)>limit:
        cutoff=np.partition(values[valid],len(valid)-limit)[-limit]
        valid=valid[values[valid]>=cutoff]
    return valid[np.lexsort((apps[valid],-values[valid]))[:limit]]


def candidate_sets(a,b,apps,sup):
    aa,bb=top(a,apps,200),top(b,apps,6630)
    simple50=np.union1d(aa[:50],bb[:50]);simple100=np.union1d(aa[:100],bb[:100])
    rare1=bb[(sup[bb]>=1)&(sup[bb]<100)][:25]
    rare2=bb[(sup[bb]>=100)&(sup[bb]<1000)][:25]
    seen=set();budget=[]
    for part in [aa[:100],bb[:50],rare1,rare2,aa,bb]:
        for j in part:
            if int(j) not in seen and len(budget)<200:budget.append(int(j));seen.add(int(j))
    return [simple50,simple100,np.sort(budget)]


def normalized(a):
    maximum=np.max(a,axis=1,keepdims=True)
    return np.divide(a,maximum,out=np.zeros_like(a),where=maximum>0)


def features(a,b,chosen,query,weights,apps,candsup,content_missing):
    qcols=query.indices
    contribution=weights[chosen][:,qcols].toarray()
    count=(contribution>0).sum(axis=1)
    total=contribution.sum(axis=1)
    square=np.square(contribution).sum(axis=1)
    eff=np.divide(total**2,square,out=np.zeros_like(total),where=square>0)
    maximum=contribution.max(axis=1) if len(qcols) else np.zeros(len(chosen))
    share=np.divide(maximum,total,out=np.zeros_like(total),where=total>0)
    ra=np.empty(len(a),dtype=int);rb=np.empty(len(b),dtype=int)
    ra[np.lexsort((apps,-a))]=np.arange(1,len(a)+1)
    rb[np.lexsort((apps,-b))]=np.arange(1,len(b)+1)
    top11=top(a,apps,11)
    gap=(a[top11[9]]-a[top11[10]]) if len(top11)==11 else 0.
    return np.column_stack([a[chosen],b[chosen],np.log1p(candsup[chosen]),
        np.full(len(chosen),np.log1p(len(qcols))),np.log1p(count),np.log1p(eff),share,
        1/(1+np.abs(ra[chosen]-rb[chosen])),np.full(len(chosen),gap),content_missing[chosen]])


def retrieve(cohort):
    folder=ROOT/'candidates'/cohort['name']; receipt=folder/'manifest.json'
    model_sources=dict(cf_matrix_sha256=digest(ROOT/'models/activity_inverse_degree_cosine_k100/similarities_csr.npz'),
        content_manifest_sha256=digest(ROOT.parent/'improvement_v3/content/metadata_extended_tfidf/manifest.json'),
        support_sha256=digest(ROOT/'models/positive_support.npy'),cohort_hashes=cohort['hashes'])
    if receipt.exists():
        m=json.loads(receipt.read_text('utf-8'))
        if m['code_sha256']!=digest(__file__) or m['protocol_sha256']!=digest(PROTOCOL) or m['model_sources']!=model_sources:
            raise ValueError('Candidate code/protocol changed')
        for f,h in m['files'].items():
            if digest(folder/f)!=h: raise ValueError('Candidate artifact changed')
        return folder
    if cohort['name']=='confirmation':raise ValueError('Retrieval for confirmation requires a separate frozen finalist runner')
    folder.mkdir(parents=True,exist_ok=True); apps,linux=catalog();ids=apps[linux];sup=support()[linux]
    weights=sparse.load_npz(ROOT/'models/activity_inverse_degree_cosine_k100/similarities_csr.npz')
    cf=MatrixScorer(weights);content=ContentScorer()
    missing=np.asarray(sum(c.matrix.getnnz(axis=1) for c in content.channels)==0)[linux]
    n=len(cohort['user_ids']);hits=np.zeros((n,5),dtype=bool);counts=np.zeros((n,3),dtype=np.int16)
    schema=pa.schema([('user_row',pa.int32()),('candidate_col',pa.int32()),('label',pa.int8())]
        +[(f,pa.float64() if f in ('cf_score','content_score') else pa.float32()) for f in FEATURES]
        +[('policy_mask',pa.int8())])
    with pq.ParquetWriter(folder/'features.parquet',schema,compression='zstd') as writer:
        for begin in range(0,n,128):
            end=min(n,begin+128);a=normalized(masked_scores(cf,cohort,begin,end));b=normalized(masked_scores(content,cohort,begin,end))
            records=[]
            for local,row in enumerate(range(begin,end)):
                sets=candidate_sets(a[local],b[local],ids,sup);target=cohort['target_candidates'][row]
                hits[row,:2]=[target in top(a[local],ids,100),target in top(b[local],ids,100)]
                hits[row,2:]=[target in part for part in sets];counts[row]=[len(part) for part in sets]
                chosen=np.union1d(np.union1d(*sets[:2]),sets[2])
                x=features(a[local],b[local],chosen,cohort['positive'][row],weights,ids,sup,missing)
                masks=sum((np.isin(chosen,part).astype(np.int8)<<p) for p,part in enumerate(sets))
                records.extend((row,int(j),int(j==target),*v,int(mask)) for j,v,mask in zip(chosen,x,masks))
            writer.write_table(pa.Table.from_pylist([dict(zip(schema.names,r)) for r in records],schema=schema))
            if begin==0 or end==n or begin%12800==0:log(f'v4 retrieval {cohort["name"]}: {end:,}/{n:,}')
    np.savez_compressed(folder/'diagnostics.npz',hits=hits,counts=counts)
    main=cohort['n_total']>=5
    labels=['cf_top100','content_top100',*POLICIES]
    diagnostic={label:dict(users=int(main.sum()),candidate_recall=float(hits[main,j].mean()),
        actual_candidates_mean=float(counts[main,j-2].mean()) if j>=2 else None)
        for j,label in enumerate(labels)}
    # Per-user flags preserve denominators for later game/support stratification.
    write_json(receipt,dict(status='complete',role=cohort['name'],features=FEATURES,
        primary_diagnostics=diagnostic,no_target_injection=True,code_sha256=digest(__file__),
        protocol_sha256=digest(PROTOCOL),cohort_hashes=cohort['hashes'],model_sources=model_sources,
        files={f:digest(folder/f) for f in ['features.parquet','diagnostics.npz']}))
    return folder


class CachedResidual:
    def __init__(self,table,eta,gate=None,columns=None,policy=2):
        self.rows=table['user_row'].to_numpy();self.cols=table['candidate_col'].to_numpy()
        self.x=np.column_stack([table[f].to_numpy() for f in FEATURES])
        self.keep=(table['policy_mask'].to_numpy()&(1<<policy))>0
        self.eta,self.gate,self.columns=eta,gate,columns;self.cursor=0
        self.spec=dict(eta=eta,feature_columns=columns,policy=POLICIES[policy],
            feature_rows=table.num_rows,gate_type=type(gate).__name__ if gate is not None else 'constant1')
    def score(self,q):
        start,end=self.cursor,self.cursor+q.shape[0];self.cursor=end
        left,right=np.searchsorted(self.rows,[start,end]);r=self.rows[left:right]-start;c=self.cols[left:right]
        x=self.x[left:right];keep=self.keep[left:right]
        g=np.ones(len(x)) if self.gate is None else self.gate.predict_proba(x[:,self.columns])[:,1]
        values=x[:,0]+self.eta*g*x[:,1]
        out=np.zeros((q.shape[0],6630));out[r[keep],c[keep]]=values[keep]
        return out


def fit_gates():
    cohort=load_cohort('fusion_fit');folder=retrieve(cohort);table=pq.read_table(folder/'features.parquet')
    x=np.column_stack([table[f].to_numpy() for f in FEATURES]);y=table['label'].to_numpy()
    policy=(table['policy_mask'].to_numpy()&4)>0;rows=table['user_row'].to_numpy()
    # Equal user mass before balanced weak-label classes. One user can retrieve
    # fewer candidates; a wide list must not silently gain more training weight.
    weight=1/np.maximum(np.bincount(rows[policy],minlength=len(cohort['user_ids']))[rows],1)
    w=weight[policy];yy=y[policy]
    for name,columns in [('prior',list(range(4))),('evidence',list(range(10)))]:
        path=ROOT/'gates'/f'{name}.pkl';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():continue
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,max_depth=3,
            l2_regularization=1.,min_samples_leaf=100,learning_rate=.05,random_state=20261108,
            early_stopping=False,class_weight='balanced')
        model.fit(x[policy][:,columns],yy,sample_weight=w)
        with path.open('wb') as f:pickle.dump(model,f)
        write_json(path.with_suffix('.json'),dict(status='complete',features=[FEATURES[j] for j in columns],
            fit_users=len(cohort['user_ids']),fit_rows=int(policy.sum()),retrieved_targets=int(yy.sum()),
            source_sha256=digest(folder/'features.parquet'),model_sha256=digest(path),
            weak_labels='Unobserved candidate label0 is not an explicit dislike',code_sha256=digest(__file__)))
    del table,x;gc.collect()


def run():
    fit_gates();cohort=load_cohort('selection');folder=retrieve(cohort)
    table=pq.read_table(folder/'features.parquet')
    for policy in range(3):
        for eta in [0,.1,.25,.5]:
            scorer=CachedResidual(table,eta,policy=policy)
            scorer.spec['candidate_manifest_sha256']=digest(folder/'manifest.json')
            evaluate(f'residual_fixed_eta{eta:g}_{POLICIES[policy]}',scorer,cohort)
    for name,columns in [('prior',list(range(4))),('evidence',list(range(10)))]:
        with (ROOT/'gates'/f'{name}.pkl').open('rb') as f:model=pickle.load(f)
        for eta in [.1,.25,.5]:
            scorer=CachedResidual(table,eta,model,columns)
            scorer.spec.update(candidate_manifest_sha256=digest(folder/'manifest.json'),
                gate_sha256=digest(ROOT/'gates'/f'{name}.pkl'))
            evaluate(f'residual_{name}_eta{eta:g}',scorer,cohort)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--retrieve',choices=['fusion_fit','selection']);args=p.parse_args()
    with threadpool_limits(limits=2):
        if args.retrieve:retrieve(load_cohort(args.retrieve))
        else:run()
