"""Query aggregation and negative-feedback ablations with frozen inputs."""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from eda_large_review_frank import EDA,digest,quote,log
from experiment_v3_common import ROOT,SPLIT,catalog,Scorer,CombinedScorer,write_json


def context(cohort):
    """Only already-visible query rows; hidden target attributes are never used."""
    name=cohort['name']
    path=ROOT/'query_context'/f'{name}.parquet'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        con=duckdb.connect(str(EDA/'data/processed/kieran_original/eda_work.duckdb'),read_only=True)
        try:
            con.execute("SET memory_limit='1500MB'")
            con.execute('SET threads=2')
            if name=='validation':
                sql=f'''SELECT q.user_key,q.app_id,q.positive,
                    min(r.timestamp_created) timestamp_created,
                    max(r.playtime_forever) playtime_forever
                    FROM read_parquet({quote(SPLIT/name/'query.parquet')}) q
                    JOIN read_parquet({quote(SPLIT/name/'review_lineage.parquet')}) l
                        USING(user_key,app_id,positive)
                    JOIN valid_raw r USING(user_key,app_id,positive,review_key)
                    GROUP BY q.user_key,q.app_id,q.positive'''
            elif name=='test':
                from prepare_query_features_v3 import fetch_frank_features
                features=fetch_frank_features()
                # Extra columns are joined to already-visible query identities;
                # hidden-target attributes remain outside the input features.
                sql=f'''SELECT q.user_key,q.app_id,q.positive,
                    min(f.timestamp_created) timestamp_created,
                    max(x.author_playtime_forever)::DOUBLE playtime_forever
                    FROM read_parquet({quote(SPLIT/name/'query.parquet')}) q
                    LEFT JOIN frank_raw f USING(user_key,app_id,positive)
                    LEFT JOIN read_parquet({quote(features)}) x
                       ON q.user_key=try_cast(x.author_steamid AS UBIGINT)
                       AND q.app_id=x.appid AND q.positive=x.voted_up
                    GROUP BY q.user_key,q.app_id,q.positive'''
            else:
                raise ValueError('Query context only for the fixed primary cohorts')
            con.execute(f'COPY ({sql} ORDER BY user_key,app_id) TO {quote(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        finally:
            con.close()
        write_json(path.with_suffix('.json'),{'status':'complete','file_sha256':digest(path),
            'query_sha256':digest(SPLIT/name/'query.parquet'),
            'source':'accepted Kieran query review lineage' if name=='validation' else 'pinned frank query-only timestamps and extra playtime projection',
            'playtime_is_snapshot_cumulative_not_at_review':True,'hidden_target_attributes_loaded':False,
            'builder_sha256':digest(__file__)})
    d=pq.read_table(path).to_pydict()
    apps,_,_=catalog()
    r=np.searchsorted(cohort['user_ids'],np.asarray(d['user_key'],dtype=np.uint64))
    c=np.searchsorted(apps,d['app_id'])
    if len(r)!=int(cohort['n_total'].sum()):raise ValueError('Query context coverage differs')
    return d,r,c


def weighted_query(cohort,method):
    d,r,c=context(cohort)
    positive=np.asarray(d['positive'])
    if method=='recency':
        t=np.array([v if v is not None else np.nan for v in d['timestamp_created']],dtype=float)
        latest=np.full(len(cohort['user_ids']),-np.inf)
        np.maximum.at(latest,r,np.nan_to_num(t,nan=-np.inf))
        weights=np.exp2(-np.maximum(0,latest[r]-t)/(365.25*86400))
        weights=np.nan_to_num(weights,nan=1.)
    elif method=='playtime':
        t=np.array([v if v is not None else np.nan for v in d['playtime_forever']],dtype=float)
        weights=np.where(np.isfinite(t)&(t>=0),1+np.log1p(np.maximum(0,t)),1.)
    else:raise ValueError(method)
    q=sparse.csr_matrix((weights[positive].astype(np.float32),(r[positive],c[positive])),shape=cohort['positive'].shape)
    if np.any(np.diff(q.indptr)<1) or (q.astype(bool)!=cohort['positive'].astype(bool)).nnz:
        raise ValueError('Weighted queries changed the visible positive games')
    return q


def build_item_kernel(base,name,block=256):
    """Exact input-item × Linux-candidate similarities for max aggregation.

    Item kernels differ from normalized mean-profile cosine; this is explicit.
    """
    out=ROOT/'query_kernels'/f'{name}.npy'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():return np.load(out,mmap_mode='r')
    apps,linux,_=catalog()
    matrix=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=(len(apps),int(linux.sum())))
    if isinstance(base,Scorer) and base.mode=='knn':
        for begin in range(0,len(apps),block):
            matrix[begin:begin+block]=base.matrix[:,begin:begin+block].T.toarray()
    else:
        children,weights=(base.children,base.weights) if isinstance(base,CombinedScorer) else ([base],[1.])
        for begin in range(0,len(apps),block):
            end=min(begin+block,len(apps))
            scores=np.zeros((end-begin,int(linux.sum())),dtype=np.float32)
            for child,w in zip(children,weights):
                values=child.matrix[begin:end]@child.matrix[linux].T
                scores+=w*(values.toarray() if sparse.issparse(values) else values)
            matrix[begin:end]=scores
    matrix.flush()
    write_json(out.with_suffix('.json'),{'status':'complete','sha256':digest(out),'base':base.spec,
        'shape':list(matrix.shape),'builder_sha256':digest(__file__)})
    return matrix


def max_similarity(kernel,query):
    result=np.zeros((query.shape[0],kernel.shape[1]),dtype=np.float64)
    for row in range(query.shape[0]):
        ids=query.indices[query.indptr[row]:query.indptr[row+1]]
        if len(ids):result[row]=np.max(kernel[ids],axis=0)
    return result


class MaxScorer:
    def __init__(self,base,kernel):
        self.base,self.kernel=base,kernel
        self.spec={'method':'max_item_similarity','base':base.spec}
    def score(self,q,negative=None):return max_similarity(self.kernel,q)


class NegativeFilterScorer:
    def __init__(self,base,kernel,threshold):
        self.base,self.kernel,self.threshold=base,kernel,threshold
        self.spec={'method':'negative_similarity_filter','threshold':threshold,'base':base.spec}
    def score(self,q,negative=None):
        s=self.base.score(q)
        if negative is not None:
            values=max_similarity(self.kernel,negative)
            s[values>=self.threshold]=0
        return s
