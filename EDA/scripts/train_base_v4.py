"""Fresh account-disjoint Item-KNN controls and V4-10 contribution weighting."""
from __future__ import annotations
import argparse
import gc
import json
import time

import duckdb
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import digest,log,quote
from item_knn import _build_user_item_csr
from improvement_common import write_json
from experiment_v4_common import ROOT,SPLIT,catalog


def training_matrix():
    work=ROOT/'training';work.mkdir(parents=True,exist_ok=True)
    path=work/'positive_csr.npz';receipt=work/'csr_manifest.json'
    train=SPLIT/'train/positive_catalog.parquet'
    if receipt.exists():
        m=json.loads(receipt.read_text('utf-8'))
        if m['train_sha256']!=digest(train) or m['csr_sha256']!=digest(path):
            raise ValueError('Training cache source changed')
        return sparse.load_npz(path),np.load(work/'support.npy')
    apps,_=catalog(); c=duckdb.connect()
    c.execute("SET memory_limit='1500MB'");c.execute('SET threads=2')
    pairs,users=c.execute(f'SELECT count(*),count(DISTINCT user_key) FROM read_parquet({quote(train)})').fetchone();c.close()
    log(f'Constructing v4 training CSR: {pairs:,} positive pairs / {users:,} users')
    x,support,_=_build_user_item_csr(pq.ParquetFile(train).iter_batches(batch_size=131072),apps,pairs,users)
    sparse.save_npz(path,x);np.save(work/'support.npy',support)
    write_json(receipt,dict(status='complete',users=users,pairs=pairs,train_sha256=digest(train),
        split_manifest_sha256=digest(SPLIT/'manifest.json'),csr_sha256=digest(path),
        code_sha256=digest(__file__)))
    return x,support


def fit(include_pair_weighting=False):
    x,sup=training_matrix(); apps,linux=catalog(); cols=np.flatnonzero(linux)
    root=ROOT/'models';root.mkdir(parents=True,exist_ok=True);np.save(root/'positive_support.npy',sup)
    degree=np.diff(x.indptr).astype(np.float64)
    x=x.astype(np.float64)
    specs=[('binary_item_knn_k100',None),('activity_inverse_degree_cosine_k100',0.)]
    if include_pair_weighting:
        specs += [(f'pair_contribution_rho{rho:g}_fixed',rho) for rho in (.25,.5,1)]
    for name,rho in specs:
        folder=root/name; receipt=folder/'manifest.json'
        if receipt.exists():
            log('Reuse v4 '+name);continue
        if rho is None: weights=np.ones(len(degree))
        else: weights=(1-rho)/degree + rho/(degree*np.maximum(1,degree-1))
        weighted=x.multiply(np.sqrt(weights)[:,None]).tocsc(); margins=np.asarray(weighted.power(2).sum(axis=0)).ravel()
        rows=weighted.T.tocsr(); indptr=[0];indices=[];data=[];start=time.perf_counter()
        # V4-10 first changes weights on the selected activity KNN's stored
        # neighbors; neighbor reselection is a separate predeclared comparison.
        fixed=(sparse.load_npz(root/'activity_inverse_degree_cosine_k100/similarities_csr.npz')
            if rho not in (None,0.) else None)
        for begin in range(0,len(cols),32):
            end=min(len(cols),begin+32); block=(rows[cols[begin:end]]@weighted).tocsr()
            for local,own in enumerate(cols[begin:end]):
                b,e=block.indptr[local:local+2]; ix=block.indices[b:e]; v=block.data[b:e]
                keep=(ix!=own)&(v>0)&(margins[ix]>0)&(margins[own]>0)
                ix,v=ix[keep],v[keep]
                v=v/np.sqrt(margins[own]*margins[ix]) if margins[own]>0 else np.zeros_like(v)
                if fixed is not None:
                    allowed=fixed.indices[fixed.indptr[begin+local]:fixed.indptr[begin+local+1]]
                    keep=np.isin(ix,allowed); ix,v=ix[keep],v[keep]
                chosen=np.lexsort((apps[ix],-v))[:100];ix,v=ix[chosen],v[chosen]
                order=np.argsort(ix);indices.extend(ix[order]);data.extend(v[order]);indptr.append(len(data))
            if begin==0 or end==len(cols) or begin%640==0:log(f'v4 {name}: {end:,}/{len(cols):,} candidates')
        matrix=sparse.csr_matrix((np.asarray(data,dtype=np.float32),np.asarray(indices,dtype=np.int32),
            np.asarray(indptr,dtype=np.int32)),shape=(len(cols),len(apps)))
        folder.mkdir(parents=True,exist_ok=True);sparse.save_npz(folder/'similarities_csr.npz',matrix)
        write_json(receipt,dict(status='complete',variant=name,rho=rho,
            weight='binary' if rho is None else '(1-rho)/d + rho/(d*max(1,d-1))',
            fixed_neighbors=fixed is not None,training_sha256=digest(SPLIT/'train/positive_catalog.parquet'),
            elapsed_seconds=time.perf_counter()-start,code_sha256=digest(__file__),
            matrix_sha256=digest(folder/'similarities_csr.npz')))
        del weighted,rows;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair-weighting',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=2):fit(args.pair_weighting)
