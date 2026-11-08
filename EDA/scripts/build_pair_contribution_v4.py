"""V4-10: mix two train-only co-occurrence statistics, then fix/reselect KNN.

Linearity avoids repeating the sparse product for every rho. The rho=0
prediction matrix is checked against the existing fresh activity control.
"""
from __future__ import annotations
import gc
import json
import shutil
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import digest, log
import experiment_linear_v4 as linear
from train_base_v4 import training_matrix


def mixture_cosine(a,b,margin_a,margin_b,own,rho):
    counts=(1-rho)*a+rho*b
    margin=(1-rho)*margin_a+rho*margin_b
    out=np.zeros_like(counts,dtype=np.float64)
    if margin[own]>0:
        denominator=np.sqrt(margin[own]*margin)
        np.divide(counts,denominator,out=out,where=denominator>0)
    out[own]=0
    return out


def run():
    linear.freeze_roster();sources=linear._source_manifest();apps,candidates=linear._catalog()
    roster=linear._variant_roster()
    names=[f'account_pair_rho{linear._name_num(rho)}_{policy}_k100'
        for rho in (0.,.25,.5,1.) for policy in ('fixed','reselected')]
    active=[name for name in names if not linear._verify_existing(linear.MODEL_ROOT/name,
        name,roster[name]['parameters'],sources)]
    if not active:return names
    control=linear.MODEL_ROOT/'activity_inverse_degree_cosine_k100'
    m=json.loads((control/'manifest.json').read_text('utf-8'))
    if m['training_sha256']!=sources['train/positive_catalog.parquet']['sha256']:
        raise ValueError('Activity control belongs to another training split')
    if digest(control/'similarities_csr.npz')!=m['matrix_sha256']:
        raise ValueError('Activity control changed')
    base=sparse.load_npz(control/'similarities_csr.npz')
    x,support=training_matrix();degree=np.diff(x.indptr).astype(np.float64)
    if np.any(degree<1) or x.shape[1]!=len(apps):raise ValueError('Positive train CSR changed')
    users=x.shape[0];pairs=x.nnz;singletons=int((degree==1).sum())
    a=x.astype(np.float64).tocsc();del x;gc.collect()
    w=1/degree;a.data=np.sqrt(w[a.indices]);del w
    w=1/(degree*np.maximum(1,degree-1))
    b=sparse.csc_matrix((np.sqrt(w[a.indices]),a.indices,a.indptr),shape=a.shape,copy=False)
    del w,degree;gc.collect()
    margin_a=linear._weighted_support(a);margin_b=linear._weighted_support(b)
    ar=a.T;br=b.T;cols=np.searchsorted(apps,candidates)
    rows={name:[] for name in active};values={name:[] for name in active}
    maximum_control_difference=0.;start=time.perf_counter()
    for begin in range(0,len(candidates),32):
        end=min(begin+32,len(candidates))
        aa=(ar[cols[begin:end]]@a).toarray();bb=(br[cols[begin:end]]@b).toarray()
        for local,own in enumerate(cols[begin:end]):
            row=begin+local;fixed=base.indices[base.indptr[row]:base.indptr[row+1]]
            for rho in (0.,.25,.5,1.):
                scores=mixture_cosine(aa[local],bb[local],margin_a,margin_b,own,rho)
                valid=np.flatnonzero(np.isfinite(scores)&(scores>0))
                selected=valid[np.lexsort((apps[valid],-scores[valid]))[:100]]
                selected=np.sort(selected)
                if rho==0:
                    if not np.array_equal(selected,fixed):
                        raise ValueError(f'Rho0 reselected neighbors differ from activity control at row {row}')
                    original=base.data[base.indptr[row]:base.indptr[row+1]]
                    difference=float(np.abs(scores[fixed].astype(np.float32)-original).max(initial=0))
                    maximum_control_difference=max(maximum_control_difference,difference)
                    if difference>1e-7:raise ValueError('Rho0 coefficients differ from activity control')
                for policy,chosen in [('fixed',fixed),('reselected',selected)]:
                    name=f'account_pair_rho{linear._name_num(rho)}_{policy}_k100'
                    if name in active:
                        rows[name].append(chosen.copy())
                        values[name].append(scores[chosen].astype(np.float32))
        if begin==0 or end==len(candidates) or begin%640==0:
            log(f'Account-pair contribution: {end:,}/{len(candidates):,}, {time.perf_counter()-start:.1f}s')
    del aa,bb,a,b,ar,br;gc.collect()
    for name in active:
        matrix=linear._csr_from_row_lists(rows[name],values[name],(len(candidates),len(apps)))
        stage=linear._new_stage(name,roster[name]['parameters'],sources)
        if stage is None:continue
        sparse.save_npz(stage/'similarities_csr.npz',matrix)
        shutil.copy2(__file__,stage/'pair_builder_snapshot.py')
        summary=dict(train_positive_users=users,train_positive_pairs=pairs,
            singleton_positive_users=singletons,catalog_items=len(apps),linux_candidates=len(candidates),
            statistics='Linear mixture of inverse-degree and inverse-degree-pair train Grams',
            rho0_control_max_absolute_difference=maximum_control_difference,
            control_manifest_sha256=digest(control/'manifest.json'),
            training_csr_receipt_sha256=digest(linear.MODEL_ROOT.parent/'training/csr_manifest.json'),
            pair_builder_sha256=digest(__file__))
        linear._finish_artifact(stage,name,'account_pair_contribution','item_similarity_csr','knn',
            roster[name]['parameters'],sources,support,apps,candidates,summary)
    return names


if __name__=='__main__':
    with threadpool_limits(limits=2):
        for name in run():print(name,flush=True)
