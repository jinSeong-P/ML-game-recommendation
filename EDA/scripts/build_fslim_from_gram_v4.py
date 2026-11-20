"""Fit the predeclared fSLIM grid from the audited binary training Gram.

This is an exact sufficient-statistic reuse, not reuse of an old fitted model.
The cyclic coordinate solver is sklearn's compiled nonnegative Elastic Net
solver. Selection/confirmation rows and labels are never opened here.
"""
from __future__ import annotations
import gc
import json
import shutil
import time
import warnings

import numpy as np
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import _cd_fast
from threadpoolctl import threadpool_limits

from eda_large_review_frank import digest, log
import experiment_linear_v4 as linear


def fit_column(gram, cross, target_support, l1, l2, max_iter=1000, tol=1e-5):
    q = np.ascontiguousarray(gram, dtype=np.float64)
    c = np.asarray(cross, dtype=np.float64)
    if q.shape != (len(c), len(c)) or l1 < 0 or l2 <= 0 or target_support < 0:
        raise ValueError('Invalid fSLIM sufficient statistics')
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(c)) or not np.array_equal(q, q.T):
        raise ValueError('fSLIM requires finite symmetric training statistics')
    # y is needed only for its squared norm in the dual-gap stopping rule.
    # For binary targets this norm equals the target's training support.
    y = np.asarray([np.sqrt(float(target_support))], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        w, gap, threshold, iterations = _cd_fast.enet_coordinate_descent_gram(
            np.zeros(len(c)), float(l1), float(l2), q, c, y,
            max_iter, tol, np.random.RandomState(20261108), random=0,
            positive=1, do_screening=0)
    gradient = q @ w - c + l1 + l2*w
    kkt = np.where(w > 0, np.abs(gradient), np.maximum(-gradient, 0))
    error = float(kkt.max(initial=0)/max(1., np.abs(c).max(initial=0)))
    return w.astype(np.float32), dict(iterations=int(iterations), dual_gap=float(gap),
        dual_threshold=float(threshold), converged=bool(gap <= threshold), normalized_kkt=error)


def neighbor_order(row, own, support, apps, limit):
    valid = np.flatnonzero((row > 0) & (support > 0))
    valid = valid[valid != own]
    if not len(valid) or support[own] == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    scores = row[valid].astype(np.float64)/np.sqrt(float(support[own])*support[valid])
    selected = np.lexsort((apps[valid], -scores))[:limit]
    return valid[selected].astype(np.int32), scores[selected].astype(np.float32)


def run():
    linear.freeze_roster(); sources=linear._source_manifest()
    apps,candidates=linear._catalog(); n=len(apps)
    reference=linear.MODEL_ROOT/'ease_lambda100_exact'
    reference_manifest=json.loads((reference/'manifest.json').read_text('utf-8'))
    if reference_manifest['sources'] != sources:
        raise ValueError('fSLIM reference belongs to a different train split')
    supfile=reference/'support.npy'
    if digest(supfile) != reference_manifest['files']['support.npy']['sha256']:
        raise ValueError('fSLIM support receipt changed')
    support=np.load(supfile)
    if support.max() >= 2**24:
        raise ValueError('Integer co-occurrence exactness in float32 is not guaranteed')
    gram_path=linear._gram_cache_path(0.)
    receipt=json.loads(linear._gram_cache_receipt(gram_path).read_text('utf-8'))
    if (receipt.get('complete') is not True or receipt['shape'] != [n,n]
        or receipt['train_sha256'] != sources['train/positive_catalog.parquet']['sha256']
        or receipt['beta'] != 0 or digest(gram_path) != receipt['sha256']):
        raise ValueError('Verified current binary train Gram required')
    roster=linear._variant_roster()
    names=[f'fslim_m{m}_l2{l2}_l1r{linear._name_num(r)}'
        for m in (200,500) for l2 in (100,1000) for r in (0.,.001,.01)]
    names += ['fslim_neighbor_cosine_k200','fslim_neighbor_cosine_k500']
    active=[name for name in names if not linear._verify_existing(
        linear.MODEL_ROOT/name,name,roster[name]['parameters'],sources)]
    if not active:return names
    gram=np.memmap(gram_path,mode='r',dtype=np.float32,shape=(n,n))
    if not np.array_equal(np.diag(gram),support):
        raise ValueError('Raw Gram diagonal differs from exact binary support')
    rows={name:[] for name in active}; values={name:[] for name in active}
    stats={name:dict(zero_rows=0,nonconverged=0,iterations=0,max_normalized_kkt=0.)
        for name in active}; start=time.perf_counter()
    for i,own in enumerate(np.searchsorted(apps,candidates)):
        cooc=np.array(gram[own],copy=True)
        if np.any(~np.isfinite(cooc)) or np.any(cooc<0) or np.any(cooc!=np.floor(cooc)):
            raise ValueError('Binary train co-occurrence is not a finite nonnegative count')
        ix,cos=neighbor_order(cooc,int(own),support,apps,500)
        q=np.array(gram[np.ix_(ix,ix)],dtype=np.float64)
        cross=cooc[ix].astype(np.float64)
        for m in (200,500):
            width=min(m,len(ix)); chosen=ix[:width]
            neighbor=f'fslim_neighbor_cosine_k{m}'
            if neighbor in active:
                order=np.argsort(chosen);rows[neighbor].append(chosen[order]);values[neighbor].append(cos[:width][order])
            for l2 in (100,1000):
                for relative in (0.,.001,.01):
                    name=f'fslim_m{m}_l2{l2}_l1r{linear._name_num(relative)}'
                    if name not in active:continue
                    if width:
                        coefficients,info=fit_column(q[:width,:width],cross[:width],support[own],
                            relative*cross[:width].max(),l2)
                        stats[name]['iterations']+=info['iterations']
                        stats[name]['nonconverged']+=int(not info['converged'])
                        stats[name]['max_normalized_kkt']=max(stats[name]['max_normalized_kkt'],info['normalized_kkt'])
                    else:coefficients=np.empty(0,np.float32)
                    keep=coefficients>0;order=np.argsort(chosen[keep])
                    rows[name].append(chosen[keep][order]);values[name].append(coefficients[keep][order])
                    stats[name]['zero_rows']+=int(not keep.any())
        if i==0 or (i+1)%500==0 or i+1==len(candidates):
            log(f'fSLIM reused Gram: {i+1:,}/{len(candidates):,}, {time.perf_counter()-start:.1f}s')
    linear._close_memmap(gram); del gram;gc.collect()
    for name in active:
        s=stats[name]
        if s['max_normalized_kkt'] > .001:
            raise ValueError(f'fSLIM did not meet normalized KKT verification: {name}: {s}')
        matrix=linear._csr_from_row_lists(rows[name],values[name],(len(candidates),n))
        stage=linear._new_stage(name,roster[name]['parameters'],sources)
        if stage is None:continue
        sparse.save_npz(stage/'similarities_csr.npz',matrix)
        shutil.copy2(__file__,stage/'gram_reuse_builder_snapshot.py')
        reference_summary=reference_manifest['fit_summary']
        summary=dict({key:reference_summary[key] for key in
            ['train_positive_users','train_positive_pairs','singleton_positive_users','catalog_items','linux_candidates']},
            gram_sha256=receipt['sha256'],gram_reused_as_sufficient_statistics=True,
            solver='sklearn compiled cyclic nonnegative Elastic Net, no screening',
            sklearn_version=sklearn.__version__,convergence='dual gap <= 1e-5 * target training support',
            candidate_zero_coefficient_rows=s['zero_rows'],candidate_zero_coefficient_rate=s['zero_rows']/len(candidates),
            nonconverged_candidate_fits=s['nonconverged'],mean_coordinate_descent_iterations=s['iterations']/len(candidates),
            max_normalized_kkt=s['max_normalized_kkt'],reuse_builder_sha256=digest(__file__))
        linear._finish_artifact(stage,name,'fslim' if name.startswith('fslim_m') else 'fslim_neighbor_baseline',
            'item_similarity_csr','linear' if name.startswith('fslim_m') else 'knn',
            roster[name]['parameters'],sources,support,apps,candidates,summary)
        del matrix
    return names


if __name__=='__main__':
    with threadpool_limits(limits=2):
        for name in run():print(name,flush=True)
