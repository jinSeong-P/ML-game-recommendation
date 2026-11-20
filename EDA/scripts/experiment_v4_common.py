"""Isolated v4 evaluation: fresh refits, full catalog, protected confirmation.

Historical v1-v3 artifacts are read-only. The v4 account split is newly held
out, not a claim that these public reviews never influenced past development.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize

from eda_large_review_frank import EDA, digest, log
from evaluate_baselines import rank_positive
from improvement_common import write_json, macro, paired_rank_ci, LOG_LABELS, log_groups

ROOT = EDA / 'data/processed/modeling/experiments_v4'
SPLIT = ROOT / 'split'
PROTOCOL = EDA / 'data/reference/improvement_protocol_v4.json'


def rank_values(ranks):
    # NumPy may return float16 for log2(int8). Saved ranks are int8, so make
    # the metric precision explicit before taking logs or averaging.
    ranks=np.asarray(ranks,dtype=np.float64)
    return (ranks>0).astype(np.float64),np.where(ranks>0,1/np.log2(np.maximum(ranks,1)+1),0.)


@lru_cache(maxsize=1)
def catalog():
    d = pq.read_table(SPLIT / 'catalog/input_apps.parquet').to_pydict()
    apps = np.asarray(d['app_id'], dtype=np.int64)
    linux = np.asarray(d['linux'], dtype=bool)
    if len(apps) != 37610 or linux.sum() != 6630 or np.any(np.diff(apps) <= 0):
        raise ValueError('Fixed catalog changed')
    return apps, linux


def support():
    return np.load(ROOT / 'models/positive_support.npy')


def load_cohort(role, root=SPLIT):
    if role not in ('fusion_fit', 'selection', 'confirmation'):
        raise ValueError('Unknown account role')
    apps, linux = catalog(); folder = root / role
    u = pq.read_table(folder / 'users.parquet').to_pydict()
    ids = np.asarray(u['user_key'], dtype=np.uint64)
    if not len(ids) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError('Users must be ascending uint64')
    t = pq.read_table(folder / 'targets.parquet').to_pydict()
    if not np.array_equal(ids, np.asarray(t['user_key'], dtype=np.uint64)):
        raise ValueError('Targets do not match account roster')
    target_ids = np.asarray(t['app_id'], dtype=np.int64)
    candidates = apps[linux]; tc = np.searchsorted(candidates, target_ids)
    if np.any(tc >= len(candidates)) or np.any(candidates[tc] != target_ids):
        raise ValueError('Target outside catalog')
    q = pq.read_table(folder / 'query.parquet').to_pydict()
    qr = np.searchsorted(ids, np.asarray(q['user_key'], dtype=np.uint64))
    qc = np.searchsorted(apps, np.asarray(q['app_id'], dtype=np.int64))
    if np.any(qr >= len(ids)) or np.any(ids[qr] != q['user_key']):
        raise ValueError('Unknown query account')
    if np.any(qc >= len(apps)) or np.any(apps[qc] != q['app_id']):
        raise ValueError('Unknown query game')
    positive = np.asarray(q['positive'], dtype=bool)
    def matrix(mask):
        m = sparse.csr_matrix((np.ones(int(mask.sum()), dtype=np.float32),
            (qr[mask], qc[mask])), shape=(len(ids),len(apps)))
        if m.nnz != int(mask.sum()) or np.any(m.data != 1):
            raise ValueError('Duplicate query pairs')
        return m
    p, n = matrix(positive), matrix(~positive)
    x = (p+n)[:,linux].tocsr()
    if np.any(x[np.arange(len(ids)),tc]):
        raise ValueError('Target is visible')
    total = np.asarray((p+n).sum(axis=1)).ravel().astype(int)
    npos = np.asarray(p.sum(axis=1)).ravel().astype(int)
    if not np.array_equal(total,u['n_total_inputs']) or not np.array_equal(npos,u['n_positive_inputs']):
        raise ValueError('Query counts differ from split')
    return dict(name=role, user_ids=ids, positive=p, negative=n, exclusions=x,
        target_ids=target_ids, target_candidates=tc, n_total=total, n_positive=npos,
        hashes={f:digest(folder/f) for f in ['users.parquet','targets.parquet','query.parquet']})


class MatrixScorer:
    def __init__(self, matrix, mode='knn', scale=None, spec=None):
        self.matrix, self.mode, self.scale = matrix, mode, scale
        self.spec=spec or {'mode':mode,'shape':list(matrix.shape)}
    def score(self,q):
        if self.mode == 'knn':
            a = q @ self.matrix.T
            a = a.toarray() if sparse.issparse(a) else np.asarray(a)
            a = a / np.maximum(np.asarray(q.sum(axis=1)).ravel(),1)[:,None]
        else:
            _,linux = catalog()
            a = normalize(q @ self.matrix) @ self.matrix[linux].T
            a = a.toarray() if sparse.issparse(a) else np.asarray(a)
        return a if self.scale is None else a*self.scale


class ContentScorer:
    def __init__(self):
        source=EDA/'data/processed/modeling/improvement_v3/content/metadata_extended_tfidf'
        manifest=json.loads((source/'manifest.json').read_text('utf-8'))
        for f in ['tfidf_csr.npz','features_csr.npz']:
            if digest(source/f)!=manifest['files'][f]['sha256']:raise ValueError('Content representation changed')
        self.spec=dict(kind='fixed_public_content',manifest_sha256=digest(source/'manifest.json'))
        self.channels=[MatrixScorer(sparse.load_npz(source/f),'content')
            for f in ['tfidf_csr.npz','features_csr.npz']]
    def score(self,q):
        return sum(.5*c.score(q) for c in self.channels)


def primary(cohort):
    mask=cohort['n_total']>=5
    return {k:(v[mask] if k in ('user_ids','positive','negative','exclusions','target_ids',
        'target_candidates','n_total','n_positive') else v) for k,v in cohort.items()}


def masked_scores(model,cohort,begin,end):
    a = np.asarray(model.score(cohort['positive'][begin:end]),dtype=np.float64)
    if a.shape != (end-begin,6630) or not np.all(np.isfinite(a)):
        raise ValueError('Nonfinite scores or wrong shape')
    a[a<=0]=0
    r,c=cohort['exclusions'][begin:end].nonzero(); a[r,c]=0
    return a


def group_metrics(ranks,targets,mask):
    rr,nn=rank_values(ranks[mask]); n=int(mask.sum())
    return dict(users=n,target_games=int(len(np.unique(targets[mask]))),
        recall=float(rr.mean()) if n else None, ndcg=float(nn.mean()) if n else None)


def metrics(ranks,lengths,cohort,sup):
    rr,nn=rank_values(ranks); targets=cohort['target_ids']; ts=sup[np.searchsorted(catalog()[0],targets)]
    main=cohort['n_total']>=5
    out=group_metrics(ranks,targets,main)
    out.update(game_macro_recall=macro(rr[main],targets[main]),
        game_macro_ndcg=macro(nn[main],targets[main]),empty_users=int((lengths[main]==0).sum()),
        average_length=float(lengths[main].mean()),all_eligible_users=len(ranks),
        tail=group_metrics(ranks,targets,main&(ts>0)&(ts<1000)),
        by_total_history={label:group_metrics(ranks,targets,mask) for label,mask in
            [('1-2',cohort['n_total']<=2),('3-4',(cohort['n_total']>=3)&(cohort['n_total']<=4)),('5+',main)]},
        by_positive_history={label:group_metrics(ranks,targets,mask) for label,mask in
            [('1',cohort['n_positive']==1),('2',cohort['n_positive']==2),
             ('3-4',(cohort['n_positive']>=3)&(cohort['n_positive']<=4)),
             ('5-9',(cohort['n_positive']>=5)&(cohort['n_positive']<=9)),('10+',cohort['n_positive']>=10)]},
        by_support={label:group_metrics(ranks,targets,main&(log_groups(ts)==i))
            for i,label in enumerate(LOG_LABELS)})
    return out


def paired_comparison(base,candidate,cohort,reps=2000):
    main=cohort['n_total']>=5
    a,b=np.asarray(base)[main],np.asarray(candidate)[main]
    out=paired_rank_ci(a,b,reps=reps)
    ar,an=rank_values(a);br,bn=rank_values(b)
    targets=cohort['target_ids'][main]
    _,inverse=np.unique(targets,return_inverse=True)
    # Fixed target-game population, resample users within each game. Multinomial
    # joint ranks are sufficient and preserve pairing without a reps x users array.
    rng=np.random.default_rng(20261108); boot=np.zeros((reps,2)); ng=int(inverse.max()+1)
    for g in range(ng):
        m=inverse==g; pair=a[m]*11+b[m]; counts=np.bincount(pair,minlength=121)
        active=np.flatnonzero(counts); ah,ad=rank_values(active//11);bh,bd=rank_values(active%11)
        samples=rng.multinomial(int(m.sum()),counts[active]/m.sum(),size=reps)
        boot[:,0]+=samples@(bh-ah)/m.sum()/ng
        boot[:,1]+=samples@(bd-ad)/m.sum()/ng
    out['game_macro']={k:dict(delta=macro(v,targets),ci95=np.quantile(boot[:,j],[.025,.975]).tolist())
        for j,(k,v) in enumerate([('Recall@10',br-ar),('NDCG@10',bn-an)])}
    out['hit_overlap']=dict(both=int(((a>0)&(b>0)).sum()),base_only=int(((a>0)&(b==0)).sum()),
        candidate_only=int(((a==0)&(b>0)).sum()),neither=int(((a==0)&(b==0)).sum()))
    return out


def evaluate(name,model,cohort,batch_size=128,confirmation_lock=None):
    protocol=json.loads(PROTOCOL.read_text('utf-8'))
    if protocol['status'] != 'fixed_before_v4_metrics':
        raise ValueError('Freeze protocol before evaluation')
    if cohort['name']=='confirmation':
        if not confirmation_lock or not Path(confirmation_lock).is_file():
            raise ValueError('Confirmation requires a selection lock')
        lock=json.loads(Path(confirmation_lock).read_text('utf-8'))
        if name not in lock['confirmation_models'] or lock['protocol_sha256']!=digest(PROTOCOL):
            raise ValueError('Only frozen finalists may open confirmation')
    folder=ROOT/'evaluations'/cohort['name']/name; folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'manifest.json'
    provenance=dict(protocol_sha256=digest(PROTOCOL),evaluator_sha256=digest(__file__),cohort=cohort['hashes'],
        split_manifest_sha256=digest(SPLIT/'manifest.json'),model=getattr(model,'spec',{}))
    if receipt.exists():
        m=json.loads(receipt.read_text('utf-8'))
        if m['provenance']!=provenance: raise ValueError('Existing evaluation changed')
        for f,h in m['files'].items():
            if digest(folder/f)!=h: raise ValueError('Existing predictions changed')
        return m
    apps,linux=catalog(); ids=apps[linux]; sup=support(); n=len(cohort['user_ids'])
    ranks=np.zeros(n,dtype=np.int8); lengths=np.zeros(n,dtype=np.int8)
    schema=pa.schema([('user_key',pa.uint64()),('app_id',pa.int64()),('rank',pa.int8()),('score',pa.float64())])
    start=time.perf_counter()
    with pq.ParquetWriter(folder/'recommendations.parquet',schema,compression='zstd') as writer:
        for begin in range(0,n,batch_size):
            end=min(n,begin+batch_size); a=masked_scores(model,cohort,begin,end); records=[]
            for local,row in enumerate(range(begin,end)):
                chosen=rank_positive(a[local],ids,10); lengths[row]=len(chosen)
                hit=np.flatnonzero(chosen==cohort['target_candidates'][row])
                if len(hit): ranks[row]=hit[0]+1
                records.extend((int(cohort['user_ids'][row]),int(ids[col]),rank+1,float(a[local,col]))
                    for rank,col in enumerate(chosen))
            writer.write_table(pa.Table.from_pylist([dict(zip(schema.names,r)) for r in records],schema=schema))
            if begin==0 or end==n or begin%(128*100)==0: log(f'v4 {cohort["name"]}/{name}: {end:,}/{n:,}')
    per=pa.table(dict(user_key=pa.array(cohort['user_ids'],type=pa.uint64()),
        target_app_id=cohort['target_ids'],target_rank=ranks,recommendation_count=lengths,
        n_total_inputs=cohort['n_total'],n_positive_inputs=cohort['n_positive'],
        target_train_support=sup[np.searchsorted(apps,cohort['target_ids'])]))
    pq.write_table(per,folder/'per_user.parquet',compression='zstd')
    m=dict(status='complete',variant=name,role=cohort['name'],provenance=provenance,
        elapsed_seconds=time.perf_counter()-start,metrics=metrics(ranks,lengths,cohort,sup),
        files={f:digest(folder/f) for f in ['per_user.parquet','recommendations.parquet']})
    write_json(receipt,m); return m
