"""Versioned v2 evaluation/guardrails; frozen v1 code is imported, never edited."""
import json
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from eda_large_review_frank import EDA, digest
from evaluate_baselines import load_cohort, evaluate_model, inventory
from item_knn import ItemKnnCatalog
from tfidf_content import TfidfCatalog

ROOT = EDA / 'data/processed/modeling/improvement_v2'
PROTOCOL = EDA / 'data/reference/improvement_protocol_v2.json'
SPLIT = EDA / 'data/processed/modeling/split_v1'
V1 = EDA / 'data/processed/modeling/baseline_v1'
LOG_LABELS = ['0','1-9','10-99','100-999','1000-9999','10000-99999','100000-999999','1000000+']

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def log_groups(support):
    return np.searchsorted([1,10,100,1000,10000,100000,1000000], support, side='right')

def rank_values(ranks):
    ranks=np.asarray(ranks)
    return (ranks>0).astype(float), np.where(ranks>0,1/np.log2(np.maximum(ranks,1)+1),0.)

def paired_rank_ci(base, candidate, reps=2000, seed=20261108):
    """Exact paired-user bootstrap via sufficient joint-rank frequencies."""
    a=np.asarray(base,dtype=int); b=np.asarray(candidate,dtype=int)
    if a.shape != b.shape or not len(a) or np.any((a<0)|(a>10)|(b<0)|(b>10)):
        raise ValueError('Invalid paired ranks')
    pairs=a*11+b; counts=np.bincount(pairs,minlength=121)
    active=np.flatnonzero(counts); aa=active//11; bb=active%11
    ar,an=rank_values(aa); br,bn=rank_values(bb)
    boot=np.random.default_rng(seed).multinomial(len(a),counts[active]/len(a),size=reps)
    result={}
    for name,delta in [('Recall@10',br-ar),('NDCG@10',bn-an)]:
        samples=boot@delta/len(a)
        result[name]={'delta':float(counts[active]@delta/len(a)), 'ci95':np.quantile(samples,[.025,.975]).tolist()}
    return result

def macro(values, targets):
    _,idx=np.unique(targets,return_inverse=True)
    return float(np.mean(np.bincount(idx,weights=values)/np.bincount(idx)))

def protective_summary(per_user, recommendations=None, baseline=None):
    d=pq.read_table(per_user).to_pydict() if isinstance(per_user,(str,Path)) else per_user
    ranks=np.asarray(d['target_rank']); recall,ndcg=rank_values(ranks)
    targets=np.asarray(d['target_app_id']); support=np.asarray(d['target_train_support'])
    lengths=np.asarray(d['recommendation_count']); groups=log_groups(support)
    def group(mask):
        n=int(mask.sum())
        return {'users':n,'target_games':int(len(np.unique(targets[mask]))),
                'Recall@10':float(recall[mask].mean()) if n else None,
                'NDCG@10':float(ndcg[mask].mean()) if n else None}
    result={'users':len(ranks),'target_games':int(len(np.unique(targets))),
       'Recall@10':float(recall.mean()),'NDCG@10':float(ndcg.mean()),
       'game_macro_Recall@10':macro(recall,targets),'game_macro_NDCG@10':macro(ndcg,targets),
       'empty_users':int((lengths==0).sum()),'short_list_users':int((lengths<10).sum()),
       'by_support_log':{label:group(groups==i) for i,label in enumerate(LOG_LABELS)},
       'tail':group((support>0)&(support<1000))}
    if recommendations:
        rec=pq.read_table(recommendations,columns=['app_id']).column(0).to_numpy()
        rows=pq.read_table(EDA/'data/processed/modeling/item_knn_v1/game_rows.parquet').to_pydict()
        apps=np.asarray(rows['app_id']); supports=np.asarray(rows['positive_user_support'])
        recsupport=supports[np.searchsorted(apps,rec)]
        result['recommendation_support_median']=float(np.median(recsupport)) if len(rec) else None
        result['recommendation_slots']=len(rec)
        result['recommended_games']=int(len(np.unique(rec)))
    if baseline:
        old=pq.read_table(baseline).to_pydict()
        for col in ['user_key','target_app_id','n_positive_inputs','n_total_inputs','target_train_support']:
            if d[col]!=old[col]: raise ValueError('Changed paired population: '+col)
        previous=protective_summary(old)
        result['paired_bootstrap']=paired_rank_ci(old['target_rank'],ranks)
        result['guard_checks']={
          'ndcg_ci_lower_positive':result['paired_bootstrap']['NDCG@10']['ci95'][0]>0,
          'whole_recall_noninferior':result['Recall@10']+1e-12>=previous['Recall@10'],
          'game_macro_ndcg_noninferior':result['game_macro_NDCG@10']+1e-12>=previous['game_macro_NDCG@10'],
          'tail_ndcg_noninferior':result['tail']['NDCG@10'] is None or result['tail']['NDCG@10']+1e-12>=previous['tail']['NDCG@10'],
          'empty_users_not_increased':result['empty_users']<=previous['empty_users']}
        result['eligible_for_automatic_adoption']=all(result['guard_checks'].values())
    return result

class CombinedCatalog:
    def __init__(self, text, metadata, alpha):
        if not np.array_equal(text.app_ids,metadata.app_ids) or not np.array_equal(text.linux,metadata.linux):
            raise ValueError('Channel catalog mismatch')
        self.text,self.metadata,self.alpha=text,metadata,float(alpha)
        self.app_ids,self.linux=text.app_ids,text.linux
        # Support diagnostics use the union of channels; scoring normalizes each
        # channel's profile independently in batch(), never this concatenation.
        self.matrix=(text.matrix if self.alpha==1 else metadata.matrix if self.alpha==0
                     else sparse.hstack([text.matrix,metadata.matrix],format='csr'))
    def recommend(self, positive_app_ids, exclude_app_ids=(), limit=10):
        from evaluate_baselines import rank_positive
        cols=np.array([self.text.row_by_app[int(a)] for a in positive_app_ids])
        q=sparse.csr_matrix((np.ones(len(cols),dtype=np.float32),(np.zeros(len(cols),dtype=int),cols)),shape=(1,len(self.app_ids)))
        scores=self.batch(q,np.array([len(cols)]))[0]
        candidates=self.app_ids[self.linux]
        scores[np.isin(candidates,list(set(exclude_app_ids)|set(positive_app_ids)))]=0
        chosen=rank_positive(scores,candidates,limit)
        return [{'app_id':int(candidates[i]),'score':float(scores[i])} for i in chosen]
    def batch(self,query,n_positive):
        from evaluate_baselines import score_batch
        return self.alpha*score_batch(self.text,query,n_positive,'tfidf')+(1-self.alpha)*score_batch(self.metadata,query,n_positive,'tfidf')

def load_knn(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
    if manifest.get('status')!='complete': raise ValueError('Incomplete model')
    files=manifest.get('files',{})
    for name,info in files.items():
        if digest(folder/name)!=info['sha256']:raise ValueError('Model file hash mismatch: '+name)
    # V2 builders may share mappings from the frozen v1 index.
    mapping=folder if (folder/'game_rows.parquet').exists() else EDA/'data/processed/modeling/item_knn_v1'
    rows=pq.read_table(mapping/'game_rows.parquet').to_pydict()
    candidates=pq.read_table(mapping/'candidate_rows.parquet').to_pydict()
    return ItemKnnCatalog(rows['app_id'],candidates['app_id'],sparse.load_npz(folder/'similarities_csr.npz'),rows['positive_user_support'])

def evaluate_variant(name,model,kind,cohort_name='validation',batch_size=256,extra=None):
    from evaluate_baselines import score_batch as original_score_batch
    import evaluate_baselines as ev
    protocol=json.loads(PROTOCOL.read_text('utf-8'))
    lock=json.loads((EDA/'data/reference/baseline_run_v1.json').read_text('utf-8'))
    frozen={'protocol_sha256':EDA/'data/reference/baseline_protocol_v1.json',
       'split_manifest_sha256':SPLIT/'manifest.json',
       'tfidf_manifest_sha256':EDA/'data/processed/modeling/tfidf_content_v1/manifest.json',
       'item_knn_manifest_sha256':EDA/'data/processed/modeling/item_knn_v1/manifest.json',
       'evaluator_sha256':Path(ev.__file__),
       'tfidf_code_sha256':EDA/'scripts/tfidf_content.py',
       'item_knn_code_sha256':EDA/'scripts/item_knn.py'}
    if any(digest(p)!=lock[k] for k,p in frozen.items()):raise ValueError('Frozen v1 changed')
    if digest(EDA/'data/reference/baseline_run_v1.json')!=protocol['baseline_run_lock_sha256']:
        raise ValueError('Frozen v1 run lock changed')
    if cohort_name=='test':
        selected=json.loads((ROOT/'selection_frozen.json').read_text('utf-8'))
        if selected['protocol_sha256']!=digest(PROTOCOL) or selected['models'][kind]['variant']!=name:
            raise ValueError('Only frozen validation-selected models can use reference test')
        if selected['models'][kind]['model']!=(extra or {}):raise ValueError('Selected model hashes changed')
    maps=pq.read_table(EDA/'data/processed/modeling/item_knn_v1/game_rows.parquet').to_pydict()
    apps=np.asarray(maps['app_id']); linux=np.asarray(maps['linux']); candidates=apps[linux]
    if not np.array_equal(model.app_ids,apps):raise ValueError('Input catalog mismatch')
    support=np.asarray(maps['positive_user_support'])[linux]
    cohort=load_cohort(SPLIT,cohort_name,apps,candidates,protocol)
    folder=ROOT/'evaluations'/cohort_name/name
    if folder.exists():
        m=json.loads((folder/'manifest.json').read_text('utf-8'))
        if (m['protocol_sha256']!=digest(PROTOCOL) or m['evaluator_v2_sha256']!=digest(__file__)
            or m['model']!=(extra or {})):raise ValueError('Existing evaluation protocol/code/model changed')
        for p,info in m['files'].items():
            if digest(folder/p)!=info['sha256']:raise ValueError('Existing evaluation changed')
        return m
    folder.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    # Reuse the audited v1 masks/top10/persistence invariants, with a local scoring dispatch.
    # No source or locked v1 code is changed.
    if isinstance(model,CombinedCatalog):
        ev.score_batch=lambda m,q,n,k: m.batch(q,n) if isinstance(m,CombinedCatalog) else original_score_batch(m,q,n,k)
    try: summary=evaluate_model(model,kind,cohort,candidates,support,np.array([19.,67.,413.75]),folder,batch_size)
    finally: ev.score_batch=original_score_batch
    protection=protective_summary(folder/'per_user.parquet',folder/'recommendations.parquet',V1/cohort_name/kind/'per_user.parquet')
    m={'status':'complete','variant':name,'kind':kind,'cohort':cohort_name,'protocol_sha256':digest(PROTOCOL),
       'evaluator_v2_sha256':digest(__file__),'evaluator_v1_sha256':digest(Path(ev.__file__)),
       'elapsed_seconds':time.perf_counter()-started,'summary':summary,'protection':protection,
       'model':extra or {},'files':inventory(folder)}
    write_json(folder/'manifest.json',m)
    return m
