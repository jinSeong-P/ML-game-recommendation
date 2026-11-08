"""Score separately versioned evaluation conditions, never select from them."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import EDA,digest,quote,log
from evaluate_baselines import checked_positions,pair_matrix,rank_positive,inventory
from experiment_v3_common import ROOT,PROTOCOL,catalog,current_models,evaluate,write_json

COHORT_ROOT=EDA/'data/processed/modeling/backlog_v3/cohorts_v3_final'


def load_supplement(name):
    folder=COHORT_ROOT/name
    m=json.loads((folder/'manifest.json').read_text('utf-8'))
    if m['status']!='complete':raise ValueError('Incomplete supplemental cohort')
    for relative,info in m['files'].items():
        if digest(folder/relative)!=info['sha256']:raise ValueError('Supplemental data changed: '+relative)
    if not m['users']:return None,m
    apps,linux,_=catalog();candidates=apps[linux]
    u=pq.read_table(folder/'users.parquet').to_pydict();users=np.asarray(u['user_key'],dtype=np.uint64)
    if np.any(users[1:]<=users[:-1]):raise ValueError('Invalid supplemental user order')
    q=pq.read_table(folder/'query.parquet').to_pydict()
    rows=checked_positions(np.asarray(q['user_key'],dtype=np.uint64),users,'query user')
    cols=checked_positions(q['app_id'],apps,'query app');positive=np.asarray(q['positive'])
    shape=(len(users),len(apps))
    qp=pair_matrix(rows[positive],cols[positive],shape,'positive query')
    qn=pair_matrix(rows[~positive],cols[~positive],shape,'negative query')
    if not np.array_equal(np.diff(qp.indptr),u['n_positive_inputs']):raise ValueError('Input positive count mismatch')
    if not np.array_equal(np.diff((qp+qn).indptr),u['n_total_inputs']):raise ValueError('Input total count mismatch')
    x=pq.read_table(folder/'exclusions.parquet').to_pydict()
    if (folder/'additional_exclusions.parquet').exists():
        extra=pq.read_table(folder/'additional_exclusions.parquet').to_pydict()
        for key in x:x[key]+=extra[key]
    er=checked_positions(np.asarray(x['user_key'],dtype=np.uint64),users,'mask user')
    ec=checked_positions(x['app_id'],apps,'mask app')
    mask=sparse.csr_matrix((np.ones(len(er),dtype=np.float32),(er,ec)),shape=shape)
    mask.data[:]=1
    if ((qp+qn).multiply(mask)!=(qp+qn)).nnz:raise ValueError('Mask misses visible input')
    t=pq.read_table(folder/'targets.parquet').to_pydict()
    tr=checked_positions(np.asarray(t['user_key'],dtype=np.uint64),users,'target user')
    tc=checked_positions(t['app_id'],candidates,'target candidate')
    target=pair_matrix(tr,tc,(len(users),len(candidates)),'target')
    counts=np.diff(target.indptr)
    if np.any(counts<1) or np.any(mask[tr,np.searchsorted(apps,t['app_id'])]):raise ValueError('Target in input/mask')
    first=target.indices[target.indptr[:-1]]
    c={'name':name,'user_ids':users,'positive':qp,'negative':qn,'n_positive':np.diff(qp.indptr),
       'n_total':np.asarray(u['n_total_inputs']),'exclusions':mask[:,linux].tocsr(),
       'target_ids':candidates[first],'target_candidates':first,'all_targets':target,
       'hashes':{p:info['sha256'] for p,info in m['files'].items()}}
    return c,m


def evaluate_multiple(name,model,c):
    out=ROOT/'evaluations'/c['name']/name
    provenance={'model':model.spec,'cohort_hashes':c['hashes'],'protocol_sha256':digest(PROTOCOL),
                'multiple_target_evaluator_sha256':digest(__file__)}
    if out.exists():
        m=json.loads((out/'manifest.json').read_text('utf-8'))
        if m['provenance']!=provenance:raise ValueError('Multi-target evaluation provenance changed')
        return m
    out.mkdir(parents=True);started=time.perf_counter()
    apps,linux,_=catalog();ids=apps[linux];n=len(c['user_ids']);targets=c['all_targets']
    hits=np.zeros(n,dtype=np.int16);ndcg=np.zeros(n);lengths=np.zeros(n,dtype=np.int8)
    schema=pa.schema([('user_key',pa.uint64()),('app_id',pa.int64()),('rank',pa.int8()),('score',pa.float64())])
    with pq.ParquetWriter(out/'recommendations.parquet',schema,compression='zstd') as writer:
        for start in range(0,n,256):
            end=min(start+256,n);scores=model.score(c['positive'][start:end],c['negative'][start:end])
            mr,mc=c['exclusions'][start:end].nonzero();scores[mr,mc]=0
            rows=[]
            for local,row in enumerate(range(start,end)):
                chosen=rank_positive(scores[local],ids)
                targetcols=targets.indices[targets.indptr[row]:targets.indptr[row+1]]
                hit=np.isin(chosen,targetcols)
                hits[row]=hit.sum();lengths[row]=len(chosen)
                gains=1/np.log2(np.arange(1,len(chosen)+1)+1)
                ideal=np.sum(1/np.log2(np.arange(1,min(10,len(targetcols))+1)+1))
                ndcg[row]=float(gains[hit].sum()/ideal)
                rows.extend((int(c['user_ids'][row]),int(ids[col]),rank,float(scores[local,col]))
                    for rank,col in enumerate(chosen,1))
            writer.write_table(pa.Table.from_arrays([pa.array([r[i] for r in rows],type=f.type)
                for i,f in enumerate(schema)],schema=schema))
            if start==0 or end==n or start%(256*250)==0:log(f'{name} {c["name"]}: {end:,}/{n:,}')
    counts=np.diff(targets.indptr);recall=hits/counts
    pq.write_table(pa.table({'user_key':c['user_ids'],'hidden_targets':counts,'retrieved_targets':hits,
        'recall_at_10':recall,'ndcg_at_10':ndcg,'recommendation_count':lengths}),out/'per_user.parquet',compression='zstd')
    con=duckdb.connect()
    try:
        con.execute('SET threads=2')
        targetcoo=targets.tocoo();mask=c['exclusions'].tocoo()
        con.register('targets',pa.table({'user_key':c['user_ids'][targetcoo.row],'app_id':ids[targetcoo.col]}))
        con.register('excluded',pa.table({'user_key':c['user_ids'][mask.row],'app_id':ids[mask.col]}))
        con.execute(f'CREATE VIEW recs AS SELECT * FROM read_parquet({quote(out/"recommendations.parquet")})')
        con.execute(f'CREATE VIEW outcomes AS SELECT * FROM read_parquet({quote(out/"per_user.parquet")})')
        checks={'observed_games':'SELECT count(*) FROM recs JOIN excluded USING(user_key,app_id)',
            'duplicate_recommendations':'SELECT count(*) FROM (SELECT user_key,app_id FROM recs GROUP BY ALL HAVING count(*)<>1)',
            'invalid_scores':'SELECT count(*) FROM recs WHERE score<=0 OR NOT isfinite(score)',
            'invalid_ranks':'SELECT count(*) FROM (SELECT user_key FROM recs GROUP BY user_key HAVING min(rank)<>1 OR max(rank)<>count(*) OR count(DISTINCT rank)<>count(*) OR count(*)>10)',
            'wrong_hit_counts':'''SELECT count(*) FROM outcomes o LEFT JOIN
                (SELECT r.user_key,count(*) n FROM recs r JOIN targets t USING(user_key,app_id) GROUP BY r.user_key) h USING(user_key)
                 WHERE o.retrieved_targets<>coalesce(h.n,0)''',
            'wrong_recall':'SELECT count(*) FROM outcomes WHERE abs(recall_at_10-retrieved_targets::DOUBLE/hidden_targets)>1e-12',
            'wrong_ndcg':'''SELECT count(*) FROM outcomes o LEFT JOIN
                (SELECT r.user_key,sum(1/log2(r.rank+1)) dcg FROM recs r JOIN targets t USING(user_key,app_id) GROUP BY r.user_key) h USING(user_key)
                WHERE abs(o.ndcg_at_10-coalesce(h.dcg,0)/(1+1/log2(3)))>1e-12'''}
        violations={k:int(con.execute(sql).fetchone()[0]) for k,sql in checks.items()}
        if any(violations.values()):raise ValueError('Multi-target saved result violation '+str(violations))
    finally:con.close()
    m={'status':'complete','variant':name,'cohort':c['name'],'provenance':provenance,'elapsed_seconds':time.perf_counter()-started,
       'metrics':{'users':n,'hidden_targets_per_user':int(counts[0]),'Recall@10':float(recall.mean()),
            'NDCG@10':float(ndcg.mean()),'HitRate@10':float(np.mean(hits>0)),
            'empty_users':int(np.sum(lengths==0))},'violations':violations,'files':inventory(out)}
    write_json(out/'manifest.json',m);return m


def run(names=None,models=None):
    models=current_models() if models is None else models
    summaries=[]
    for folder in sorted(COHORT_ROOT.iterdir()):
        if not folder.is_dir() or names and folder.name not in names:continue
        c,source=load_supplement(folder.name)
        if c is None:
            summaries.append({'cohort':folder.name,'status':'not_measurable','users':0,'reason':'No target after maximum full TRAIN timestamp under fixed rules.'})
            continue
        for name,model in models.items():
            if np.max(np.diff(c['all_targets'].indptr))>1:
                m=evaluate_multiple(name,model,c)
            else:m=evaluate(name,model,c,family='supplemental')
            summaries.append({'cohort':folder.name,'model':name,'evaluation_manifest_sha256':digest(ROOT/'evaluations'/folder.name/name/'manifest.json')})
    write_json(ROOT/'supplemental_evaluations_index.json',{'status':'controls_complete' if models.keys()==current_models().keys() else 'selected_complete',
        'cohort_root':str(COHORT_ROOT.relative_to(EDA)),'outcomes':summaries})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',action='append');args=p.parse_args()
    with threadpool_limits(limits=2):run(args.cohort)
