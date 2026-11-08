"""Audit a completed temporal experiment by saving independently ranked predictions.

The original temporal report/model/cohort remain immutable. This does not refit
or select a model; it adds an auditable replay beside the completed run.
"""
import json
import math
import shutil
from pathlib import Path
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from eda_large_review_frank import EDA, digest, quote
from experiment_v3_common import ROOT, write_json
from experiment_temporal_v3 import iter_query_batches


def rank_row(indices, values, candidates, excluded):
    """Use finite positive scores and ascending App ID to break equal scores."""
    keep=np.isfinite(values)&(values>0)&~np.isin(candidates[indices],excluded)
    indices,values=indices[keep],values[keep]
    order=np.lexsort((candidates[indices],-values))[:10]
    return candidates[indices[order]],values[order]


def run():
    base=ROOT/'temporal_cutoff2022';result=base/'result';out=base/'prediction_replay'
    report=json.loads((result/'report.json').read_text('utf-8'))
    report_sha=digest(result/'report.json')
    if out.exists():
        m=json.loads((out/'audit.json').read_text('utf-8'))
        if m['report_sha256']!=report_sha:raise ValueError('Temporal report changed')
        for rel,sha in m['files'].items():
            if digest(out/rel)!=sha:raise ValueError('Temporal replay changed')
        return m
    for rel,info in report['files'].items():
        if digest(result/rel)!=info['sha256']:raise ValueError('Temporal source changed: '+rel)
    stage=base/'.prediction_replay.building'
    if stage.exists():raise FileExistsError('Inspect partial temporal replay before retrying')
    stage.mkdir()
    source=EDA/'scripts/experiment_temporal_v3.py'
    # Copy only when this is the exact builder fingerprint recorded at completion.
    if digest(source)!=report['builder_sha256']:raise ValueError('Original temporal builder changed')
    shutil.copyfile(source,stage/'temporal_builder_snapshot.py')
    apps=pq.read_table(result/'model/game_rows.parquet')['app_id'].to_numpy()
    candidates=pq.read_table(result/'model/candidate_rows.parquet')['app_id'].to_numpy()
    supports=pq.read_table(result/'model/game_rows.parquet')['positive_user_support'].to_numpy()
    targets=pq.read_table(result/'validation/targets.parquet').to_pydict()
    target_map=dict(zip(targets['user_key'],targets['app_id']))
    matrix=sparse.load_npz(result/'model/similarities_csr.npz').T.tocsr()
    user_schema=pa.schema([('user_key',pa.uint64()),('target_app_id',pa.int64()),
        ('target_rank',pa.int16()),('recommendation_count',pa.int16()),('target_train_support',pa.int64())])
    rec_schema=pa.schema([('user_key',pa.uint64()),('app_id',pa.int64()),('rank',pa.int16()),('score',pa.float64())])
    with pq.ParquetWriter(stage/'per_user.parquet',user_schema,compression='zstd') as uw, \
         pq.ParquetWriter(stage/'recommendations.parquet',rec_schema,compression='zstd') as rw:
        for groups in iter_query_batches(result/'validation/query.parquet'):
            rows=[];cols=[]
            for i,(_,ids,positive) in enumerate(groups):
                positions=np.searchsorted(apps,ids[positive])
                if np.any(positions>=len(apps)) or np.any(apps[positions]!=ids[positive]):raise ValueError('Unknown input')
                rows.extend([i]*len(positions));cols.extend(positions)
            q=sparse.csr_matrix((np.ones(len(cols),dtype=np.uint8),(rows,cols)),shape=(len(groups),len(apps)))
            scores=(q@matrix).tocsr();scores.sum_duplicates();scores.sort_indices()
            users=[];recommendations=[]
            for i,(user,ids,positive) in enumerate(groups):
                lo,hi=scores.indptr[i:i+2]
                chosen,values=rank_row(scores.indices[lo:hi],scores.data[lo:hi],candidates,ids)
                target=target_map[user];hits=np.flatnonzero(chosen==target)
                users.append({'user_key':user,'target_app_id':target,'target_rank':int(hits[0]+1) if len(hits) else 0,
                    'recommendation_count':len(chosen),'target_train_support':int(supports[np.searchsorted(apps,target)])})
                # Stored scores are average similarities, with the same ordering as sums.
                recommendations.extend({'user_key':user,'app_id':int(a),'rank':r,'score':float(s)/sum(positive)}
                    for r,(a,s) in enumerate(zip(chosen,values),1))
            uw.write_table(pa.Table.from_pylist(users,schema=user_schema))
            rw.write_table(pa.Table.from_pylist(recommendations,schema=rec_schema))
    con=duckdb.connect()
    try:
        con.execute("SET memory_limit='500MB'");con.execute('SET threads=2')
        for name,path in [('u',stage/'per_user.parquet'),('r',stage/'recommendations.parquet'),
                          ('t',result/'validation/targets.parquet'),('q',result/'validation/query.parquet'),
                          ('c',result/'model/candidate_rows.parquet')]:
            con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({quote(path)})')
        violations={
            'missing_or_extra_user':con.execute('SELECT count(*) FROM u FULL JOIN t USING(user_key) WHERE u.user_key IS NULL OR t.user_key IS NULL OR u.target_app_id<>t.app_id').fetchone()[0],
            'duplicate_user':con.execute('SELECT count(*)-count(DISTINCT user_key) FROM u').fetchone()[0],
            'duplicate_game':con.execute('SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM r GROUP BY user_key,app_id)').fetchone()[0],
            'seen_game':con.execute('SELECT count(*) FROM r JOIN q USING(user_key,app_id)').fetchone()[0],
            'outside_candidates':con.execute('SELECT count(*) FROM r ANTI JOIN c USING(app_id)').fetchone()[0],
            'nonpositive_or_nonfinite_score':con.execute('SELECT count(*) FROM r WHERE score<=0 OR NOT isfinite(score)').fetchone()[0],
            'rank_or_count':con.execute('''SELECT count(*) FROM u LEFT JOIN
                (SELECT user_key,count(*) n,min(rank) lo,max(rank) hi,count(DISTINCT rank) nr FROM r GROUP BY user_key) x USING(user_key)
                WHERE coalesce(n,0)<>recommendation_count OR coalesce(n,0)>10 OR (n>0 AND (lo<>1 OR hi<>n OR nr<>n))''').fetchone()[0],
            'wrong_target_rank':con.execute('''SELECT count(*) FROM u LEFT JOIN r
                ON u.user_key=r.user_key AND u.target_app_id=r.app_id WHERE u.target_rank<>coalesce(r.rank,0)''').fetchone()[0],
            'wrong_order':con.execute('''SELECT count(*) FROM (SELECT *,lag(score) OVER w ps,lag(app_id) OVER w pa
                FROM r WINDOW w AS(PARTITION BY user_key ORDER BY rank)) WHERE score>ps OR (score=ps AND app_id<pa)''').fetchone()[0]}
        n,recall,ndcg,hits,empty=con.execute('''SELECT count(*),avg((target_rank>0)::INTEGER),
            avg(CASE WHEN target_rank>0 THEN 1/log2(target_rank+1) ELSE 0 END),
            count(*) FILTER(WHERE target_rank>0),count(*) FILTER(WHERE recommendation_count=0) FROM u''').fetchone()
    finally:con.close()
    expected=report['metrics']
    if n!=expected['cohort_users'] or hits!=expected['target_hits_at_10'] or not math.isclose(recall,expected['recall_at_10'],abs_tol=1e-12) or not math.isclose(ndcg,expected['ndcg_at_10'],abs_tol=1e-12):raise ValueError('Temporal replay metrics differ')
    if any(violations.values()):raise ValueError('Temporal prediction audit failed: '+str(violations))
    audit={'status':'passed','report_sha256':report_sha,'metrics':{'users':n,'Recall@10':recall,'NDCG@10':ndcg,'hits':hits,'empty_users':empty},
        'violations':violations,'code_sha256':digest(__file__),
        'files':{p.name:digest(p) for p in stage.iterdir() if p.is_file()},
        'scope':'Independent saved-prediction replay; original report/model/cohort unchanged; no refit or selection.'}
    write_json(stage/'audit.json',audit);stage.rename(out)
    return audit


if __name__=='__main__':print(json.dumps(run(),indent=2))
