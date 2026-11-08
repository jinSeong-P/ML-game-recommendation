"""Reconstruct hits from actual recommendations, then add reporting metrics."""
from __future__ import annotations
import argparse
import json

import duckdb
import numpy as np
import pyarrow.parquet as pq

from eda_large_review_frank import EDA,digest,quote
from improvement_common import write_json,rank_values
from experiment_v4_common import ROOT,SPLIT,catalog,support,load_cohort,paired_comparison


def exposure_metrics(counts,sup):
    counts=np.asarray(counts,dtype=float);slots=counts.sum();n=len(counts)
    if not slots:return dict(coverage=0.,recommended_games=0,slots=0,gini=None,entropy=None,novelty=None,mean_log_support=None)
    ordered=np.sort(counts);gini=2*np.dot(np.arange(1,n+1),ordered)/(n*slots)-(n+1)/n
    p=counts[counts>0]/slots;prob=(sup+1)/(sup.sum()+len(sup))
    return dict(coverage=float((counts>0).mean()),recommended_games=int((counts>0).sum()),slots=int(slots),
        gini=float(gini),entropy=float(-(p*np.log(p)).sum()),
        novelty=float(counts@(-np.log2(prob))/slots),mean_log_support=float(counts@np.log1p(sup)/slots))


def weighted_sensitivity(ranks,targets,sup):
    nref=float(np.median(sup[sup>0]));r,n=rank_values(ranks);result={}
    for gamma in [0,.5,1]:
        for cap in [10,50]:
            w=np.minimum(cap,(nref/(targets+1))**gamma)
            mass=w.sum();_,ix=np.unique(targets,return_inverse=True)
            result[f'gamma{gamma:g}_cap{cap}']=dict(recall=float(w@r/mass),ndcg=float(w@n/mass),
                effective_sample_size=float(mass**2/(w@w)),maximum_support_group_contribution=float(np.bincount(ix,weights=w).max()/mass))
    return result


def audit(folder,cohort):
    apps,linux=catalog();candidates=apps[linux];sup=support()[linux]
    manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
    for f,h in manifest['files'].items():
        if digest(folder/f)!=h:raise ValueError('Saved evaluation changed')
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
    con.execute(f'CREATE VIEW r AS SELECT * FROM read_parquet({quote(folder/"recommendations.parquet")})')
    con.execute(f'CREATE VIEW p AS SELECT * FROM read_parquet({quote(folder/"per_user.parquet")})')
    con.execute(f'CREATE VIEW q AS SELECT * FROM read_parquet({quote(SPLIT/cohort["name"]/"query.parquet")})')
    con.execute(f'CREATE VIEW t AS SELECT * FROM read_parquet({quote(SPLIT/cohort["name"]/"targets.parquet")})')
    con.execute(f'CREATE VIEW c AS SELECT * FROM read_parquet({quote(SPLIT/"catalog/linux_candidates.parquet")})')
    checks={
        'unknown_candidates':con.execute('SELECT count(*) FROM r ANTI JOIN c USING(app_id)').fetchone()[0],
        'observed_recommended':con.execute('SELECT count(*) FROM r JOIN q USING(user_key,app_id)').fetchone()[0],
        'unknown_users':con.execute('SELECT count(*) FROM r ANTI JOIN p USING(user_key)').fetchone()[0],
        'nonpositive_nonfinite':con.execute('SELECT count(*) FROM r WHERE score<=0 OR NOT isfinite(score) OR rank<1 OR rank>10').fetchone()[0],
        'duplicate_recs':con.execute('SELECT count(*) FROM (SELECT user_key,app_id,count(*) n FROM r GROUP BY ALL HAVING n>1)').fetchone()[0],
        'rank_order':con.execute('''SELECT count(*) FROM (SELECT *,lag(score) OVER(PARTITION BY user_key ORDER BY rank) previous,
            lag(app_id) OVER(PARTITION BY user_key ORDER BY rank) previous_app FROM r)
            WHERE score>previous OR (score=previous AND app_id<previous_app)''').fetchone()[0],
        'reconstructed_hit_mismatch':con.execute('''SELECT count(*) FROM p LEFT JOIN
            (SELECT r.user_key,r.rank FROM r JOIN t USING(user_key,app_id)) h USING(user_key)
            WHERE p.target_rank<>coalesce(h.rank,0)''').fetchone()[0],
        'recommendation_count_mismatch':con.execute('''SELECT count(*) FROM p LEFT JOIN
            (SELECT user_key,count(*) n,max(rank) maxrank FROM r GROUP BY user_key) h USING(user_key)
            WHERE recommendation_count<>coalesce(n,0) OR coalesce(n,0)<>coalesce(maxrank,0)''').fetchone()[0]}
    if any(checks.values()):raise ValueError(f'Actual recommendation replay failed: {checks}')
    counts=con.execute('''SELECT app_id,count(*) FROM r SEMI JOIN
        (SELECT user_key FROM p WHERE n_total_inputs>=5) u USING(user_key) GROUP BY app_id ORDER BY app_id''').fetchall()
    exposures=np.zeros(len(candidates));
    for app,n in counts:exposures[np.searchsorted(candidates,app)]=n
    tagfile=EDA/'data/processed/modeling/improvement_v2/tfidf/metadata_channel/metadata_source_audit.parquet'
    tags={int(row['app_id']):set(row['normalized_tags'] or []) for row in pq.read_table(tagfile,columns=['app_id','normalized_tags']).to_pylist()}
    lists=con.execute('''SELECT user_key,list(app_id ORDER BY rank) FROM r SEMI JOIN
        (SELECT user_key FROM p WHERE n_total_inputs>=5) u USING(user_key) GROUP BY user_key HAVING count(*)>=2''').fetchall()
    values=[];missing_users=0
    for user,items in lists:
        if any(not tags.get(int(i)) for i in items):missing_users+=1;continue
        distances=[]
        for j,a in enumerate(items):
            for b in items[j+1:]:
                ta,tb=tags[a],tags[b];distances.append(1-len(ta&tb)/len(ta|tb))
        values.append(np.mean(distances))
    per=pq.read_table(folder/'per_user.parquet').to_pydict();main=np.asarray(per['n_total_inputs'])>=5
    ranks=np.asarray(per['target_rank'])[main];ts=np.asarray(per['target_train_support'])[main]
    sensitivity=weighted_sensitivity(ranks,ts,sup)
    for v in sensitivity.values():v.pop('maximum_support_group_contribution',None)
    # Maximum actual target GAME weight, not the aggregate support-value bucket.
    targetapps=np.asarray(per['target_app_id'])[main];_,idx=np.unique(targetapps,return_inverse=True)
    nref=np.median(sup[sup>0])
    for gamma in [0,.5,1]:
        for cap in [10,50]:
            w=np.minimum(cap,(nref/(ts+1))**gamma)
            sensitivity[f'gamma{gamma:g}_cap{cap}']['maximum_target_game_contribution']=float(np.bincount(idx,weights=w).max()/w.sum())
    result=dict(status='passed_actual_recommendation_replay',checks=checks,
        exposure=exposure_metrics(exposures,sup),
        diversity=dict(tag_jaccard_distance=float(np.mean(values)) if values else None,
            users_with_at_least2_recommendations=len(lists),users_all_recommended_tags_available=len(values),
            users_excluded_missing_tags=missing_users),
        inverse_popularity_sensitivity=sensitivity,audit_code_sha256=digest(__file__))
    write_json(folder/'reporting_metrics.json',result);con.close();return result


def run(role='selection'):
    cohort=load_cohort(role);base=ROOT/'evaluations'/role/'activity_inverse_degree_cosine_k100/per_user.parquet'
    a=pq.read_table(base,columns=['target_rank'])['target_rank'].to_numpy() if base.exists() else None
    for m in sorted((ROOT/'evaluations'/role).glob('*/manifest.json')):
        if not (m.parent/'reporting_metrics.json').exists():audit(m.parent,cohort)
        if a is not None:
            b=pq.read_table(m.parent/'per_user.parquet',columns=['target_rank'])['target_rank'].to_numpy()
            write_json(m.parent/'paired_comparison.json',paired_comparison(a,b,cohort))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='selection',choices=['selection','confirmation']);args=p.parse_args();run(args.role)
