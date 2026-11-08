"""Independently recalculate saved aggregate metrics and verify frozen choices."""
import json
import math
import duckdb
from eda_large_review_frank import EDA,digest,quote
from experiment_v3_common import ROOT,PROTOCOL,SPLIT,catalog,write_json
from select_experiments_v3 import manifests,verify_evidence,expected_variants


def run():
    selection=json.loads((ROOT/'selection_frozen.json').read_text('utf-8'))
    if digest(ROOT/'selection_frozen.json')!=digest(EDA/'data/reference/improvement_selection_v3.json'):
        raise ValueError('Public/private selections differ')
    primary=manifests()
    if {m['variant'] for m in primary}!=expected_variants():raise ValueError('Primary roster mismatch')
    if verify_evidence(primary)!=selection['validation_evidence']:raise ValueError('Evidence changed')
    for name,sha in selection['code_sha256'].items():
        if digest(EDA/'scripts'/name)!=sha:raise ValueError('Code changed after selection: '+name)
    content_audit_path=ROOT/'content_source_audit.json'
    content_audit=json.loads(content_audit_path.read_text('utf-8'))
    if content_audit['status']!='passed':raise ValueError('Content lineage audit incomplete')
    ease_audit_path=ROOT/'ease_solution_audit.json'
    ease_audit=json.loads(ease_audit_path.read_text('utf-8'))
    if ease_audit['status']!='passed':raise ValueError('EASE solution audit incomplete')
    from replay_temporal_v3 import run as temporal_replay
    temporal=temporal_replay()
    if temporal['status']!='passed' or any(temporal['violations'].values()):raise ValueError('Temporal prediction audit incomplete')
    supplements=json.loads((ROOT/'selected_supplemental_index.json').read_text('utf-8'))
    if supplements['selection_sha256']!=digest(ROOT/'selection_frozen.json'):raise ValueError('Supplement selection changed')
    # The cohort directory names, rather than shorthand aliases, are authoritative.
    from evaluate_supplements_v3 import COHORT_ROOT
    expected_supplements={p.name for p in COHORT_ROOT.iterdir() if p.is_dir()
        and json.loads((p/'manifest.json').read_text('utf-8')).get('users',0)>0}
    actual_supplements={(x['cohort'],x['variant']) for x in supplements['outcomes']}
    variants={x['variant'] for x in selection['models'].values()}
    if actual_supplements!={(c,v) for c in expected_supplements for v in variants}:raise ValueError('Selected supplemental coverage incomplete')
    checks=[];con=duckdb.connect()
    try:
        con.execute("SET memory_limit='500MB'");con.execute('SET threads=2')
        import pyarrow as pa
        apps,linux,_=catalog();con.register('fixed_candidates',pa.table({'app_id':apps[linux]}))
        for path in sorted((ROOT/'evaluations').glob('*/*/manifest.json')):
            m=json.loads(path.read_text('utf-8'));p=m.get('protection',m.get('metrics',{}))
            if m['status']!='complete' or any(m['violations'].values()):raise ValueError('Failed saved audit '+str(path))
            for rel,info in m['files'].items():
                if digest(path.parent/rel)!=info['sha256']:raise ValueError('Changed evaluation artifact')
            src=quote(path.parent/'per_user.parquet')
            cohort_folder=(SPLIT/m['cohort'] if m['cohort'] in ['validation','test'] else COHORT_ROOT/m['cohort'])
            for view,file in [('saved_users',path.parent/'per_user.parquet'),('saved_recs',path.parent/'recommendations.parquet'),
                              ('expected_users',cohort_folder/'users.parquet'),('expected_targets',cohort_folder/'targets.parquet'),
                              ('excluded_games',cohort_folder/'exclusions.parquet')]:
                con.execute(f'CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM read_parquet({quote(file)})')
            sql_checks={
                'missing_users':'SELECT count(*) FROM expected_users ANTI JOIN saved_users USING(user_key)',
                'extra_users':'SELECT count(*) FROM saved_users ANTI JOIN expected_users USING(user_key)',
                'duplicate_users':'SELECT count(*)-count(DISTINCT user_key) FROM saved_users',
                'unknown_recommendation_users':'SELECT count(*) FROM saved_recs ANTI JOIN saved_users USING(user_key)',
                'outside_catalog':'SELECT count(*) FROM saved_recs ANTI JOIN fixed_candidates USING(app_id)',
                'seen_games':'SELECT count(*) FROM saved_recs JOIN excluded_games USING(user_key,app_id)',
                'duplicate_games':'SELECT count(*) FROM (SELECT user_key,app_id FROM saved_recs GROUP BY ALL HAVING count(*)<>1)',
                'nonpositive_scores':'SELECT count(*) FROM saved_recs WHERE score<=0 OR NOT isfinite(score)',
                'rank_or_count':'''SELECT count(*) FROM saved_users o LEFT JOIN (SELECT user_key,count(*) n,min(rank) lo,max(rank) hi,
                    count(DISTINCT rank) nr FROM saved_recs GROUP BY user_key) r USING(user_key)
                    WHERE o.recommendation_count<>coalesce(n,0) OR coalesce(n,0)>10 OR (n>0 AND (lo<>1 OR hi<>n OR nr<>n))''',
                'out_of_order':'''SELECT count(*) FROM(SELECT *,lag(score) OVER w ps,lag(app_id) OVER w pa FROM saved_recs
                    WINDOW w AS(PARTITION BY user_key ORDER BY rank)) WHERE score>ps OR(score=ps AND app_id<pa)'''}
            extra=cohort_folder/'additional_exclusions.parquet'
            if extra.exists():sql_checks['seen_additional_games']=f'SELECT count(*) FROM saved_recs JOIN read_parquet({quote(extra)}) USING(user_key,app_id)'
            multiple='hidden_targets_per_user' in p
            if multiple:
                sql_checks['wrong_target_counts']='''SELECT count(*) FROM saved_users o LEFT JOIN
                    (SELECT user_key,count(*) n FROM expected_targets GROUP BY user_key) t USING(user_key)
                    WHERE o.hidden_targets<>coalesce(n,0) OR o.hidden_targets<>2'''
                sql_checks['wrong_saved_gains']='''SELECT count(*) FROM saved_users o LEFT JOIN
                    (SELECT r.user_key,count(*) hits,sum(1/log2(r.rank+1)) dcg FROM saved_recs r JOIN expected_targets t USING(user_key,app_id)
                     GROUP BY r.user_key) h USING(user_key) WHERE o.retrieved_targets<>coalesce(hits,0)
                     OR abs(o.ndcg_at_10-coalesce(dcg,0)/(1+1/log2(3)))>1e-12'''
                sql=f'''SELECT count(*),avg(retrieved_targets::DOUBLE/hidden_targets),avg(ndcg_at_10),
                    count(*) FILTER(WHERE recommendation_count=0) FROM read_parquet({src})'''
            else:
                sql_checks['wrong_target']='SELECT count(*) FROM saved_users o JOIN expected_targets t USING(user_key) WHERE o.target_app_id<>t.app_id'
                sql_checks['wrong_saved_target_rank']='''SELECT count(*) FROM saved_users o LEFT JOIN saved_recs r
                    ON o.user_key=r.user_key AND o.target_app_id=r.app_id WHERE o.target_rank<>coalesce(r.rank,0)'''
                sql=f'''SELECT count(*),avg((target_rank>0)::INTEGER),
                    avg(CASE WHEN target_rank>0 THEN 1/log2(target_rank+1) ELSE 0 END),
                    count(*) FILTER(WHERE recommendation_count=0) FROM read_parquet({src})'''
            n,recall,ndcg,empty=con.execute(sql).fetchone()
            independent_violations={k:int(con.execute(q).fetchone()[0]) for k,q in sql_checks.items()}
            if any(independent_violations.values()):raise ValueError('Independent saved recommendation audit failed: '+str(path)+' '+str(independent_violations))
            for name,actual in [('users',n),('Recall@10',recall),('NDCG@10',ndcg),('empty_users',empty)]:
                if not math.isclose(float(actual),float(p[name]),abs_tol=1e-12,rel_tol=1e-12):
                    raise ValueError(f'Saved aggregate differs: {path} {name} {actual} {p[name]}')
            if not multiple:
                macro=con.execute(f'''SELECT avg(ndcg) FROM (SELECT target_app_id,
                    avg(CASE WHEN target_rank>0 THEN 1/log2(target_rank+1) ELSE 0 END) ndcg
                    FROM read_parquet({src}) GROUP BY target_app_id)''').fetchone()[0]
                if not math.isclose(macro,p['game_macro_NDCG@10'],abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Macro metric differs')
            checks.append({'cohort':m['cohort'],'variant':m['variant'],'users':n,'metrics_reproduced':True,
                'independent_saved_prediction_violations':independent_violations,
                'manifest_sha256':digest(path)})
    finally:con.close()
    result={'status':'passed','primary_evaluations':len(primary),'saved_evaluations':len(checks),
        'aggregate_differences':0,'recorded_sql_violations':0,'code_pins_verified':True,
        'protocol_sha256':digest(PROTOCOL),'selection_sha256':digest(ROOT/'selection_frozen.json'),
        'content_source_audit_sha256':digest(content_audit_path),
        'ease_solution_audit_sha256':digest(ease_audit_path),
        'temporal_replay_audit_sha256':digest(ROOT/'temporal_cutoff2022/prediction_replay/audit.json'),
        'selected_supplemental_evaluations':len(supplements['outcomes']),
        'audit_code_sha256':digest(__file__),'evaluations':checks}
    write_json(EDA/'reports/improvement_v3_artifact_audit.json',result)
    return result


if __name__=='__main__':run()
