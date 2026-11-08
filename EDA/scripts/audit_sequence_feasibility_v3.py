"""Assess the conditional sequential-model backlog without inventing model scores."""
import json
import duckdb
from eda_large_review_frank import digest,quote
from experiment_v3_common import ROOT,write_json


def run():
    base=ROOT/'temporal_cutoff2022/result'
    report=json.loads((base/'report.json').read_text('utf-8'))
    con=duckdb.connect()
    try:
        con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
        degree=con.execute(f'''WITH d AS(SELECT user_key,count(*) n FROM
            read_parquet({quote(base/'training/positive_catalog.parquet')}) GROUP BY user_key)
            SELECT count(*),sum(n),count(*) FILTER(WHERE n>=2),count(*) FILTER(WHERE n>=5),
                count(*) FILTER(WHERE n>=10),sum(greatest(n-1,0)),approx_quantile(n,[.25,.5,.75,.9,.99]) FROM d''').fetchone()
        query=con.execute(f'''WITH d AS(SELECT user_key,count(*) n,sum(positive::INTEGER) p FROM
            read_parquet({quote(base/'validation/query.parquet')}) GROUP BY user_key)
            SELECT count(*),approx_quantile(n,[.25,.5,.75]),approx_quantile(p,[.25,.5,.75]) FROM d''').fetchone()
    finally:con.close()
    result={'status':'feasibility_audit_complete_model_not_run','code_sha256':digest(__file__),
        'temporal_report_sha256':digest(base/'report.json'),
        'training':{'users':degree[0],'positive_pairs':degree[1],'users_at_least2':degree[2],
            'users_at_least5':degree[3],'users_at_least10':degree[4],
            'possible_adjacent_pairs_before_timestamp_ties_or_dedup':degree[5],
            'approximate_degree_quartiles_p25_p50_p75_p90_p99':degree[6]},
        'validation':{'users':query[0],'approximate_visible_total_p25_p50_p75':query[1],
            'approximate_visible_positive_p25_p50_p75':query[2]},
        'decision':'SASRec is potentially trainable, not shown infeasible. Full model training remains a conditional follow-up, not a measured v3 candidate.',
        'missing_prerequisites':['Create and audit accepted train event sequences with stable event-time tie rules; current temporal model input stores user/app pairs, not sequences.',
            'Freeze sequence length, negative sampling, architecture, optimizer and train/validation chronology before SASRec metrics.',
            'Address the existing temporal experiment snapshot-label/transductive-catalog limitations; timestamp cutoff alone is not an independent historical replay.'],
        'metric':None,'scope':'A review-created timestamp is not a play-session/order event. Do not claim SASRec performance from sequence counts.'}
    write_json(ROOT/'sequence_feasibility_audit.json',result)
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
