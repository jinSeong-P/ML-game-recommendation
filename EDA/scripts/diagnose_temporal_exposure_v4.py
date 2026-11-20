"""Quantify future-input and future-training exposure in old random holdout.

This is an aggregate diagnostic of the already observed validation set, not
a new performance test or proof of reconstructing historical review labels.
"""
import json
import duckdb
import numpy as np
from eda_large_review_frank import EDA,digest,quote
from improvement_common import write_json


def run():
    root=EDA/'data/processed/modeling/experiments_v4/temporal_diagnostic';root.mkdir(parents=True,exist_ok=True)
    c=duckdb.connect(str(EDA/'data/processed/kieran_original/eda_work.duckdb'),read_only=True)
    c.execute("SET memory_limit='1500MB'");c.execute('SET threads=2')
    c.execute(f'SET temp_directory={quote(root/"temp")}')
    split=EDA/'data/processed/modeling/split_v1'
    context=EDA/'data/processed/modeling/improvement_v3/query_context/validation.parquet'
    c.execute(f'CREATE TEMP VIEW h AS SELECT * FROM read_parquet({quote(context)})')
    c.execute(f'CREATE TEMP VIEW t AS SELECT * FROM read_parquet({quote(split/"validation/targets.parquet")})')
    c.execute(f'CREATE TEMP VIEW vl AS SELECT * FROM read_parquet({quote(split/"validation/review_lineage.parquet")})')
    c.execute('''CREATE TEMP TABLE target_time AS SELECT t.user_key,t.app_id,min(r.timestamp_created) target_time
        FROM t JOIN vl USING(user_key,app_id,positive)
        JOIN valid_raw r USING(user_key,app_id,review_key,positive) GROUP BY t.user_key,t.app_id''')
    if c.execute('SELECT count(*) FROM target_time').fetchone()[0]!=325639:
        raise ValueError('Old validation target timestamp roster differs')
    cols=['input_pairs','pairs_with_known_times','later_input_pairs','positive_input_pairs','later_positive_input_pairs',
        'users','users_with_later_input']
    row=c.execute('''SELECT count(*),count(*) FILTER(WHERE h.timestamp_created IS NOT NULL AND t.target_time IS NOT NULL),
        count(*) FILTER(WHERE h.timestamp_created>t.target_time),count(*) FILTER(WHERE h.positive),
        count(*) FILTER(WHERE h.positive AND h.timestamp_created>t.target_time),
        count(DISTINCT h.user_key),count(DISTINCT h.user_key) FILTER(WHERE h.timestamp_created>t.target_time)
        FROM h JOIN target_time t USING(user_key) WHERE h.app_id<>t.app_id''').fetchone()
    result=dict(zip(cols,map(int,row)))
    result['later_input_fraction']=result['later_input_pairs']/result['pairs_with_known_times']
    result['users_with_later_input_fraction']=result['users_with_later_input']/result['users']
    # accepted original lineage only, current catalog, positive pairs. MIN
    # creation is the earliest accepted positive review; MAX update discloses
    # when the latest snapshot label could have become known.
    c.execute(f'CREATE TEMP VIEW l AS SELECT * FROM read_parquet({quote(split/"train/review_lineage.parquet")})')
    c.execute(f'CREATE TEMP VIEW apps AS SELECT * FROM read_parquet({quote(split/"catalog/input_apps.parquet")})')
    c.execute('''CREATE TEMP TABLE train_times AS SELECT l.user_key,l.app_id,min(r.timestamp_created) created,
        max(r.timestamp_updated) updated FROM l JOIN valid_raw r USING(user_key,app_id,review_key,positive)
        SEMI JOIN apps USING(app_id) WHERE l.positive GROUP BY l.user_key,l.app_id''')
    times=np.asarray(c.execute('SELECT created,count(*) FROM train_times WHERE created IS NOT NULL GROUP BY created ORDER BY created').fetchall(),dtype=np.int64)
    targets=np.asarray(c.execute('SELECT target_time FROM target_time WHERE target_time IS NOT NULL ORDER BY target_time').fetchall(),dtype=np.int64).ravel()
    total=int(times[:,1].sum());cumulative=np.r_[0,np.cumsum(times[:,1])]
    later=total-cumulative[np.searchsorted(times[:,0],targets,side='right')]
    result['train_positive_pairs_known_creation']=total
    result['mean_fraction_train_created_after_target']=float(np.mean(later/total))
    result['train_future_fraction_quantiles_p0_p25_p50_p75_p100']=np.quantile(later/total,[0,.25,.5,.75,1]).tolist()
    result['training_updated_after_created_pairs']=int(c.execute('SELECT count(*) FROM train_times WHERE updated>created').fetchone()[0])
    meta=EDA/'legacy/data/processed/C_games_2025.parquet'
    c.execute(f'''CREATE TEMP TABLE release AS SELECT cast(appid AS BIGINT) app_id,
        epoch(try_cast(release_date AS DATE)) release_time FROM read_parquet({quote(meta)})''')
    c.execute(f'CREATE TEMP VIEW candidates AS SELECT * FROM read_parquet({quote(split/"catalog/linux_candidates.parquet")})')
    release=np.asarray(c.execute('SELECT release_time FROM release SEMI JOIN candidates USING(app_id) WHERE release_time IS NOT NULL ORDER BY release_time').fetchall()).ravel()
    result['candidate_release_dates_known']=len(release)
    result['mean_fraction_all_candidates_released_after_target']=float(np.mean((len(release)-np.searchsorted(release,targets,side='right'))/6630))
    result['unknown_release_candidates']=6630-len(release)
    result['status']='diagnostic_complete_not_temporal_performance'
    result['limitations']=[
        'Cached original created timestamps do not establish when current positive/negative labels were first observed.',
        'Current 2025 metadata does not reconstruct historical release records or historical Linux support.',
        'Future-training fraction counts current positive training pairs; it is exposure, not a measured causal effect on Recall.',
        'Forgemaster source is absent, so no Forgemaster temporal exposure is calculated.']
    result['sources']={str(p.relative_to(EDA)):digest(p) for p in [context,split/'validation/targets.parquet',split/'train/review_lineage.parquet',meta]}
    result['code_sha256']=digest(__file__)
    write_json(EDA/'reports/temporal_exposure_v4.json',result);c.close();return result


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
