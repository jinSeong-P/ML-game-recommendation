"""Verify the full Kieran EDA artifacts and recompute frank eligibility.

This publishes aggregates and split previews; it does not construct final splits.
The former candidate corpora are not read by this main workflow.
"""
import json
from datetime import datetime, timezone

import duckdb
import pyarrow.parquet as pq

from eda_large_review_frank import EDA, digest, many, one, prepare_frank, quote


def verify_kieran(metrics):
    folder = EDA/'data/processed/kieran_original'
    manifest = folder/'ingestion_manifest.json'
    if digest(manifest) != metrics['ingestion_manifest_sha256']:
        raise ValueError('Kieran ingestion manifest changed')
    paths = sorted(folder.glob('reviews_*.parquet'))
    expected = {item['file']:item for item in metrics['canonical_inputs']}
    if set(expected) != {p.name for p in paths}:
        raise ValueError('Kieran shard inventory changed')
    rows = 0
    for path in paths:
        if digest(path) != expected[path.name]['sha256']:
            raise ValueError(f'Kieran shard changed: {path.name}')
        rows += pq.ParquetFile(path).metadata.num_rows
    if rows != metrics['raw_quality']['raw_rows']:
        raise ValueError('Kieran total row count changed')
    table = EDA/'reports/tables/16_kieran_original_app_support.csv'
    if digest(table) != metrics['app_support_table_sha256']:
        raise ValueError('Kieran item-support output changed')
    if not metrics['whole_account_and_cross_review_identity_checks_passed'] or any(metrics['post_exclusion_checks'].values()):
        raise ValueError('Full-source identity/exclusion audit did not pass')
    return {'canonical_shards':len(paths),'verified_rows':rows,
            'canonical_hashes_verified':True,'support_table_hash_verified':True,
            'ingestion_manifest_hash_verified':True,
            'full_analysis_code_sha256':metrics['analysis_code_sha256']}


def main():
    source_path = EDA/'reports/kieran_original_frank_eda_metrics.json'
    metrics = json.loads(source_path.read_text('utf-8'))
    verified = verify_kieran(metrics)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    frank = prepare_frank(con,EDA/'data/processed')
    for key in ['commit','projection_sha256','raw_rows','reserved_test_users']:
        if frank[key] != metrics['frank_provenance'][key]:
            raise ValueError(f'frank differs from the full Kieran exclusion audit: {key}')
    reference = json.loads((EDA/'data/reference/manifest.json').read_text('utf-8'))
    if metrics['analysis_version'] == 'large-original-frank-eda-v1':
        if reference['former_catalog_inputs'] != metrics['reference_catalog_inputs']:
            raise ValueError('Frozen catalog provenance differs from the full-source audit')
    else:
        for item in metrics['reference_catalog_inputs']:
            if digest(EDA/'data/reference'/item['file']) != item['sha256']:
                raise ValueError('Frozen catalog differs from the full-source audit')
    support = EDA/'reports/tables/16_kieran_original_app_support.csv'
    con.execute(f'CREATE TABLE support AS SELECT * FROM read_csv_auto({quote(support)})')
    quarantine = EDA/'data/reference/review_app_quarantine.parquet'
    con.execute(f'CREATE TABLE quarantine AS SELECT * FROM read_parquet({quote(quarantine)})')
    con.execute('''CREATE TABLE ambiguous_pairs AS SELECT DISTINCT f.user_key,f.app_id
        FROM frank_raw f JOIN quarantine q USING(review_key)''')
    con.execute('''CREATE TABLE comparison AS SELECT f.*,c.linux,
        b.user_key IS NOT NULL ambiguous,
        coalesce(s.final_train_reviews,0) train_reviews,
        coalesce(s.final_train_positive,0) train_positive
        FROM frank f JOIN test_users t USING(user_key) JOIN catalog c USING(app_id)
        LEFT JOIN support s USING(app_id)
        LEFT JOIN ambiguous_pairs b ON f.user_key=b.user_key AND f.app_id=b.app_id
        WHERE c.reviewed''')
    cohorts = {}
    for name,condition in {
        'safe_mapping_fixed_catalog':'NOT ambiguous',
        'safe_mapping_kieran_review_support':'NOT ambiguous AND train_reviews>0',
        'safe_mapping_kieran_positive_support_5':'NOT ambiguous AND train_positive>=5',
    }.items():
        cohorts[name] = many(con,f'''WITH per_user AS (
            SELECT user_key,count(*) FILTER(WHERE {condition}) n_games,
                count(*) FILTER(WHERE {condition} AND positive) n_positive,
                count(*) FILTER(WHERE {condition} AND positive AND linux) n_targets
            FROM comparison GROUP BY user_key)
            SELECT k minimum_inputs,
                count(*) FILTER(WHERE n_games>=k+1 AND n_targets>0) potential_users,
                count(*) FILTER(WHERE n_games>=k+1 AND n_targets>0 AND n_positive>=2) potential_users_with_positive_query,
                count(*) FILTER(WHERE n_games>=k+1 AND n_targets>0 AND n_positive=1) potential_users_with_only_negative_query,
                coalesce(sum(n_targets) FILTER(WHERE n_games>=k+1 AND n_targets>0),0) possible_target_pairs
            FROM per_user CROSS JOIN (VALUES (1),(3),(5),(10)) minimum(k)
            GROUP BY k ORDER BY k''')
    audit = one(con,'''SELECT (SELECT count(*) FROM quarantine) ambiguous_original_review_ids_in_frank,
        count(*) FILTER(WHERE ambiguous) ambiguous_pairs_in_fixed_test_histories,
        count(DISTINCT user_key) FILTER(WHERE ambiguous) fixed_test_users_with_any_ambiguous_pair,
        count(*) FILTER(WHERE ambiguous AND positive AND linux) ambiguous_possible_linux_target_pairs FROM comparison''')
    result = {
        'selection':'ADOPTED_KIERAN_ORIGINAL_TO_FRANK_REVIEWS',
        'checked_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Verified full-source Kieran EDA, recomputed frank review eligibility and training-support diagnostics. No model fit or final split files.',
        'source_metric_file':source_path.name,'source_metric_sha256':digest(source_path),
        'reference_manifest_sha256':digest(EDA/'data/reference/manifest.json'),
        'verification':verified,'frank_provenance':frank,
        'frank_full_review_summary':one(con,'''SELECT count(*) clean_pairs,count(DISTINCT user_key) users,
            count(DISTINCT app_id) apps,count(*) FILTER(WHERE positive) positives FROM frank'''),
        'catalog':one(con,'''SELECT count(*) all_apps,count(*) FILTER(WHERE reviewed) input_apps,
            count(*) FILTER(WHERE reviewed AND linux) linux_target_apps FROM catalog'''),
        'reserved_test_users':frank['reserved_test_users'],
        'evaluation_mapping_audit':audit,'cohorts':cohorts,
        'protocol':'Hide exactly one Linux-positive review; all other in-catalog positive/negative reviews may be inputs. No chronological split claimed.',
        'training_exclusion':'Reserve the original 87,573-account superset, even when evaluation is a cleaned subset.',
        'status':{'dataset_choice_final':True,'final_train_validation_test_files_written':False,'model_performance_measured':False}}
    output = EDA/'reports/kieran_frank_eda_metrics.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    con.close()
    print(json.dumps({'output':str(output),'verification':verified,'catalog':result['catalog'],
        'safe_cohort':cohorts['safe_mapping_fixed_catalog']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
