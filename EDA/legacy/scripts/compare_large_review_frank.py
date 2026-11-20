"""Compare complete-source EDA using a common fixed frank evaluation roster."""

import json
from pathlib import Path

import duckdb

from eda_large_review_frank import EDA, many, prepare_frank, quote


def main():
    con=duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    prepare_frank(con,EDA/'data/processed')
    for name,alias in [('artermiloff_2024','a'),('kieran_original','b')]:
        path=EDA/'reports/tables'/f'16_{name}_app_support.csv'
        con.execute(f"CREATE TABLE {alias} AS SELECT * FROM read_csv_auto({quote(path)})")
        database=EDA/'data/processed/candidate_identity'/name/'eda_work.duckdb'
        con.execute(f"ATTACH {quote(database)} AS src_{alias} (READ_ONLY)")
    # Ambiguous original review-to-app mappings also affect the credibility of
    # the evaluation labels. Remove the union from the evaluation preview;
    # training has conservatively reserved the original 87,573-account superset.
    con.execute("""CREATE TABLE ambiguous_frank_review_keys AS
        SELECT f.review_key FROM frank_raw f JOIN src_a.main.review_keys k USING(review_key)
            WHERE k.min_app<>k.max_app OR k.min_user<>k.max_user
        UNION SELECT f.review_key FROM frank_raw f JOIN src_b.main.review_keys k USING(review_key)
            WHERE k.min_app<>k.max_app OR k.min_user<>k.max_user
        UNION SELECT f.review_key FROM src_a.main.valid_raw r JOIN frank_raw f USING(review_key) WHERE r.app_id<>f.app_id
        UNION SELECT f.review_key FROM src_b.main.valid_raw r JOIN frank_raw f USING(review_key) WHERE r.app_id<>f.app_id""")
    con.execute("""CREATE TABLE ambiguous_frank_pairs AS SELECT DISTINCT f.user_key,f.app_id
        FROM frank_raw f JOIN ambiguous_frank_review_keys k USING(review_key)""")
    con.execute("""CREATE TABLE comparison AS SELECT f.user_key,f.app_id,f.positive,c.linux,
        bad.user_key IS NOT NULL review_app_mapping_ambiguous,
        coalesce(a.final_train_reviews,0) a_reviews,coalesce(b.final_train_reviews,0) b_reviews,
        coalesce(a.final_train_positive,0) a_positive,coalesce(b.final_train_positive,0) b_positive
        FROM frank f JOIN test_users t USING(user_key) JOIN catalog c USING(app_id)
        LEFT JOIN a USING(app_id) LEFT JOIN b USING(app_id)
        LEFT JOIN ambiguous_frank_pairs bad ON f.user_key=bad.user_key AND f.app_id=bad.app_id
        WHERE c.reviewed""")
    output={"scope":"Common fixed test roster; source support is diagnostic, never silently drops users from primary evaluation",
            "reserved_test_users":87573,"protocol":"Hide exactly one Linux positive, all other reviewed games may be inputs",
            "validation":"Same global Steam ID hash rule in both complete-source audits",
            "cohorts":{},
            "safe_evaluation_note":"Remove the union of known original-review-to-app ambiguities from frank before selecting query/target. Keep the original 87,573-user training exclusion superset in both sources; safe evaluation is a subset, so no further training rows need to be restored."}
    output['source_identity_checks']={name:json.loads((EDA/'reports'/f'{name}_frank_eda_metrics.json').read_text('utf-8'))['whole_account_and_cross_review_identity_checks_passed']
        for name in ['artermiloff_2024','kieran_original']}
    output['both_sources_are_valid_for_exact_account_evaluation']=all(output['source_identity_checks'].values())
    output['valid_primary_training_source']='kieran_original'
    output['invalid_source_note']='Artermiloff integer-looking original CSV author IDs have float64 rounding. Its account counts, split counts, validation buckets and support counts are nominal diagnostics, not a certified train/test construction. Do not interpret a zero nominal ID intersection as independence.'
    result=con.execute("""SELECT
        (SELECT count(*) FROM ambiguous_frank_review_keys) ambiguous_original_review_ids_in_frank,
        count(*) FILTER(WHERE review_app_mapping_ambiguous) ambiguous_pairs_in_fixed_test_histories,
        count(DISTINCT user_key) FILTER(WHERE review_app_mapping_ambiguous) fixed_test_users_with_any_ambiguous_pair,
        count(*) FILTER(WHERE review_app_mapping_ambiguous AND positive AND linux) ambiguous_possible_linux_target_pairs
        FROM comparison""")
    output['evaluation_mapping_audit']=dict(zip([d[0] for d in result.description],result.fetchone()))
    output['positive_training_histories']={}
    for alias,name in [('a','artermiloff_2024'),('b','kieran_original')]:
        result=con.execute(f"""SELECT count(*) FILTER(WHERE n_positive>=2) users_with_at_least_2_positive_games,
            count(*) FILTER(WHERE n_positive>=5) users_with_at_least_5_positive_games
            FROM src_{alias}.main.user_stats WHERE NOT is_test AND NOT is_validation""")
        output['positive_training_histories'][name]=dict(zip([d[0] for d in result.description],result.fetchone()))
    conditions={'fixed_A_reviewed_catalog':'true',
                'review_support_in_both_final_training_sources':'a_reviews>0 AND b_reviews>0',
                'positive_support_in_both_1':'a_positive>=1 AND b_positive>=1',
                'positive_support_in_both_5':'a_positive>=5 AND b_positive>=5',
                'positive_support_in_both_20':'a_positive>=20 AND b_positive>=20'}
    conditions.update({
        'safe_mapping_fixed_catalog':'NOT review_app_mapping_ambiguous',
        'safe_mapping_review_support_in_both':'NOT review_app_mapping_ambiguous AND a_reviews>0 AND b_reviews>0',
        'safe_mapping_positive_support_in_both_5':'NOT review_app_mapping_ambiguous AND a_positive>=5 AND b_positive>=5',
        'safe_mapping_kieran_review_support':'NOT review_app_mapping_ambiguous AND b_reviews>0',
        'safe_mapping_kieran_positive_support_5':'NOT review_app_mapping_ambiguous AND b_positive>=5'})
    for label,condition in conditions.items():
        output['cohorts'][label]=many(con,f"""WITH per_user AS (
            SELECT user_key,count(*) FILTER(WHERE {condition}) supported_games,
                count(*) FILTER(WHERE {condition} AND positive) supported_positive_games,
                count(*) FILTER(WHERE {condition} AND positive AND linux) target_pairs
            FROM comparison GROUP BY user_key)
            SELECT k minimum_inputs,count(*) FILTER(WHERE supported_games>=k+1 AND target_pairs>0) potential_users,
                count(*) FILTER(WHERE supported_games>=k+1 AND target_pairs>0 AND supported_positive_games>=2) potential_users_with_positive_query,
                count(*) FILTER(WHERE supported_games>=k+1 AND target_pairs>0 AND supported_positive_games=1) potential_users_with_only_negative_query,
                coalesce(sum(target_pairs) FILTER(WHERE supported_games>=k+1 AND target_pairs>0),0) possible_target_pairs
            FROM per_user CROSS JOIN (VALUES (1),(3),(5),(10)) minimum(k) GROUP BY k ORDER BY k""")
    baseline=output['cohorts']['fixed_A_reviewed_catalog']
    assert [r['potential_users'] for r in baseline]==[87573,39456,24577,11647]
    path=EDA/'reports/large_original_frank_comparison_metrics.json'
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))
    con.close()


if __name__=='__main__':
    main()
