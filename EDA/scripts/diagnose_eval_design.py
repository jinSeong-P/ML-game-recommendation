"""Diagnose evaluation-design risks of the Kieran -> frank protocol.

Reads the existing EDA work database read-only and publishes aggregates only:
target popularity by split, unobserved positives, time ranges, quarantine
structure, Item-KNN input scope and description coverage. No model fitting,
split files or account-level exports. Text coverage reads the archived A/C
Parquet files under legacy/, because the main reference holds flags only.
"""
import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import duckdb

from eda_large_review_frank import EDA, digest, many, one, quote

TOP_K = (10, 50, 500)


def prepare_cohorts(con, quarantine_path, min_games=6):
    """Build the frank input>=5 cohort used by the published eligibility counts.

    min_games counts the hidden target too, so 6 means at least 5 other inputs.
    """
    con.execute(f"CREATE TEMP TABLE quar AS SELECT * FROM read_parquet({quote(quarantine_path)})")
    con.execute("CREATE TEMP TABLE amb AS SELECT DISTINCT user_key,app_id FROM frank_raw JOIN quar USING(review_key)")
    con.execute("""CREATE TEMP TABLE test_hist AS SELECT f.user_key,f.app_id,f.positive,c.linux
        FROM frank f JOIN test_users USING(user_key) JOIN catalog c USING(app_id)
        ANTI JOIN amb ON f.user_key=amb.user_key AND f.app_id=amb.app_id WHERE c.reviewed""")
    con.execute("""CREATE TEMP TABLE validation_hist AS SELECT c.user_key,c.app_id,c.positive,cat.linux
        FROM clean c JOIN user_stats u USING(user_key) JOIN catalog cat USING(app_id)
        WHERE cat.reviewed AND u.is_validation AND NOT u.is_test""")
    con.execute("""CREATE TEMP TABLE reserved_hist AS SELECT c.user_key,c.app_id,c.positive,cat.linux
        FROM clean c SEMI JOIN test_users USING(user_key) JOIN catalog cat USING(app_id)
        WHERE cat.reviewed""")
    for prefix, history in [('test', 'test_hist'), ('validation', 'validation_hist'), ('reserved', 'reserved_hist')]:
        con.execute(f"""CREATE TEMP TABLE {prefix}_cohort AS SELECT user_key,count(*) n,
            count(*) FILTER(WHERE positive) n_positive,
            count(*) FILTER(WHERE positive AND linux) n_targets
            FROM {history} GROUP BY user_key HAVING n_targets>0 AND n>={int(min_games)}""")
    con.execute("""CREATE TEMP TABLE linux_rank AS SELECT app_id,train_positive,
        row_number() OVER(ORDER BY train_positive DESC,app_id) rank_ FROM (
        SELECT c.app_id,coalesce(a.final_train_positive,0) train_positive
        FROM catalog c LEFT JOIN app_stats a USING(app_id) WHERE c.reviewed AND c.linux)""")


def target_concentration(con):
    """Share of a uniformly drawn per-user Linux target falling on top-k popular games."""
    def targets(history, cohort):
        columns = ",".join(f"sum(w) FILTER(WHERE rank_<={k})/sum(w) top_{k}" for k in TOP_K)
        return one(con, f"""SELECT {columns} FROM (SELECT 1.0/c.n_targets w,l.rank_
            FROM {history} h JOIN {cohort} c USING(user_key) JOIN linux_rank l USING(app_id)
            WHERE h.positive AND h.linux)""")
    columns = ",".join(f"sum(train_positive) FILTER(WHERE rank_<={k})*1.0/sum(train_positive) top_{k}" for k in TOP_K)
    catalog = con.execute("SELECT count(*) FROM linux_rank").fetchone()[0]
    return {'train_positive_reviews': one(con, f"SELECT {columns} FROM linux_rank"),
            'validation_targets': targets('validation_hist', 'validation_cohort'),
            'frank_targets': targets('test_hist', 'test_cohort'),
            'reserved_account_kieran_targets': targets('reserved_hist', 'reserved_cohort'),
            'uniform_over_linux_catalog': {f'top_{k}': min(1.0, k/catalog) for k in TOP_K}}


def unobserved_positives(con):
    """Compare frank histories of the input>=5 cohort with the same accounts in Kieran."""
    con.execute("""CREATE TEMP TABLE kieran_side AS SELECT c.user_key,c.app_id,c.positive,cat.linux
        FROM clean c SEMI JOIN test_cohort USING(user_key) JOIN catalog cat USING(app_id) WHERE cat.reviewed""")
    con.execute("""CREATE TEMP TABLE history_join AS SELECT coalesce(k.user_key,f.user_key) user_key,
        count(k.app_id) n_kieran,count(f.app_id) n_frank,
        count(*) FILTER(WHERE k.app_id IS NOT NULL AND f.app_id IS NOT NULL) n_both,
        count(*) FILTER(WHERE k.app_id IS NOT NULL AND f.app_id IS NULL) kieran_only,
        count(*) FILTER(WHERE k.app_id IS NOT NULL AND f.app_id IS NULL AND k.linux AND k.positive) kieran_only_linux_positive,
        count(*) FILTER(WHERE f.app_id IS NOT NULL AND k.app_id IS NULL) frank_only
        FROM kieran_side k FULL JOIN (SELECT user_key,app_id FROM test_hist SEMI JOIN test_cohort USING(user_key)) f
        USING(user_key,app_id) GROUP BY 1""")
    return one(con, """SELECT count(*) users,count(*) FILTER(WHERE n_kieran>0) users_in_kieran,
        quantile_disc(n_frank,[0.25,0.5,0.75,0.9]) frank_history_quantiles,
        quantile_disc(n_kieran,[0.25,0.5,0.75,0.9]) kieran_history_quantiles,
        sum(n_both)*1.0/sum(n_kieran) kieran_pairs_also_in_frank_share,
        quantile_disc(kieran_only_linux_positive,[0.25,0.5,0.75,0.9]) kieran_only_linux_positive_quantiles,
        avg((kieran_only_linux_positive>0)::INT) users_with_kieran_only_linux_positive_share,
        sum(kieran_only) kieran_only_pairs,sum(kieran_only_linux_positive) kieran_only_linux_positive_pairs,
        sum(frank_only) frank_only_pairs FROM history_join""")


def time_ranges(con):
    day = "'%Y-%m-%d'"
    return {
        'kieran_created': one(con, f"""SELECT strftime(to_timestamp(min(timestamp_created)),{day}) first_day,
            strftime(to_timestamp(max(timestamp_created)),{day}) last_day FROM valid_raw"""),
        'frank_created': one(con, f"""SELECT strftime(to_timestamp(min(timestamp_created)),{day}) first_day,
            strftime(to_timestamp(quantile_disc(timestamp_created,0.5)),{day}) median_day,
            strftime(to_timestamp(max(timestamp_created)),{day}) last_day FROM frank_raw"""),
        'frank_rows_by_year': many(con, """SELECT year(to_timestamp(timestamp_created)) review_year,count(*) n_rows
            FROM frank_raw GROUP BY 1 ORDER BY 1"""),
        'frank_target_share_after_kieran_end': con.execute("""SELECT sum(w) FILTER(WHERE r.timestamp_created>(SELECT max(timestamp_created) FROM valid_raw))/sum(w)
            FROM (SELECT h.user_key,h.app_id,1.0/c.n_targets w FROM test_hist h JOIN test_cohort c USING(user_key)
                  WHERE h.positive AND h.linux) t JOIN frank_raw r USING(user_key,app_id)""").fetchone()[0],
    }


def quarantine_structure(con, legacy, table_path):
    """Describe app groups erased by whole-review-ID quarantine and export the groups."""
    con.execute("""CREATE TEMP TABLE bad_ids AS SELECT review_key FROM review_keys
        WHERE min_app<>max_app OR min_user<>max_user""")
    con.execute("""CREATE TEMP TABLE id_groups AS SELECT review_key,count(*) n_rows,
        list_sort(list(DISTINCT app_id)) apps FROM valid_raw SEMI JOIN bad_ids USING(review_key) GROUP BY 1""")
    con.execute("CREATE TEMP TABLE alias_groups AS SELECT apps,count(*) review_ids,sum(n_rows) n_rows FROM id_groups GROUP BY apps")
    con.execute("""CREATE TEMP TABLE alias_apps AS SELECT a.apps,u.app_id FROM alias_groups a,
        LATERAL (SELECT unnest(a.apps) app_id) u""")
    titles = (f"(SELECT app_id,title FROM read_parquet({quote(legacy/'A_games.parquet')}) UNION ALL "
              f"SELECT appid,name FROM read_parquet({quote(legacy/'C_games_2025.parquet')}))")
    con.execute(f"""CREATE TEMP TABLE alias_summary AS SELECT g.apps,g.review_ids,g.n_rows,
        count(DISTINCT x.app_id) FILTER(WHERE c.reviewed) catalog_apps,
        count(DISTINCT x.app_id) FILTER(WHERE c.reviewed AND c.linux) linux_apps,
        string_agg(DISTINCT t.title,' / ') titles FROM alias_groups g JOIN alias_apps x USING(apps)
        LEFT JOIN catalog c USING(app_id) LEFT JOIN {titles} t USING(app_id)
        GROUP BY g.apps,g.review_ids,g.n_rows""")
    con.execute(f"""COPY (SELECT array_to_string(apps,'|') app_ids,titles,review_ids,n_rows raw_rows,catalog_apps,linux_apps
        FROM alias_summary ORDER BY n_rows DESC,app_ids) TO {quote(table_path)} (HEADER,DELIMITER ',')""")
    result = one(con, """SELECT count(*) app_groups,sum(review_ids) review_ids,sum(n_rows) raw_rows,
        count(*) FILTER(WHERE catalog_apps=1) groups_with_one_catalog_app,
        count(*) FILTER(WHERE catalog_apps>1) groups_with_several_catalog_apps,
        count(*) FILTER(WHERE catalog_apps=0) groups_without_catalog_app,
        coalesce(sum(review_ids) FILTER(WHERE catalog_apps=1),0) review_ids_with_one_catalog_app,
        count(*) FILTER(WHERE linux_apps>0) groups_with_linux_catalog_app FROM alias_summary""")
    result.update(one(con, "SELECT count(DISTINCT app_id) apps_in_groups FROM alias_apps"))
    result.update(one(con, """SELECT count(*) test_cohort_history_pairs,count(DISTINCT user_key) test_cohort_users,
        count(*) FILTER(WHERE positive AND linux) test_cohort_linux_positive_pairs
        FROM test_hist SEMI JOIN test_cohort USING(user_key) SEMI JOIN alias_apps USING(app_id)"""))
    return result


def item_knn_scope(con):
    con.execute("""CREATE TEMP TABLE train_degree AS SELECT c.user_key,count(*) d FROM clean c
        JOIN user_stats u USING(user_key) JOIN catalog cat USING(app_id)
        WHERE c.positive AND cat.reviewed AND NOT u.is_test AND NOT u.is_validation GROUP BY 1""")
    return one(con, """SELECT count(*) catalog_positive_users,sum(d) catalog_positive_pairs,
        count(*) FILTER(WHERE d>=2) users_with_two_or_more,sum(d) FILTER(WHERE d>=2) pairs_from_users_with_two_or_more,
        sum(d*d) sum_squared_user_degree,max(d) max_user_degree FROM train_degree""")


def text_coverage(con, legacy):
    con.execute(f"""CREATE TEMP TABLE text_lengths AS SELECT c.app_id,c.linux,c.reviewed,
        length(trim(coalesce(a.description,''))) a_description,
        length(trim(coalesce(g.about_the_game,''))) c_about,
        length(trim(coalesce(g.detailed_description,''))) c_detailed,
        length(trim(coalesce(g.short_description,''))) c_short,
        trim(a.description)=trim(g.short_description) a_equals_c_short,
        g.linux c_linux,g.appid IS NOT NULL in_c
        FROM catalog c LEFT JOIN read_parquet({quote(legacy/'A_games_metadata.parquet')}) a USING(app_id)
        LEFT JOIN read_parquet({quote(legacy/'C_games_2025.parquet')}) g ON c.app_id=g.appid""")
    result = {}
    for name, condition in [('input_catalog', 'reviewed'), ('linux_catalog', 'reviewed AND linux')]:
        result[name] = one(con, f"""SELECT count(*) apps,count(*) FILTER(WHERE a_description>0) a_description_apps,
            count(*) FILTER(WHERE c_about>0) c_about_apps,count(*) FILTER(WHERE c_short>0) c_short_apps,
            count(*) FILTER(WHERE a_description>0 OR c_short>0) a_or_c_short_apps,
            count(*) FILTER(WHERE a_description>0 OR c_about>0 OR c_detailed>0 OR c_short>0) any_text_apps,
            quantile_disc(a_description,[0.1,0.5,0.9]) FILTER(WHERE a_description>0) a_description_length_quantiles,
            quantile_disc(c_about,[0.1,0.5,0.9]) FILTER(WHERE c_about>0) c_about_length_quantiles
            FROM text_lengths WHERE {condition}""")
    result['a_description_vs_c_short'] = one(con, """SELECT count(*) both_present,count(*) FILTER(WHERE a_equals_c_short) identical
        FROM text_lengths WHERE reviewed AND a_description>0 AND c_short>0""")
    result['linux_flag_a_vs_c'] = many(con, """SELECT linux a_linux,c_linux,count(*) apps FROM text_lengths
        WHERE reviewed GROUP BY 1,2 ORDER BY 1,2""")
    return result


def main():
    metrics_path = EDA/'reports/kieran_frank_eda_metrics.json'
    full_path = EDA/'reports/kieran_original_frank_eda_metrics.json'
    work = EDA/'data/processed/kieran_original/eda_work.duckdb'
    legacy = EDA/'legacy/data/processed'
    full = json.loads(full_path.read_text('utf-8'))
    summary = json.loads(metrics_path.read_text('utf-8'))
    table = EDA/'reports/tables/17_quarantined_app_groups.csv'
    with tempfile.TemporaryDirectory() as temp:
        con = duckdb.connect(str(work), read_only=True)
        con.execute("SET threads=6")
        con.execute("SET memory_limit='10GB'")
        con.execute(f"SET temp_directory={quote(temp)}")
        clean_pairs = con.execute("SELECT count(*) FROM clean").fetchone()[0]
        if clean_pairs != full['clean_source']['pairs']:
            raise ValueError('EDA work database differs from the published full-source metrics')
        prepare_cohorts(con, EDA/'data/reference/review_app_quarantine.parquet')
        cohort = next(r for r in summary['cohorts']['safe_mapping_fixed_catalog'] if r['minimum_inputs']==5)
        expected = one(con, """SELECT count(*) potential_users,count(*) FILTER(WHERE n_positive>=2) potential_users_with_positive_query,
            sum(n_targets) possible_target_pairs FROM test_cohort""")
        if any(expected[k] != cohort[k] for k in expected):
            raise ValueError(f'Cohort differs from published eligibility: {expected}')
        result = {
            'scope': 'Aggregate evaluation-design diagnostics. No model fit, split files or account-level exports.',
            'checked_utc': datetime.now(timezone.utc).isoformat(),
            'source_metric_sha256': digest(full_path),
            'reference_manifest_sha256': digest(EDA/'data/reference/manifest.json'),
            'code_sha256': digest(__file__),
            'cohort_definition': 'frank input>=5 and validation/reserved-account analogues: at least 5 catalog reviews remain after hiding one positive Linux game; each user contributes one uniformly drawn target',
            'cohort_sizes': {name: one(con, f"SELECT count(*) users,count(*) FILTER(WHERE n_positive>=2) users_with_positive_query,quantile_disc(n,[0.25,0.5,0.75,0.9]) history_quantiles,quantile_disc(n_positive-1,[0.25,0.5,0.75]) positive_query_quantiles,count(*) FILTER(WHERE n_positive-1>=5) users_with_five_positive_queries FROM {name}")
                             for name in ['test_cohort', 'validation_cohort', 'reserved_cohort']},
            'frank_sampling': one(con, """SELECT count(*) apps,max(n) max_rows_per_app,count(*) FILTER(WHERE n>=100) apps_at_cap,
                sum(n) FILTER(WHERE n>=100) rows_at_cap,sum(n) total_rows FROM (SELECT app_id,count(*) n FROM frank_raw GROUP BY 1)"""),
            'frank_catalog_scope': one(con, """SELECT count(*) test_account_pairs,count(*) FILTER(WHERE c.reviewed) in_input_catalog
                FROM frank f SEMI JOIN test_users USING(user_key) LEFT JOIN catalog c USING(app_id)"""),
            'time_ranges': time_ranges(con),
            'target_concentration': target_concentration(con),
            'unobserved_positives': unobserved_positives(con),
            'account_history_degree': many(con, """SELECT is_test,is_validation,count(*) users,quantile_disc(n_reviews,[0.5,0.9,0.99]) history_quantiles
                FROM user_stats GROUP BY 1,2 ORDER BY 1,2"""),
            'item_knn_scope': item_knn_scope(con),
            'quarantine_structure': quarantine_structure(con, legacy, table),
            'linux_training_support': one(con, """SELECT count(*) linux_apps,count(*) FILTER(WHERE train_positive=0) without_train_positive,
                count(*) FILTER(WHERE train_positive<5) under_five_train_positive,
                count(*) FILTER(WHERE coalesce(a.all_reviews,0)=0) without_kieran_review,
                count(*) FILTER(WHERE coalesce(a.all_reviews,0)=0 AND f.app_id IS NOT NULL) without_kieran_review_but_in_frank
                FROM linux_rank l LEFT JOIN app_stats a USING(app_id)
                LEFT JOIN (SELECT DISTINCT app_id FROM frank) f USING(app_id)"""),
            'catalog_last_release': con.execute(f"SELECT strftime(max(date_release),'%Y-%m-%d') FROM read_parquet({quote(legacy/'A_games.parquet')})").fetchone()[0],
            'text_coverage': text_coverage(con, legacy),
        }
        con.close()
    output = EDA/'reports/kieran_frank_eval_diagnostics.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 default=lambda v: float(v) if isinstance(v, Decimal) else str(v))+'\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
