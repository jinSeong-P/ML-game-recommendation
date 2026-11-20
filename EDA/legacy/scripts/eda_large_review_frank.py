"""Full original-source EDA and whole-account exclusion against frozen frank.

Consumes complete canonical Parquet shards produced by the two original-source
ingesters. Exports aggregates only; internal DuckDB files stay in ignored data/.
No model fitting, final train/test files, or individual account exports.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


EDA = Path(__file__).resolve().parents[1]
BASE = 76561197960265728
VERSION = "large-original-frank-eda-v1"


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def one(con, sql):
    result = con.execute(sql)
    keys = [d[0] for d in result.description]
    row = result.fetchone()
    return dict(zip(keys, row))


def many(con, sql):
    result = con.execute(sql)
    keys = [d[0] for d in result.description]
    return [dict(zip(keys, row)) for row in result.fetchall()]


def steam_valid(column):
    return (f"regexp_full_match({column}, '[0-9]{{17}}') AND "
            f"try_cast({column} AS UBIGINT) BETWEEN {BASE} AND {BASE + 2**32 - 1}")


def prepare_frank(con, processed):
    folder = processed / "candidate_identity/frankjc2022"
    manifest = json.loads((folder / "projection_manifest.json").read_text("utf-8"))
    path = folder / "reviews_identity_columns.parquet"
    expected = manifest["files"]["reviews"]
    if digest(path) != expected["local_projection"]["sha256"]:
        raise ValueError("Frozen frank projection checksum mismatch")
    con.execute(f"CREATE OR REPLACE VIEW frank_input AS SELECT * FROM read_parquet({quote(path)})")
    if con.execute("SELECT count(*) FROM frank_input").fetchone()[0] != expected["rows"]:
        raise ValueError("Frozen frank row count mismatch")
    bad = con.execute(f"SELECT count(*) FROM frank_input WHERE NOT coalesce(({steam_valid('author_steamid')}),false)").fetchone()[0]
    if bad:
        raise ValueError("Noncanonical frank Steam ID")
    con.execute("""CREATE OR REPLACE TABLE frank_raw AS SELECT
        cast(author_steamid AS UBIGINT) user_key, cast(appid AS BIGINT) app_id,
        cast(recommendationid AS UBIGINT) review_key, voted_up positive,
        timestamp_created, timestamp_updated FROM frank_input""")
    con.execute("""CREATE OR REPLACE TABLE frank AS SELECT user_key,app_id,
        bool_or(positive) positive FROM frank_raw GROUP BY user_key,app_id
        HAVING bool_or(positive)=bool_and(positive)""")
    # The comparison catalog is frozen to the preceding cross-source audit.
    con.execute(f"""CREATE OR REPLACE TABLE catalog AS SELECT g.app_id,g.linux,
        r.app_id IS NOT NULL reviewed FROM read_parquet({quote(processed / 'A_games.parquet')}) g
        LEFT JOIN (SELECT DISTINCT app_id FROM read_parquet({quote(processed / 'A_recommendations.parquet')})) r USING(app_id)""")
    con.execute(f"""CREATE OR REPLACE TABLE content_catalog AS WITH a AS (
        SELECT app_id,length(trim(coalesce(description,'')))>0 has_description,
            coalesce(array_length(tags),0)>0 has_tags
        FROM read_parquet({quote(processed / 'A_games_metadata.parquet')})),
        c AS (SELECT appid app_id,
            length(trim(coalesce(nullif(about_the_game,''),nullif(detailed_description,''),short_description,'')))>0 has_description,
            coalesce(trim(tags),'') NOT IN ('','[]','{{}}','None','null') has_tags
            FROM read_parquet({quote(processed / 'C_games_2025.parquet')}))
        SELECT coalesce(a.app_id,c.app_id) app_id,a.app_id IS NOT NULL in_A_metadata,
            c.app_id IS NOT NULL in_C_metadata,
            coalesce(a.has_description,false) OR coalesce(c.has_description,false) has_description,
            coalesce(a.has_tags,false) OR coalesce(c.has_tags,false) has_tags
        FROM a FULL OUTER JOIN c USING(app_id)""")
    if con.execute("SELECT count(*)<>count(DISTINCT app_id) FROM content_catalog").fetchone()[0]:
        raise ValueError("Metadata catalog has duplicate App IDs; refuse weighted coverage inflation")
    con.execute("""CREATE OR REPLACE TABLE test_users AS SELECT f.user_key
        FROM frank f JOIN catalog c USING(app_id) WHERE c.reviewed
        GROUP BY user_key HAVING count(*)>=2 AND count(*) FILTER(WHERE f.positive AND c.linux)>0""")
    count = con.execute("SELECT count(*) FROM test_users").fetchone()[0]
    if count != 87573:
        raise ValueError(f"Frozen test cohort changed: {count}")
    return {"commit": manifest["commit"],
            "projection_sha256": expected["local_projection"]["sha256"],
            "raw_rows": expected["rows"], "reserved_test_users": count,
            "cohort": "Fixed A-reviewed input apps; at least one other review after hiding one positive A-Linux game"}


def build_source(con, glob):
    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM read_parquet({quote(glob)}, union_by_name=true)")
    valid = f"({steam_valid('user_id')}) AND app_id>0 AND positive IS NOT NULL AND regexp_full_match(review_id,'[0-9]+') AND try_cast(review_id AS UBIGINT)>0"
    quality = one(con, f"""SELECT count(*) raw_rows,
        count(*) FILTER(WHERE NOT coalesce(({steam_valid('user_id')}),false)) invalid_or_missing_users,
        count(*) FILTER(WHERE app_id IS NULL OR app_id<=0) invalid_or_missing_apps,
        count(*) FILTER(WHERE positive IS NULL) missing_or_unrecognized_labels,
        count(*) FILTER(WHERE NOT coalesce(regexp_full_match(review_id,'[0-9]+') AND try_cast(review_id AS UBIGINT)>0,false)) invalid_or_missing_review_ids,
        count(*) FILTER(WHERE NOT coalesce(({valid}),false)) quarantined_rows,
        count(*) FILTER(WHERE timestamp_created IS NULL) missing_created_times,
        count(*) FILTER(WHERE timestamp_updated IS NULL) missing_updated_times,
        count(*) FILTER(WHERE timestamp_updated<timestamp_created) updated_before_created,
        count(*) FILTER(WHERE playtime_forever<0) negative_playtime_rows,
        min(timestamp_created) min_created_unix,max(timestamp_created) max_created_unix,
        min(timestamp_updated) min_updated_unix,max(timestamp_updated) max_updated_unix
        FROM raw""")
    con.execute(f"""CREATE OR REPLACE TABLE valid_raw AS SELECT
        cast(user_id AS UBIGINT) user_key,app_id,cast(review_id AS UBIGINT) review_key,
        positive,timestamp_created,timestamp_updated,playtime_forever,language
        FROM raw WHERE {valid}""")
    log("Grouping original review identifiers")
    con.execute("""CREATE OR REPLACE TABLE review_keys AS SELECT review_key,count(*) n,
        min(user_key) min_user,max(user_key) max_user,min(app_id) min_app,max(app_id) max_app,
        bool_and(positive) min_label,bool_or(positive) max_label FROM valid_raw GROUP BY review_key""")
    quality.update(one(con,"""SELECT sum(n) identity_valid_rows,count(*) unique_review_ids,
        sum(n-1) repeated_original_review_id_rows,
        count(*) FILTER(WHERE min_user<>max_user) review_ids_with_conflicting_owner,
        count(*) FILTER(WHERE min_app<>max_app) review_ids_with_conflicting_app,
        count(*) FILTER(WHERE min_user<>max_user OR min_app<>max_app) review_ids_with_conflicting_owner_or_app,
        coalesce(sum(n) FILTER(WHERE min_user<>max_user OR min_app<>max_app),0) rows_with_conflicting_review_identity,
        count(*) FILTER(WHERE min_label<>max_label) review_ids_with_changed_labels FROM review_keys"""))
    if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='frank_raw'").fetchone()[0]:
        con.execute("""CREATE OR REPLACE TABLE cross_review_conflicts AS SELECT DISTINCT r.review_key
            FROM valid_raw r JOIN frank_raw f USING(review_key)
            WHERE r.user_key<>f.user_key OR r.app_id<>f.app_id""")
    else:
        con.execute("CREATE OR REPLACE TABLE cross_review_conflicts(review_key UBIGINT)")
    quality.update(one(con,"""SELECT count(*) cross_source_ambiguous_review_ids,
        count(*) FILTER(WHERE k.min_user=k.max_user AND k.min_app=k.max_app) additional_cross_source_review_ids_quarantined
        FROM cross_review_conflicts x JOIN review_keys k USING(review_key)"""))
    con.execute("""CREATE OR REPLACE VIEW compatible_raw AS SELECT r.* FROM valid_raw r
        JOIN review_keys k USING(review_key) ANTI JOIN cross_review_conflicts x USING(review_key)
        WHERE k.min_user=k.max_user AND k.min_app=k.max_app""")
    log("Grouping complete user-game histories")
    con.execute("""CREATE OR REPLACE TABLE pair_groups AS SELECT user_key,app_id,
        count(*) raw_rows,bool_and(positive) min_label,bool_or(positive) max_label
        FROM compatible_raw GROUP BY user_key,app_id""")
    quality.update(one(con,"""SELECT count(*) distinct_pairs_before_label_conflict,
        sum(raw_rows-CASE WHEN min_label<>max_label THEN 2 ELSE 1 END) same_label_duplicate_rows,
        count(*) FILTER(WHERE min_label<>max_label) conflicting_label_pairs,
        coalesce(sum(raw_rows) FILTER(WHERE min_label<>max_label),0) rows_in_conflicting_label_pairs
        FROM pair_groups"""))
    con.execute("""CREATE OR REPLACE TABLE clean AS SELECT user_key,app_id,max_label positive
        FROM pair_groups WHERE min_label=max_label""")
    return quality


def degree_summary(con, where="true"):
    return one(con,f"""SELECT count(*) users,coalesce(sum(n_reviews),0) pairs,coalesce(sum(n_positive),0) positives,
        quantile_disc(n_reviews,[0.25,0.5,0.75,0.9,0.99,1]) history_quantiles,
        count(*) FILTER(WHERE n_reviews=1) single_game_users,
        count(*) FILTER(WHERE n_reviews>=5) users_with_at_least_5_games,
        count(*) FILTER(WHERE n_reviews>=10) users_with_at_least_10_games,
        count(*) FILTER(WHERE n_positive=n_reviews) positive_only_users,
        count(*) FILTER(WHERE n_positive=0) negative_only_users,
        count(*) FILTER(WHERE n_positive>0 AND n_positive<n_reviews) mixed_label_users
        FROM user_stats WHERE {where}""")


def cohort_coverage(con, support, positive_support):
    """Hide exactly ONE supported positive target; the other games are inputs.

    Supported games >= k+1 subtracts that one target exactly, even if several
    other positive Linux games are also valid alternative targets. Counts for
    simultaneously hiding all positives require a different protocol.
    """
    result = {}
    for split, column, pcolumn in [("after_test_exclusion", "post_test_reviews", "post_test_positive"),
                                     ("after_validation_reservation", "final_train_reviews", "final_train_positive")]:
        condition = f"a.{column}>={support}"
        if positive_support:
            condition = f"a.{pcolumn}>={support}"
        rows = many(con,f"""WITH per_user AS (
            SELECT f.user_key,count(*) FILTER(WHERE {condition}) supported_games,
                count(*) FILTER(WHERE {condition} AND f.positive) supported_positive_games,
                count(*) FILTER(WHERE {condition} AND f.positive AND c.linux) target_pairs
            FROM frank f JOIN test_users u USING(user_key)
            JOIN catalog c USING(app_id) LEFT JOIN app_stats a USING(app_id)
            WHERE c.reviewed GROUP BY f.user_key)
            SELECT k minimum_inputs,count(*) FILTER(WHERE supported_games>=k+1 AND target_pairs>0) potential_users,
                count(*) FILTER(WHERE supported_games>=k+1 AND target_pairs>0 AND supported_positive_games>=2) potential_users_with_positive_query,
                count(*) FILTER(WHERE supported_games>=k+1 AND target_pairs>0 AND supported_positive_games=1) potential_users_with_only_negative_query,
                coalesce(sum(target_pairs) FILTER(WHERE supported_games>=k+1 AND target_pairs>0),0) possible_target_pairs
            FROM per_user CROSS JOIN (VALUES (1),(3),(5),(10)) minimum(k) GROUP BY k ORDER BY k""")
        result[split] = rows
    return result


def analyze(con):
    log("Counting source/test account, pair and original-review overlaps")
    overlap = one(con,"""SELECT count(DISTINCT r.review_key) common_original_review_ids,
        count(DISTINCT r.review_key) FILTER(WHERE r.user_key<>f.user_key) common_review_ids_with_different_owner,
        count(DISTINCT r.review_key) FILTER(WHERE r.app_id<>f.app_id) common_review_ids_with_different_app,
        count(DISTINCT r.review_key) FILTER(WHERE r.user_key<>f.user_key OR r.app_id<>f.app_id) common_review_ids_with_different_user_or_app,
        count(DISTINCT r.review_key) FILTER(WHERE r.user_key=f.user_key AND r.app_id=f.app_id AND r.positive<>f.positive) common_review_ids_with_changed_label,
        count(DISTINCT r.review_key) FILTER(WHERE r.timestamp_created IS DISTINCT FROM f.timestamp_created) common_review_ids_with_changed_created_time
        FROM valid_raw r JOIN frank_raw f USING(review_key)""")
    overlap.update(one(con,"""SELECT count(DISTINCT r.review_key) retained_clean_review_ids_with_incompatible_owner_or_app
        FROM compatible_raw r SEMI JOIN clean s USING(user_key,app_id)
        JOIN frank_raw f USING(review_key) WHERE r.user_key<>f.user_key OR r.app_id<>f.app_id"""))
    overlap.update(one(con,"""SELECT count(DISTINCT s.user_key) common_clean_steam_users
        FROM clean s SEMI JOIN frank f USING(user_key)"""))
    overlap.update(one(con,"""SELECT count(*) common_clean_user_app_pairs,
        count(*) FILTER(WHERE s.positive<>f.positive) common_clean_pairs_with_changed_label
        FROM clean s JOIN frank f USING(user_key,app_id)"""))
    # Stable common Steam-account namespace, independent of source file or order.
    con.execute("""CREATE OR REPLACE TABLE user_stats AS WITH degrees AS (
        SELECT user_key,count(*) n_reviews,count(*) FILTER(WHERE positive) n_positive
        FROM clean GROUP BY user_key)
        SELECT d.*,t.user_key IS NOT NULL is_test,
        cast('0x'||substr(md5('steam-eda-validation-v1:'||cast(d.user_key AS VARCHAR)),1,15) AS UBIGINT)%10=0 is_validation
        FROM degrees d LEFT JOIN test_users t USING(user_key)""")
    log("Aggregating item support before/after account exclusion")
    con.execute("""CREATE OR REPLACE TABLE app_stats AS SELECT c.app_id,
        count(*) all_reviews,count(*) FILTER(WHERE c.positive) all_positive,
        count(*) FILTER(WHERE NOT u.is_test) post_test_reviews,
        count(*) FILTER(WHERE NOT u.is_test AND c.positive) post_test_positive,
        count(*) FILTER(WHERE NOT u.is_test AND NOT u.is_validation) final_train_reviews,
        count(*) FILTER(WHERE NOT u.is_test AND NOT u.is_validation AND c.positive) final_train_positive
        FROM clean c JOIN user_stats u USING(user_key) GROUP BY c.app_id""")
    source = degree_summary(con)
    post = degree_summary(con,"NOT is_test")
    validation = degree_summary(con,"NOT is_test AND is_validation")
    train = degree_summary(con,"NOT is_test AND NOT is_validation")
    for metrics, column, pcolumn in [(source,"all_reviews","all_positive"),
                                      (post,"post_test_reviews","post_test_positive"),
                                      (train,"final_train_reviews","final_train_positive")]:
        metrics.update(one(con,f"""SELECT count(*) FILTER(WHERE {column}>0) apps,
            count(*) FILTER(WHERE {pcolumn}>0) positive_apps,
            quantile_disc({column},[0.25,0.5,0.75,0.9,0.99,1]) FILTER(WHERE {column}>0) app_review_quantiles,
            coalesce(sum({column}) FILTER(WHERE catalog.reviewed),0) reviews_in_fixed_A_input_catalog,
            count(*) FILTER(WHERE {column}>0 AND catalog.reviewed) apps_in_fixed_A_input_catalog,
            count(*) FILTER(WHERE {column}>0 AND catalog.reviewed AND catalog.linux) linux_apps_in_fixed_catalog,
            coalesce(sum({pcolumn}) FILTER(WHERE catalog.reviewed AND catalog.linux),0) positive_reviews_on_fixed_linux_catalog
            FROM app_stats LEFT JOIN catalog USING(app_id)"""))
        metrics['top_50_review_share'] = con.execute(f"SELECT sum(n)/(SELECT sum({column}) FROM app_stats) FROM (SELECT {column} n FROM app_stats ORDER BY n DESC LIMIT 50)").fetchone()[0]
    post['removed_test_users'] = source['users']-post['users']
    post['removed_test_pairs'] = source['pairs']-post['pairs']
    checks = one(con,"""SELECT count(*) remaining_train_test_user_intersection
        FROM user_stats u JOIN test_users t USING(user_key) WHERE NOT u.is_test""")
    checks.update(one(con,"""SELECT count(*) remaining_train_test_user_app_intersection
        FROM clean c JOIN user_stats u USING(user_key) JOIN frank f USING(user_key,app_id)
        JOIN test_users t USING(user_key) WHERE NOT u.is_test"""))
    checks.update(one(con,"""SELECT count(DISTINCT r.review_key) remaining_train_test_original_review_id_intersection
        FROM compatible_raw r SEMI JOIN clean s USING(user_key,app_id) JOIN user_stats u USING(user_key)
        JOIN (SELECT f.review_key FROM frank_raw f JOIN test_users t USING(user_key)) f USING(review_key)
        WHERE NOT u.is_test"""))
    target_coverage = one(con,"""SELECT count(*) possible_target_pairs,count(DISTINCT f.app_id) target_apps,
        count(*) FILTER(WHERE coalesce(a.post_test_reviews,0)=0) targets_without_training_reviews,
        count(*) FILTER(WHERE coalesce(a.post_test_positive,0)=0) targets_without_positive_training_reviews,
        count(*) FILTER(WHERE coalesce(a.post_test_positive,0)<5) targets_with_fewer_than_5_positive_training_users,
        count(*) FILTER(WHERE coalesce(a.post_test_positive,0)<20) targets_with_fewer_than_20_positive_training_users,
        count(*) FILTER(WHERE coalesce(a.final_train_positive,0)=0) targets_without_positive_training_reviews_after_validation
        FROM frank f JOIN test_users t USING(user_key) JOIN catalog c USING(app_id)
        LEFT JOIN app_stats a USING(app_id) WHERE f.positive AND c.reviewed AND c.linux""")
    temporal = many(con,"""SELECT CASE WHEN timestamp_created BETWEEN 0 AND 4102444800
        THEN cast(year(to_timestamp(timestamp_created)) AS VARCHAR) ELSE 'invalid_or_missing' END review_created_year,
        count(*) raw_rows,count(*) FILTER(WHERE positive) positive_rows FROM valid_raw GROUP BY 1 ORDER BY 1""")
    languages = many(con,"SELECT coalesce(language,'<missing>') AS language_name,count(*) AS n_rows FROM valid_raw GROUP BY 1 ORDER BY n_rows DESC")
    playtime = one(con,"""SELECT count(*) FILTER(WHERE playtime_forever IS NOT NULL) rows_with_playtime,
        count(*) FILTER(WHERE playtime_forever=0) zero_playtime_rows,
        count(*) FILTER(WHERE playtime_forever<0) negative_playtime_rows,
        quantile_cont(playtime_forever,[0.25,0.5,0.75,0.9,0.99]) playtime_minutes_quantiles
        FROM valid_raw""")
    coverage = {"any_review_support": cohort_coverage(con,1,False),
                "positive_support_1": cohort_coverage(con,1,True),
                "positive_support_5": cohort_coverage(con,5,True),
                "positive_support_20": cohort_coverage(con,20,True)}
    return {"cross_source_overlap": overlap,"clean_source": source,
            "after_test_exclusion": post,"validation_preview": validation,
            "after_validation_reservation": train,"post_exclusion_checks": checks,
            "heldout_target_item_support": target_coverage,"supported_cohort_sizes": coverage,
            "raw_created_years": temporal,"raw_language_counts": languages,"raw_playtime": playtime}


def metadata_summary(con):
    return one(con,"""SELECT count(*) source_apps,
        count(*) FILTER(WHERE m.in_A_metadata) source_apps_with_A_metadata,
        count(*) FILTER(WHERE m.in_C_metadata) source_apps_with_C_metadata,
        count(*) FILTER(WHERE NOT coalesce(m.in_A_metadata,false) AND m.in_C_metadata) C_only_source_apps,
        count(*) FILTER(WHERE m.app_id IS NULL) source_apps_without_A_or_C_metadata,
        count(*) FILTER(WHERE m.has_description) source_apps_with_nonempty_A_or_C_description,
        count(*) FILTER(WHERE m.has_tags) source_apps_with_nonempty_A_or_C_tags,
        sum(a.final_train_reviews) FILTER(WHERE NOT coalesce(m.has_description,false)) final_train_pairs_without_description,
        sum(a.final_train_reviews) FILTER(WHERE NOT coalesce(m.has_tags,false)) final_train_pairs_without_tags,
        sum(a.final_train_reviews) FILTER(WHERE m.app_id IS NULL) final_train_pairs_without_A_or_C_metadata
        FROM app_stats a LEFT JOIN content_catalog m USING(app_id)""")


def reuse_verified_report(report, manifest_path, shards, table):
    if not report.exists() or not table.exists():
        return False
    previous=json.loads(report.read_text('utf-8'))
    if previous.get('analysis_code_sha256') != digest(__file__) or previous.get('ingestion_manifest_sha256') != digest(manifest_path):
        return False
    expected={p['file']:p for p in previous.get('canonical_inputs',[])}
    if set(expected)!=set(p.name for p in shards):
        return False
    if any(digest(p)!=expected[p.name]['sha256'] for p in shards):
        return False
    if any(digest(EDA/'data/processed'/p['file'])!=p['sha256'] for p in previous['reference_catalog_inputs']):
        return False
    frank=EDA/'data/processed/candidate_identity/frankjc2022/reviews_identity_columns.parquet'
    if digest(frank)!=previous['frank_provenance']['projection_sha256']:
        return False
    return digest(table)==previous.get('app_support_table_sha256')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",choices=["artermiloff_2024","kieran_original"])
    parser.add_argument("--memory",default="8GB")
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    folder = EDA / "data/processed/candidate_identity" / args.source
    manifest_path = folder / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if not (manifest.get("complete") is True or manifest.get("status") == "complete"):
        raise ValueError("Refusing partial source ingestion")
    shards = sorted(folder.glob("reviews_*.parquet"))
    if not shards:
        raise ValueError("No complete canonical source shards")
    report = EDA / "reports" / f"{args.source}_frank_eda_metrics.json"
    table = EDA / "reports/tables" / f"16_{args.source}_app_support.csv"
    if reuse_verified_report(report,manifest_path,shards,table):
        log(f"Reused complete EDA after verifying source, catalog, frank, code and output fingerprints: {args.source}")
        return
    log(f"Starting complete-source EDA: {args.source}, {len(shards)} shards")
    con = duckdb.connect(str(folder / "eda_work.duckdb"))
    con.execute(f"SET memory_limit={quote(args.memory)}")
    con.execute(f"SET threads={args.threads}")
    con.execute("SET preserve_insertion_order=false")
    temp = folder / "duckdb_temp"
    temp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory={quote(temp)}")
    provenance = prepare_frank(con, EDA / "data/processed")
    quality = build_source(con,str(folder / "reviews_*.parquet"))
    row_info = manifest.get('rows',manifest.get('canonical_rows_written'))
    expected_rows = row_info.get('canonical_rows_written') if isinstance(row_info,dict) else row_info
    if quality['raw_rows'] != expected_rows:
        raise ValueError("Ingestion manifest/canonical row count mismatch")
    result = analyze(con)
    result['metadata_coverage'] = metadata_summary(con)
    # Per-file archive inventories stay in ignored local data; publish summary
    # provenance and canonical file fingerprints rather than tens of thousands
    # of repetitive ZIP entries.
    public_manifest = {k:v for k,v in manifest.items() if k not in ['csv_entry_inventory','entry_inventory']}
    result.update({"analysis_version": VERSION,"checked_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code_sha256": digest(__file__),
        "source": args.source,"ingestion_manifest": public_manifest,"ingestion_manifest_sha256": digest(manifest_path),
        "canonical_inputs": [{"file":p.name,"bytes":p.stat().st_size,"sha256":digest(p)} for p in shards],
        "reference_catalog_inputs": [{"file":name,"sha256":digest(EDA/'data/processed'/name)} for name in
            ['A_games.parquet','A_recommendations.parquet','A_games_metadata.parquet','C_games_2025.parquet']],
        "frank_provenance": provenance,"raw_quality": quality,
        "scope": "Complete canonical columns from complete original archive; aggregate EDA and split previews, no model fitting or final split files",
        "validation_rule": "Non-test whole Steam accounts with first 15 hexadecimal MD5 digits of steam-eda-validation-v1:<exact ID> modulo 10 equal zero; same namespace across sources",
        "degree_quantile_order": [0.25,0.5,0.75,0.9,0.99,1],
        "whole_account_and_cross_review_identity_checks_passed": not quality['review_ids_with_conflicting_owner'] and not result['cross_source_overlap']['common_review_ids_with_different_owner'] and not result['cross_source_overlap']['retained_clean_review_ids_with_incompatible_owner_or_app'] and not any(result['post_exclusion_checks'].values()),
        "identity_check_scope": "No raw original-review-ID owner conflicts internally or against frank; after invalid identities, ambiguous internal/cross-source review-to-App-ID mappings and pair-label conflicts are quarantined: no retained shared review ID with incompatible owner/app; zero post-exclusion user, user-app and original-review-ID intersections. Does not certify representative sampling, temporal independence or labels unchanged.",
        "potential_eligibility_protocol": "Hide exactly one supported positive Linux game; every other reviewed game, including other positives, may be an input. No simultaneous multi-target holdout is measured."})
    con.execute(f"COPY (SELECT a.*,coalesce(c.reviewed,false) in_fixed_A_input_catalog,coalesce(c.linux,false) linux_in_A_catalog FROM app_stats a LEFT JOIN catalog c USING(app_id) ORDER BY a.app_id) TO {quote(table)} (HEADER,DELIMITER ',')")
    result['app_support_table_sha256']=digest(table)
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    log(f"Completed: {report.name}; identity/exclusion checks={result['whole_account_and_cross_review_identity_checks_passed']}")
    con.close()


if __name__ == "__main__":
    main()
