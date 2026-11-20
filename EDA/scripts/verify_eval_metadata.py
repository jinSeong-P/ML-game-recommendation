"""Independently verify metadata/time concerns; export aggregates, never histories.

Reads the frozen work DB and archived game metadata read-only. This audit does
not change the adopted catalog, review quarantine, split rules, or model plan.
"""

import json
import tempfile
from datetime import datetime, timezone

import duckdb

from eda_large_review_frank import EDA, digest, many, one, quote


def main():
    reference = EDA / "data/reference"
    legacy = EDA / "legacy/data/processed"
    manifest = json.loads((reference / "manifest.json").read_text("utf-8"))
    source_hashes = {r["file"]: r["sha256"] for r in manifest["former_catalog_inputs"]}
    sources = [legacy / name for name in
               ("A_games.parquet", "A_games_metadata.parquet", "C_games_2025.parquet")]
    for source in sources:
        if digest(source) != source_hashes[source.name]:
            raise ValueError(f"Archived metadata changed: {source.name}")
    with tempfile.TemporaryDirectory() as temp:
        con = duckdb.connect(str(EDA / "data/processed/kieran_original/eda_work.duckdb"), read_only=True)
        con.execute("SET threads=2")
        con.execute("SET memory_limit='2GB'")
        con.execute(f"SET temp_directory={quote(temp)}")
        con.execute(f"CREATE TEMP TABLE ref AS SELECT * FROM read_parquet({quote(reference / 'catalog.parquet')})")
        con.execute(f"CREATE TEMP TABLE quarantine AS SELECT * FROM read_parquet({quote(reference / 'review_app_quarantine.parquet')})")
        con.execute("""CREATE TEMP TABLE safe_history AS
            SELECT f.user_key, f.app_id, f.positive, r.linux
            FROM frank f JOIN ref r USING(app_id) SEMI JOIN test_users USING(user_key)
            ANTI JOIN (SELECT DISTINCT x.user_key,x.app_id FROM frank_raw x JOIN quarantine q USING(review_key)) bad
            ON f.user_key=bad.user_key AND f.app_id=bad.app_id WHERE r.reviewed""")
        con.execute("""CREATE TEMP TABLE eligible AS SELECT user_key,count(*) n,
            count(*) FILTER(WHERE positive) n_positive,
            count(*) FILTER(WHERE positive AND linux) n_targets
            FROM safe_history GROUP BY user_key
            HAVING n>=6 AND n_targets>0""")
        con.execute(f"""CREATE TEMP TABLE texts AS SELECT r.app_id,r.reviewed,r.linux,
            a.description,g.about_the_game,g.short_description,g.linux c_linux,
            g.appid IS NOT NULL in_c
            FROM ref r LEFT JOIN read_parquet({quote(legacy / 'A_games_metadata.parquet')}) a USING(app_id)
            LEFT JOIN read_parquet({quote(legacy / 'C_games_2025.parquet')}) g ON r.app_id=g.appid""")
        if con.execute("SELECT count(*)<>count(DISTINCT app_id) FROM texts").fetchone()[0]:
            raise ValueError("Metadata join duplicates App IDs")
        result = {
            "scope": "Read-only independent metadata/time audit; no models or final splits.",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": digest(__file__),
            "reference_manifest_sha256": digest(reference / "manifest.json"),
            "metadata_sha256": {p.name: digest(p) for p in sources},
            "cohort_counts": one(con, "SELECT count(*) all_users,count(*) FILTER(WHERE n_positive>=2) positive_query_users FROM eligible"),
        }
        result["text"] = many(con, """SELECT linux,count(*) apps,
            count(*) FILTER(WHERE length(trim(coalesce(description,'')))>0) a_short,
            count(*) FILTER(WHERE length(trim(coalesce(short_description,'')))>0) c_short,
            count(*) FILTER(WHERE length(trim(coalesce(about_the_game,'')))>0) c_about,
            count(*) FILTER(WHERE length(trim(coalesce(description,'')))>0 OR length(trim(coalesce(short_description,'')))>0) a_or_c_short,
            count(*) FILTER(WHERE length(trim(coalesce(description,'')))>0 OR length(trim(coalesce(about_the_game,'')))>0) a_or_c_about,
            count(*) FILTER(WHERE length(trim(coalesce(description,'')))=0 AND length(trim(coalesce(short_description,'')))>0) filled_by_c_short,
            quantile_disc(length(trim(description)),0.5) FILTER(WHERE length(trim(coalesce(description,'')))>0) a_median_chars,
            quantile_disc(length(trim(short_description)),0.5) FILTER(WHERE length(trim(coalesce(short_description,'')))>0) c_short_median_chars,
            quantile_disc(length(trim(about_the_game)),0.5) FILTER(WHERE length(trim(coalesce(about_the_game,'')))>0) c_about_median_chars
            FROM texts WHERE reviewed GROUP BY GROUPING SETS ((linux),()) ORDER BY linux NULLS FIRST""")
        result["short_text_agreement"] = one(con, """SELECT count(*) both_present,
            count(*) FILTER(WHERE trim(description)=trim(short_description)) identical,
            avg((trim(description)=trim(short_description))::INT) identical_share
            FROM texts WHERE reviewed AND length(trim(coalesce(description,'')))>0
            AND length(trim(coalesce(short_description,'')))>0""")
        result["linux_disagreement"] = many(con, """SELECT linux a_linux,c_linux,count(*) apps
            FROM texts WHERE reviewed GROUP BY 1,2 ORDER BY 1,2""")
        result["linux_disagreement_target_weights"] = many(con, """SELECT e.n_positive>=2 has_positive_query,
            sum(1.0/e.n_targets) expected_targets,
            sum(CASE WHEN t.in_c AND NOT t.c_linux THEN 1.0/e.n_targets ELSE 0 END) c_disagrees_targets,
            sum(CASE WHEN NOT t.in_c THEN 1.0/e.n_targets ELSE 0 END) c_absent_targets,
            sum(CASE WHEN length(trim(coalesce(t.description,'')))=0 AND length(trim(coalesce(t.short_description,'')))=0
                THEN 1.0/e.n_targets ELSE 0 END) missing_short_targets
            FROM safe_history h JOIN eligible e USING(user_key) JOIN texts t USING(app_id)
            WHERE h.positive AND h.linux GROUP BY GROUPING SETS ((e.n_positive>=2),())""")
        result["time_bounds"] = one(con, """SELECT
            (SELECT min(timestamp_created) FROM valid_raw) kieran_first_created,
            (SELECT max(timestamp_created) FROM valid_raw) kieran_last_created,
            (SELECT max(timestamp_updated) FROM valid_raw) kieran_last_updated,
            (SELECT min(timestamp_created) FROM frank_raw) frank_first_created,
            (SELECT max(timestamp_created) FROM frank_raw) frank_last_created,
            (SELECT max(timestamp_updated) FROM frank_raw) frank_last_updated,
            (SELECT avg((year(to_timestamp(timestamp_created))=2025)::INT) FROM frank_raw) frank_2025_raw_share""")
        con.execute("""CREATE TEMP TABLE frank_pair_time AS SELECT user_key,app_id,
            min(timestamp_created) first_created,max(timestamp_created) last_created,count(*) raw_rows
            FROM frank_raw GROUP BY user_key,app_id""")
        result["target_times"] = many(con, """SELECT e.n_positive>=2 has_positive_query,
            sum(1.0/e.n_targets) expected_targets,
            sum(CASE WHEN p.first_created>(SELECT max(timestamp_created) FROM valid_raw)
                THEN 1.0/e.n_targets ELSE 0 END)/sum(1.0/e.n_targets) after_kieran_end_share,
            count(*) FILTER(WHERE p.first_created<>p.last_created) targets_with_disagreeing_created_time
            FROM safe_history h JOIN eligible e USING(user_key) JOIN frank_pair_time p USING(user_key,app_id)
            WHERE h.positive AND h.linux GROUP BY GROUPING SETS ((e.n_positive>=2),())""")
        result["frank_catalog_scope"] = one(con, """SELECT count(*) reserved_account_pairs,
            count(*) FILTER(WHERE NOT coalesce(r.reviewed,false)) outside_catalog_pairs,
            count(*) FILTER(WHERE e.user_key IS NOT NULL) main_cohort_pairs,
            count(*) FILTER(WHERE e.user_key IS NOT NULL AND NOT coalesce(r.reviewed,false)) main_cohort_outside_catalog_pairs
            FROM frank f SEMI JOIN test_users USING(user_key)
            LEFT JOIN ref r USING(app_id) LEFT JOIN eligible e USING(user_key)""")
        result["catalog_release"] = one(con, f"""SELECT max(a.date_release) all_a_last_release,
            max(a.date_release) FILTER(WHERE r.reviewed) input_last_release,
            max(a.date_release) FILTER(WHERE r.reviewed AND r.linux) linux_last_release
            FROM read_parquet({quote(legacy / 'A_games.parquet')}) a JOIN ref r USING(app_id)""")
        con.close()
    output = EDA / "reports/kieran_frank_metadata_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
