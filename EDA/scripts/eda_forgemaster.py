"""Full-corpus EDA and within-source random per-game-cap sensitivity.

Requires duckdb, numpy, pandas, matplotlib in the project environment.
Never changes the adopted split, model, or source manifest. Steam IDs stay
VARCHAR; exact duplicate user/app pairs with conflicting labels are excluded.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EDA = Path(__file__).resolve().parents[1]
RAW = EDA / "data/raw/forgemaster_v1"
OUT = EDA / "data/processed/forgemaster_v1"
TABLES = EDA / "reports/tables/forgemaster"
FIGURES = EDA / "figures/forgemaster"
METRICS = EDA / "reports/forgemaster_eda_metrics.json"
FIELDS = ["steamid", "appid", "voted_up", "votes_up", "votes_funny", "weighted_vote_score",
          "playtime_forever", "playtime_at_review", "num_games_owned", "num_reviews", "review",
          "unix_timestamp_created", "unix_timestamp_updated"]
SEEDS = [20261108, 20261109, 20261110, 20261111, 20261112]


def quote(value):
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one(con, sql):
    result = con.execute(sql)
    return dict(zip([column[0] for column in result.description], result.fetchone()))


def table(con, name, sql):
    frame = con.execute(sql).fetchdf()
    frame.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    return frame


def summary(con, relation, field):
    return one(con, f"""SELECT min({field}) minimum, max({field}) maximum,
        avg({field}) mean, quantile_disc({field}, [0.25,0.5,0.75,0.9,0.95,0.99]) quantiles
        FROM {relation}""")


def make_histories(con, source, name):
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {name} AS SELECT steamid,
        count(*) n, count(*) FILTER(WHERE positive) p,
        count(*) FILTER(WHERE positive AND linux) targets
        FROM {source} JOIN catalog USING(app_id) WHERE reviewed GROUP BY steamid""")


def cohort(con, hist):
    return [one(con, f"""SELECT {n} minimum_inputs_after_holdout,
        count(*) FILTER(WHERE n>={n}+1 AND targets>0) potential_users,
        count(*) FILTER(WHERE n>={n}+1 AND targets>0 AND p>=2) eligible_users,
        coalesce(sum(targets) FILTER(WHERE n>={n}+1 AND targets>0 AND p>=2),0) possible_target_pairs
        FROM {hist}""") for n in [1,3,5,10]]


def profile(con, relation):
    return one(con, f"""SELECT count(*) AS "rows", count(DISTINCT steamid) users,
        count(DISTINCT app_id) apps, count(*) FILTER(WHERE positive) positives,
        avg(positive::INT) positive_rate FROM {relation}""")


def main():
    for directory in [OUT, TABLES, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW / "manifest.json").read_text("utf-8"))
    print("Verifying complete source files", flush=True)
    for entry in manifest["files"]:
        path = RAW / entry["file"]
        if path.stat().st_size != entry["listed_bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Source fingerprint mismatch: {path.name}")
    con = duckdb.connect(str(OUT / "eda.duckdb"))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory={quote(OUT / 'duckdb_temp')}")
    paths = "[" + ",".join(quote(RAW / entry["file"]) for entry in manifest["files"]) + "]"
    column_types = "{" + ",".join(f"{quote(field)}:'VARCHAR'" for field in FIELDS) + "}"
    con.execute(f"""CREATE OR REPLACE VIEW raw AS SELECT * FROM read_csv({paths},
        header=true, columns={column_types}, auto_detect=false, delim=',',quote='"',escape='"', filename=true,
        strict_mode=false, ignore_errors=false, max_line_size=16777216)""")
    columns = [item[0] for item in con.execute("DESCRIBE raw").fetchall()]
    if columns != FIELDS + ["filename"]:
        raise ValueError(f"Unexpected source schema: {columns}")
    metrics = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "analysis_version": 1, "source_manifest": manifest,
               "source_manifest_sha256": sha(RAW / "manifest.json"),
               "script_sha256": sha(Path(__file__)),
               "environment": {"duckdb": duckdb.__version__, "memory_limit": "4GB", "threads": 4},
               "policies": {"steamid": "Read and store as VARCHAR; never floating point",
                            "parser": "DuckDB strict parser rejected some input; explicit 13-column schema with strict_mode=false and ignore_errors=false. Independently cross-checked with Python csv.reader including strict mode; no skipped records.",
                            "deduplication": "One steamid/app_id pair; conflicting positive/negative pairs excluded entirely",
                            "time": "UTC; no chronological train/test claim",
                            "playtime": "Raw Steam API minutes; publisher card says hours, not assumed correct",
                            "cap_simulation": "Uniform random maximum 100 clean user/app pairs per game using hash ordering; 5 seeds. Not a reconstruction of Frank API selection.",
                            "model_performance": "No models trained or ranking metrics evaluated"}}
    print("Reading all rows and checking nulls, numeric values, and text", flush=True)
    null_sql = ",".join(f"count(*) FILTER(WHERE {field} IS NULL OR trim({field})='') AS {field}" for field in FIELDS)
    metrics["raw"] = one(con, 'SELECT count(*) AS "rows",count(DISTINCT steamid) users,count(DISTINCT appid) apps FROM raw')
    metrics["null_or_blank"] = one(con, f"SELECT {null_sql} FROM raw")
    types = {field: "BIGINT" for field in FIELDS if field not in ["steamid", "review", "voted_up", "weighted_vote_score"]}
    types.update({"voted_up": "BOOLEAN", "weighted_vote_score": "DOUBLE"})
    invalid_sql = ",".join(f"count(*) FILTER(WHERE {field} IS NOT NULL AND try_cast({field} AS {kind}) IS NULL) AS {field}" for field, kind in types.items())
    metrics["invalid_numeric_or_boolean"] = one(con, f"SELECT {invalid_sql} FROM raw")
    source_counts = table(con, "01_source_files", 'SELECT regexp_extract(filename,\'[^/\\\\]+$\') file,count(*) AS "rows" FROM raw GROUP BY filename ORDER BY file')
    null_records = [{"column": key, "null_or_blank": value, "invalid_cast": metrics["invalid_numeric_or_boolean"].get(key,0)} for key,value in metrics["null_or_blank"].items()]
    import pandas as pd
    pd.DataFrame(null_records).to_csv(TABLES / "02_schema_quality.csv", index=False, encoding="utf-8-sig")
    print("Creating typed compact staging table", flush=True)
    casts = ",".join(f"try_cast({field} AS {kind}) {field}" for field, kind in types.items())
    con.execute(f"""CREATE OR REPLACE TABLE typed AS SELECT steamid,{casts},
        length(review) text_length, regexp_matches(coalesce(review,''),'[^\\x00-\\x7F]') non_ascii_text,
        filename FROM raw""")
    con.execute("""CREATE OR REPLACE VIEW valid AS SELECT * FROM typed WHERE
        regexp_full_match(steamid,'[0-9]{17}') AND appid>0 AND voted_up IS NOT NULL""")
    metrics["quality"] = one(con, """SELECT
        count(*) FILTER(WHERE NOT coalesce(regexp_full_match(steamid,'[0-9]{17}'),false)) invalid_steamid_rows,
        count(*) FILTER(WHERE appid IS NULL OR appid<=0) invalid_appid_rows,
        count(*) FILTER(WHERE voted_up IS NULL) missing_label_rows,
        count(*) FILTER(WHERE playtime_forever<0 OR playtime_at_review<0) negative_playtime_rows,
        count(*) FILTER(WHERE playtime_at_review>playtime_forever) playtime_at_review_gt_forever,
        count(*) FILTER(WHERE unix_timestamp_updated<unix_timestamp_created) updated_before_created,
        count(*) FILTER(WHERE unix_timestamp_created<0 OR unix_timestamp_created>1619568000) created_outside_publish_window,
        count(*) FILTER(WHERE num_reviews<0 OR num_games_owned<0) negative_user_metadata,
        count(*) FILTER(WHERE votes_up<0 OR votes_funny<0) negative_votes,
        count(*) FILTER(WHERE NOT isfinite(weighted_vote_score) OR weighted_vote_score<0 OR weighted_vote_score>1) invalid_helpfulness_score,
        count(*) FILTER(WHERE non_ascii_text) non_ascii_text_rows,
        min(unix_timestamp_created) first_created_unix,max(unix_timestamp_created) last_created_unix,
        min(unix_timestamp_updated) first_updated_unix,max(unix_timestamp_updated) last_updated_unix
        FROM typed""")
    metrics["text_length_chars"] = summary(con, "typed", "text_length")
    metrics["playtime_minutes"] = {field: summary(con, "typed", field) for field in ["playtime_forever", "playtime_at_review"]}
    print("Deduplicating user/game pairs, preserving both labels until conflict checks", flush=True)
    con.execute("""CREATE OR REPLACE TABLE pair_groups AS SELECT steamid,appid app_id,
        count(*) raw_rows,count(DISTINCT voted_up) labels,bool_or(voted_up) positive,
        min(unix_timestamp_created) created,max(unix_timestamp_updated) updated,
        max(num_reviews) reported_reviews,max(playtime_forever) playtime_minutes
        FROM valid GROUP BY steamid,appid""")
    metrics["duplicates"] = one(con, """SELECT count(*) distinct_pairs,
        sum(raw_rows-1) extra_user_game_rows,
        count(*) FILTER(WHERE raw_rows>1) repeated_pairs,
        count(*) FILTER(WHERE labels>1) conflicting_pairs,
        coalesce(sum(raw_rows) FILTER(WHERE labels>1),0) rows_in_conflicting_pairs FROM pair_groups""")
    con.execute("CREATE OR REPLACE TABLE clean AS SELECT * EXCLUDE(labels) FROM pair_groups WHERE labels=1")
    metrics["clean"] = profile(con, "clean")
    con.execute(f"CREATE OR REPLACE TEMP VIEW catalog AS SELECT * FROM read_parquet({quote(EDA / 'data/reference/catalog.parquet')})")
    con.execute(f"CREATE OR REPLACE TEMP VIEW names AS SELECT app_id,title,date_release,user_reviews FROM read_parquet({quote(EDA / 'data/processed/A_games.parquet')})")
    con.execute("CREATE OR REPLACE TABLE apps AS SELECT app_id,count(*) reviews,sum(positive::INT) positives,avg(positive::INT) positive_rate FROM clean GROUP BY app_id")
    con.execute("CREATE OR REPLACE TABLE users AS SELECT steamid,count(*) reviews,sum(positive::INT) positives,max(reported_reviews) reported_reviews FROM clean GROUP BY steamid")
    metrics["reviews_per_game"] = summary(con, "apps", "reviews")
    metrics["reviews_per_user"] = summary(con, "users", "reviews")
    metrics["user_history_completeness_proxy"] = one(con, """SELECT
        count(*) FILTER(WHERE reviews=1) singleton_users,
        count(*) FILTER(WHERE reported_reviews>0) users_with_positive_reported_reviews,
        count(*) FILTER(WHERE reported_reviews>reviews) observed_less_than_reported_users,
        count(*) FILTER(WHERE reported_reviews<reviews) observed_more_than_reported_users,
        median(least(reviews*1.0/reported_reviews,1.0)) FILTER(WHERE reported_reviews>0) median_observed_to_reported_clipped_at_one
        FROM users""")
    games = table(con,"03_game_support", """SELECT a.*,n.title,c.reviewed in_input_catalog,
        coalesce(c.reviewed AND c.linux,false) in_linux_candidates FROM apps a
        LEFT JOIN catalog c USING(app_id) LEFT JOIN names n USING(app_id)
        ORDER BY reviews DESC,app_id""")
    metrics["cap_shape"] = one(con, """SELECT count(*) games,
        count(*) FILTER(WHERE reviews<=100) games_le_100,
        count(*) FILTER(WHERE reviews=100) games_exact_100,
        count(*) FILTER(WHERE reviews>100) games_gt_100,
        sum(reviews) FILTER(WHERE reviews>100) reviews_on_games_gt_100,
        max(reviews) maximum,
        sum(least(reviews,100)) expected_rows_after_cap,
        avg(positive_rate) macro_positive_rate,
        sum(positives)*1.0/sum(reviews) micro_positive_rate FROM apps""")
    freq=table(con,"04_game_count_frequencies", "SELECT reviews,count(*) games FROM apps GROUP BY reviews ORDER BY games DESC,reviews")
    userfreq=table(con,"05_user_history_frequencies", "SELECT reviews,count(*) users FROM users GROUP BY reviews ORDER BY reviews")
    years=table(con,"06_review_years", '''SELECT year(to_timestamp(created)) AS "year",count(*) reviews,
        avg(positive::INT) positive_rate FROM clean GROUP BY "year" ORDER BY "year"''')
    metrics["catalog_coverage"] = one(con, """SELECT count(*) source_apps,
        count(*) FILTER(WHERE c.reviewed) input_apps_covered,
        count(*) FILTER(WHERE c.reviewed AND c.linux) linux_candidates_covered,
        sum(a.reviews) FILTER(WHERE c.reviewed) rows_in_input_catalog,
        sum(a.positives) FILTER(WHERE c.reviewed AND c.linux) positive_linux_rows,
        (SELECT count(*) FROM catalog WHERE reviewed) total_input_apps,
        (SELECT count(*) FROM catalog WHERE reviewed AND linux) total_linux_candidates
        FROM apps a LEFT JOIN catalog c USING(app_id)""")
    make_histories(con, "clean", "hist")
    metrics["eligibility"] = cohort(con,"hist")
    pd.DataFrame(metrics["eligibility"]).to_csv(TABLES / "07_evaluation_eligibility.csv",index=False,encoding="utf-8-sig")
    metrics["full_eligible_history"] = one(con,"SELECT count(*) users,avg(n-1) mean_inputs_after_holdout,median(n-1) median_inputs_after_holdout,sum(targets) possible_target_pairs FROM hist WHERE n>=6 AND p>=2 AND targets>0")
    con.execute(f"""CREATE OR REPLACE TEMP VIEW kieran_support AS SELECT * FROM read_csv(
        {quote(EDA / 'reports/tables/16_kieran_original_app_support.csv')},header=true)""")
    metrics["kieran_game_support_overlap"] = one(con,"""SELECT count(*) apps,
        sum(a.reviews) source_pairs_on_kieran_apps,
        count(*) FILTER(WHERE final_train_positive>0) apps_with_recorded_train_positive_support,
        corr(log10(a.reviews+1),log10(k.final_train_reviews+1)) log_support_pearson
        FROM apps a JOIN kieran_support k USING(app_id)""")
    table(con,"08_source_vs_recorded_kieran_game_support", """SELECT a.*,k.all_reviews kieran_all_reviews,
        k.final_train_reviews recorded_kieran_train_reviews,k.final_train_positive recorded_kieran_train_positive
        FROM apps a LEFT JOIN kieran_support k USING(app_id) ORDER BY a.reviews DESC""")
    # Identity-level Kieran source/splits must be present to certify leakage.
    metrics["training_identity_audit"] = {"status":"NOT_VERIFIED",
        "reason":"The recorded Kieran and split_v1 identity artifacts are absent in this checkout. Aggregate per-app support cannot establish user or pair independence.",
        "consequence":"Do not evaluate the existing trained models as an independent test. Audit exact Steam IDs against train and validation; reserve new test accounts and retrain as needed."}
    print("Running five uniform within-source 100-review cap simulations",flush=True)
    cap_results=[]
    full_top10=set(games.head(10).app_id.astype(int))
    metrics["full_top10_review_share"]=float(games.head(10).reviews.sum()/games.reviews.sum())
    for seed in SEEDS:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE capped AS SELECT * FROM clean
            QUALIFY row_number() OVER(PARTITION BY app_id ORDER BY hash(steamid,app_id,{seed}),steamid)<=100""")
        make_histories(con,"capped","capped_hist")
        item={"seed":seed,**profile(con,"capped"),"eligibility":cohort(con,"capped_hist")}
        item.update(one(con,"""SELECT count(*) users,avg(n-1) mean_inputs_after_holdout,
            median(n-1) median_inputs_after_holdout,sum(targets) possible_target_pairs
            FROM capped_hist WHERE n>=6 AND p>=2 AND targets>0"""))
        # Profile uses 'users' for all users; retain it explicitly before cohort metrics.
        item["all_sample_users"]=con.execute("SELECT count(DISTINCT steamid) FROM capped").fetchone()[0]
        item["eligible_users"]=item.pop("users")
        item["full_history_mean_inputs_for_same_cap_eligible_users"]=con.execute("""SELECT avg(h.n-1) FROM hist h JOIN capped_hist c USING(steamid)
            WHERE c.n>=6 AND c.p>=2 AND c.targets>0""").fetchone()[0]
        item["positive_rate_error_games_gt100"]=one(con,"""WITH r AS (
            SELECT app_id,avg(positive::INT) rate FROM capped GROUP BY app_id)
            SELECT avg(abs(a.positive_rate-r.rate)) mean_absolute_error,
            median(abs(a.positive_rate-r.rate)) median_absolute_error,
            quantile_disc(abs(a.positive_rate-r.rate),0.9) p90_absolute_error,
            count(*) games FROM apps a JOIN r USING(app_id) WHERE a.reviews>100""")
        top_ids=",".join(map(str,sorted(full_top10)))
        item["full_top10_review_share_after_cap"]=con.execute(f"SELECT avg((app_id IN ({top_ids}))::INT) FROM capped").fetchone()[0]
        cap_results.append(item)
        print(f"seed={seed}: {item['rows']:,} sampled pairs, {item['eligible_users']:,} eligible users",flush=True)
    metrics["cap_sensitivity"]=cap_results
    cap_flat=[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} | row["positive_rate_error_games_gt100"] for row in cap_results]
    pd.DataFrame(cap_flat).to_csv(TABLES / "09_cap_sensitivity.csv",index=False,encoding="utf-8-sig")
    frank=EDA / "data/processed/forgemaster_eda/frank_reviews_identity.parquet"
    if frank.exists():
        print("Comparing exact Steam IDs with the pinned Frank review projection",flush=True)
        con.execute(f"CREATE OR REPLACE TEMP VIEW frank_raw AS SELECT * FROM read_parquet({quote(frank)})")
        con.execute("""CREATE OR REPLACE TEMP TABLE frank_groups AS SELECT author_steamid steamid,
            appid app_id,count(*) raw_rows,count(DISTINCT voted_up) labels,bool_or(voted_up) positive,
            min(timestamp_created) created FROM frank_raw GROUP BY author_steamid,appid""")
        con.execute("CREATE OR REPLACE TEMP TABLE frank_clean AS SELECT * FROM frank_groups WHERE labels=1")
        metrics["frank_comparison"]={"projection_sha256":sha(frank),"profile":profile(con,"frank_clean")}
        metrics["frank_comparison"]["overlap"]=one(con,"""SELECT
            (SELECT count(*) FROM users SEMI JOIN (SELECT DISTINCT steamid FROM frank_clean) f USING(steamid)) shared_users,
            (SELECT count(*) FROM clean SEMI JOIN frank_clean USING(steamid,app_id)) shared_user_game_pairs,
            (SELECT count(*) FROM clean c JOIN frank_clean f USING(steamid,app_id) WHERE c.positive!=f.positive) shared_pair_label_disagreements,
            (SELECT count(*) FROM apps SEMI JOIN (SELECT DISTINCT app_id FROM frank_clean) f USING(app_id)) shared_games""")
        # Apply the existing review/app quarantine to Frank only. Forgemaster has
        # no recommendationid and cannot be audited by this identifier.
        con.execute(f"""CREATE OR REPLACE TEMP VIEW quarantine AS SELECT * FROM read_parquet(
            {quote(EDA / 'data/reference/review_app_quarantine.parquet')})""")
        q_columns=[row[0] for row in con.execute("DESCRIBE quarantine").fetchall()]
        q_key="recommendationid" if "recommendationid" in q_columns else "review_key"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE frank_safe AS SELECT f.* FROM frank_clean f
            ANTI JOIN (SELECT DISTINCT author_steamid steamid,appid app_id FROM frank_raw r
                SEMI JOIN quarantine q ON cast(r.recommendationid AS VARCHAR)=cast(q.{q_key} AS VARCHAR)) bad
                USING(steamid,app_id)""")
        make_histories(con,"frank_safe","frank_hist")
        metrics["frank_comparison"]["eligibility_after_existing_quarantine"]=cohort(con,"frank_hist")
        metrics["frank_comparison"]["source_eligible_overlap"]=one(con,"""SELECT count(*) source_eligible_users_with_frank_catalog_history,
            count(*) FILTER(WHERE f.n>=6 AND f.p>=2 AND f.targets>0) eligible_in_both
            FROM hist h JOIN frank_hist f USING(steamid) WHERE h.n>=6 AND h.p>=2 AND h.targets>0""")
        metrics["frank_comparison"]["source_eligible_overlap"]["source_eligible_users_in_any_frank"]=con.execute("""SELECT count(*) FROM hist h
            SEMI JOIN (SELECT DISTINCT steamid FROM frank_clean) f USING(steamid)
            WHERE h.n>=6 AND h.p>=2 AND h.targets>0""").fetchone()[0]
    else:
        metrics["frank_comparison"]={"status":"Projection absent; existing reported aggregates only"}
    # Export exact account IDs only to ignored local data for future split audits.
    con.execute(f"""COPY (SELECT h.* FROM hist h WHERE n>=6 AND p>=2 AND targets>0)
        TO {quote(OUT / 'potential_test_accounts.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute(f"COPY clean TO {quote(OUT / 'clean_pairs.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    python_validation=json.loads((OUT / "python_csv_validation.json").read_text("utf-8"))
    metrics["independent_python_csv_validation"]=python_validation
    python_counts={row["file"]:row["rows"] for row in python_validation["files"]}
    checks={"source_rows_match_publisher":metrics["raw"]["rows"]==15437471,
        "source_users_match_publisher":metrics["raw"]["users"]==6976390,
        "source_apps_match_publisher":metrics["raw"]["apps"]==8183,
        "file_row_sum_matches":int(source_counts.rows.sum())==metrics["raw"]["rows"],
        "python_csv_total_matches":python_validation["rows"]==metrics["raw"]["rows"],
        "python_csv_each_file_matches":all(python_counts[row.file]==row.rows for row in source_counts.itertuples()),
        "game_pair_sum_matches":int(games.reviews.sum())==metrics["clean"]["rows"],
        "user_pair_sum_matches":int((userfreq.reviews*userfreq.users).sum())==metrics["clean"]["rows"],
        "years_sum_matches":int(years.reviews.sum())==metrics["clean"]["rows"],
        "cap_row_counts_match":all(row["rows"]==metrics["cap_shape"]["expected_rows_after_cap"] for row in cap_results),
        "clean_pair_uniqueness":con.execute("SELECT count(*)=count(DISTINCT (steamid,app_id)) FROM clean").fetchone()[0],
        "clean_labels_unambiguous":con.execute("SELECT count(*) FROM pair_groups WHERE labels=1").fetchone()[0]==metrics["clean"]["rows"]}
    metrics["reconciliation"]=checks
    if not all(value for key,value in checks.items() if not key.endswith("match_publisher")):
        raise AssertionError(checks)
    print("Rendering EDA figures",flush=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    axes[0,0].hist(games.reviews,bins=np.geomspace(1,max(games.reviews)+1,55),color="#3764a3")
    axes[0,0].set(xscale="log",yscale="log",xlabel="Reviews per game",ylabel="Games",title="Game support: full corpus")
    axes[0,0].axvline(100,color="#c34835",ls="--",label="100 reviews");axes[0,0].legend()
    xx=userfreq.reviews.to_numpy();yy=userfreq.users.to_numpy()
    axes[0,1].plot(xx,np.cumsum(yy[::-1])[::-1]/yy.sum(),color="#377f72")
    axes[0,1].set(xscale="log",xlabel="Observed reviews per user (at least)",ylabel="Fraction of users",title="User history is incomplete")
    axes[1,0].bar(years.year.astype(str),years.reviews/1e6,color="#3764a3")
    axes[1,0].set(xlabel="Creation year (UTC)",ylabel="Million clean reviews",title="Historical coverage")
    axes[1,0].tick_params(axis="x",rotation=50)
    capped_users=np.array([row["eligible_users"] for row in cap_results])
    full_users=metrics["eligibility"][2]["eligible_users"]
    axes[1,1].bar(["Full source","Random cap = 100"],[full_users,capped_users.mean()],color=["#377f72","#c34835"])
    axes[1,1].errorbar([1],[capped_users.mean()],yerr=[[capped_users.mean()-capped_users.min()],[capped_users.max()-capped_users.mean()]],color="black",capsize=5)
    axes[1,1].set(ylabel="Eligible users",title="Same-source cap sensitivity (5 seeds)")
    for i,value in enumerate([full_users,capped_users.mean()]):
        axes[1,1].text(i,value,f"{value:,.0f}",ha="center",va="bottom")
    fig.suptitle("Forgemaster Steam Reviews v1 | full-corpus EDA",fontsize=17)
    fig.savefig(FIGURES / "01_overview.png",bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    axes[0].plot(np.arange(1,len(games)+1)/len(games),games.reviews.cumsum()/games.reviews.sum(),label="Full corpus",color="#3764a3")
    cap_counts=np.minimum(games.reviews,100)
    axes[0].plot(np.arange(1,len(games)+1)/len(games),cap_counts.cumsum()/cap_counts.sum(),label="Cap 100 (same game order)",color="#c34835")
    axes[0].set(xlabel="Games in descending full support",ylabel="Cumulative review share",title="Cap changes the observed popularity distribution");axes[0].legend()
    axes[1].bar(["Full source"]+[str(row["seed"])[-2:] for row in cap_results],
        [metrics["clean"]["positive_rate"]]+[row["positive_rate"] for row in cap_results],color=["#3764a3"]+["#c34835"]*5)
    axes[1].set(ylim=(0,1),ylabel="Positive recommendation fraction",xlabel="Full / random cap seed suffix",title="Review-weighted label distribution")
    fig.savefig(FIGURES / "02_cap_distribution.png",bbox_inches="tight");plt.close(fig)
    metrics["outputs"]={"clean_pairs": "data/processed/forgemaster_v1/clean_pairs.parquet",
        "potential_accounts":"data/processed/forgemaster_v1/potential_test_accounts.parquet",
        "tables":"reports/tables/forgemaster", "figures":"figures/forgemaster"}
    METRICS.write_text(json.dumps(metrics,indent=2,ensure_ascii=False,default=lambda value:value.isoformat() if isinstance(value,(datetime,date)) else value.item())+"\n","utf-8")
    con.close()
    print(f"COMPLETE {METRICS}",flush=True)


if __name__=="__main__":
    main()
