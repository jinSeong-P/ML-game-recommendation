"""Recompute user-weighted baseline metrics with per-user exclusion masks.

This audit is read-only against the EDA DuckDB. It exports aggregate metrics
only; no account identifiers, model fits, or split files are produced.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from diagnose_eval_design import EDA, prepare_cohorts
from eda_large_review_frank import digest, quote


WORK = EDA / "data/processed/kieran_original/eda_work.duckdb"
QUARANTINE = EDA / "data/reference/review_app_quarantine.parquet"
OUTPUT = EDA / "reports/kieran_frank_audit_distribution.json"
TOP_K = (10, 50, 500)


def one(con, sql):
    result = con.execute(sql)
    keys = [item[0] for item in result.description]
    return dict(zip(keys, result.fetchone()))


def prepare_additional_tables(con):
    con.execute("""CREATE TEMP TABLE test_query_cohort AS
        SELECT * FROM test_cohort WHERE n_positive>=2""")
    con.execute("""CREATE TEMP TABLE validation_query_cohort AS
        SELECT * FROM validation_cohort WHERE n_positive>=2""")
    con.execute("""CREATE TEMP TABLE reserved_query_cohort AS
        SELECT * FROM reserved_cohort WHERE n_positive>=2""")
    con.execute("""CREATE TEMP TABLE kieran_side AS
        SELECT c.user_key,c.app_id,c.positive,cat.linux
        FROM clean c SEMI JOIN test_cohort USING(user_key)
        JOIN catalog cat USING(app_id) WHERE cat.reviewed""")
    con.execute("""CREATE TEMP TABLE kieran_only_positive_linux AS
        SELECT k.user_key,k.app_id FROM kieran_side k
        ANTI JOIN test_hist f USING(user_key,app_id)
        WHERE k.positive AND k.linux""")
    con.execute("""CREATE TEMP TABLE kieran_all_linux AS
        SELECT user_key,app_id FROM kieran_side WHERE linux""")


def target_distribution(con, hist, cohort):
    shares = ",".join(
        f"sum(1.0/c.n_targets) FILTER(WHERE l.rank_<={k})/count(DISTINCT c.user_key) top_{k}"
        for k in TOP_K
    )
    return one(con, f"""SELECT {shares}
        FROM {hist} h JOIN {cohort} c USING(user_key)
        JOIN linux_rank l USING(app_id)
        WHERE h.positive AND h.linux""")


def user_mask_sensitivity(con, cohort, base_hist, additions=None):
    additions = additions or []
    parts = [f"SELECT user_key,app_id FROM {base_hist} WHERE linux"]
    parts.extend(f"SELECT user_key,app_id FROM {addition}" for addition in additions)
    union_sql = " UNION ALL ".join(parts)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _mask_union AS
        SELECT DISTINCT user_key,app_id FROM ({union_sql})""")
    total_linux = con.execute("SELECT count(*) FROM linux_rank").fetchone()[0]
    user_masks = one(con, f"""SELECT count(*) users,
        min({total_linux}-m.n_masked+1) minimum_candidate_count,
        quantile_disc({total_linux}-m.n_masked+1,0.5) median_candidate_count,
        quantile_disc({total_linux}-m.n_masked+1,0.9) p90_candidate_count,
        avg({total_linux}-m.n_masked+1) mean_candidate_count,
        avg(least(10.0,{total_linux}-m.n_masked+1)/({total_linux}-m.n_masked+1)) random_recall_at_10,
        avg(4.543559338088345/({total_linux}-m.n_masked+1)) random_ndcg_at_10
        FROM {cohort} c JOIN
             (SELECT user_key,count(*) n_masked FROM _mask_union GROUP BY user_key) m
             USING(user_key)""")
    if user_masks["minimum_candidate_count"] <= 10:
        # This is not expected for this catalog, but keep the random NDCG exact
        # if a future source materially changes the candidate pool.
        user_masks.update(one(con, f"""SELECT avg(
            (SELECT sum(1.0/log2(r+1)) FROM range(1,least(10,{total_linux}-m.n_masked+1)+1) rs(r))
            /({total_linux}-m.n_masked+1)) random_ndcg_at_10
            FROM {cohort} c JOIN
                 (SELECT user_key,count(*) n_masked FROM _mask_union GROUP BY user_key) m
                 USING(user_key)"""))

    con.execute(f"""CREATE OR REPLACE TEMP TABLE _targets AS
        SELECT h.user_key,h.app_id,c.n_targets,l.rank_
        FROM {base_hist} h JOIN {cohort} c USING(user_key)
        JOIN linux_rank l USING(app_id)
        WHERE h.positive AND h.linux""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _target_scores AS
        SELECT t.user_key,t.app_id,t.n_targets,t.rank_,
            t.rank_-count(DISTINCT m.app_id) FILTER(WHERE mr.rank_<t.rank_) masked_rank
        FROM _targets t LEFT JOIN _mask_union m USING(user_key)
        LEFT JOIN linux_rank mr ON mr.app_id=m.app_id
        GROUP BY t.user_key,t.app_id,t.n_targets,t.rank_""")
    score = one(con, f"""SELECT count(DISTINCT user_key) users,count(*) target_pairs,
        sum(1.0/n_targets) target_weight_sum,
        sum((rank_<=10)::INT*1.0/n_targets)/count(DISTINCT user_key) unmasked_popularity_recall_at_10,
        sum((rank_<=50)::INT*1.0/n_targets)/count(DISTINCT user_key) unmasked_popularity_top_50_share,
        sum((rank_<=500)::INT*1.0/n_targets)/count(DISTINCT user_key) unmasked_popularity_top_500_share,
        sum((masked_rank<=10)::INT*1.0/n_targets)/count(DISTINCT user_key) masked_popularity_recall_at_10,
        sum((masked_rank<=50)::INT*1.0/n_targets)/count(DISTINCT user_key) masked_popularity_top_50_share,
        sum((masked_rank<=500)::INT*1.0/n_targets)/count(DISTINCT user_key) masked_popularity_top_500_share,
        sum((masked_rank BETWEEN 1 AND 10)::INT*1.0/n_targets)/count(DISTINCT user_key) masked_rank_1_10,
        sum((masked_rank BETWEEN 11 AND 50)::INT*1.0/n_targets)/count(DISTINCT user_key) masked_rank_11_50,
        sum((masked_rank BETWEEN 51 AND 500)::INT*1.0/n_targets)/count(DISTINCT user_key) masked_rank_51_500,
        sum((masked_rank>500)::INT*1.0/n_targets)/count(DISTINCT user_key) masked_rank_over_500,
        sum(CASE WHEN masked_rank<=10 THEN 1.0/log2(masked_rank+1) ELSE 0 END
            /n_targets)/count(DISTINCT user_key) masked_popularity_ndcg_at_10
        FROM _target_scores""")
    if abs(score["target_weight_sum"]-score["users"]) > 1e-5:
        raise AssertionError(f"Per-user target weights did not sum to the cohort size: {score}")
    score.update(user_masks)
    return score


def unobserved_positive_summary(con, cohort):
    return one(con, f"""WITH per_user AS (
        SELECT c.user_key,count(k.app_id) n_kieran_only_positive_linux
        FROM {cohort} c LEFT JOIN kieran_only_positive_linux k USING(user_key)
        GROUP BY c.user_key)
        SELECT count(*) users,
            count(*) FILTER(WHERE n_kieran_only_positive_linux>0) users_with_unobserved_positive,
            avg((n_kieran_only_positive_linux>0)::INT) share_with_unobserved_positive,
            sum(n_kieran_only_positive_linux) unobserved_positive_linux_pairs,
            quantile_disc(n_kieran_only_positive_linux,[0.25,0.5,0.75,0.9])
                unobserved_positive_linux_quantiles
        FROM per_user""")


def main():
    with tempfile.TemporaryDirectory(prefix="kieran_frank_audit_") as temp_dir:
        con = duckdb.connect(str(WORK), read_only=True)
        con.execute("SET threads=2")
        con.execute("SET memory_limit='2GB'")
        con.execute(f"SET temp_directory={quote(temp_dir)}")
        prepare_cohorts(con, QUARANTINE)
        prepare_additional_tables(con)

        cohorts = {
            "frank_all_24571": ("test_hist", "test_cohort"),
            "frank_positive_query_24342": ("test_hist", "test_query_cohort"),
            "kieran_validation_all": ("validation_hist", "validation_cohort"),
            "kieran_validation_positive_query": ("validation_hist", "validation_query_cohort"),
            "reserved_kieran_all": ("reserved_hist", "reserved_cohort"),
            "reserved_kieran_positive_query": ("reserved_hist", "reserved_query_cohort"),
        }
        summary = {}
        for name, (hist, cohort) in cohorts.items():
            base = user_mask_sensitivity(con, cohort, hist)
            base["unmasked_target_popularity"] = target_distribution(con, hist, cohort)
            summary[name] = base

        summary["frank_all_24571"]["mask_sensitivity_add_kieran_only_positive_linux"] = \
            user_mask_sensitivity(con, "test_cohort", "test_hist", ["kieran_only_positive_linux"])
        summary["frank_positive_query_24342"]["mask_sensitivity_add_kieran_only_positive_linux"] = \
            user_mask_sensitivity(con, "test_query_cohort", "test_hist", ["kieran_only_positive_linux"])
        summary["frank_all_24571"]["mask_sensitivity_add_all_kieran_linux_reviews"] = \
            user_mask_sensitivity(con, "test_cohort", "test_hist", ["kieran_all_linux"])
        summary["frank_positive_query_24342"]["mask_sensitivity_add_all_kieran_linux_reviews"] = \
            user_mask_sensitivity(con, "test_query_cohort", "test_hist", ["kieran_all_linux"])

        summary["unobserved_kieran_positive_linux"] = {
            "frank_all_24571": unobserved_positive_summary(con, "test_cohort"),
            "frank_positive_query_24342": unobserved_positive_summary(con, "test_query_cohort"),
        }
        summary["reserved_overlap"] = one(con, """SELECT
            (SELECT count(*) FROM reserved_cohort) reserved_users,
            (SELECT count(*) FROM reserved_query_cohort) reserved_users_with_positive_query,
            (SELECT count(*) FROM test_cohort t JOIN reserved_cohort r USING(user_key))
                frank_cohort_users_also_in_reserved,
            (SELECT count(*) FROM test_query_cohort t JOIN reserved_query_cohort r USING(user_key))
                frank_positive_query_users_also_in_reserved,
            (SELECT count(*) FROM test_cohort t JOIN reserved_query_cohort r USING(user_key))
                frank_cohort_users_in_reserved_positive_query""")

        summary["definitions"] = {
            "target_selection": "Each user contributes one uniformly sampled positive Linux target from that user's cohort history. Cohort averages weight each user equally; target pairs are weighted 1/n_targets.",
            "candidate_mask": "All other reviewed Linux games for that user are excluded; the selected target is always restored as a candidate. Popularity rank is global train-positive count descending with app_id tie-break, then adjusted by removing the user's masked higher-ranked items.",
            "random_baseline": "Expected Recall@10 and NDCG@10 over a uniformly random ranking of the user's remaining Linux candidates, averaged equally by user.",
            "mask_sensitivity": "Two supplemental masks: add Kieran-only positive Linux interactions omitted from the full Frank history, or add every catalog-reviewed Kieran Linux interaction. The selected Frank target is restored after masking, including when it also appears in Kieran.",
            "supplemental_set": "Reserved Kieran cohort is drawn from accounts designated as Frank test users and excluded from Kieran training; it is a same-source, overlapping-account evaluation, not a new independent user sample.",
            "limits": "No recommender is trained. Baseline shifts quantify target-distribution and candidate-mask sensitivity only; they do not estimate Item-KNN or TF-IDF model performance.",
        }
        result = {
            "scope": "Aggregate-only read-only audit of the first four evaluation-design suspicions.",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "duckdb_path": "EDA/data/processed/kieran_original/eda_work.duckdb",
                "clean_pairs": con.execute("SELECT count(*) FROM clean").fetchone()[0],
                "diagnostics_sha256": digest(EDA / "reports/kieran_frank_eval_diagnostics.json"),
                "reference_manifest_sha256": digest(EDA / "data/reference/manifest.json"),
                "threads": 2,
                "memory_limit": "2GB",
            },
            "linux_catalog_apps": con.execute("SELECT count(*) FROM linux_rank").fetchone()[0],
            "cohorts": summary,
        }
        con.close()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
