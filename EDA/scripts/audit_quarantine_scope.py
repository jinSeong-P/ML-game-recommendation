"""Audit quarantine identity, Item-KNN count scope, and Linux target support.

Reads the published Kieran DuckDB work database in read-only mode and creates
only connection-local temporary tables. Writes aggregate findings only; it
does not modify source data, the database, split files, or modeling decisions.
"""

import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal

import duckdb

from eda_large_review_frank import EDA, digest, many, one, quote


def build_quarantine_tables(con):
    con.execute("""CREATE TEMP TABLE bad_ids AS SELECT review_key FROM review_keys
        WHERE min_app<>max_app OR min_user<>max_user""")
    con.execute("""CREATE TEMP TABLE bad_rows AS SELECT v.* FROM valid_raw v
        SEMI JOIN bad_ids USING(review_key)""")

    fields = [row[0] for row in con.execute("DESCRIBE valid_raw").fetchall()
              if row[0] not in {"review_key", "app_id"}]
    consistency_exprs = []
    for field in fields:
        # Treat NULL as a value so that one missing and one present value is
        # considered a field disagreement.
        consistency_exprs.append(
            f"(count(DISTINCT {field}) + "
            f"CASE WHEN count(*) FILTER(WHERE {field} IS NULL)>0 THEN 1 ELSE 0 END) "
            f"<= 1 AS {field}_consistent"
        )
    con.execute("CREATE TEMP TABLE id_consistency AS SELECT review_key, count(*) n_rows, "
                "count(DISTINCT app_id) n_apps, " + ", ".join(consistency_exprs) +
                " FROM bad_rows GROUP BY review_key")

    all_fields_consistent = " AND ".join(f"{field}_consistent" for field in fields)
    owner_label_consistent = "user_key_consistent AND positive_consistent"
    consistency = one(con, f"""SELECT count(*) review_ids,sum(n_rows) raw_rows,
        count(*) FILTER(WHERE n_apps>1) ids_with_multiple_apps,
        count(*) FILTER(WHERE {owner_label_consistent}) ids_owner_and_label_consistent,
        count(*) FILTER(WHERE NOT user_key_consistent) ids_with_owner_conflict,
        count(*) FILTER(WHERE NOT positive_consistent) ids_with_label_conflict,
        count(*) FILTER(WHERE {all_fields_consistent}) ids_identical_except_app_id,
        count(*) FILTER(WHERE NOT ({all_fields_consistent})) ids_differ_on_other_fields
        FROM id_consistency""")
    consistency["available_fields_compared_except_app_id_and_grouped_review_key"] = fields
    consistency["fields_with_any_inconsistent_review_id"] = many(
        con,
        "SELECT field_name, count(*) inconsistent_review_ids FROM (" +
        " UNION ALL ".join(
            f"SELECT '{field}' field_name, NOT {field}_consistent inconsistent FROM id_consistency"
            for field in fields
        ) + ") WHERE inconsistent GROUP BY field_name ORDER BY field_name",
    )

    con.execute("""CREATE TEMP TABLE id_groups AS SELECT review_key,count(*) n_rows,
        list_sort(list(DISTINCT app_id)) apps FROM valid_raw
        SEMI JOIN bad_ids USING(review_key) GROUP BY review_key""")
    con.execute("""CREATE TEMP TABLE alias_groups AS SELECT apps,count(*) review_ids,
        sum(n_rows) quarantined_rows FROM id_groups GROUP BY apps""")
    con.execute("""CREATE TEMP TABLE alias_apps AS SELECT DISTINCT app_id FROM alias_groups,
        LATERAL (SELECT unnest(apps) app_id) u""")
    con.execute("""CREATE TEMP TABLE alias_summary AS SELECT g.apps,g.review_ids,
        g.quarantined_rows,count(DISTINCT x.app_id) FILTER(WHERE c.reviewed) catalog_apps,
        count(DISTINCT x.app_id) app_ids FROM alias_groups g,
        LATERAL (SELECT unnest(g.apps) app_id) x LEFT JOIN catalog c USING(app_id)
        GROUP BY g.apps,g.review_ids,g.quarantined_rows""")

    groups = one(con, """SELECT count(*) app_groups,sum(review_ids) review_ids,
        sum(quarantined_rows) raw_rows,
        count(*) FILTER(WHERE catalog_apps=1) groups_with_one_catalog_app,
        count(*) FILTER(WHERE catalog_apps>1) groups_with_several_catalog_apps,
        count(*) FILTER(WHERE catalog_apps=0) groups_without_catalog_app,
        coalesce(sum(review_ids) FILTER(WHERE catalog_apps=1),0) review_ids_with_one_catalog_app,
        count(*) FILTER(WHERE app_ids>1) groups_with_multiple_app_ids
        FROM alias_summary""")
    groups.update(one(con, "SELECT count(*) apps_in_groups FROM alias_apps"))

    con.execute("""CREATE TEMP TABLE app_loss AS SELECT v.app_id,count(*) total_rows,
        count(*) FILTER(WHERE b.review_key IS NOT NULL) quarantined_rows
        FROM valid_raw v JOIN alias_apps a USING(app_id)
        LEFT JOIN bad_ids b USING(review_key) GROUP BY v.app_id""")
    con.execute("""CREATE TEMP TABLE app_loss_rate AS SELECT *,
        quarantined_rows*1.0/total_rows loss_fraction FROM app_loss""")
    app_history_loss = one(con, """SELECT count(*) apps,sum(total_rows) available_history_rows,
        sum(quarantined_rows) rows_removed_by_quarantine,
        count(*) FILTER(WHERE quarantined_rows=total_rows) fully_quarantined_apps,
        count(*) FILTER(WHERE loss_fraction>=0.9) apps_with_at_least_90pct_lost,
        count(*) FILTER(WHERE quarantined_rows>0) apps_with_any_rows_lost,
        quantile_cont(loss_fraction,[0,0.25,0.5,0.75,0.9,1]) app_loss_fraction_quantiles
        FROM app_loss_rate""")
    app_history_loss["share_of_available_linked_app_history_rows_removed"] = (
        app_history_loss["rows_removed_by_quarantine"] / app_history_loss["available_history_rows"]
    )
    app_history_loss["available_history_rows_remaining"] = (
        app_history_loss["available_history_rows"] - app_history_loss["rows_removed_by_quarantine"]
    )
    group_history_loss = one(con, """SELECT count(*) app_groups,
        count(*) FILTER(WHERE all_linked_apps_fully_quarantined) groups_where_all_linked_app_histories_lost,
        quantile_cont(group_loss_fraction,[0,0.25,0.5,0.75,0.9,1]) group_loss_fraction_quantiles
        FROM (SELECT g.apps,bool_and(a.quarantined_rows=a.total_rows) all_linked_apps_fully_quarantined,
            sum(a.quarantined_rows)*1.0/sum(a.total_rows) group_loss_fraction
            FROM alias_summary g,LATERAL (SELECT unnest(g.apps) app_id) x
            JOIN app_loss_rate a USING(app_id) GROUP BY g.apps) q""")
    groups["available_valid_raw_history_loss_by_linked_app"] = app_history_loss
    groups["available_valid_raw_history_loss_by_alias_group"] = group_history_loss

    return {"duplicate_field_consistency": consistency,
            "alias_group_structure": groups}


def item_knn_scope(con):
    # A full-cardinality user hash aggregate exceeds the 2 GB audit budget.
    # Partition users by a stable DuckDB hash and aggregate one disjoint slice
    # at a time. This preserves exact counts while bounding memory use.
    con.execute("SET preserve_insertion_order=false")
    partitions = 8
    summed_fields = [
        "training_users", "training_pairs", "all_app_positive_training_pairs",
        "catalog_positive_training_pairs", "outside_catalog_positive_training_pairs",
        "catalog_positive_users", "catalog_positive_pairs", "singleton_users",
        "singleton_user_item_pairs", "users_with_two_or_more_catalog_positive_apps",
        "pairs_from_users_with_two_or_more_catalog_positive_apps", "sum_squared_user_degree",
        "per_user_item_pair_cooccurrence_incidences",
    ]
    combined = {field: 0 for field in summed_fields}
    max_user_degree = 0
    for partition in range(partitions):
        part = one(con, f"""WITH train_users AS (
                SELECT user_key FROM user_stats
                WHERE NOT is_test AND NOT is_validation AND hash(user_key)%{partitions}={partition}
            ),
            train_rows AS (
                SELECT c.user_key,c.app_id,c.positive,coalesce(cat.reviewed,false) reviewed
                FROM clean c JOIN train_users u USING(user_key)
                LEFT JOIN catalog cat USING(app_id)
                WHERE hash(c.user_key)%{partitions}={partition}
            ),
            degrees AS (
                SELECT user_key,count(*) d FROM train_rows
                WHERE positive AND reviewed GROUP BY user_key
            )
            SELECT (SELECT count(*) FROM train_users) training_users,
                (SELECT count(*) FROM train_rows) training_pairs,
                count(*) FILTER(WHERE positive) all_app_positive_training_pairs,
                count(*) FILTER(WHERE positive AND reviewed) catalog_positive_training_pairs,
                count(*) FILTER(WHERE positive AND NOT reviewed) outside_catalog_positive_training_pairs,
                (SELECT count(*) FROM degrees) catalog_positive_users,
                (SELECT sum(d) FROM degrees) catalog_positive_pairs,
                (SELECT count(*) FROM degrees WHERE d=1) singleton_users,
                (SELECT sum(d) FROM degrees WHERE d=1) singleton_user_item_pairs,
                (SELECT count(*) FROM degrees WHERE d>=2) users_with_two_or_more_catalog_positive_apps,
                (SELECT sum(d) FROM degrees WHERE d>=2) pairs_from_users_with_two_or_more_catalog_positive_apps,
                (SELECT sum(d*d) FROM degrees) sum_squared_user_degree,
                (SELECT max(d) FROM degrees) max_user_degree,
                (SELECT sum(d*(d-1)/2) FROM degrees) per_user_item_pair_cooccurrence_incidences
            FROM train_rows""")
        for field in summed_fields:
            combined[field] += part[field] or 0
        max_user_degree = max(max_user_degree, part["max_user_degree"] or 0)
    all_scope = {field: combined[field] for field in [
        "training_users", "training_pairs", "all_app_positive_training_pairs",
        "catalog_positive_training_pairs", "outside_catalog_positive_training_pairs"]}
    degree_scope = {field: combined[field] for field in summed_fields if field not in {
        "training_users", "training_pairs", "all_app_positive_training_pairs",
        "outside_catalog_positive_training_pairs"}}
    degree_scope["max_user_degree"] = max_user_degree
    if all_scope["all_app_positive_training_pairs"] != (
            all_scope["catalog_positive_training_pairs"] +
            all_scope["outside_catalog_positive_training_pairs"]):
        raise ValueError("Item-KNN positive row scopes do not sum")
    return {"all_training_scope": all_scope,
            "fixed_input_catalog_positive_scope": degree_scope,
            "catalog_positive_review_share_of_all_positive_training_rows":
                degree_scope["catalog_positive_pairs"] / all_scope["all_app_positive_training_pairs"],
            "singleton_share_of_catalog_positive_rows":
                degree_scope["singleton_user_item_pairs"] / degree_scope["catalog_positive_pairs"],
            "semantics": {
                "all_app_positive_training_pairs": "positive user-app training observations from non-test, non-validation Kieran accounts across all apps",
                "catalog_positive_pairs": "positive user-app observations restricted to reviewed apps in the fixed input catalog",
                "pairs_from_users_with_two_or_more_catalog_positive_apps": "observations from users with catalog positive degree >=2; these are still user-app rows, not the count of co-occurrence edges",
                "singleton_user_item_pairs": "catalog positive observations from users with catalog positive degree 1; they create no off-diagonal co-occurrence for that user, while still contributing to item support/norms in common cosine constructions",
                "per_user_item_pair_cooccurrence_incidences": "sum over users of choose(catalog positive degree, 2), counting user-specific item-pair incidences rather than unique item-pair edges",
                "degree_aggregation": f"exactly aggregated in {partitions} disjoint hash partitions to stay within the 2 GB memory limit",
            }}


def linux_training_support(con):
    return one(con, """SELECT count(*) linux_catalog_apps,
        count(*) FILTER(WHERE coalesce(a.final_train_positive,0)=0) apps_without_train_positive,
        count(*) FILTER(WHERE coalesce(a.final_train_positive,0) BETWEEN 1 AND 4) apps_with_one_to_four_train_positives,
        count(*) FILTER(WHERE coalesce(a.final_train_positive,0)<5) apps_with_fewer_than_five_train_positives,
        count(*) FILTER(WHERE coalesce(a.final_train_positive,0)>=5) apps_with_at_least_five_train_positives,
        count(*) FILTER(WHERE coalesce(a.all_reviews,0)=0) apps_without_any_kieran_review,
        count(*) FILTER(WHERE coalesce(a.all_reviews,0)=0 AND f.app_id IS NOT NULL) apps_without_kieran_review_but_present_in_frank,
        sum(coalesce(a.final_train_positive,0)) total_train_positive_rows_on_linux_catalog
        FROM catalog c LEFT JOIN app_stats a USING(app_id)
        LEFT JOIN (SELECT DISTINCT app_id FROM frank) f USING(app_id)
        WHERE c.reviewed AND c.linux""")


def prepare_frank_cohort(con, quarantine_path):
    con.execute("CREATE TEMP TABLE quar AS SELECT DISTINCT review_key FROM read_parquet(" +
                quote(quarantine_path) + ")")
    con.execute("""CREATE TEMP TABLE amb AS SELECT DISTINCT user_key,app_id
        FROM frank_raw JOIN quar USING(review_key)""")
    con.execute("""CREATE TEMP TABLE test_hist AS SELECT f.user_key,f.app_id,f.positive,c.linux
        FROM frank f JOIN test_users USING(user_key) JOIN catalog c USING(app_id)
        ANTI JOIN amb ON f.user_key=amb.user_key AND f.app_id=amb.app_id
        WHERE c.reviewed""")
    con.execute("""CREATE TEMP TABLE test_cohort AS SELECT user_key,count(*) n,
        count(*) FILTER(WHERE positive) n_positive,
        count(*) FILTER(WHERE positive AND linux) n_targets
        FROM test_hist GROUP BY user_key HAVING n_targets>0 AND n>=6""")
    con.execute("""CREATE TEMP TABLE linux_support AS SELECT c.app_id,
        coalesce(a.final_train_positive,0) train_positive,
        coalesce(a.all_reviews,0) kieran_reviews
        FROM catalog c LEFT JOIN app_stats a USING(app_id)
        WHERE c.reviewed AND c.linux""")
    con.execute("""CREATE TEMP TABLE potential_targets AS SELECT h.user_key,h.app_id,
        co.n_targets,co.n_positive,s.train_positive,
        CASE WHEN s.train_positive=0 THEN 'zero_train_positive'
             WHEN s.train_positive<5 THEN 'one_to_four_train_positives'
             ELSE 'at_least_five_train_positives' END support_bucket
        FROM test_hist h JOIN test_cohort co USING(user_key)
        JOIN linux_support s USING(app_id)
        WHERE h.positive AND h.linux""")


def frank_target_support(con, published_metrics):
    prepare_frank_cohort(con, EDA / "data/reference/review_app_quarantine.parquet")
    cohort_stats = one(con, """SELECT count(*) potential_users,
        count(*) FILTER(WHERE n_positive>=2) users_with_positive_query,
        sum(n_targets) possible_target_pairs,
        sum(n_targets) FILTER(WHERE n_positive>=2) target_pairs_for_positive_query_users
        FROM test_cohort""")
    expected = next(r for r in published_metrics["cohorts"]["safe_mapping_fixed_catalog"]
                    if r["minimum_inputs"] == 5)
    checks = {
        "potential_users": "potential_users",
        "users_with_positive_query": "potential_users_with_positive_query",
        "possible_target_pairs": "possible_target_pairs",
    }
    for actual_key, expected_key in checks.items():
        if cohort_stats[actual_key] != expected[expected_key]:
            raise ValueError(f"Frank cohort disagrees with published metrics for {actual_key}: {cohort_stats[actual_key]} vs {expected[expected_key]}")

    outcomes = {}
    for cohort_name, cohort_filter in (("all_potential_users", "TRUE"),
                                       ("positive_query_required", "n_positive>=2")):
        eligible_users = one(con, f"SELECT count(*) users FROM test_cohort WHERE {cohort_filter}")["users"]
        eligible_target_pairs = one(con, f"""SELECT count(*) target_pairs FROM potential_targets t
            WHERE {cohort_filter.replace('n_positive', 't.n_positive')}""")["target_pairs"]
        buckets = many(con, f"""SELECT support_bucket,count(*) possible_target_pairs,
            count(DISTINCT t.user_key) users_with_any_target_in_bucket,
            count(DISTINCT t.app_id) distinct_target_apps,
            sum(1.0/t.n_targets) uniformly_selected_target_weight
            FROM potential_targets t WHERE {cohort_filter.replace('n_positive', 't.n_positive')}
            GROUP BY support_bucket ORDER BY support_bucket""")
        for bucket in buckets:
            bucket["share_of_uniformly_selected_targets"] = bucket["uniformly_selected_target_weight"] / eligible_users
            bucket["share_of_possible_target_pairs"] = bucket["possible_target_pairs"] / eligible_target_pairs
        outcomes[cohort_name] = {"eligible_users": eligible_users,
                                 "possible_target_pairs": eligible_target_pairs,
                                 "support_distribution": {b["support_bucket"]: b for b in buckets}}
    return {"cohort_validation": cohort_stats,
            "support_distribution_for_uniformly_selected_frank_target": outcomes,
            "selection_semantics": "Within each eligible user, each positive Linux game in the fixed reviewed catalog is a possible hidden target with equal probability; bucket weights sum one probability per user."}


def main():
    work = EDA / "data/processed/kieran_original/eda_work.duckdb"
    quarantine_path = EDA / "data/reference/review_app_quarantine.parquet"
    full_metrics_path = EDA / "reports/kieran_original_frank_eda_metrics.json"
    cohort_metrics_path = EDA / "reports/kieran_frank_eda_metrics.json"
    diagnostics_path = EDA / "reports/kieran_frank_eval_diagnostics.json"
    backlog_path = EDA / "docs/modeling_backlog.md"
    full_metrics = json.loads(full_metrics_path.read_text("utf-8"))
    cohort_metrics = json.loads(cohort_metrics_path.read_text("utf-8"))

    with tempfile.TemporaryDirectory(prefix="kieran-frank-quarantine-audit-") as temp:
        con = duckdb.connect(str(work), read_only=True)
        con.execute("SET threads=2")
        con.execute("SET memory_limit='2GB'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET temp_directory=" + quote(temp))

        training_scope = item_knn_scope(con)
        expected_training = full_metrics["after_validation_reservation"]
        actual_training = training_scope["all_training_scope"]
        if actual_training["training_pairs"] != expected_training["pairs"]:
            raise ValueError("Training pair scope does not match published full-source metrics")
        if actual_training["all_app_positive_training_pairs"] != expected_training["positives"]:
            raise ValueError("Training positive scope does not match published full-source metrics")

        quarantine = build_quarantine_tables(con)
        target_support = linux_training_support(con)
        expected_linux_positive = expected_training["positive_reviews_on_fixed_linux_catalog"]
        if (target_support["linux_catalog_apps"] != 6630 or
                target_support["apps_without_train_positive"] != 36 or
                target_support["apps_with_fewer_than_five_train_positives"] != 224 or
                target_support["total_train_positive_rows_on_linux_catalog"] != expected_linux_positive):
            raise ValueError("Linux target support does not match the published fixed-catalog training metrics")
        frank_targets = frank_target_support(con, cohort_metrics)
        identity = quarantine["duplicate_field_consistency"]
        identical_except_app_count = identity["ids_identical_except_app_id"]
        inconsistent_ids = identity["ids_differ_on_other_fields"]
        payload_identity_status = (
            "refuted_as_universal_claim" if inconsistent_ids else "supported"
        )
        knn = training_scope["all_training_scope"]
        catalog_scope = training_scope["fixed_input_catalog_positive_scope"]
        support = target_support
        potential = frank_targets["support_distribution_for_uniformly_selected_frank_target"]["all_potential_users"]["support_distribution"]
        positive_query = frank_targets["support_distribution_for_uniformly_selected_frank_target"]["positive_query_required"]["support_distribution"]
        unsupported_target_share = (potential.get("zero_train_positive", {}).get("share_of_uniformly_selected_targets", 0.0) +
                                    potential.get("one_to_four_train_positives", {}).get("share_of_uniformly_selected_targets", 0.0))
        unsupported_target_share_query = (positive_query.get("zero_train_positive", {}).get("share_of_uniformly_selected_targets", 0.0) +
                                          positive_query.get("one_to_four_train_positives", {}).get("share_of_uniformly_selected_targets", 0.0))

        result = {
            "scope": "Aggregate read-only audit of review-ID quarantine, positive Item-KNN training row scopes, and Linux target training support. No model fit, split files, or account-level output.",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "work_database": "data/processed/kieran_original/eda_work.duckdb",
                "database_opened_read_only": True,
                "duckdb_version": duckdb.__version__,
                "threads": 2,
                "memory_limit": "2GB",
                "unique_temporary_directory": True,
                "source_metrics_sha256": digest(full_metrics_path),
                "cohort_metrics_sha256": digest(cohort_metrics_path),
                "existing_diagnostics_sha256": digest(diagnostics_path),
                "reference_manifest_sha256": digest(EDA / "data/reference/manifest.json"),
                "quarantine_file_sha256": digest(quarantine_path),
                "diagnostic_code_sha256": digest(EDA / "scripts/diagnose_eval_design.py"),
                "backlog_sha256": digest(backlog_path),
                "audit_code_sha256": digest(__file__),
            },
            "quarantine": quarantine,
            "item_knn_scope": training_scope,
            "linux_training_support": {
                "fixed_linux_candidate_support": target_support,
                "frank_potential_cohort_targets": frank_targets,
            },
            "verdicts": {
                "quarantined_review_payload_identity": {
                    "status": payload_identity_status,
                    "finding": f"{identical_except_app_count:,} of {identity['review_ids']:,} flagged Kieran review IDs match across every available valid_raw field except app_id; {inconsistent_ids:,} have at least one other-field disagreement.",
                    "owner_label": f"Owner conflicts: {identity['ids_with_owner_conflict']:,}; label conflicts: {identity['ids_with_label_conflict']:,}.",
                },
                "one_catalog_app_proves_original_app": {
                    "status": "unproven",
                    "finding": f"{quarantine['alias_group_structure']['groups_with_one_catalog_app']} of {quarantine['alias_group_structure']['app_groups']} app-ID groups have one currently reviewed catalog app, but the inspected review payload and catalog do not independently establish which app ID was original.",
                    "limitation": "No authoritative source mapping independent of this review-ID collision was present in the evidence inspected. Recovery should not be inferred from catalog cardinality alone.",
                },
                "item_knn_positive_pair_scope": {
                    "status": "scope_difference_confirmed_count_not_inherently_erroneous",
                    "finding": f"{knn['all_app_positive_training_pairs']:,} counts all-app positive user-app training observations; {catalog_scope['catalog_positive_pairs']:,} are within the fixed reviewed input catalog and {knn['outside_catalog_positive_training_pairs']:,} are outside it.",
                    "singleton_semantics": f"{catalog_scope['singleton_users']:,} singleton users contribute {catalog_scope['singleton_user_item_pairs']:,} catalog rows and no off-diagonal co-occurrence for that user; they can still contribute item norms/support. The {catalog_scope['pairs_from_users_with_two_or_more_catalog_positive_apps']:,} rows from degree-2+ users are observations, not co-occurrence-edge counts.",
                },
                "linux_targets_without_positive_training_support": {
                    "status": "supported_as_a_coverage_gap",
                    "finding": f"{support['apps_without_train_positive']} of {support['linux_catalog_apps']} Linux candidates have zero final-train positives; {support['apps_with_fewer_than_five_train_positives']} have fewer than five.",
                    "frank_uniform_target_probability_below_five_train_positives": unsupported_target_share,
                    "frank_positive_query_uniform_target_probability_below_five_train_positives": unsupported_target_share_query,
                    "limitation": "Support counts describe item availability, not measured ranking harm; no model has been evaluated.",
                },
            },
        }
        con.close()

    output = EDA / "reports/kieran_frank_audit_quarantine_scope.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 default=lambda value: float(value) if isinstance(value, Decimal) else str(value)) + "\n",
                       encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
