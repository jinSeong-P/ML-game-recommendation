"""Validate saved modeling Parquets, without reading model results or fitting.

Checks operate on the exported files, not the builder's temporary SQL tables.
Only aggregate counts are returned; no account identifiers are printed.
"""

import argparse
import json
import tempfile
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from eda_large_review_frank import digest, log, one, quote


def verify_split(root, verify_hashes=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    files = manifest["files"]
    required = {"train/interactions.parquet", "train/positive_catalog.parquet",
                "train/review_lineage.parquet", "validation/interactions.parquet",
                "validation/review_lineage.parquet", "test/review_lineage.parquet",
                "catalog/input_apps.parquet", "catalog/linux_candidates.parquet",
                "reserved_test_accounts.parquet"}
    required.update(f"{split}/{name}.parquet" for split in ("test", "validation")
                    for name in ("users", "query", "targets", "exclusions"))
    if set(files) != required:
        raise ValueError("Unexpected or missing split artifact inventory")
    for relative, info in files.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Missing or resized artifact: {relative}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != info["rows"]:
            raise ValueError(f"Parquet row count changed: {relative}")
        schema = [{"name": f.name, "type": str(f.type)} for f in parquet.schema_arrow]
        if schema != info["schema"]:
            raise ValueError(f"Parquet schema changed: {relative}")
        for field in parquet.schema_arrow:
            if field.name in {"user_key", "review_key"} and str(field.type) != "uint64":
                raise ValueError(f"Identifier must be lossless uint64: {relative}: {field.name}")
        if verify_hashes and digest(path) != info["sha256"]:
            raise ValueError(f"Parquet fingerprint changed: {relative}")
    log("Read-back verification: artifact inventory and fingerprints passed")
    checks = {}
    with tempfile.TemporaryDirectory(prefix="verify_split_v1_") as temp:
        con = duckdb.connect()
        try:
            con.execute("SET threads=2")
            con.execute("SET memory_limit='4GB'")
            con.execute("SET preserve_insertion_order=false")
            con.execute(f"SET temp_directory={quote(temp)}")
            for relative in sorted(files):
                name = relative.removesuffix(".parquet").replace("/", "_")
                con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({quote(root/relative)})")

            def zero(name, sql):
                value = con.execute(sql).fetchone()[0]
                checks[name] = int(value or 0)
                if value:
                    raise ValueError(f"Saved split validation failed: {name}={value}")

            def difference(a, b):
                return f"SELECT count(*) FROM (({a} EXCEPT {b}) UNION ALL ({b} EXCEPT {a}))"

            zero("input_app_duplicates", "SELECT count(*)-count(DISTINCT app_id) FROM catalog_input_apps")
            zero("linux_candidate_duplicates", "SELECT count(*)-count(DISTINCT app_id) FROM catalog_linux_candidates")
            zero("linux_candidate_set_mismatch", difference(
                "SELECT app_id FROM catalog_input_apps WHERE linux",
                "SELECT app_id FROM catalog_linux_candidates"))
            zero("reserved_account_duplicates", "SELECT count(*)-count(DISTINCT user_key) FROM reserved_test_accounts")
            total_counts = {"train_accounts": 0, "validation_accounts": 0,
                            "positive_train_accounts": 0, "positive_singleton_accounts": 0}
            # Bound exact pair grouping and lineage checks by disjoint account
            # partitions; full 100M-pair DISTINCT may exceed local RAM.
            for split in ("train", "validation"):
                for part in range(8):
                    log(f"Read-back verification: {split} identity/lineage partition {part+1}/8")
                    con.execute(f"""CREATE OR REPLACE TEMP TABLE _pairs AS
                        SELECT user_key,app_id,count(*) n,bool_and(positive) min_label,
                            bool_or(positive) max_label
                        FROM {split}_interactions WHERE user_key%8={part}
                        GROUP BY user_key,app_id""")
                    con.execute(f"""CREATE OR REPLACE TEMP TABLE _lineage AS
                        SELECT user_key,app_id,bool_and(positive) min_label,
                            bool_or(positive) max_label
                        FROM {split}_review_lineage WHERE user_key%8={part}
                        GROUP BY user_key,app_id""")
                    zero(f"{split}_pair_lineage_errors_{part}", """SELECT count(*)
                        FROM _pairs i FULL JOIN _lineage l USING(user_key,app_id)
                        WHERE i.user_key IS NULL OR l.user_key IS NULL OR i.n<>1
                            OR i.min_label IS NULL OR i.min_label<>i.max_label
                            OR l.min_label IS NULL OR l.min_label<>l.max_label
                            OR i.max_label<>l.max_label OR coalesce(i.app_id,l.app_id)<=0""")
                    condition = "=0" if split == "validation" else "<>0"
                    zero(f"{split}_wrong_account_assignment_{part}", f"""SELECT count(*) FROM _pairs
                        WHERE NOT (cast('0x'||substr(md5('steam-eda-validation-v1:'||
                            cast(user_key AS VARCHAR)),1,15) AS UBIGINT)%10{condition})""")
                    total_counts[f"{split}_accounts"] += con.execute(
                        "SELECT count(DISTINCT user_key) FROM _pairs").fetchone()[0]
                    if split == "train":
                        con.execute(f"""CREATE OR REPLACE TEMP TABLE _positive AS
                            SELECT user_key,app_id,count(*) n,bool_and(positive) min_label,
                                bool_or(positive) max_label FROM train_positive_catalog
                            WHERE user_key%8={part} GROUP BY user_key,app_id""")
                        zero(f"train_positive_projection_errors_{part}", """SELECT count(*) FROM
                            (SELECT i.* FROM _pairs i JOIN catalog_input_apps USING(app_id)
                             WHERE i.max_label) expected
                            FULL JOIN _positive p USING(user_key,app_id)
                            WHERE expected.user_key IS NULL OR p.user_key IS NULL OR p.n<>1
                                OR NOT coalesce(p.min_label AND p.max_label,false)""")
                        support = one(con, """SELECT count(*) users,
                            count(*) FILTER(WHERE n=1) singletons FROM (
                            SELECT user_key,count(*) n FROM _positive GROUP BY user_key)""")
                        total_counts["positive_train_accounts"] += support["users"]
                        total_counts["positive_singleton_accounts"] += support["singletons"]
                zero(f"{split}_reserved_account_leaks", f"""SELECT count(*) FROM {split}_interactions
                    SEMI JOIN reserved_test_accounts USING(user_key)""")
                zero(f"{split}_invalid_identifiers", f"""SELECT count(*) FROM {split}_interactions
                    WHERE user_key IS NULL OR user_key<76561197960265728
                        OR user_key>76561197960265728+4294967295
                        OR app_id IS NULL OR app_id<=0 OR positive IS NULL""")
            for key in ("_pairs", "_lineage", "_positive"):
                con.execute(f"DROP TABLE IF EXISTS {key}")
            zero("train_validation_account_intersections", """SELECT count(*) FROM train_interactions
                SEMI JOIN (SELECT DISTINCT user_key FROM validation_interactions) USING(user_key)""")
            zero("train_validation_pair_intersections", """SELECT count(*) FROM train_interactions
                SEMI JOIN validation_interactions USING(user_key,app_id)""")
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                zero(f"{left}_{right}_original_review_intersections", f"""SELECT count(*)
                    FROM {left}_review_lineage SEMI JOIN {right}_review_lineage USING(review_key)""")
            for split in ("validation", "test"):
                query, target, users, exclusions = (f"{split}_{name}" for name in
                                                    ("query", "targets", "users", "exclusions"))
                zero(f"{split}_user_duplicates", f"SELECT count(*)-count(DISTINCT user_key) FROM {users}")
                zero(f"{split}_target_count_errors", f"""SELECT count(*) FROM
                    (SELECT user_key,count(*) n FROM {target} GROUP BY user_key HAVING n<>1)""")
                zero(f"{split}_target_user_set_mismatch", difference(
                    f"SELECT user_key FROM {users}", f"SELECT user_key FROM {target}"))
                zero(f"{split}_query_user_set_mismatch", difference(
                    f"SELECT user_key FROM {users}", f"SELECT user_key FROM {query}"))
                zero(f"{split}_target_query_overlap", f"SELECT count(*) FROM {target} JOIN {query} USING(user_key,app_id)")
                zero(f"{split}_target_exclusion_overlap", f"SELECT count(*) FROM {target} JOIN {exclusions} USING(user_key,app_id)")
                zero(f"{split}_exclusion_set_mismatch", difference(
                    f"SELECT user_key,app_id FROM {query}", f"SELECT * FROM {exclusions}"))
                zero(f"{split}_exclusion_duplicates", f"""SELECT count(*) FROM
                    (SELECT user_key,app_id,count(*) n FROM {exclusions} GROUP BY 1,2 HAVING n<>1)""")
                zero(f"{split}_invalid_targets", f"""SELECT count(*) FROM {target} t
                    LEFT JOIN catalog_linux_candidates c USING(app_id)
                    WHERE c.app_id IS NULL OR NOT coalesce(t.positive AND t.linux,false)""")
                zero(f"{split}_invalid_query_apps", f"""SELECT count(*) FROM {query} q
                    LEFT JOIN catalog_input_apps c USING(app_id)
                    WHERE c.app_id IS NULL OR q.linux IS DISTINCT FROM c.linux OR q.positive IS NULL""")
                zero(f"{split}_query_pair_duplicates", f"""SELECT count(*) FROM
                    (SELECT user_key,app_id,count(*) n FROM {query} GROUP BY 1,2 HAVING n<>1)""")
                zero(f"{split}_ineligible_query_counts", f"""SELECT count(*) FROM (
                    SELECT u.user_key FROM {users} u LEFT JOIN {query} q USING(user_key)
                    GROUP BY u.user_key,u.n_total_inputs,u.n_positive_inputs,u.n_eligible_targets
                    HAVING count(q.app_id)<>u.n_total_inputs OR count(q.app_id)<5
                        OR count(q.app_id) FILTER(WHERE q.positive)<>u.n_positive_inputs
                        OR count(q.app_id) FILTER(WHERE q.positive)<1 OR u.n_eligible_targets<1)""")
                full = f"(SELECT user_key,app_id,positive,linux FROM {query} UNION ALL SELECT * FROM {target})"
                zero(f"{split}_lineage_history_mismatch", difference(
                    f"SELECT user_key,app_id,positive FROM {full}",
                    f"SELECT user_key,app_id,positive FROM {split}_review_lineage" +
                    (f" SEMI JOIN {users} USING(user_key) JOIN catalog_input_apps USING(app_id)"
                     if split == "validation" else "")))
                zero(f"{split}_target_candidate_count_errors", f"""SELECT count(*) FROM
                    (SELECT u.user_key FROM {users} u JOIN {full} h USING(user_key)
                     GROUP BY u.user_key,u.n_eligible_targets HAVING
                     count(*) FILTER(WHERE h.positive AND h.linux)<>u.n_eligible_targets)""")
                if split == "test":
                    zero("test_unreserved_users", f"SELECT count(*) FROM {users} ANTI JOIN reserved_test_accounts USING(user_key)")
                else:
                    zero("validation_query_not_in_validation", f"""SELECT count(*) FROM
                        (SELECT user_key,app_id,positive FROM {full}) h
                        ANTI JOIN validation_interactions USING(user_key,app_id,positive)""")
            counts = {
                **total_counts,
                "train_pairs_all_apps_all_labels": files["train/interactions.parquet"]["rows"],
                "train_positive_catalog_pairs": files["train/positive_catalog.parquet"]["rows"],
                "validation_pairs_all_apps_all_labels": files["validation/interactions.parquet"]["rows"],
                "validation_users": files["validation/users.parquet"]["rows"],
                "validation_query_rows": files["validation/query.parquet"]["rows"],
                "test_users": files["test/users.parquet"]["rows"],
                "test_query_rows": files["test/query.parquet"]["rows"],
                "reserved_test_accounts": files["reserved_test_accounts.parquet"]["rows"],
                "input_apps": files["catalog/input_apps.parquet"]["rows"],
                "linux_candidates": files["catalog/linux_candidates.parquet"]["rows"],
            }
            if "audited_source_counts" in manifest:
                expected = {"train_pairs_all_apps_all_labels": 98857701,
                            "train_positive_catalog_pairs": 78315268,
                            "validation_pairs_all_apps_all_labels": 10977308,
                            "train_accounts": 34014115, "validation_accounts": 3781518,
                            "positive_singleton_accounts": 17042189,
                            "test_users": 24342, "validation_users": 325639,
                            "reserved_test_accounts": 87573, "input_apps": 37610,
                            "linux_candidates": 6630}
                for key, value in expected.items():
                    if counts[key] != value:
                        raise ValueError(f"Final count differs from audited expectation: {key}: {counts[key]} != {value}")
            log("Read-back verification: all saved split invariants passed")
            return {"passed": True, "hashes_verified": verify_hashes,
                    "files_verified": len(files), "counts": counts, "zero_violation_checks": checks}
        finally:
            con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_split(args.root), indent=2))


if __name__ == "__main__":
    main()
