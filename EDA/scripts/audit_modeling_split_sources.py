"""Cross-check published split files against the read-only audited source DB."""

import hashlib
import json
import tempfile
from datetime import datetime, timezone

import duckdb

from build_modeling_splits import DEFAULT_OUTPUT, WORK, prepare_histories
from eda_large_review_frank import EDA, digest, log, quote


def main():
    root = DEFAULT_OUTPUT
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if manifest["status"] != "complete" or not manifest["validation"]["passed"]:
        raise ValueError("A published, verified split is required")
    checks = {}
    with tempfile.TemporaryDirectory(prefix="split_source_audit_") as temp:
        con = duckdb.connect(str(WORK), read_only=True)
        try:
            con.execute("SET threads=2")
            con.execute("SET memory_limit='3GB'")
            con.execute(f"SET temp_directory={quote(temp)}")
            con.execute(f"CREATE TEMP TABLE _frozen_quarantine AS SELECT * FROM read_parquet({quote(EDA/'data/reference/review_app_quarantine.parquet')})")
            prepare_histories(con)
            def zero(name, sql):
                checks[name] = con.execute(sql).fetchone()[0]
                if checks[name]:
                    raise ValueError(f"Source audit failed: {name}={checks[name]}")
            for split in ("train", "validation", "test"):
                con.execute(f"CREATE TEMP VIEW saved_{split}_lineage AS SELECT * FROM read_parquet({quote(root/split/'review_lineage.parquet')})")
            con.execute("""CREATE TEMP TABLE reserved_original_reviews AS SELECT f.review_key
                FROM frank_raw f SEMI JOIN test_users USING(user_key)""")
            con.execute("""CREATE TEMP TABLE all_bad_kieran_ids AS SELECT review_key FROM review_keys
                WHERE min_user<>max_user OR min_app<>max_app
                UNION SELECT review_key FROM cross_review_conflicts""")
            con.execute("""CREATE TEMP TABLE bad_kieran_pairs AS SELECT user_key,app_id FROM pair_groups
                WHERE min_label<>max_label""")
            for split in ("train", "validation"):
                zero(f"{split}_all_reserved_frank_review_id_intersections", f"""SELECT count(*)
                    FROM saved_{split}_lineage SEMI JOIN reserved_original_reviews USING(review_key)""")
                zero(f"{split}_quarantined_original_ids", f"SELECT count(*) FROM saved_{split}_lineage SEMI JOIN all_bad_kieran_ids USING(review_key)")
                zero(f"{split}_conflicting_label_pairs", f"SELECT count(*) FROM saved_{split}_lineage SEMI JOIN bad_kieran_pairs USING(user_key,app_id)")
            zero("test_quarantined_pairs", "SELECT count(*) FROM saved_test_lineage SEMI JOIN _ambiguous_frank_pairs USING(user_key,app_id)")
            for split in ("validation", "test"):
                for name in ("query", "targets", "users"):
                    con.execute(f"CREATE TEMP VIEW saved_{split}_{name} AS SELECT * FROM read_parquet({quote(root/split/(name+'.parquet'))})")
                zero(f"{split}_missing_or_extra_history_rows", f"""SELECT count(*) FROM (
                    ((SELECT * FROM {split}_hist SEMI JOIN saved_{split}_users USING(user_key))
                        EXCEPT ALL (SELECT * FROM saved_{split}_query UNION ALL SELECT * FROM saved_{split}_targets))
                    UNION ALL ((SELECT * FROM saved_{split}_query UNION ALL SELECT * FROM saved_{split}_targets)
                        EXCEPT ALL (SELECT * FROM {split}_hist SEMI JOIN saved_{split}_users USING(user_key))))""")
                rows = con.execute(f"""SELECT h.user_key,t.app_id,
                    list(h.app_id ORDER BY h.app_id) FILTER(WHERE h.positive AND h.linux) apps
                    FROM {split}_hist h JOIN saved_{split}_targets t USING(user_key)
                    GROUP BY h.user_key,t.app_id""").fetchall()
                errors = 0
                for user, selected, apps in rows:
                    n = len(apps)
                    limit = 2**256 - 2**256 % n
                    counter = 0
                    while True:
                        raw = f"steam-hidden-positive-linux-v1:{manifest['seed']}:{split}:{user}:{counter}".encode()
                        value = int(hashlib.sha256(raw).hexdigest(), 16)
                        if value < limit:
                            break
                        counter += 1
                    errors += selected != apps[value % n]
                checks[f"{split}_target_reproduction_errors"] = errors
                if errors:
                    raise ValueError(f"Target reproduction failed: {split}: {errors}")
                log(f"Source audit: {split} complete histories and {len(rows):,} seeded targets reproduced")
        finally:
            con.close()
    result = {"version": manifest["version"], "checked_utc": datetime.now(timezone.utc).isoformat(),
              "split_manifest_sha256": digest(root / "manifest.json"),
              "source_database_sha256_from_generation": manifest["sources"]["database_sha256"],
              "source_database_opened_read_only": True, "seed": manifest["seed"],
              "audit_code_sha256": digest(__file__), "passed": True,
              "zero_violation_checks": checks}
    path = EDA / "reports/modeling_split_v1_source_audit.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
