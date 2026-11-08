"""Count text support in frozen cohorts; no recommendations or ranking metrics."""

import json

import duckdb

from eda_large_review_frank import EDA, digest, one, quote
from tfidf_content import DEFAULT_INDEX


def main():
    root = DEFAULT_INDEX
    content = json.loads((root / "manifest.json").read_text("utf-8"))
    source = json.loads((EDA / "data/source_manifest.json").read_text("utf-8"))
    split = EDA / source["split_artifacts"]["directory"]
    if digest(split / "manifest.json") != content["sources"]["split_manifest_sha256"]:
        raise ValueError("Text index and evaluation split catalog versions differ")
    con = duckdb.connect()
    try:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='1GB'")
        con.execute(f"CREATE VIEW games AS SELECT * FROM read_parquet({quote(root/'game_rows.parquet')})")
        summaries = {}
        for name in ("validation", "test"):
            con.execute(f"CREATE OR REPLACE TEMP VIEW q AS SELECT * FROM read_parquet({quote(split/name/'query.parquet')})")
            con.execute(f"CREATE OR REPLACE TEMP VIEW t AS SELECT * FROM read_parquet({quote(split/name/'targets.parquet')})")
            con.execute("""CREATE OR REPLACE TEMP TABLE user_support AS SELECT q.user_key,
                count(*) query_rows,count(*) FILTER(WHERE q.positive) positive_inputs,
                count(*) FILTER(WHERE q.positive AND g.feature_count>0) supported_positive_inputs
                FROM q LEFT JOIN games g USING(app_id) GROUP BY q.user_key""")
            summaries[name] = one(con, """SELECT count(*) users,sum(query_rows) query_rows,
                sum(positive_inputs) positive_query_rows,sum(supported_positive_inputs) supported_positive_query_rows,
                count(*) FILTER(WHERE supported_positive_inputs=0) users_with_zero_text_profile,
                count(*) FILTER(WHERE supported_positive_inputs<positive_inputs) users_with_some_unsupported_positive_input
                FROM user_support""")
            summaries[name].update(one(con, """SELECT count(*) target_rows,
                count(*) FILTER(WHERE g.feature_count=0) unsupported_target_rows,
                count(*) FILTER(WHERE g.app_id IS NULL) targets_outside_text_catalog FROM t
                LEFT JOIN games g USING(app_id)"""))
            summaries[name]["query_rows_outside_text_catalog"] = con.execute(
                "SELECT count(*) FROM q ANTI JOIN games USING(app_id)").fetchone()[0]
            if summaries[name]["targets_outside_text_catalog"] or summaries[name]["query_rows_outside_text_catalog"]:
                raise ValueError(f"{name} contains apps outside the content catalog")
    finally:
        con.close()
    result = {"content_manifest_sha256": digest(root / "manifest.json"),
              "split_manifest_sha256": digest(split / "manifest.json"),
              "audit_code_sha256": digest(__file__), "cohorts": summaries,
              "scope": "Support counts only, after index construction; no feature/parameter selection, no cohort filtering and no ranking metrics.",
              "cohorts_unchanged": True, "ranking_metrics_computed": False}
    path = EDA / "reports/tfidf_content_v1_support.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
