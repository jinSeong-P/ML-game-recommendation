"""Build versioned supplemental evaluation cohorts from the frozen v1 sources.

This builder only materializes evaluation inputs. It does not score, select, or
train models. It keeps split_v1 untouched and reads its audited DuckDB cache in
read-only mode, with DuckDB capped at 2 GB and two threads.

Default output: EDA/data/processed/modeling/backlog_v3/cohorts_v3_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from eda_large_review_frank import EDA, digest, log, one, quote


VERSION = "modeling-backlog-supplement-v3"
DEFAULT_SEED = 20261107
DEFAULT_OUTPUT = EDA / "data/processed/modeling/backlog_v3/cohorts_v3_final"
SPLIT = EDA / "data/processed/modeling/split_v1"
WORK = EDA / "data/processed/kieran_original/eda_work.duckdb"
BASELINE_PROTOCOL = EDA / "data/reference/baseline_protocol_v1.json"
REFERENCE = EDA / "data/reference"
FRANK_PROJECTION = EDA / "data/processed/frank/reviews_identity_columns.parquet"

# Predeclared before any v3 outcome generation. Single-target selection reuses
# the v1 SHA256 rejection sampler and split namespace; multi-target selection
# uses SHA256 rank order without replacement.
HOLDOUT_NAMESPACE = "steam-hidden-positive-linux-v1"
MULTI_NAMESPACE = "steam-backlog-v3-two-targets-v1"
CAP_NAMESPACE = "steam-backlog-v3-app-cap100-v1"
MIN3_TOTAL_INPUTS = 3
MIN3_POSITIVE_INPUTS = 1
MULTI_TARGET_COUNT = 2
MULTI_MIN_TOTAL_INPUTS = 5
MULTI_MIN_POSITIVE_INPUTS = 1
APP_REVIEW_CAP = 100
DEFAULT_MEMORY_LIMIT = "2GB"
DEFAULT_THREADS = 2
ROW_GROUP_SIZE = 122_880

BASE_QUERY_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()),
    ("positive", pa.bool_()), ("linux", pa.bool_()),
])
BASE_TARGET_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()),
    ("positive", pa.bool_()), ("linux", pa.bool_()),
    ("target_order", pa.int64()),
])
USER_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("n_total_inputs", pa.int64()),
    ("n_positive_inputs", pa.int64()), ("n_eligible_targets", pa.int64()),
])
PAIR_SCHEMA = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64())])


def uniform_index(user_key: int, n: int, seed: int, split: str) -> int:
    """v1's stable unbiased target index, duplicated to keep this builder local."""
    if n < 1:
        raise ValueError("A target must have at least one eligible candidate")
    limit = (1 << 256) - (1 << 256) % n
    counter = 0
    while True:
        payload = f"{HOLDOUT_NAMESPACE}:{seed}:{split}:{int(user_key)}:{counter}"
        value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        if value < limit:
            return value % n
        counter += 1


def _safe_name(name: str) -> None:
    if not name or not (name[0].isalpha() or name[0] == "_") or not name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe cohort identifier: {name!r}")


def _insert_arrow(con: duckdb.DuckDBPyConnection, table: str, rows: list[tuple], schema: pa.Schema) -> None:
    """Insert a bounded batch of Python rows into a temporary DuckDB table."""
    if not rows:
        return
    batch = pa.Table.from_arrays(
        [pa.array([row[i] for row in rows], type=field.type) for i, field in enumerate(schema)],
        schema=schema,
    )
    con.register("_v3_batch", batch)
    try:
        con.execute(f"INSERT INTO {table} SELECT * FROM _v3_batch")
    finally:
        con.unregister("_v3_batch")


def create_single_holdout(
    con: duckdb.DuckDBPyConnection,
    history: str,
    prefix: str,
    seed: int,
    min_total: int = 5,
    min_positive: int = 1,
    selection_split: str | None = None,
) -> None:
    """Select one positive Linux target and export SQL temp cohort tables.

    Thresholds count unique catalog games after removing the one target. The
    history table must contain one row per user/app pair.
    """
    for name in (history, prefix):
        _safe_name(name)
    if min_total < 1 or min_positive < 1 or min_positive > min_total:
        raise ValueError("Invalid minimum query counts")
    split = selection_split or prefix
    function_name = f"_v3_uniform_target_{prefix}"

    def target_app(user_key: int, apps: list[int]) -> int:
        ordered = list(apps)
        return int(ordered[uniform_index(int(user_key), len(ordered), seed, split)])

    con.create_function(
        function_name, target_app,
        [duckdb.sqltype("UBIGINT"), duckdb.list_type(duckdb.sqltype("BIGINT"))],
        duckdb.sqltype("BIGINT"),
    )
    con.execute(f"""CREATE TEMP TABLE {prefix}_candidates AS
        SELECT user_key,
            count(*)-1 AS n_total_inputs,
            count(*) FILTER (WHERE positive)-1 AS n_positive_inputs,
            count(*) FILTER (WHERE positive AND linux) AS n_eligible_targets,
            list(app_id ORDER BY app_id) FILTER (WHERE positive AND linux) AS target_apps
        FROM {history}
        GROUP BY user_key
        HAVING count(*) >= {int(min_total)+1}
           AND count(*) FILTER (WHERE positive) >= {int(min_positive)+1}
           AND count(*) FILTER (WHERE positive AND linux) > 0
        ORDER BY user_key""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_users AS
        SELECT user_key,n_total_inputs,n_positive_inputs,n_eligible_targets
        FROM {prefix}_candidates""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_targets AS
        SELECT user_key,{function_name}(user_key,target_apps) AS app_id,
               true AS positive,true AS linux,1::BIGINT AS target_order
        FROM {prefix}_candidates""")
    con.remove_function(function_name)
    con.execute(f"DROP TABLE {prefix}_candidates")
    con.execute(f"""CREATE TEMP TABLE {prefix}_query AS
        SELECT h.user_key,h.app_id,h.positive,h.linux
        FROM {history} h SEMI JOIN {prefix}_users USING(user_key)
        ANTI JOIN {prefix}_targets USING(user_key,app_id)""")
    con.execute(f"CREATE TEMP TABLE {prefix}_exclusions AS SELECT user_key,app_id FROM {prefix}_query")


def create_multi_holdout(
    con: duckdb.DuckDBPyConnection,
    history: str,
    prefix: str,
    seed: int = DEFAULT_SEED,
    min_total: int = MULTI_MIN_TOTAL_INPUTS,
    min_positive: int = MULTI_MIN_POSITIVE_INPUTS,
    target_count: int = MULTI_TARGET_COUNT,
) -> None:
    """Select two targets by deterministic SHA256 candidate rank, without replacement."""
    for name in (history, prefix):
        _safe_name(name)
    if target_count < 2 or min_total < 1 or min_positive < 1 or min_positive > min_total:
        raise ValueError("Invalid multiple-target parameters")
    con.execute(f"""CREATE TEMP TABLE {prefix}_users AS
        SELECT user_key,
               count(*)-{int(target_count)} AS n_total_inputs,
               count(*) FILTER (WHERE positive)-{int(target_count)} AS n_positive_inputs,
               count(*) FILTER (WHERE positive AND linux) AS n_eligible_targets
        FROM {history}
        GROUP BY user_key
        HAVING count(*) >= {int(min_total)+int(target_count)}
           AND count(*) FILTER (WHERE positive) >= {int(min_positive)+int(target_count)}
           AND count(*) FILTER (WHERE positive AND linux) >= {int(target_count)}""")
    # SHA256 ordering gives a row-order-independent sample without materializing
    # every user's positive target list in Python memory.
    con.execute(f"""CREATE TEMP TABLE {prefix}_targets AS
        WITH ranked AS (
            SELECT h.user_key,h.app_id,h.positive,h.linux,
                   row_number() OVER (PARTITION BY h.user_key ORDER BY
                       sha256('{MULTI_NAMESPACE}:{seed}:{prefix}:' ||
                              CAST(h.user_key AS VARCHAR) || ':' || CAST(h.app_id AS VARCHAR)),
                       h.app_id) AS target_order
            FROM {history} h SEMI JOIN {prefix}_users USING(user_key)
            WHERE h.positive AND h.linux
        )
        SELECT user_key,app_id,positive,linux,target_order
        FROM ranked WHERE target_order <= {int(target_count)}""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_query AS
        SELECT h.user_key,h.app_id,h.positive,h.linux
        FROM {history} h SEMI JOIN {prefix}_users USING(user_key)
        ANTI JOIN {prefix}_targets USING(user_key,app_id)""")
    con.execute(f"CREATE TEMP TABLE {prefix}_exclusions AS SELECT user_key,app_id FROM {prefix}_query")


def create_capped_history(
    con: duckdb.DuckDBPyConnection,
    history: str,
    capped: str,
    cap: int = APP_REVIEW_CAP,
    seed: int = DEFAULT_SEED,
) -> None:
    """Keep the first cap distinct user/app pairs per app by SHA256 rank."""
    for name in (history, capped):
        _safe_name(name)
    if cap < 1:
        raise ValueError("Cap must be positive")
    con.execute(f"""CREATE TEMP TABLE {capped} AS
        WITH ranked AS (
            SELECT h.*,
                   row_number() OVER (PARTITION BY h.app_id ORDER BY
                       sha256('{CAP_NAMESPACE}:{seed}:' || CAST(h.app_id AS VARCHAR) || ':' ||
                              CAST(h.user_key AS VARCHAR)), h.user_key) AS cap_rank
            FROM {history} h
        )
        SELECT * EXCLUDE(cap_rank) FROM ranked WHERE cap_rank <= {int(cap)}""")


def create_temporal_holdout(
    con: duckdb.DuckDBPyConnection,
    history: str,
    train_cutoff: int,
    prefix: str = "temporal_validation",
    min_total: int = 5,
    min_positive: int = 1,
) -> dict[str, int | None]:
    """Use each validation user's latest post-cutoff positive Linux game as target.

    History has one row per user/app and first accepted review timestamp. Only
    targets later than the maximum accepted TRAIN review timestamp are eligible.
    Only events with timestamp strictly before the target enter the query. Ties
    in target timestamp resolve to ascending App ID. Users whose latest future
    positive Linux game lacks the required earlier history are omitted.
    """
    for name in (history, prefix):
        _safe_name(name)
    con.execute(f"""CREATE TEMP TABLE {prefix}_latest AS
        SELECT user_key,app_id,positive,linux,timestamp_created
        FROM {history}
        WHERE positive AND linux AND timestamp_created IS NOT NULL
          AND timestamp_created>{int(train_cutoff)}
        QUALIFY row_number() OVER (PARTITION BY user_key
            ORDER BY timestamp_created DESC,app_id ASC)=1""")
    availability = one(con, f"""WITH latest AS (
            SELECT user_key,app_id,timestamp_created FROM {history}
            WHERE positive AND linux AND timestamp_created IS NOT NULL
              AND timestamp_created>{int(train_cutoff)}
            QUALIFY row_number() OVER (PARTITION BY user_key
                ORDER BY timestamp_created DESC,app_id ASC)=1
        ), prior AS (
            SELECT t.user_key,t.app_id,t.timestamp_created,
                   count(h.app_id) AS n_prior_total,
                   count(h.app_id) FILTER (WHERE h.positive) AS n_prior_positive
            FROM latest t LEFT JOIN {history} h
              ON h.user_key=t.user_key AND h.timestamp_created<t.timestamp_created
            GROUP BY t.user_key,t.app_id,t.timestamp_created
        )
        SELECT (SELECT count(*) FROM {history}
                WHERE positive AND linux AND timestamp_created>{int(train_cutoff)}) AS post_cutoff_positive_linux_pairs,
               count(*) AS users_with_latest_post_cutoff_target,
               count(*) FILTER (WHERE n_prior_total>={int(min_total)}
                                AND n_prior_positive>={int(min_positive)}) AS users_meeting_prior_input_rules,
               count(*) FILTER (WHERE n_prior_total<{int(min_total)}) AS users_short_on_prior_total,
               count(*) FILTER (WHERE n_prior_positive<{int(min_positive)}) AS users_short_on_prior_positive,
               coalesce(sum(n_prior_total),0) AS sum_prior_pairs,
               max(timestamp_created) AS latest_target_timestamp_created
        FROM prior""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_users AS
        SELECT t.user_key,
               count(h.app_id) AS n_total_inputs,
               count(h.app_id) FILTER (WHERE h.positive) AS n_positive_inputs,
               max(x.n_eligible_targets) AS n_eligible_targets
        FROM {prefix}_latest t
        JOIN {history} h ON h.user_key=t.user_key AND h.timestamp_created<t.timestamp_created
        JOIN (SELECT user_key,count(*) FILTER (WHERE positive AND linux
                         AND timestamp_created>{int(train_cutoff)}) n_eligible_targets
              FROM {history} GROUP BY user_key) x ON x.user_key=t.user_key
        GROUP BY t.user_key
        HAVING count(h.app_id)>={int(min_total)}
           AND count(h.app_id) FILTER (WHERE h.positive)>={int(min_positive)}""")
    # Query rows deliberately contain no same-time or future review events.
    con.execute(f"""CREATE TEMP TABLE {prefix}_targets AS
        SELECT t.user_key,t.app_id,t.positive,t.linux,1::BIGINT AS target_order,
               t.timestamp_created
        FROM {prefix}_latest t SEMI JOIN {prefix}_users USING(user_key)""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_query AS
        SELECT h.user_key,h.app_id,h.positive,h.linux,h.timestamp_created,h.playtime_forever
        FROM {history} h SEMI JOIN {prefix}_users USING(user_key)
        JOIN {prefix}_targets t ON t.user_key=h.user_key
        WHERE h.timestamp_created<t.timestamp_created""")
    con.execute(f"CREATE TEMP TABLE {prefix}_exclusions AS SELECT user_key,app_id FROM {prefix}_query")
    return {key: (int(value) if value is not None else None) for key, value in availability.items()}


def validate_cohort(
    con: duckdb.DuckDBPyConnection,
    prefix: str,
    min_total: int,
    min_positive: int,
    target_count: int = 1,
    input_apps: str = "_v3_input_apps",
    linux_candidates: str = "_v3_linux_candidates",
) -> dict[str, int]:
    """Check cohort uniqueness, thresholds, masks, and frozen-catalog membership."""
    _safe_name(prefix)
    checks = one(con, f"""SELECT
        (SELECT count(*)-count(DISTINCT user_key) FROM {prefix}_users) duplicate_users,
        (SELECT count(*) FROM (SELECT user_key,app_id,count(*) n FROM {prefix}_query GROUP BY 1,2 HAVING n<>1)) duplicate_query_pairs,
        (SELECT count(*) FROM (SELECT user_key,app_id,count(*) n FROM {prefix}_targets GROUP BY 1,2 HAVING n<>1)) duplicate_targets,
        (SELECT count(*) FROM (SELECT user_key,count(*) n FROM {prefix}_targets GROUP BY 1 HAVING n<>{int(target_count)})) target_count_errors,
        (SELECT count(*) FROM {prefix}_targets WHERE positive IS DISTINCT FROM TRUE OR linux IS DISTINCT FROM TRUE) invalid_targets,
        (SELECT count(*) FROM {prefix}_targets t JOIN {prefix}_query q USING(user_key,app_id)) target_query_overlap,
        (SELECT count(*) FROM {prefix}_targets t JOIN {prefix}_exclusions x USING(user_key,app_id)) masked_targets,
        (SELECT count(*) FROM {prefix}_users WHERE n_total_inputs<{int(min_total)} OR n_positive_inputs<{int(min_positive)}) ineligible_users,
        (SELECT count(*) FROM (
            SELECT u.user_key FROM {prefix}_users u LEFT JOIN {prefix}_query q USING(user_key)
            GROUP BY u.user_key,u.n_total_inputs,u.n_positive_inputs
            HAVING count(q.app_id)<>u.n_total_inputs OR
                   count(q.app_id) FILTER (WHERE q.positive)<>u.n_positive_inputs
        )) bad_user_counts,
        (SELECT count(*) FROM (
            (SELECT user_key,app_id FROM {prefix}_query EXCEPT SELECT * FROM {prefix}_exclusions)
            UNION ALL
            (SELECT * FROM {prefix}_exclusions EXCEPT SELECT user_key,app_id FROM {prefix}_query)
        )) mask_differences,
        (SELECT count(*) FROM {prefix}_query q ANTI JOIN {input_apps} c USING(app_id)) query_outside_input_catalog,
        (SELECT count(*) FROM {prefix}_targets t ANTI JOIN {linux_candidates} c USING(app_id)) target_outside_linux_catalog""")
    if any(checks.values()):
        raise ValueError(f"Cohort validation failed: {prefix}: {checks}")
    return checks


def write_parquet(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO {quote(path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})")
    log(f"Wrote {path.parent.name}/{path.name}: {pq.ParquetFile(path).metadata.num_rows:,} rows")


def file_inventory(root: Path, exclude_manifest: bool = True) -> dict[str, dict]:
    inventory = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root).as_posix()
        parquet = pq.ParquetFile(path)
        inventory[relative] = {
            "rows": parquet.metadata.num_rows,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "schema": [{"name": f.name, "type": str(f.type)} for f in parquet.schema_arrow],
        }
    return inventory


def export_cohort(
    con: duckdb.DuckDBPyConnection,
    stage: Path,
    name: str,
    prefix: str,
    metadata: dict,
    extras: dict[str, str] | None = None,
) -> dict:
    """Verify then export standardized cohort files plus named SQL extra tables."""
    root = stage / name
    root.mkdir(parents=True, exist_ok=False)
    for table, filename, order in [
        (f"{prefix}_users", "users.parquet", "user_key"),
        (f"{prefix}_query", "query.parquet", "user_key,app_id"),
        (f"{prefix}_targets", "targets.parquet", "user_key,target_order"),
        (f"{prefix}_exclusions", "exclusions.parquet", "user_key,app_id"),
    ]:
        write_parquet(con, f"SELECT * FROM {table} ORDER BY {order}", root / filename)
    for filename, table in (extras or {}).items():
        write_parquet(con, f"SELECT * FROM {table} ORDER BY ALL", root / filename)
    inv = file_inventory(root)
    n_users = inv["users.parquet"]["rows"]
    cohort_manifest = {
        "version": VERSION,
        "status": "complete",
        "cohort": name,
        "role": "supplemental; does not replace or mutate split_v1 or any v1/v2 result",
        "parameters": metadata,
        "users": n_users,
        "files": inv,
    }
    (root / "manifest.json").write_text(json.dumps(cohort_manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return cohort_manifest


def _setup_sources(con: duckdb.DuckDBPyConnection, split_manifest: dict) -> dict:
    """Load only bounded cohort-relevant rows and pin the frozen sources."""
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("split_v1 is not a verified completed split")
    baseline = json.loads(BASELINE_PROTOCOL.read_text("utf-8"))
    split_sha = digest(SPLIT / "manifest.json")
    if split_sha != baseline.get("split_manifest_sha256"):
        raise ValueError("split_v1 manifest differs from the frozen baseline protocol")
    ref_manifest = json.loads((REFERENCE / "manifest.json").read_text("utf-8"))
    if digest(REFERENCE / "manifest.json") != split_manifest["sources"]["reference_manifest_sha256"]:
        raise ValueError("Reference manifest no longer matches split_v1")
    if digest(WORK) != split_manifest["sources"]["database_sha256"]:
        raise ValueError("Read-only Kieran/frank DuckDB cache differs from split_v1")
    if digest(FRANK_PROJECTION) != split_manifest["sources"]["frank_projection_sha256"]:
        raise ValueError("Frozen frank projection differs from split_v1")
    for filename, info in ref_manifest["files"].items():
        path = REFERENCE / filename
        if digest(path) != info["sha256"]:
            raise ValueError(f"Frozen reference input changed: {filename}")

    con.execute(f"CREATE TEMP TABLE _v3_input_apps AS SELECT * FROM read_parquet({quote(SPLIT/'catalog/input_apps.parquet')})")
    con.execute(f"CREATE TEMP TABLE _v3_linux_candidates AS SELECT * FROM read_parquet({quote(SPLIT/'catalog/linux_candidates.parquet')})")
    catalog_diff = one(con, """SELECT
        (SELECT count(*) FROM ((SELECT app_id,linux FROM catalog WHERE reviewed EXCEPT SELECT * FROM _v3_input_apps)
                               UNION ALL (SELECT * FROM _v3_input_apps EXCEPT SELECT app_id,linux FROM catalog WHERE reviewed))) input_differences,
        (SELECT count(*) FROM ((SELECT app_id FROM catalog WHERE reviewed AND linux EXCEPT SELECT app_id FROM _v3_linux_candidates)
                               UNION ALL (SELECT app_id FROM _v3_linux_candidates EXCEPT SELECT app_id FROM catalog WHERE reviewed AND linux))) candidate_differences""")
    if any(catalog_diff.values()):
        raise ValueError(f"Frozen catalog mismatch: {catalog_diff}")

    con.execute(f"CREATE TEMP TABLE _v3_quarantine AS SELECT * FROM read_parquet({quote(REFERENCE/'review_app_quarantine.parquet')})")
    con.execute("""CREATE TEMP TABLE _v3_bad_original_ids AS
        SELECT review_key FROM review_keys WHERE min_user<>max_user OR min_app<>max_app""")
    con.execute("""CREATE TEMP TABLE _v3_bad_label_pairs AS
        SELECT user_key,app_id FROM pair_groups WHERE min_label<>max_label""")
    train_cutoff = one(con, """SELECT max(r.timestamp_created) AS timestamp_created,
               count(*) AS accepted_train_review_events
        FROM valid_raw r
        SEMI JOIN _v3_input_apps a USING(app_id)
        ANTI JOIN _v3_bad_original_ids USING(review_key)
        ANTI JOIN cross_review_conflicts USING(review_key)
        ANTI JOIN _v3_bad_label_pairs USING(user_key,app_id)
        ANTI JOIN test_users USING(user_key)
        WHERE try_cast('0x' || substr(md5('steam-eda-validation-v1:' || cast(r.user_key AS VARCHAR)),1,15)
                      AS UBIGINT) % 10 <> 0""")
    if train_cutoff["timestamp_created"] is None:
        raise ValueError("No accepted fixed-catalog training review timestamp for temporal cutoff")
    con.execute("""CREATE TEMP TABLE _v3_eval_users AS
        SELECT user_key FROM user_stats
        WHERE is_test OR (NOT is_test AND is_validation)""")
    # Bound the large canonical review join to held-out accounts and the frozen
    # 37,610-app input catalog before touching per-review timestamps.
    con.execute("""CREATE TEMP TABLE _v3_eval_pairs AS
        SELECT c.user_key,c.app_id,c.positive,a.linux
        FROM clean c SEMI JOIN _v3_eval_users USING(user_key)
        JOIN catalog a USING(app_id)
        WHERE a.reviewed""")
    # First accepted review per user/app supplies temporal ordering and, where
    # available, a Kieran review-time playtime snapshot. Work stays inside SQL;
    # no full raw review table is fetched into Python.
    con.execute("""CREATE TEMP TABLE _v3_kieran_events AS
        SELECT user_key,app_id,positive,linux,timestamp_created,playtime_forever
        FROM (
            SELECT r.user_key,r.app_id,r.positive,p.linux,r.timestamp_created,r.playtime_forever,
                   row_number() OVER (PARTITION BY r.user_key,r.app_id
                       ORDER BY r.timestamp_created ASC NULLS LAST,r.review_key ASC) AS rn
            FROM valid_raw r
            JOIN _v3_eval_pairs p USING(user_key,app_id,positive)
            ANTI JOIN _v3_bad_original_ids USING(review_key)
            ANTI JOIN cross_review_conflicts USING(review_key)
        ) ranked WHERE rn=1""")
    con.execute("""CREATE TEMP TABLE _v3_validation_hist AS
        SELECT e.* FROM _v3_kieran_events e JOIN user_stats u USING(user_key)
        WHERE NOT u.is_test AND u.is_validation""")
    con.execute("""CREATE TEMP TABLE _v3_reserved_kieran_hist AS
        SELECT e.* FROM _v3_kieran_events e JOIN user_stats u USING(user_key)
        WHERE u.is_test""")

    con.execute("""CREATE TEMP TABLE _v3_ambiguous_frank_pairs AS
        SELECT DISTINCT f.user_key,f.app_id FROM frank_raw f JOIN _v3_quarantine USING(review_key)""")
    con.execute("""CREATE TEMP TABLE _v3_frank_hist AS
        SELECT user_key,app_id,positive,linux,timestamp_created,playtime_forever
        FROM (
            SELECT r.user_key,r.app_id,r.positive,c.linux,r.timestamp_created,
                   NULL::DOUBLE AS playtime_forever,
                   row_number() OVER (PARTITION BY r.user_key,r.app_id
                       ORDER BY r.timestamp_created ASC NULLS LAST,r.review_key ASC) AS rn
            FROM frank_raw r
            JOIN frank p USING(user_key,app_id,positive)
            SEMI JOIN test_users USING(user_key)
            JOIN catalog c USING(app_id)
            ANTI JOIN _v3_ambiguous_frank_pairs USING(user_key,app_id)
            WHERE c.reviewed
        ) ranked WHERE rn=1""")
    counts = one(con, """SELECT
        (SELECT count(*) FROM _v3_input_apps) input_apps,
        (SELECT count(*) FROM _v3_linux_candidates) linux_candidates,
        (SELECT count(*) FROM _v3_validation_hist) validation_history_pairs,
        (SELECT count(DISTINCT user_key) FROM _v3_validation_hist) validation_accounts_with_catalog_history,
        (SELECT count(*) FROM _v3_reserved_kieran_hist) reserved_kieran_pairs,
        (SELECT count(DISTINCT user_key) FROM _v3_reserved_kieran_hist) reserved_accounts_with_kieran_history,
        (SELECT count(*) FROM _v3_frank_hist) frank_history_pairs,
        (SELECT count(DISTINCT user_key) FROM _v3_frank_hist) frank_accounts_with_catalog_history""")
    if counts["input_apps"] != 37610 or counts["linux_candidates"] != 6630:
        raise ValueError(f"Unexpected frozen catalog sizes: {counts}")
    return {
        "split_manifest_sha256": split_sha,
        "split_version": split_manifest["version"],
        "database_sha256": split_manifest["sources"]["database_sha256"],
        "reference_manifest_sha256": digest(REFERENCE / "manifest.json"),
        "frank_projection_sha256": digest(FRANK_PROJECTION),
        "input_catalog": {"path": "split_v1/catalog/input_apps.parquet", "sha256": split_manifest["files"]["catalog/input_apps.parquet"]["sha256"], "rows": counts["input_apps"]},
        "linux_candidate_catalog": {"path": "split_v1/catalog/linux_candidates.parquet", "sha256": split_manifest["files"]["catalog/linux_candidates.parquet"]["sha256"], "rows": counts["linux_candidates"]},
        "counts": counts,
        "temporal_train_cutoff": {
            "timestamp_created": int(train_cutoff["timestamp_created"]),
            "utc": datetime.fromtimestamp(int(train_cutoff["timestamp_created"]), timezone.utc).isoformat(),
            "accepted_train_review_events_scanned": int(train_cutoff["accepted_train_review_events"]),
            "definition": "Maximum timestamp_created over accepted Kieran review rows from non-reserved, non-validation accounts in the frozen input catalog.",
        },
    }


def _copy_base_frank(con: duckdb.DuckDBPyConnection) -> None:
    """Attach exact v1 test cohort, then append timestamps to selected targets."""
    for table, path in [
        ("_v3_frank_base_users", SPLIT/"test/users.parquet"),
        ("_v3_frank_base_query", SPLIT/"test/query.parquet"),
        ("_v3_frank_base_targets", SPLIT/"test/targets.parquet"),
        ("_v3_frank_base_exclusions", SPLIT/"test/exclusions.parquet"),
    ]:
        con.execute(f"CREATE TEMP TABLE {table} AS SELECT * FROM read_parquet({quote(path)})")
    con.execute("""CREATE TEMP TABLE _v3_frank_base_target_times AS
        SELECT t.user_key,t.app_id,t.positive,t.linux,1::BIGINT AS target_order,
               min(r.timestamp_created) AS timestamp_created
        FROM _v3_frank_base_targets t
        JOIN frank_raw r USING(user_key,app_id,positive)
        ANTI JOIN _v3_ambiguous_frank_pairs USING(user_key,app_id)
        GROUP BY t.user_key,t.app_id,t.positive,t.linux""")
    missing = con.execute("""SELECT count(*) FROM _v3_frank_base_targets t
        ANTI JOIN _v3_frank_base_target_times x USING(user_key,app_id)""").fetchone()[0]
    if missing:
        raise ValueError(f"Frank v1 targets missing accepted timestamps: {missing}")


def _create_frank_augmentation(con: duckdb.DuckDBPyConnection, suffix: str, before_target: bool) -> None:
    prefix = f"frank_aug_{suffix}"
    time_predicate = "AND k.timestamp_created<t.timestamp_created" if before_target else ""
    con.execute(f"""CREATE TEMP TABLE {prefix}_added AS
        SELECT k.user_key,k.app_id,k.positive,k.linux,k.timestamp_created,k.playtime_forever
        FROM _v3_reserved_kieran_hist k
        JOIN _v3_frank_base_target_times t ON t.user_key=k.user_key
        WHERE NOT EXISTS (SELECT 1 FROM _v3_frank_base_query f
                          WHERE f.user_key=k.user_key AND f.app_id=k.app_id)
          AND NOT EXISTS (SELECT 1 FROM _v3_frank_base_targets target
                          WHERE target.user_key=k.user_key AND target.app_id=k.app_id)
          {time_predicate}""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_users AS
        SELECT u.user_key,
               count(*) FILTER (WHERE q.app_id IS NOT NULL) AS n_total_inputs,
               count(*) FILTER (WHERE q.positive) AS n_positive_inputs,
               max(u.n_eligible_targets) AS n_eligible_targets
        FROM _v3_frank_base_users u
        LEFT JOIN (
            SELECT user_key,app_id,positive FROM _v3_frank_base_query
            UNION ALL
            SELECT user_key,app_id,positive FROM {prefix}_added
        ) q USING(user_key)
        GROUP BY u.user_key""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_query AS
        SELECT user_key,app_id,positive,linux FROM _v3_frank_base_query
        UNION ALL
        SELECT a.user_key,a.app_id,a.positive,a.linux FROM {prefix}_added a""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_targets AS
        SELECT user_key,app_id,positive,linux,target_order FROM _v3_frank_base_target_times""")
    con.execute(f"CREATE TEMP TABLE {prefix}_exclusions AS SELECT user_key,app_id FROM {prefix}_query")


def _create_kieran_only_mask(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""CREATE TEMP TABLE frank_kieran_only_positive_mask_users AS
        SELECT * FROM _v3_frank_base_users""")
    con.execute("""CREATE TEMP TABLE frank_kieran_only_positive_mask_query AS
        SELECT * FROM _v3_frank_base_query""")
    con.execute("""CREATE TEMP TABLE frank_kieran_only_positive_mask_targets AS
        SELECT user_key,app_id,positive,linux,1::BIGINT AS target_order
        FROM _v3_frank_base_target_times""")
    con.execute("""CREATE TEMP TABLE frank_kieran_only_positive_mask_exclusions AS
        SELECT * FROM _v3_frank_base_exclusions""")
    con.execute("""CREATE TEMP TABLE _v3_kieran_only_positive_mask AS
        SELECT k.user_key,k.app_id
        FROM _v3_reserved_kieran_hist k
        JOIN _v3_frank_base_users u USING(user_key)
        WHERE k.positive AND k.linux
          AND NOT EXISTS (SELECT 1 FROM _v3_frank_base_query f
                          WHERE f.user_key=k.user_key AND f.app_id=k.app_id)
          AND NOT EXISTS (SELECT 1 FROM _v3_frank_base_targets t
                          WHERE t.user_key=k.user_key AND t.app_id=k.app_id)""")


def _write_all(con: duckdb.DuckDBPyConnection, stage: Path, seed: int, source_info: dict, split_manifest: dict) -> dict:
    reports: dict[str, dict] = {}
    account_checks: dict[str, dict[str, int]] = {}

    def make(name: str, prefix: str, history: str, min_total: int, min_positive: int,
             selection_split: str, rule: dict) -> None:
        create_single_holdout(con, history, prefix, seed, min_total, min_positive, selection_split)
        checks = validate_cohort(con, prefix, min_total, min_positive)
        # Validation and reserved-account cohorts must remain in their reserved partition.
        if selection_split == "validation":
            leak = con.execute(f"""SELECT count(*) FROM {prefix}_users u
                JOIN user_stats s USING(user_key) WHERE s.is_test OR NOT s.is_validation""").fetchone()[0]
            if leak:
                raise ValueError(f"{name} includes non-validation accounts: {leak}")
        elif selection_split == "reserved_kieran":
            leak = con.execute(f"""SELECT count(*) FROM {prefix}_users u
                LEFT JOIN test_users t USING(user_key) WHERE t.user_key IS NULL""").fetchone()[0]
            if leak:
                raise ValueError(f"{name} includes non-reserved accounts: {leak}")
        else:
            leak = con.execute(f"""SELECT count(*) FROM {prefix}_users u
                LEFT JOIN test_users t USING(user_key) WHERE t.user_key IS NULL""").fetchone()[0]
            if leak:
                raise ValueError(f"{name} includes non-frank-reserved accounts: {leak}")
        account_checks[name] = {"non_reserved_or_wrong_partition_users": int(leak)}
        reports[name] = export_cohort(con, stage, name, prefix, {**rule, "seed": seed, "validation_checks": checks})

    make("min3_validation", "min3_validation", "_v3_validation_hist",
         MIN3_TOTAL_INPUTS, MIN3_POSITIVE_INPUTS, "validation",
         {"rule": "One uniformly selected positive Linux target; after hiding it, require at least 3 total inputs and 1 positive input.",
          "target_selection_namespace": f"{HOLDOUT_NAMESPACE}:{seed}:validation"})
    make("min3_frank", "min3_frank", "_v3_frank_hist",
         MIN3_TOTAL_INPUTS, MIN3_POSITIVE_INPUTS, "test",
         {"rule": "One uniformly selected positive Linux frank target; after hiding it, require at least 3 total inputs and 1 positive input.",
          "target_selection_namespace": f"{HOLDOUT_NAMESPACE}:{seed}:test"})

    for name, history, selection_split in [
        ("multi2_validation", "_v3_validation_hist", "validation"),
        ("multi2_frank", "_v3_frank_hist", "test"),
    ]:
        prefix = name
        create_multi_holdout(con, history, prefix, seed)
        checks = validate_cohort(con, prefix, MULTI_MIN_TOTAL_INPUTS, MULTI_MIN_POSITIVE_INPUTS, MULTI_TARGET_COUNT)
        if selection_split == "validation":
            leak = con.execute(f"""SELECT count(*) FROM {prefix}_users u JOIN user_stats s USING(user_key)
                WHERE s.is_test OR NOT s.is_validation""").fetchone()[0]
        else:
            leak = con.execute(f"""SELECT count(*) FROM {prefix}_users u LEFT JOIN test_users t USING(user_key)
                WHERE t.user_key IS NULL""").fetchone()[0]
        if leak:
            raise ValueError(f"{name} account partition mismatch: {leak}")
        account_checks[name] = {"non_reserved_or_wrong_partition_users": int(leak)}
        reports[name] = export_cohort(con, stage, name, prefix,
            {"rule": "Hide exactly 2 positive Linux targets selected by SHA256 rank without replacement; require at least 5 total and 1 positive remaining input.",
             "target_count": MULTI_TARGET_COUNT, "target_selection_namespace": f"{MULTI_NAMESPACE}:{seed}:{prefix}",
             "validation_checks": checks})

    create_capped_history(con, "_v3_validation_hist", "_v3_validation_cap100_hist", APP_REVIEW_CAP, seed)
    cap_counts = one(con, """SELECT count(*) apps,count(*) FILTER(WHERE cap_rows>100) apps_over_cap,
        max(cap_rows) max_rows_per_app,sum(cap_rows) retained_rows FROM
        (SELECT app_id,count(*) cap_rows FROM _v3_validation_cap100_hist GROUP BY app_id)""")
    if cap_counts["apps_over_cap"]:
        raise ValueError(f"Review cap exceeded: {cap_counts}")
    con.execute("""CREATE TEMP TABLE _v3_validation_cap100_audit AS
        SELECT b.app_id,count(*) AS validation_rows_before,
               count(*) FILTER (WHERE c.user_key IS NOT NULL) AS retained_rows_after
        FROM _v3_validation_hist b LEFT JOIN _v3_validation_cap100_hist c USING(user_key,app_id)
        GROUP BY b.app_id""")
    create_single_holdout(con, "_v3_validation_cap100_hist", "cap100_validation", seed, 5, 1, "cap100_validation")
    checks = validate_cohort(con, "cap100_validation", 5, 1)
    leak = con.execute("""SELECT count(*) FROM cap100_validation_users u JOIN user_stats s USING(user_key)
        WHERE s.is_test OR NOT s.is_validation""").fetchone()[0]
    if leak:
        raise ValueError(f"cap100_validation includes non-validation accounts: {leak}")
    account_checks["cap100_validation"] = {"non_reserved_or_wrong_partition_users": int(leak)}
    reports["cap100_validation"] = export_cohort(con, stage, "cap100_validation", "cap100_validation",
        {"rule": "First cap distinct held-out Kieran validation user/app pairs per app by ascending SHA256 rank; then hide one positive Linux target and require at least 5 total and 1 positive remaining input.",
         "per_app_review_cap": APP_REVIEW_CAP, "cap_namespace": f"{CAP_NAMESPACE}:{seed}",
         "cap_counts": cap_counts, "validation_checks": checks},
        {"app_cap_audit.parquet": "_v3_validation_cap100_audit"})

    _copy_base_frank(con)
    _create_frank_augmentation(con, "all", False)
    _create_frank_augmentation(con, "before_target_time", True)
    for name, prefix, history_mode in [
        ("frank_augmented_all", "frank_aug_all", "all accepted reserved-account Kieran catalog history, excluding Frank query/target duplicates"),
        ("frank_augmented_before_target_time", "frank_aug_before_target_time", "reserved-account Kieran catalog history with first accepted timestamp strictly earlier than the Frank target; Frank v1 query remains intact"),
    ]:
        checks = validate_cohort(con, prefix, 5, 1)
        leak = con.execute(f"""SELECT count(*) FROM {prefix}_users u LEFT JOIN test_users t USING(user_key)
            WHERE t.user_key IS NULL""").fetchone()[0]
        if leak:
            raise ValueError(f"{name} includes non-reserved accounts: {leak}")
        account_checks[name] = {"non_reserved_or_wrong_partition_users": int(leak)}
        reports[name] = export_cohort(con, stage, name, prefix,
            {"rule": "Exact frozen frank v1 test users and targets; union their frank query with additional Kieran reviewed-catalog pairs. Frank rows win user/app overlaps; hidden target is always removed from augmentation.",
             "kieran_history_mode": history_mode,
             "uses_kieran_review_time_playtime": "augmentation.parquet includes first-review timestamp and Kieran review-time playtime snapshot where present; frank review-time playtime is absent from the frozen review projection.",
             "validation_checks": checks},
            {"augmentation.parquet": f"{prefix}_added"})

    _create_kieran_only_mask(con)
    prefix = "frank_kieran_only_positive_mask"
    checks = validate_cohort(con, prefix, 5, 1)
    addmask = one(con, """SELECT count(*) additional_mask_rows,count(DISTINCT user_key) affected_users
        FROM _v3_kieran_only_positive_mask""")
    reports[prefix] = export_cohort(con, stage, prefix, prefix,
        {"rule": "Exact frank v1 users, query, and targets; add only reserved-account Kieran-only positive Linux catalog games to the candidate exclusion mask. This is a candidate-mask sensitivity cohort, not an input augmentation.",
         "additional_exclusion_rows": addmask["additional_mask_rows"], "affected_users": addmask["affected_users"],
         "selected_targets_reincluded": True, "validation_checks": checks},
        {"additional_exclusions.parquet": "_v3_kieran_only_positive_mask"})

    make("reserved_kieran", "reserved_kieran", "_v3_reserved_kieran_hist", 5, 1,
         "reserved_kieran",
         {"rule": "Auxiliary Kieran holdout among frank-reserved accounts only; one positive Linux target hidden, then require at least 5 total and 1 positive remaining input.",
          "target_selection_namespace": f"{HOLDOUT_NAMESPACE}:{seed}:reserved_kieran"})

    temporal_cutoff = source_info["temporal_train_cutoff"]["timestamp_created"]
    temporal_availability = create_temporal_holdout(con, "_v3_validation_hist", temporal_cutoff,
                                                    "temporal_validation", 5, 1)
    temporal_checks = validate_cohort(con, "temporal_validation", 5, 1)
    temporal_leak = con.execute("""SELECT count(*) FROM temporal_validation_users u JOIN user_stats s USING(user_key)
        WHERE s.is_test OR NOT s.is_validation""").fetchone()[0]
    if temporal_leak:
        raise ValueError(f"temporal_validation includes non-validation accounts: {temporal_leak}")
    account_checks["temporal_validation"] = {"non_reserved_or_wrong_partition_users": int(temporal_leak)}
    reports["temporal_validation"] = export_cohort(con, stage, "temporal_validation", "temporal_validation",
        {"rule": "For each held-out validation account, choose its latest positive Linux game whose first accepted review timestamp is later than the global accepted TRAIN review cutoff; include only unique app reviews strictly earlier than that timestamp and require at least 5 total and 1 positive input.",
         "global_train_cutoff": source_info["temporal_train_cutoff"],
         "temporal_availability_audit": temporal_availability,
         "future_target_rule": "timestamp_created > global maximum TRAIN timestamp_created",
         "target_timestamp_semantics": "Earliest accepted timestamp_created per user/app; not timestamp_updated.",
         "query_timestamp_comparison": "strictly less than target timestamp; same-time events are excluded.",
         "validation_checks": temporal_checks},
        {"target_timestamps.parquet": "temporal_validation_targets"})

    # Temporal and augmented variants carry a few columns beyond the common
    # schema; reports make these explicit instead of hiding the data semantics.
    return {"cohorts": reports, "account_partition_checks": account_checks,
            "source_counts": source_info["counts"],
            "unchanged_primary": {"split_v1_manifest_sha256": source_info["split_manifest_sha256"],
                                 "validation_users": split_manifest["files"]["validation/users.parquet"]["rows"],
                                 "frank_users": split_manifest["files"]["test/users.parquet"]["rows"]}}


def build(output: Path = DEFAULT_OUTPUT, seed: int = DEFAULT_SEED,
          memory_limit: str = DEFAULT_MEMORY_LIMIT, threads: int = DEFAULT_THREADS) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing supplemental cohort folder: {output}")
    if threads != DEFAULT_THREADS:
        raise ValueError("This builder is fixed at two DuckDB threads")
    if memory_limit not in {"512MB", "1GB", "1536MB", "2GB"}:
        raise ValueError("DuckDB memory_limit must be at or below 2GB")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.building-{uuid.uuid4().hex[:10]}"
    stage.mkdir()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        split_manifest = json.loads((SPLIT / "manifest.json").read_text("utf-8"))
        manifest = {
            "version": VERSION, "status": "building", "seed": seed,
            "started_utc": started,
            "scope": "Supplemental evaluation cohorts only. No model fitting or ranking; split_v1/v2 outputs are immutable references.",
            "predeclared_parameters": {
                "min3": {"minimum_total_inputs_after_one_target": MIN3_TOTAL_INPUTS, "minimum_positive_inputs_after_one_target": MIN3_POSITIVE_INPUTS},
                "multi_target": {"target_count": MULTI_TARGET_COUNT, "minimum_total_inputs_after_hiding_targets": MULTI_MIN_TOTAL_INPUTS, "minimum_positive_inputs_after_hiding_targets": MULTI_MIN_POSITIVE_INPUTS, "target_selection": "ascending SHA256 rank per user, without replacement"},
                "app_cap": {"maximum_validation_pairs_per_app": APP_REVIEW_CAP, "selection": "ascending SHA256 rank by app_id/user_key; then existing single target rules"},
                "temporal": {"target": "latest positive Linux app by first accepted timestamp_created after the global accepted TRAIN timestamp cutoff", "strictly_prior_inputs": True, "minimum_prior_total": 5, "minimum_prior_positive": 1},
                "frank_augmentation": {"base_user_target_catalog_and_ranker_protocol": "exact split_v1/test", "new_inputs": "deduplicated reserved-account Kieran user/app rows absent from frank query and hidden target"},
            },
            "runtime": {"python": platform.python_version(), "duckdb": duckdb.__version__, "pyarrow": pa.__version__, "threads": threads, "memory_limit": memory_limit},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="backlog_v3_duckdb_") as temp:
            con = duckdb.connect(str(WORK), read_only=True)
            try:
                con.execute(f"SET threads={threads}")
                con.execute(f"SET memory_limit={quote(memory_limit)}")
                con.execute("SET preserve_insertion_order=false")
                con.execute(f"SET temp_directory={quote(temp)}")
                log("Verifying frozen split, catalog, frank projection, and DuckDB fingerprints")
                source_info = _setup_sources(con, split_manifest)
                result = _write_all(con, stage, seed, source_info, split_manifest)
            finally:
                con.close()
        manifest["sources"] = source_info
        manifest["cohorts"] = result["cohorts"]
        manifest["account_partition_checks"] = result["account_partition_checks"]
        manifest["unchanged_primary"] = result["unchanged_primary"]
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest["builder_sha256"] = digest(Path(__file__))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        stage.rename(output)
        log(f"Published supplemental cohorts: {output}")
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--memory", default=DEFAULT_MEMORY_LIMIT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = parser.parse_args()
    result = build(args.output, args.seed, args.memory, args.threads)
    summary = {
        "version": result["version"], "status": result["status"],
        "cohorts": {name: info["users"] for name, info in result["cohorts"].items()},
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
