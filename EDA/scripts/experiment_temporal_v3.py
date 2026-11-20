"""Build and evaluate an isolated as-of Item-KNN temporal experiment.

The experiment uses accepted Kieran TRAIN/validation review lineage only,
filters visible reviews by both review timestamps, and keeps the global v1
catalog fixed so that the cutoff model is comparable with the existing suite.
It never reads Frank review data or the full-data Item-KNN model for scoring.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy import sparse

from eda_large_review_frank import EDA, digest, log, quote
from item_knn import _build_user_item_csr, _top_cosine_neighbors


VERSION = "temporal-cutoff-item-knn-v1"
DEFAULT_OUTPUT = EDA / "data/processed/modeling/improvement_v3/temporal_cutoff2022"
SPLIT = EDA / "data/processed/modeling/split_v1"
WORK = EDA / "data/processed/kieran_original/eda_work.duckdb"
INPUT_APPS = SPLIT / "catalog/input_apps.parquet"
LINUX_CANDIDATES = SPLIT / "catalog/linux_candidates.parquet"
TRAIN_LINEAGE = SPLIT / "train/review_lineage.parquet"
VALIDATION_LINEAGE = SPLIT / "validation/review_lineage.parquet"
VALIDATION_USERS = SPLIT / "validation/users.parquet"
SOURCE_MANIFEST = EDA / "data/source_manifest.json"
SORT_BATCH_SIZE = 131_072
DUCKDB_MEMORY_LIMIT = "2GB"
DUCKDB_THREADS = 2
DEFAULT_BLOCK_SIZE = 24
TRAIN_PARTITION_BUCKETS = 64
DEFAULT_SEED = 20261107
CUTOFF_UNIX = 1_672_531_199  # 2022-12-31 23:59:59 UTC
CUTOFF_UTC = "2022-12-31T23:59:59+00:00"
TARGET_NAMESPACE = "steam-temporal-cutoff-v1"
MIN_HISTORY_TOTAL = 5
MIN_HISTORY_POSITIVE = 1
NEIGHBORS = 100
EVAL_K = 10

QUERY_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()),
    ("positive", pa.bool_()), ("linux", pa.bool_()),
])
TARGET_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()),
    ("positive", pa.bool_()), ("linux", pa.bool_()),
    ("target_order", pa.int64()), ("timestamp_created", pa.int64()),
])
USER_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("n_total_inputs", pa.int64()),
    ("n_positive_inputs", pa.int64()), ("n_eligible_targets", pa.int64()),
])
PAIR_SCHEMA = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64())])

# This is the protocol frozen before source inspection, materialization, model
# fitting, or metric calculation. Existing copies must match byte-for-byte in
# parsed structure; the builder will never silently revise it after outcomes.
TEMPORAL_PROTOCOL = {
    "version": "temporal-cutoff-protocol-v1",
    "created_before_any_experiment_metrics": True,
    "cutoff": {"unix_seconds": CUTOFF_UNIX, "utc": CUTOFF_UTC,
                "comparison": "inclusive for visible history and training rows"},
    "sources": {
        "training_accounts": "the accepted train/review_lineage.parquet account partition from split_v1",
        "validation_accounts": "the fixed validation/users.parquet accounts from split_v1",
        "accepted_lineage": "split_v1 accepted Kieran review lineage joined to valid_raw by user, App ID, review ID, and label; existing quarantine and split assignment are preserved",
        "catalog": "fixed split_v1 catalog: 37,610 input games and 6,630 Linux candidates",
        "frank_reviews": "never read for training, cohort construction, or evaluation",
    },
    "training": {
        "eligibility": "accepted TRAIN review events whose timestamp_created and timestamp_updated are both non-null and <= cutoff",
        "model_rows": "distinct positive (user, App ID) pairs in the fixed input catalog from eligible events",
        "model": "binary positive Item-KNN, exact cosine count(i,j)/sqrt(support(i)*support(j)); candidate rows are Linux games; top 100 non-self neighbors per candidate",
        "support": "all time-eligible positive TRAIN users, including users with a single positive catalog game",
        "query_score": "mean stored binary-cosine similarity from each positive query game; positive scores only; no fill; exclude every visible query game",
        "ties": "App ID ascending",
    },
    "validation_cohort": {
        "history": "accepted validation review events in the fixed input catalog whose timestamp_created and timestamp_updated are both non-null and <= cutoff; both positive and negative rows are visible",
        "target": "one uniformly sampled positive Linux App ID per eligible account, with an accepted review timestamp_created > cutoff; the target App ID must not already occur in that account's visible history",
        "sampling": f"SHA256 counter-mode rejection sampling over ascending eligible App IDs; namespace={TARGET_NAMESPACE}; seed={DEFAULT_SEED}; split=validation; exact uint64 user key",
        "minimum_visible_history": {"total_games": MIN_HISTORY_TOTAL, "positive_games": MIN_HISTORY_POSITIVE},
        "mask": "visible history rows only; all other future review rows are excluded from query and exclusion masks",
    },
    "metrics": {
        "ranking": "each account's selected target is ranked among positive-score Linux candidates after excluding all visible query games",
        "recall_at_10": "mean indicator that the one target appears in the first 10 recommendations",
        "ndcg_at_10": "mean 1/log2(rank+1) for a target in the first 10, otherwise zero",
        "evaluation_model": "only the cutoff-refit Item-KNN model defined above",
    },
    "known_scope_limits": [
        "The fixed catalog was frozen from the full project snapshot, so app inclusion and Linux-candidate membership are transductive with respect to the cutoff.",
        "Validation accounts come from the fixed full-snapshot split_v1 assignment; this is a restricted snapshot-label temporal test, not a prospectively sampled population.",
        "Accepted lineage already applies the project's quarantine policy, including checks that use the complete snapshot and Frank identity overlap; the cutoff refit cannot undo that retrospective provenance.",
        "A positive target is observed in the later Kieran snapshot; this experiment does not reconstruct historical recommendation impressions or labels beyond the accepted review timestamps.",
    ],
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def freeze_protocol(output: Path = DEFAULT_OUTPUT) -> tuple[Path, str]:
    """Write the predeclared temporal protocol or verify its frozen copy."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text("utf-8"))
        if saved != TEMPORAL_PROTOCOL:
            raise ValueError(f"Frozen temporal protocol differs from this builder: {path}")
    else:
        _write_json(path, TEMPORAL_PROTOCOL)
    return path, digest(path)


def uniform_index(user_key: int, count: int, seed: int = DEFAULT_SEED,
                  split: str = "validation") -> int:
    """Select an unbiased deterministic index with SHA256 rejection sampling."""
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("count must be a positive integer")
    limit = (1 << 256) - (1 << 256) % int(count)
    counter = 0
    while True:
        payload = f"{TARGET_NAMESPACE}:{int(seed)}:{split}:{int(user_key)}:{counter}"
        value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        if value < limit:
            return value % int(count)
        counter += 1


def choose_target_app(user_key: int, eligible_app_ids, seed: int = DEFAULT_SEED,
                      split: str = "validation") -> int:
    """Select from a unique ascending target list using the frozen sampler."""
    apps = [int(app) for app in eligible_app_ids]
    if not apps or any(app <= 0 for app in apps):
        raise ValueError("Eligible target App IDs must be positive and non-empty")
    if any(apps[i] >= apps[i + 1] for i in range(len(apps) - 1)):
        raise ValueError("Eligible target App IDs must be unique and ascending")
    return apps[uniform_index(user_key, len(apps), seed, split)]


def _source_info(path: Path, relative_to: Path = EDA) -> dict:
    return {"file": Path(path).resolve().relative_to(relative_to.resolve()).as_posix(),
            "bytes": Path(path).stat().st_size, "sha256": digest(path)}


def _verify_sources() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Verify split and catalog lineage before loading any model data."""
    source_manifest = json.loads(SOURCE_MANIFEST.read_text("utf-8"))
    split_config = source_manifest["split_artifacts"]
    split_dir = (EDA / split_config["directory"]).resolve()
    if split_dir != SPLIT.resolve():
        raise ValueError("The temporal builder only accepts the pinned split_v1 directory")
    split_manifest_path = SPLIT / "manifest.json"
    split_sha = digest(split_manifest_path)
    if split_sha != split_config.get("manifest_sha256"):
        raise ValueError("split_v1 manifest does not match source_manifest.json")
    split_manifest = json.loads(split_manifest_path.read_text("utf-8"))
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("Only a verified complete split_v1 is accepted")
    if digest(WORK) != split_manifest.get("sources", {}).get("database_sha256"):
        raise ValueError("Audited Kieran cache differs from the database fingerprint pinned by split_v1")

    expected_files = {
        "train/review_lineage.parquet": TRAIN_LINEAGE,
        "validation/review_lineage.parquet": VALIDATION_LINEAGE,
        "validation/users.parquet": VALIDATION_USERS,
        "catalog/input_apps.parquet": INPUT_APPS,
        "catalog/linux_candidates.parquet": LINUX_CANDIDATES,
    }
    for relative, path in expected_files.items():
        info = split_manifest.get("files", {}).get(relative)
        if not info or digest(path) != info.get("sha256") or path.stat().st_size != info.get("bytes"):
            raise ValueError(f"Frozen split artifact changed: {relative}")

    catalog_table = pq.read_table(INPUT_APPS)
    apps = np.asarray(catalog_table.column("app_id").to_numpy(), dtype=np.int64)
    linux = np.asarray(catalog_table.column("linux").to_numpy(), dtype=bool)
    candidates = np.asarray(pq.read_table(LINUX_CANDIDATES).column("app_id").to_numpy(), dtype=np.int64)
    if len(apps) != 37_610 or len(candidates) != 6_630:
        raise ValueError("The fixed 37,610 by 6,630 catalog dimensions changed")
    if np.any(apps[1:] <= apps[:-1]) or not np.array_equal(candidates, apps[linux]):
        raise ValueError("Frozen catalog mappings are not ascending or Linux-consistent")
    counts = split_config.get("counts", {})
    if (counts.get("input_apps"), counts.get("linux_candidates")) != (37_610, 6_630):
        raise ValueError("Source manifest catalog counts differ from the temporal contract")

    sources = {
        "source_manifest": _source_info(SOURCE_MANIFEST),
        "split_manifest": _source_info(split_manifest_path),
        "kieran_cache": {"file": WORK.relative_to(EDA).as_posix(),
                         "bytes": WORK.stat().st_size,
                         "sha256": split_manifest["sources"]["database_sha256"]},
        "accepted_train_review_lineage": _source_info(TRAIN_LINEAGE),
        "accepted_validation_review_lineage": _source_info(VALIDATION_LINEAGE),
        "fixed_validation_accounts": _source_info(VALIDATION_USERS),
        "fixed_input_catalog": _source_info(INPUT_APPS),
        "fixed_linux_candidate_catalog": _source_info(LINUX_CANDIDATES),
    }
    return {"fingerprints": sources, "split_manifest": split_manifest,
            "split_manifest_sha256": split_sha}, apps, linux, candidates


def _sql(con: duckdb.DuckDBPyConnection, statement: str) -> dict:
    result = con.execute(statement)
    return dict(zip([col[0] for col in result.description], result.fetchone()))


def _copy_query(con: duckdb.DuckDBPyConnection, select_sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({select_sql}) TO {quote(path)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")


def _asof_history_predicate(alias: str, cutoff: int) -> str:
    """Shared strict review visibility rule for training and query history."""
    return (f"{alias}.timestamp_created IS NOT NULL AND {alias}.timestamp_updated IS NOT NULL "
            f"AND {alias}.timestamp_created <= {int(cutoff)} "
            f"AND {alias}.timestamp_updated <= {int(cutoff)}")


def _materialize_train(con: duckdb.DuckDBPyConnection, destination: Path,
                       cutoff: int = CUTOFF_UNIX) -> dict:
    """Export distinct eligible positive train pairs, sorted for streamed CSR.

    The 98M-row accepted lineage/cache join is hash-partitioned first. This
    keeps each exact event join and deduplication inside the fixed 2 GB DuckDB
    ceiling, then an external merge sort creates one user-ordered pair file.
    """
    parent = destination.parent.resolve()
    lineage_parts = destination.parent / "_train_lineage_buckets"
    raw_parts = destination.parent / "_cutoff_raw_buckets"
    pair_parts = destination.parent / "_train_pair_buckets"
    for temporary in (lineage_parts, raw_parts, pair_parts):
        if temporary.exists():
            raise FileExistsError(f"Refusing to reuse temporal build staging path: {temporary}")
    pair_parts.mkdir(parents=True)
    bucket_n = int(TRAIN_PARTITION_BUCKETS)
    lineage_sql = f"""SELECT l.user_key,l.app_id,l.review_key,l.positive,
            CAST(hash(l.user_key) % {bucket_n} AS INTEGER) AS bucket
        FROM read_parquet({quote(TRAIN_LINEAGE)}) l
        JOIN read_parquet({quote(INPUT_APPS)}) a USING(app_id)
        WHERE l.positive"""
    raw_sql = f"""SELECT r.user_key,r.app_id,r.review_key,r.positive,
            CAST(hash(r.user_key) % {bucket_n} AS INTEGER) AS bucket
        FROM valid_raw r
        JOIN read_parquet({quote(INPUT_APPS)}) a USING(app_id)
        WHERE r.positive AND {_asof_history_predicate('r', cutoff)}"""
    con.execute(f"COPY ({lineage_sql}) TO {quote(lineage_parts)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket), ROW_GROUP_SIZE 122880)")
    con.execute(f"COPY ({raw_sql}) TO {quote(raw_parts)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket), ROW_GROUP_SIZE 122880)")

    eligible_events = 0
    for bucket in range(bucket_n):
        lineage_path = lineage_parts / f"bucket={bucket}" / "*.parquet"
        raw_path = raw_parts / f"bucket={bucket}" / "*.parquet"
        if not lineage_path.parent.is_dir() or not raw_path.parent.is_dir():
            continue
        event_join = f"""FROM read_parquet({quote(lineage_path)}, hive_partitioning=true) l
            JOIN read_parquet({quote(raw_path)}, hive_partitioning=true) r
              USING(user_key,app_id,review_key,positive)"""
        bucket_events = _sql(con, f"""SELECT count(*) AS rows FROM (
            SELECT DISTINCT l.user_key,l.app_id,l.review_key,l.positive
            {event_join}) e""")
        eligible_events += int(bucket_events["rows"])
        bucket_output = pair_parts / f"pairs_{bucket:02d}.parquet"
        _copy_query(con, f"""SELECT DISTINCT l.user_key,l.app_id,true AS positive
            {event_join}
            ORDER BY l.user_key,l.app_id""", bucket_output)

    available_parts = sorted(pair_parts.glob("pairs_*.parquet"))
    if not available_parts:
        raise ValueError("No positive TRAIN pair is eligible at the fixed temporal cutoff")
    # Every pair was already deduplicated in its user-key bucket. A plain
    # external sort now avoids the memory-intensive combined GROUP BY+ORDER BY.
    _copy_query(con, f"""SELECT user_key,app_id,positive
        FROM read_parquet({quote(pair_parts / '*.parquet')})
        ORDER BY user_key,app_id""", destination)

    def remove_temporal_stage(path: Path) -> None:
        resolved = path.resolve()
        if resolved == parent or parent not in resolved.parents:
            raise ValueError(f"Refusing to remove path outside temporal build staging: {resolved}")
        shutil.rmtree(resolved)

    remove_temporal_stage(lineage_parts)
    remove_temporal_stage(raw_parts)
    remove_temporal_stage(pair_parts)
    parquet = pq.ParquetFile(destination)
    pairs = int(parquet.metadata.num_rows)
    parquet.close()
    if pairs < 1:
        raise ValueError("No positive TRAIN pair is eligible at the fixed temporal cutoff")
    pair_counts = _sql(con, f"""SELECT count(DISTINCT user_key) AS users
        FROM read_parquet({quote(destination)})""")
    users = int(pair_counts["users"])
    if pairs >= np.iinfo(np.int32).max or users >= np.iinfo(np.int32).max:
        raise ValueError("Temporal training exceeds SciPy CSR index limits")
    return {"eligible_review_events": int(eligible_events),
            "positive_pairs": pairs, "positive_users": users,
            "lineage_rows_input": int(pq.ParquetFile(TRAIN_LINEAGE).metadata.num_rows),
            "filtered_positive_catalog": destination}


def _materialize_validation(con: duckdb.DuckDBPyConnection, folder: Path,
                            seed: int = DEFAULT_SEED, cutoff: int = CUTOFF_UNIX) -> dict:
    """Create past-only validation history and one future target per account."""
    folder.mkdir(parents=True, exist_ok=True)
    history_filter = _asof_history_predicate("r", cutoff)
    future_from = f"""FROM read_parquet({quote(VALIDATION_LINEAGE)}) l
        JOIN valid_raw r USING(user_key,app_id,review_key,positive)
        JOIN read_parquet({quote(LINUX_CANDIDATES)}) a USING(app_id)
        JOIN read_parquet({quote(VALIDATION_USERS)}) v USING(user_key)"""
    history_from = f"""FROM read_parquet({quote(VALIDATION_LINEAGE)}) l
        JOIN valid_raw r USING(user_key,app_id,review_key,positive)
        JOIN read_parquet({quote(INPUT_APPS)}) a USING(app_id)
        JOIN read_parquet({quote(VALIDATION_USERS)}) v USING(user_key)"""
    con.execute(f"""CREATE TEMP TABLE _temporal_validation_history AS
        SELECT l.user_key,l.app_id,l.positive,a.linux
        {history_from}
        WHERE a.app_id IS NOT NULL AND {history_filter}
        GROUP BY l.user_key,l.app_id,l.positive,a.linux""")
    con.execute(f"""CREATE TEMP TABLE _temporal_future_target_pairs AS
        SELECT l.user_key,l.app_id,min(r.timestamp_created)::BIGINT AS timestamp_created
        {future_from}
        WHERE l.positive AND r.timestamp_created IS NOT NULL
          AND r.timestamp_created > {int(cutoff)}
        GROUP BY l.user_key,l.app_id""")
    con.execute("""CREATE TEMP TABLE _temporal_history_counts AS
        SELECT user_key,count(*)::BIGINT AS n_total_inputs,
               count(*) FILTER(WHERE positive)::BIGINT AS n_positive_inputs
        FROM _temporal_validation_history GROUP BY user_key""")
    con.execute(f"""CREATE TEMP TABLE _temporal_target_candidates AS
        SELECT f.user_key,h.n_total_inputs,h.n_positive_inputs,
               count(*)::BIGINT AS n_eligible_targets,
               list(f.app_id ORDER BY f.app_id) AS target_apps
        FROM _temporal_future_target_pairs f
        JOIN _temporal_history_counts h USING(user_key)
        WHERE h.n_total_inputs >= {MIN_HISTORY_TOTAL}
          AND h.n_positive_inputs >= {MIN_HISTORY_POSITIVE}
          AND NOT EXISTS (SELECT 1 FROM _temporal_validation_history p
                          WHERE p.user_key=f.user_key AND p.app_id=f.app_id)
        GROUP BY f.user_key,h.n_total_inputs,h.n_positive_inputs""")

    selector_name = "_temporal_uniform_target_app"
    def target_app(user_key, app_ids):
        return choose_target_app(int(user_key), app_ids, seed=seed, split="validation")
    con.create_function(selector_name, target_app,
                        [duckdb.sqltype("UBIGINT"),
                         duckdb.list_type(duckdb.sqltype("BIGINT"))],
                        duckdb.sqltype("BIGINT"))
    try:
        con.execute(f"""CREATE TEMP TABLE _temporal_targets AS
            SELECT c.user_key,{selector_name}(c.user_key,c.target_apps) AS app_id,
                   true AS positive,true AS linux,1::BIGINT AS target_order
            FROM _temporal_target_candidates c""")
    finally:
        con.remove_function(selector_name)

    con.execute("""CREATE TEMP TABLE _temporal_users AS
        SELECT c.user_key,c.n_total_inputs,c.n_positive_inputs,c.n_eligible_targets
        FROM _temporal_target_candidates c""")
    con.execute("""CREATE TEMP TABLE _temporal_selected_targets AS
        SELECT t.user_key,t.app_id,t.positive,t.linux,t.target_order,
               f.timestamp_created
        FROM _temporal_targets t JOIN _temporal_future_target_pairs f
          USING(user_key,app_id)""")

    _copy_query(con, f"""SELECT user_key,n_total_inputs,n_positive_inputs,n_eligible_targets
        FROM _temporal_users ORDER BY user_key""", folder / "users.parquet")
    _copy_query(con, f"""SELECT user_key,app_id,positive,linux,target_order,timestamp_created
        FROM _temporal_selected_targets ORDER BY user_key,app_id""", folder / "targets.parquet")
    _copy_query(con, f"""SELECT h.user_key,h.app_id,h.positive,h.linux
        FROM _temporal_validation_history h SEMI JOIN _temporal_users u USING(user_key)
        ORDER BY h.user_key,h.app_id""", folder / "query.parquet")
    _copy_query(con, f"""SELECT user_key,app_id FROM read_parquet({quote(folder/'query.parquet')})
        ORDER BY user_key,app_id""", folder / "exclusions.parquet")

    totals = _sql(con, f"""SELECT
        (SELECT count(*) FROM read_parquet({quote(folder/'users.parquet')})) cohort_users,
        (SELECT count(*) FROM _temporal_validation_history) visible_history_pairs,
        (SELECT count(*) FROM _temporal_history_counts) accounts_with_visible_history,
        (SELECT count(*) FROM _temporal_future_target_pairs) future_positive_linux_pairs,
        (SELECT count(*) FROM _temporal_target_candidates) eligible_accounts""")
    n_users = int(totals["cohort_users"])
    query_pf = pq.ParquetFile(folder / "query.parquet")
    target_pf = pq.ParquetFile(folder / "targets.parquet")
    exclusion_pf = pq.ParquetFile(folder / "exclusions.parquet")
    query_rows = int(query_pf.metadata.num_rows)
    target_rows = int(target_pf.metadata.num_rows)
    exclusion_rows = int(exclusion_pf.metadata.num_rows)
    if target_rows != n_users or exclusion_rows != query_rows:
        raise ValueError("Temporal cohort target or exclusion cardinality check failed")
    if n_users:
        target_check = _sql(con, f"""SELECT count(*) AS bad FROM (
            SELECT t.user_key,t.app_id
            FROM read_parquet({quote(folder/'targets.parquet')}) t
            JOIN read_parquet({quote(folder/'query.parquet')}) q USING(user_key,app_id)) x""")
        if int(target_check["bad"]):
            raise ValueError("A future target overlaps its visible query history")
    return {
        "validation_accounts_in_fixed_split": int(pq.ParquetFile(VALIDATION_USERS).metadata.num_rows),
        "accounts_with_visible_history": int(totals["accounts_with_visible_history"]),
        "visible_history_pairs_all_accounts": int(totals["visible_history_pairs"]),
        "future_positive_linux_target_pairs": int(totals["future_positive_linux_pairs"]),
        "eligible_accounts": int(totals["eligible_accounts"]),
        "cohort_users": n_users,
        "query_pairs": query_rows,
        "query_positive_pairs": int(pq.read_table(folder / "query.parquet", columns=["positive"])
                                      .column("positive").to_numpy().sum()),
        "targets": target_rows,
        "exclusion_pairs": exclusion_rows,
        "target_app_overlap_with_visible_history": 0,
        "future_reviews_in_query_or_exclusions": 0,
        "minimum_total_visible_games": MIN_HISTORY_TOTAL,
        "minimum_positive_visible_games": MIN_HISTORY_POSITIVE,
        "seed": int(seed),
        "target_selection_namespace": f"{TARGET_NAMESPACE}:{seed}:validation",
        "files": {name: folder / name for name in
                  ("users.parquet", "query.parquet", "targets.parquet", "exclusions.parquet")},
    }


def _write_model_mappings(folder: Path, apps: np.ndarray, linux: np.ndarray,
                          candidates: np.ndarray, support: np.ndarray) -> None:
    input_rows = pa.table({
        "row_id": pa.array(np.arange(len(apps), dtype=np.int32)),
        "app_id": pa.array(apps, type=pa.int64()),
        "linux": pa.array(linux, type=pa.bool_()),
        "positive_user_support": pa.array(support, type=pa.int64()),
    })
    candidate_rows = pa.table({
        "row_id": pa.array(np.arange(len(candidates), dtype=np.int32)),
        "app_id": pa.array(candidates, type=pa.int64()),
        "input_row_id": pa.array(np.searchsorted(apps, candidates).astype(np.int32), type=pa.int32()),
    })
    pq.write_table(input_rows, folder / "game_rows.parquet", compression="zstd")
    pq.write_table(candidate_rows, folder / "candidate_rows.parquet", compression="zstd")


def _fit_knn(sorted_pairs: Path, model_folder: Path, apps: np.ndarray,
             linux: np.ndarray, candidates: np.ndarray, expected_pairs: int,
             expected_users: int, block_size: int = DEFAULT_BLOCK_SIZE) -> dict:
    """Refit the exact full-catalog binary-cosine candidate-by-input matrix."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    sorted_file = pq.ParquetFile(sorted_pairs)
    user_items, support, user_count = _build_user_item_csr(
        sorted_file.iter_batches(batch_size=SORT_BATCH_SIZE, columns=["user_key", "app_id"]),
        apps, expected_pairs=expected_pairs, expected_users=expected_users)
    sorted_file.close()
    if user_count != expected_users or user_items.nnz != expected_pairs:
        raise ValueError("Streamed temporal CSR differs from the filtered positive catalog")
    singleton_users = int(np.count_nonzero(np.diff(user_items.indptr) == 1))
    user_items = user_items.tocsc()
    candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
    output_capacity = len(candidates) * min(NEIGHBORS, max(0, len(apps) - 1))
    indices = np.empty(output_capacity, dtype=np.int32)
    values = np.empty(output_capacity, dtype=np.float64)
    indptr = np.zeros(len(candidates) + 1, dtype=np.int32)
    nnz = 0
    for begin in range(0, len(candidates), int(block_size)):
        end = min(begin + int(block_size), len(candidates))
        candidate_by_user = user_items[:, candidate_cols[begin:end]].T.tocsr()
        candidate_by_user = candidate_by_user.astype(np.int32, copy=False)
        cooccurrence = (candidate_by_user @ user_items).tocsr()
        cooccurrence.sum_duplicates()
        for local_row in range(end - begin):
            global_row = begin + local_row
            row_begin, row_end = cooccurrence.indptr[local_row:local_row + 2]
            out_cols, out_values = _top_cosine_neighbors(
                int(candidate_cols[global_row]), cooccurrence.indices[row_begin:row_end],
                cooccurrence.data[row_begin:row_end], apps, support, NEIGHBORS)
            size = len(out_cols)
            if nnz + size > output_capacity:
                raise ValueError("Temporal Item-KNN output exceeded its top-100 capacity")
            indices[nnz:nnz + size] = out_cols
            values[nnz:nnz + size] = out_values
            nnz += size
            indptr[global_row + 1] = nnz
        log(f"Temporal KNN candidates {end:,}/{len(candidates):,}")
        del candidate_by_user, cooccurrence
        gc.collect()
    matrix = sparse.csr_matrix((values[:nnz], indices[:nnz], indptr),
                               shape=(len(candidates), len(apps)), copy=False)
    matrix.sort_indices()
    _write_model_mappings(model_folder, apps, linux, candidates, support)
    sparse.save_npz(model_folder / "similarities_csr.npz", matrix, compressed=True)
    _write_json(model_folder / "parameters.json", {
        "version": VERSION, "neighbors": NEIGHBORS,
        "input_catalog_rows": len(apps), "linux_candidate_rows": len(candidates),
        "training_rows": "accepted TRAIN positive user/App ID pairs with both review timestamps <= cutoff",
        "binary_interactions": True, "single_positive_users_retained": True,
        "similarity": "count(i,j)/sqrt(count(i)*count(j)); self-similarity zero",
        "neighbor_order": "cosine descending, then App ID ascending",
        "candidate_block_size": int(block_size),
        "query_score": "mean cosine from positive visible history games only",
    })
    return {
        "positive_train_users": int(user_count), "positive_train_pairs": int(user_items.nnz),
        "positive_train_singleton_users": singleton_users,
        "input_games_with_positive_support": int(np.count_nonzero(support)),
        "input_games_without_positive_support": int(len(support) - np.count_nonzero(support)),
        "linux_candidates_with_positive_support": int(np.count_nonzero(support[candidate_cols])),
        "linux_candidates_without_positive_support": int(len(candidates) - np.count_nonzero(support[candidate_cols])),
        "similarity_nnz": int(matrix.nnz),
        "candidates_with_neighbors": int(np.count_nonzero(np.diff(matrix.indptr))),
        "input_columns_with_neighbor_links": int(np.count_nonzero(np.diff(matrix.tocsc().indptr))),
        "similarity_shape": list(matrix.shape),
        "similarity_file_sha256": digest(model_folder / "similarities_csr.npz"),
    }


def iter_query_groups(path: Path, scan_batch_size: int = 65_536):
    """Stream ordered query rows into complete per-user groups."""
    parquet = pq.ParquetFile(path)
    pending_user = None
    pending_apps = []
    pending_positive = []
    for batch in parquet.iter_batches(batch_size=scan_batch_size,
                                      columns=["user_key", "app_id", "positive"]):
        users = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.uint64)
        apps = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)
        positives = np.asarray(batch.column(2).to_numpy(zero_copy_only=False), dtype=bool)
        if not len(users):
            continue
        if np.any(users[1:] < users[:-1]):
            raise ValueError("Temporal query file is not ordered by user key")
        starts = np.concatenate(([0], np.flatnonzero(users[1:] != users[:-1]) + 1))
        ends = np.concatenate((starts[1:], [len(users)]))
        for start, end in zip(starts, ends):
            user = int(users[start])
            if pending_user is not None and user != pending_user:
                yield pending_user, np.concatenate(pending_apps), np.concatenate(pending_positive)
                pending_user, pending_apps, pending_positive = None, [], []
            if pending_user is None:
                pending_user = user
            pending_apps.append(apps[start:end])
            pending_positive.append(positives[start:end])
            if end < len(users):
                yield pending_user, np.concatenate(pending_apps), np.concatenate(pending_positive)
                pending_user, pending_apps, pending_positive = None, [], []
    if pending_user is not None:
        yield pending_user, np.concatenate(pending_apps), np.concatenate(pending_positive)


def iter_query_batches(path: Path, users_per_batch: int = 256):
    group_batch = []
    for group in iter_query_groups(path):
        group_batch.append(group)
        if len(group_batch) >= users_per_batch:
            yield group_batch
            group_batch = []
    if group_batch:
        yield group_batch


def rank_score_row(candidate_app_ids: np.ndarray, score_indices: np.ndarray,
                   score_values: np.ndarray, target_app: int,
                   excluded_app_ids) -> tuple[int | None, int, float]:
    """Rank one positive-score target under the frozen exclusion and tie rules."""
    excluded = np.asarray(list(excluded_app_ids), dtype=np.int64)
    excluded_rows = np.searchsorted(candidate_app_ids, excluded)
    valid_exclusion = ((excluded_rows < len(candidate_app_ids)) &
                       (candidate_app_ids[np.minimum(excluded_rows, len(candidate_app_ids)-1)] == excluded))
    excluded_rows = excluded_rows[valid_exclusion]
    keep = ~np.isin(score_indices, excluded_rows, assume_unique=False)
    valid_indices = score_indices[keep]
    valid_scores = score_values[keep]
    available = int(len(valid_indices))
    target_row = int(np.searchsorted(candidate_app_ids, int(target_app)))
    if target_row >= len(candidate_app_ids) or int(candidate_app_ids[target_row]) != int(target_app):
        raise ValueError("Target App ID is outside the fixed Linux catalog")
    target_pos = int(np.searchsorted(score_indices, target_row))
    if target_pos >= len(score_indices) or int(score_indices[target_pos]) != target_row:
        return None, available, 0.0
    target_score = float(score_values[target_pos])
    if not keep[target_pos] or target_score <= 0:
        return None, available, target_score
    before = (valid_scores > target_score) | ((valid_scores == target_score) & (valid_indices < target_row))
    return int(np.count_nonzero(before) + 1), available, target_score


def _evaluate(model_folder: Path, cohort_folder: Path, apps: np.ndarray,
              candidates: np.ndarray, cohort_counts: dict,
              users_per_batch: int = 256) -> tuple[dict, dict]:
    """Evaluate one cutoff KNN model over the full fixed Linux candidate set."""
    matrix = sparse.load_npz(model_folder / "similarities_csr.npz").tocsr()
    if matrix.shape != (len(candidates), len(apps)):
        raise ValueError("Temporal similarity matrix has an unexpected full-catalog shape")
    by_input = matrix.T.tocsr()
    target_table = pq.read_table(cohort_folder / "targets.parquet",
                                 columns=["user_key", "app_id"])
    target_users = np.asarray(target_table.column("user_key").to_numpy(), dtype=np.uint64)
    target_apps = np.asarray(target_table.column("app_id").to_numpy(), dtype=np.int64)
    target_rows = np.searchsorted(candidates, target_apps)
    if len(target_rows) and (np.any(target_rows >= len(candidates)) or
                             np.any(candidates[target_rows] != target_apps)):
        raise ValueError("Temporal targets are outside the frozen Linux catalog")
    if len(target_users) != int(cohort_counts["cohort_users"]):
        raise ValueError("Temporal target count differs from the selected user count")
    if not np.array_equal(target_users[1:] > target_users[:-1], np.ones(max(0, len(target_users)-1), dtype=bool)):
        raise ValueError("Temporal targets are not sorted and unique by exact user key")

    n_users = 0
    target_hit = 0
    target_linkable = 0
    target_train_supported = int(np.count_nonzero(
        pq.read_table(model_folder / "game_rows.parquet", columns=["positive_user_support"])
        .column("positive_user_support").to_numpy()[np.searchsorted(
            apps, target_apps)] > 0)) if len(target_apps) else 0
    available_positive_score = 0
    available_top10 = 0
    available_recommendation_sum = 0
    ndcg_gain_sum = 0.0
    query_total_sum = 0
    query_positive_sum = 0
    target_rank_histogram = {}
    observed_user_index = 0
    for groups in iter_query_batches(cohort_folder / "query.parquet", users_per_batch):
        group_users = np.asarray([group[0] for group in groups], dtype=np.uint64)
        group_target_pos = np.searchsorted(target_users, group_users)
        if np.any(group_target_pos >= len(target_users)) or not np.array_equal(target_users[group_target_pos], group_users):
            raise ValueError("Query users differ from the fixed validation target accounts")
        local_targets = target_apps[group_target_pos]
        q_rows = []
        q_cols = []
        exclusions = []
        positive_counts = np.zeros(len(groups), dtype=np.int32)
        total_counts = np.zeros(len(groups), dtype=np.int32)
        for local, (_, query_apps, query_positive) in enumerate(groups):
            cols = np.searchsorted(apps, query_apps)
            if np.any(cols >= len(apps)) or np.any(apps[cols] != query_apps):
                raise ValueError("A visible query App ID is outside the fixed input catalog")
            positive_cols = cols[query_positive]
            q_rows.extend([local] * len(positive_cols))
            q_cols.extend(positive_cols.tolist())
            positive_counts[local] = len(positive_cols)
            total_counts[local] = len(query_apps)
            exclusions.append(query_apps)
        if np.any(positive_counts < MIN_HISTORY_POSITIVE) or np.any(total_counts < MIN_HISTORY_TOTAL):
            raise ValueError("A selected query falls below its frozen visible-history thresholds")
        query_matrix = sparse.csr_matrix(
            (np.ones(len(q_rows), dtype=np.uint8),
             (np.asarray(q_rows, dtype=np.int32), np.asarray(q_cols, dtype=np.int32))),
            shape=(len(groups), len(apps)), dtype=np.uint8)
        score_matrix = (query_matrix @ by_input).tocsr()
        score_matrix.sum_duplicates()
        score_matrix.sort_indices()
        for local, (_, query_apps, _) in enumerate(groups):
            start, end = score_matrix.indptr[local:local + 2]
            indices = score_matrix.indices[start:end]
            values = score_matrix.data[start:end]
            rank, available, score = rank_score_row(
                candidates, indices, values, int(local_targets[local]), query_apps)
            if score > 0:
                target_linkable += 1
            if available:
                available_positive_score += 1
            available_recommendation_sum += min(EVAL_K, available)
            if rank is not None:
                target_rank_histogram[str(rank)] = target_rank_histogram.get(str(rank), 0) + 1
                if rank <= EVAL_K:
                    target_hit += 1
                    ndcg_gain_sum += 1.0 / math.log2(rank + 1)
            n_users += 1
            query_total_sum += int(total_counts[local])
            query_positive_sum += int(positive_counts[local])
        observed_user_index += len(groups)
        del query_matrix, score_matrix
    if n_users != int(cohort_counts["cohort_users"]) or observed_user_index != n_users:
        raise ValueError("Scored query user count differs from the frozen temporal cohort")
    metrics = {
        "cohort_users": int(n_users),
        "recall_at_10": (float(target_hit / n_users) if n_users else None),
        "ndcg_at_10": (float(ndcg_gain_sum / n_users) if n_users else None),
        "target_hits_at_10": int(target_hit),
        "targets_with_positive_model_score": int(target_linkable),
        "targets_with_positive_model_score_fraction": (float(target_linkable / n_users) if n_users else None),
        "targets_with_positive_train_support": int(target_train_supported),
        "accounts_with_at_least_one_recommendable_item": int(available_positive_score),
        "mean_available_top10_recommendations": (float(available_recommendation_sum / n_users) if n_users else None),
        "mean_visible_history_games": (float(query_total_sum / n_users) if n_users else None),
        "mean_visible_positive_history_games": (float(query_positive_sum / n_users) if n_users else None),
        "target_ranks_with_positive_score": target_rank_histogram,
        "candidate_scope": "all 6,630 fixed Linux candidates; positive-score-only; visible-history games excluded; no fill; App ID tie-break",
    }
    supports = np.asarray(pq.read_table(model_folder / "game_rows.parquet",
                                        columns=["positive_user_support"])
                          .column("positive_user_support").to_numpy(), dtype=np.int64)
    model_support = {
        "positive_train_user_app_pairs": int(np.sum(supports)),
        "input_games_with_positive_support": int(np.count_nonzero(supports)),
        "input_games_without_positive_support": int(len(supports) - np.count_nonzero(supports)),
        "positive_train_support_min_nonzero": int(np.min(supports[supports > 0])) if np.any(supports > 0) else 0,
        "positive_train_support_median_nonzero": float(np.median(supports[supports > 0])) if np.any(supports > 0) else 0.0,
        "positive_train_support_max": int(np.max(supports)) if len(supports) else 0,
        "linux_candidates_with_positive_support": int(np.count_nonzero(supports[np.searchsorted(apps, candidates)])),
        "similarity_matrix_nonzeros": int(matrix.nnz),
        "linux_candidates_with_neighbors": int(np.count_nonzero(np.diff(matrix.indptr))),
        "input_games_with_incoming_neighbor_links": int(np.count_nonzero(np.diff(matrix.tocsc().indptr))),
    }
    return metrics, model_support


def _inventory(root: Path, exclude: set[str] | None = None) -> dict:
    exclude = exclude or set()
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in exclude:
            continue
        if "duckdb_temp" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        info = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix == ".parquet":
            info["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
        result[relative] = info
    return result


def build(output: Path = DEFAULT_OUTPUT, memory_limit: str = DUCKDB_MEMORY_LIMIT,
          threads: int = DUCKDB_THREADS, block_size: int = DEFAULT_BLOCK_SIZE) -> dict:
    """Freeze protocol, create temporal artifacts, fit KNN, and report metrics."""
    if memory_limit != DUCKDB_MEMORY_LIMIT:
        raise ValueError("The temporal run is fixed at DuckDB memory_limit=2GB")
    if threads != DUCKDB_THREADS:
        raise ValueError("The temporal run is fixed at two DuckDB threads")
    if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    output = Path(output).resolve()
    protocol_path, protocol_sha = freeze_protocol(output)
    result_path = output / "result"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite temporal experiment result: {result_path}")
    stage = output / f".result.building-{uuid.uuid4().hex[:10]}"
    training_dir = stage / "training"
    cohort_dir = stage / "validation"
    model_dir = stage / "model"
    training_dir.mkdir(parents=True)
    cohort_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    temp_dir = stage / "duckdb_temp"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        log("Verifying pinned split, accepted lineage, cache, and fixed catalog")
        sources, apps, linux, candidates = _verify_sources()
        train_pairs_path = training_dir / "positive_catalog.parquet"
        con = duckdb.connect(str(WORK), read_only=True)
        try:
            con.execute(f"SET memory_limit={quote(memory_limit)}")
            con.execute(f"SET threads={int(threads)}")
            con.execute("SET preserve_insertion_order=false")
            temp_dir.mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory={quote(temp_dir)}")
            con.execute("SET max_temp_directory_size='30GB'")
            log("Materializing timestamp-eligible positive TRAIN pairs")
            train_counts = _materialize_train(con, train_pairs_path)
            log("Materializing a past-only validation cohort with one future target")
            cohort_counts = _materialize_validation(con, cohort_dir)
        finally:
            con.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

        log("Refitting binary-cosine Item-KNN K100 over the full fixed catalog")
        model_counts = _fit_knn(
            train_pairs_path, model_dir, apps, linux, candidates,
            expected_pairs=train_counts["positive_pairs"],
            expected_users=train_counts["positive_users"], block_size=int(block_size))
        train_counts.pop("filtered_positive_catalog")
        _write_json(model_dir / "manifest.json", {
            "version": VERSION, "status": "complete", "fit_scope": "accepted TRAIN lineage as-of fixed cutoff",
            "cutoff_unix": CUTOFF_UNIX, "protocol_sha256": protocol_sha,
            "source_fingerprints": sources["fingerprints"],
            "train_support": train_counts, "model_support": model_counts,
            "builder_sha256": digest(Path(__file__)),
            "runtime": {"python": platform.python_version(), "duckdb": duckdb.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "pyarrow": pa.__version__, "duckdb_memory_limit": memory_limit,
                        "duckdb_threads": threads, "candidate_block_size": int(block_size)},
        })

        log("Evaluating only the cutoff-refit model on the fixed Linux catalog")
        metrics, scoring_support = _evaluate(model_dir, cohort_dir, apps, candidates, cohort_counts)
        # The final train pair parquet is the auditable temporal training input;
        # preserve it under training/ while the builder temp directory is removed.
        report = {
            "version": VERSION,
            "status": "complete",
            "created_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cutoff": {"unix_seconds": CUTOFF_UNIX, "utc": CUTOFF_UTC},
            "protocol": {"file": protocol_path.relative_to(output).as_posix(),
                         "sha256": protocol_sha},
            "source_fingerprints": sources["fingerprints"],
            "catalog": {"input_games": len(apps), "linux_candidates": len(candidates),
                        "input_catalog_sha256": digest(INPUT_APPS),
                        "candidate_catalog_sha256": digest(LINUX_CANDIDATES)},
            "training": train_counts,
            "validation_cohort": {key: value for key, value in cohort_counts.items() if key != "files"},
            "model_support": scoring_support,
            "metrics": metrics,
            "limitations": TEMPORAL_PROTOCOL["known_scope_limits"],
            "lineage_note": (
                "Training and validation rows come from the accepted split_v1 Kieran lineage; "
                "the lineage preserves the project quarantine and assignment rules. The cache was "
                "opened read-only and fingerprinted against split_v1. No Frank review rows or full-data "
                "Item-KNN model were used by this experiment. The fixed global catalog remains transductive."
            ),
            "builder_sha256": digest(Path(__file__)),
            "files": _inventory(stage, exclude={"report.json"}),
        }
        _write_json(stage / "report.json", report)
        # Preserve the protocol adjacent to its results and publish atomically.
        stage.rename(result_path)
        log(f"Published temporal K100 report: {result_path / 'report.json'}")
        return report
    except Exception as error:
        _write_json(stage / "FAILED.json", {
            "version": VERSION, "status": "failed", "started_utc": started,
            "error": f"{type(error).__name__}: {error}",
            "protocol_sha256": protocol_sha,
            "builder_sha256": digest(Path(__file__)),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory", default=DUCKDB_MEMORY_LIMIT)
    parser.add_argument("--threads", type=int, default=DUCKDB_THREADS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--freeze-protocol-only", action="store_true",
                        help="Write or verify protocol.json without accessing data or fitting models")
    args = parser.parse_args()
    if args.freeze_protocol_only:
        path, sha = freeze_protocol(args.output)
        print(json.dumps({"protocol": str(path), "sha256": sha}, indent=2))
        return
    result = build(args.output, args.memory, args.threads, args.block_size)
    print(json.dumps({"version": result["version"], "status": result["status"],
                      "cohort_users": result["metrics"]["cohort_users"],
                      "Recall@10": result["metrics"]["recall_at_10"],
                      "NDCG@10": result["metrics"]["ndcg_at_10"],
                      "report": str((Path(args.output).resolve() / "result/report.json"))}, indent=2))


if __name__ == "__main__":
    main()
