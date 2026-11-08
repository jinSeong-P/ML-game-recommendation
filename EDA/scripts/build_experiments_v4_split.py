"""Reserve new v4 Kieran account roles and materialize fresh training/cohorts.

This is a split builder only. It does not fit or score a model. All role
accounts are removed from the new base-training positive pairs before the
cohort files are written. Existing v1-v3 files are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EDA = Path(__file__).resolve().parents[1]
MODELING = EDA / "data/processed/modeling"
DEFAULT_SPLIT = MODELING / "split_v1"
DEFAULT_OUTPUT = MODELING / "experiments_v4/split"
DEFAULT_SEED = 20261108
VERSION = "experiments-v4-account-split-v1"
TARGET_NAMESPACE = "steam-experiments-v4-hidden-positive-linux-v1"
ASSIGN_NAMESPACE = "steam-experiments-v4-account-role-v1"
ROLE_NAMES = ("fusion_fit", "selection", "confirmation")
STRATA = ("1-2", "3-4", "5+")
DEFAULT_QUOTAS = {
    "1-2": {"fusion_fit": 5_000, "selection": 5_000, "confirmation": 5_000},
    "3-4": {"fusion_fit": 5_000, "selection": 5_000, "confirmation": 5_000},
    "5+": {"fusion_fit": 10_000, "selection": 10_000, "confirmation": 10_000},
}
MEMORY_LIMIT = "2GB"
THREADS = 2
ROW_GROUP_SIZE = 122_880

PAIR_SCHEMA = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64()), ("positive", pa.bool_())])
USER_SCHEMA = pa.schema([
    ("user_key", pa.uint64()),
    ("n_total_inputs", pa.int64()),
    ("n_positive_inputs", pa.int64()),
    ("n_eligible_targets", pa.int64()),
])
QUERY_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()),
    ("positive", pa.bool_()), ("linux", pa.bool_()),
])
TARGET_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()),
    ("positive", pa.bool_()), ("linux", pa.bool_()), ("target_order", pa.int64()),
])
EXCLUSION_SCHEMA = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64())])


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def digest(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def uniform_index(user_key: int, count: int, seed: int, role: str) -> int:
    """Unbiased, repeatable index into ascending target App IDs."""
    if count < 1:
        raise ValueError("Cannot sample a target from an empty candidate list")
    limit = (1 << 256) - (1 << 256) % count
    counter = 0
    while True:
        payload = f"{TARGET_NAMESPACE}:{seed}:{role}:{int(user_key)}:{counter}"
        value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        if value < limit:
            return value % count
        counter += 1


def uniform_target(user_key: int, app_ids: list[int] | tuple[int, ...], seed: int, role: str) -> int:
    ordered = [int(app_id) for app_id in app_ids]
    if not ordered or ordered != sorted(ordered) or len(ordered) != len(set(ordered)):
        raise ValueError("Target candidates must be a nonempty ascending unique list")
    return ordered[uniform_index(int(user_key), len(ordered), seed, role)]


def role_for_rank(stratum: str, rank: int, quotas: dict[str, dict[str, int]]) -> str | None:
    """Map a one-based hash rank to a nonoverlapping role quota."""
    if stratum not in quotas or rank < 1:
        return None
    offset = 0
    for role in ROLE_NAMES:
        stop = offset + int(quotas[stratum][role])
        if rank <= stop:
            return role
        offset = stop
    return None


def validate_quotas(quotas: dict[str, dict[str, int]]) -> None:
    if set(quotas) != set(STRATA):
        raise ValueError(f"Quotas must define exactly these post-hide strata: {STRATA}")
    for stratum in STRATA:
        if set(quotas[stratum]) != set(ROLE_NAMES):
            raise ValueError(f"Quotas for {stratum} must define exactly these roles: {ROLE_NAMES}")
        if any(int(quotas[stratum][role]) < 0 for role in ROLE_NAMES):
            raise ValueError("Role quotas cannot be negative")


def prior_evaluation_user_files(split_root: Path, modeling_root: Path) -> list[Path]:
    """Return every locally materialized evaluation account list from v1-v3.

    In this workspace the evaluation cohorts are persisted as users.parquet
    files in split_v1 and backlog_v3, plus the cutoff-2022 validation cohort.
    Their results use those same cohorts. Forgemaster account identities are
    not present in the local source snapshot.
    """
    paths = {split_root / "validation/users.parquet", split_root / "test/users.parquet"}
    for subtree in (
        modeling_root / "backlog_v3",
        modeling_root / "improvement_v2",
        modeling_root / "improvement_v3",
    ):
        if subtree.exists():
            paths.update(subtree.rglob("users.parquet"))
    paths.add(split_root / "reserved_test_accounts.parquet")
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing prior-evaluation account source(s): " + ", ".join(map(str, missing)))
    return sorted(paths)


def build_prior_account_table(con: duckdb.DuckDBPyConnection, files: list[Path]) -> None:
    file_list = ",".join(quote(path) for path in files)
    con.execute(f"""CREATE TEMP TABLE prior_evaluation_accounts AS
        SELECT DISTINCT CAST(user_key AS UBIGINT) AS user_key
        FROM read_parquet([{file_list}], union_by_name=true)
        WHERE user_key IS NOT NULL""")


def build_user_stats(con: duckdb.DuckDBPyConnection, split_root: Path) -> None:
    """Count fixed-catalog histories only for accounts with positive Linux targets."""
    positive = split_root / "train/positive_catalog.parquet"
    interactions = split_root / "train/interactions.parquet"
    input_apps = split_root / "catalog/input_apps.parquet"
    candidates = split_root / "catalog/linux_candidates.parquet"
    con.execute(f"""CREATE TEMP TABLE target_support AS
        SELECT p.user_key, count(*)::BIGINT AS n_eligible_targets
        FROM read_parquet({quote(positive)}) p
        SEMI JOIN read_parquet({quote(candidates)}) c USING(app_id)
        GROUP BY p.user_key""")
    con.execute(f"""CREATE TEMP TABLE user_stats AS
        SELECT h.user_key,
               count(*)::BIGINT AS n_total_before_hide,
               count(*) FILTER (WHERE h.positive)::BIGINT AS n_positive_before_hide,
               max(t.n_eligible_targets)::BIGINT AS n_eligible_targets
        FROM read_parquet({quote(interactions)}) h
        JOIN read_parquet({quote(input_apps)}) c USING(app_id)
        JOIN target_support t USING(user_key)
        GROUP BY h.user_key""")


def build_role_assignments(
    con: duckdb.DuckDBPyConnection,
    seed: int,
    quotas: dict[str, dict[str, int]],
    stats_table: str = "user_stats",
    prior_table: str = "prior_evaluation_accounts",
) -> None:
    """Stratify the eligible train pool by SHA256 rank and reserve role ranges."""
    validate_quotas(quotas)
    if stats_table != "user_stats" or prior_table != "prior_evaluation_accounts":
        raise ValueError("Only internal v4 table names are supported")
    quota_rows = []
    for stratum in STRATA:
        for role in ROLE_NAMES:
            quota_rows.append(f"('{stratum}', '{role}', {int(quotas[stratum][role])})")
    con.execute("CREATE TEMP TABLE v4_quotas(stratum VARCHAR, role VARCHAR, n BIGINT)")
    con.execute("INSERT INTO v4_quotas VALUES " + ",".join(quota_rows))
    con.execute(f"""CREATE TEMP TABLE v4_ranked_users AS
        SELECT s.user_key,
               s.n_total_before_hide,
               s.n_positive_before_hide,
               s.n_eligible_targets,
               CASE WHEN s.n_total_before_hide - 1 <= 2 THEN '1-2'
                    WHEN s.n_total_before_hide - 1 <= 4 THEN '3-4'
                    ELSE '5+' END AS input_stratum,
               row_number() OVER (
                   PARTITION BY CASE WHEN s.n_total_before_hide - 1 <= 2 THEN '1-2'
                                     WHEN s.n_total_before_hide - 1 <= 4 THEN '3-4'
                                     ELSE '5+' END
                   ORDER BY sha256('{ASSIGN_NAMESPACE}:{int(seed)}:' ||
                                   CASE WHEN s.n_total_before_hide - 1 <= 2 THEN '1-2'
                                        WHEN s.n_total_before_hide - 1 <= 4 THEN '3-4'
                                        ELSE '5+' END || ':' || CAST(s.user_key AS VARCHAR)),
                            s.user_key) AS sample_rank
        FROM user_stats s
        ANTI JOIN prior_evaluation_accounts p USING(user_key)
        WHERE s.n_total_before_hide >= 2
          AND s.n_positive_before_hide >= 2
          AND s.n_eligible_targets >= 1""")
    con.execute("""CREATE TEMP TABLE role_assignments AS
        WITH bounds AS (
            SELECT stratum, role, n,
                   coalesce(sum(n) OVER (
                       PARTITION BY stratum
                       ORDER BY CASE role WHEN 'fusion_fit' THEN 1
                                           WHEN 'selection' THEN 2 ELSE 3 END
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                   ), 0) AS rank_start,
                   sum(n) OVER (
                       PARTITION BY stratum
                       ORDER BY CASE role WHEN 'fusion_fit' THEN 1
                                           WHEN 'selection' THEN 2 ELSE 3 END
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS rank_stop
            FROM v4_quotas
        )
        SELECT r.user_key, q.role, r.input_stratum, r.sample_rank::BIGINT AS sample_rank
        FROM v4_ranked_users r
        JOIN bounds q ON q.stratum = r.input_stratum
        WHERE r.sample_rank > q.rank_start AND r.sample_rank <= q.rank_stop""")
    requested = sum(int(quotas[s][r]) for s in STRATA for r in ROLE_NAMES)
    actual = con.execute("SELECT count(*) FROM role_assignments").fetchone()[0]
    if actual != requested:
        available = con.execute("""SELECT input_stratum, count(*) AS available
            FROM v4_ranked_users GROUP BY input_stratum ORDER BY input_stratum""").fetchall()
        raise ValueError(f"Only {actual:,}/{requested:,} role accounts were available; by stratum: {available}")


def build_selected_history(con: duckdb.DuckDBPyConnection, split_root: Path) -> None:
    interactions = split_root / "train/interactions.parquet"
    input_apps = split_root / "catalog/input_apps.parquet"
    candidates = split_root / "catalog/linux_candidates.parquet"
    con.execute(f"""CREATE TEMP TABLE selected_history AS
        SELECT h.user_key, h.app_id, h.positive,
               (c.app_id IS NOT NULL) AS linux,
               a.role
        FROM read_parquet({quote(interactions)}) h
        JOIN role_assignments a USING(user_key)
        JOIN read_parquet({quote(input_apps)}) i USING(app_id)
        LEFT JOIN read_parquet({quote(candidates)}) c USING(app_id)""")


def create_role_cohort_tables(
    con: duckdb.DuckDBPyConnection,
    role: str,
    seed: int,
    stats_table: str = "user_stats",
) -> None:
    if role not in ROLE_NAMES or stats_table != "user_stats":
        raise ValueError("Unknown role or internal stats table")
    function_name = f"_v4_target_{role}"
    con.create_function(
        function_name,
        lambda user_key, apps: uniform_target(user_key, apps, seed, role),
        [duckdb.sqltype("UBIGINT"), duckdb.list_type(duckdb.sqltype("BIGINT"))],
        duckdb.sqltype("BIGINT"),
    )
    con.execute(f"""CREATE TEMP TABLE {role}_targets AS
        SELECT user_key,
               {function_name}(user_key, list(app_id ORDER BY app_id)) AS app_id,
               true AS positive,
               true AS linux,
               1::BIGINT AS target_order
        FROM selected_history
        WHERE role = '{role}' AND positive AND linux
        GROUP BY user_key""")
    con.remove_function(function_name)
    con.execute(f"""CREATE TEMP TABLE {role}_query AS
        SELECT h.user_key, h.app_id, h.positive, h.linux
        FROM selected_history h
        ANTI JOIN {role}_targets t USING(user_key, app_id)
        WHERE h.role = '{role}'""")
    con.execute(f"""CREATE TEMP TABLE {role}_users AS
        SELECT a.user_key,
               (s.n_total_before_hide - 1)::BIGINT AS n_total_inputs,
               (s.n_positive_before_hide - 1)::BIGINT AS n_positive_inputs,
               s.n_eligible_targets::BIGINT AS n_eligible_targets
        FROM role_assignments a
        JOIN user_stats s USING(user_key)
        WHERE a.role = '{role}'""")
    con.execute(f"CREATE TEMP TABLE {role}_exclusions AS SELECT user_key, app_id FROM {role}_query")


def validate_role_cohort(
    con: duckdb.DuckDBPyConnection,
    role: str,
    quotas: dict[str, dict[str, int]],
    input_apps_table: str = "input_apps",
    candidate_table: str = "linux_candidates",
) -> dict[str, int]:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown role: {role}")
    checks = con.execute(f"""SELECT
        (SELECT count(*) - count(DISTINCT user_key) FROM {role}_users) duplicate_users,
        (SELECT count(*) FROM (
            SELECT user_key, app_id FROM {role}_query GROUP BY 1,2 HAVING count(*) <> 1
        )) duplicate_query_pairs,
        (SELECT count(*) FROM (
            SELECT user_key, app_id FROM {role}_targets GROUP BY 1,2 HAVING count(*) <> 1
        )) duplicate_targets,
        (SELECT count(*) FROM {role}_targets t JOIN {role}_query q USING(user_key,app_id)) target_query_overlap,
        (SELECT count(*) FROM {role}_targets WHERE positive IS DISTINCT FROM TRUE OR linux IS DISTINCT FROM TRUE) invalid_targets,
        (SELECT count(*) FROM {role}_targets t ANTI JOIN {candidate_table} c USING(app_id)) target_outside_candidate_catalog,
        (SELECT count(*) FROM {role}_query q ANTI JOIN {input_apps_table} c USING(app_id)) query_outside_input_catalog,
        (SELECT count(*) FROM (
            SELECT u.user_key FROM {role}_users u LEFT JOIN {role}_query q USING(user_key)
            GROUP BY u.user_key, u.n_total_inputs, u.n_positive_inputs
            HAVING count(q.app_id) <> u.n_total_inputs OR
                   count(q.app_id) FILTER (WHERE q.positive) <> u.n_positive_inputs
        )) query_count_mismatch,
        (SELECT count(*) FROM (
            (SELECT user_key, app_id FROM {role}_query EXCEPT SELECT user_key, app_id FROM {role}_exclusions)
            UNION ALL
            (SELECT user_key, app_id FROM {role}_exclusions EXCEPT SELECT user_key, app_id FROM {role}_query)
        )) exclusion_mismatch,
        (SELECT count(*) FROM {role}_users WHERE n_total_inputs < 1 OR n_positive_inputs < 1) ineligible_users,
        (SELECT count(*) FROM role_assignments a JOIN prior_evaluation_accounts p USING(user_key)
            WHERE a.role = '{role}') prior_evaluation_overlap""").fetchone()
    names = [item[0] for item in con.description]
    result = {key: int(value) for key, value in zip(names, checks)}
    if any(result.values()):
        raise ValueError(f"Cohort validation failed for {role}: {result}")
    expected_size = sum(quotas[stratum][role] for stratum in STRATA)
    actual_size = con.execute(f"SELECT count(*) FROM {role}_users").fetchone()[0]
    if actual_size != expected_size:
        raise ValueError(f"{role} has {actual_size} users; expected {expected_size}")
    for stratum in STRATA:
        actual = con.execute(
            f"SELECT count(*) FROM role_assignments WHERE role = ? AND input_stratum = ?",
            [role, stratum],
        ).fetchone()[0]
        expected = quotas[stratum][role]
        if actual != expected:
            raise ValueError(f"{role}/{stratum} has {actual} users; expected {expected}")
    return result


def export_table(con: duckdb.DuckDBPyConnection, table: str, path: Path, order: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT * FROM {table} ORDER BY {order}) TO {quote(path)} "
        f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})"
    )
    log(f"Wrote {path.parent.name}/{path.name}: {pq.ParquetFile(path).metadata.num_rows:,} rows")


def filter_training_positive(
    source: Path,
    output: Path,
    reserved_user_ids: np.ndarray,
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, int]:
    """Remove reserved users and externally sort the new base-training pairs."""
    reserved = np.asarray(reserved_user_ids, dtype=np.uint64)
    if len(reserved) and np.any(reserved[1:] <= reserved[:-1]):
        raise ValueError("Reserved account IDs must be unique and ascending")
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow != PAIR_SCHEMA:
        raise ValueError(f"Unexpected training schema: {parquet.schema_arrow}")
    output.parent.mkdir(parents=True, exist_ok=True)
    own_connection = con is None
    if own_connection:
        con = duckdb.connect()
        sort_temp = output.parent / ".duckdb_sort"
        sort_temp.mkdir(exist_ok=True)
        con.execute("SET memory_limit='1GB'")
        con.execute(f"SET threads={THREADS}")
        con.execute(f"SET temp_directory={quote(sort_temp)}")
        con.execute("SET max_temp_directory_size='20GB'")
    assert con is not None
    try:
        reserved_table = pa.table({"user_key": pa.array(reserved, type=pa.uint64())})
        con.register("_v4_reserved_batch", reserved_table)
        try:
            con.execute("CREATE OR REPLACE TEMP TABLE reserved_v4_accounts AS SELECT * FROM _v4_reserved_batch")
        finally:
            con.unregister("_v4_reserved_batch")
        con.execute(f"""COPY (
            SELECT p.user_key, p.app_id, p.positive
            FROM read_parquet({quote(source)}) p
            ANTI JOIN reserved_v4_accounts r USING(user_key)
            ORDER BY p.user_key, p.app_id
        ) TO {quote(output)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})""")
    finally:
        if own_connection:
            con.close()
    output_pf = pq.ParquetFile(output)
    if output_pf.schema_arrow != PAIR_SCHEMA:
        raise ValueError("Filtered base-training Parquet schema changed")
    kept_rows = output_pf.metadata.num_rows
    kept_users = 0
    last_user = None
    last_app = None
    for batch in output_pf.iter_batches(batch_size=ROW_GROUP_SIZE, columns=["user_key", "app_id"]):
        users = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.uint64)
        apps = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)
        if len(users) > 1:
            if np.any(users[1:] < users[:-1]):
                raise ValueError("New base-training pairs are not sorted by user_key")
            same = users[1:] == users[:-1]
            if np.any(apps[1:][same] <= apps[:-1][same]):
                raise ValueError("New base-training pairs contain duplicate or unsorted user/App pairs")
        if len(users):
            if last_user is None:
                kept_users += 1
            else:
                if users[0] < last_user or (users[0] == last_user and apps[0] <= last_app):
                    raise ValueError("New base-training pairs are not globally sorted")
                if int(users[0]) != last_user:
                    kept_users += 1
            kept_users += int(np.count_nonzero(users[1:] != users[:-1]))
            last_user, last_app = int(users[-1]), int(apps[-1])
    return int(kept_rows), int(kept_users)


def file_inventory(root: Path) -> dict[str, dict]:
    inventory: dict[str, dict] = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root).as_posix()
        parquet = pq.ParquetFile(path)
        inventory[relative] = {
            "rows": parquet.metadata.num_rows,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "schema": [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow],
        }
    return inventory


def expected_quota_counts(quotas: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {role: {stratum: int(quotas[stratum][role]) for stratum in STRATA} for role in ROLE_NAMES}


def run_builder(
    split_root: Path = DEFAULT_SPLIT,
    output: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
    quotas: dict[str, dict[str, int]] | None = None,
    memory_limit: str = MEMORY_LIMIT,
    threads: int = THREADS,
) -> dict:
    quotas = quotas or DEFAULT_QUOTAS
    validate_quotas(quotas)
    split_root = split_root.resolve()
    output = output.resolve()
    modeling_root = split_root.parent
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite v4 split output: {output}")
    if threads != THREADS:
        raise ValueError(f"This split builder is fixed at {THREADS} DuckDB threads")
    if memory_limit not in ("1GB", "2GB"):
        raise ValueError("Set DuckDB memory_limit to 1GB or 2GB")
    split_manifest_path = split_root / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text("utf-8"))
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("split_v1 is not a verified completed split")
    baseline_protocol = EDA / "data/reference/baseline_protocol_v1.json"
    if baseline_protocol.exists():
        protocol = json.loads(baseline_protocol.read_text("utf-8"))
        if protocol.get("split_manifest_sha256") != digest(split_manifest_path):
            raise ValueError("split_v1 manifest differs from the frozen v1 baseline protocol")
    required = [
        split_root / "train/interactions.parquet",
        split_root / "train/positive_catalog.parquet",
        split_root / "catalog/input_apps.parquet",
        split_root / "catalog/linux_candidates.parquet",
    ]
    for source in required:
        relative = source.relative_to(split_root).as_posix()
        expected = split_manifest.get("files", {}).get(relative)
        if not expected or expected.get("sha256") != digest(source):
            raise ValueError(f"Frozen split source fingerprint mismatch: {relative}")
    if pq.ParquetFile(required[2]).metadata.num_rows != 37_610:
        raise ValueError("Expected the frozen 37,610-app input catalog")
    if pq.ParquetFile(required[3]).metadata.num_rows != 6_630:
        raise ValueError("Expected the frozen 6,630-app Linux candidate catalog")

    prior_files = prior_evaluation_user_files(split_root, modeling_root)
    source_files = [*required, split_manifest_path, *prior_files]
    source_info = []
    for path in prior_files:
        relative = path.relative_to(modeling_root).as_posix()
        source_info.append({"path": relative, "rows": pq.ParquetFile(path).metadata.num_rows, "sha256": digest(path)})

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f".building-{uuid.uuid4().hex[:10]}")
    stage.mkdir(parents=True, exist_ok=False)
    temp_dir = stage / ".duckdb_tmp"
    temp_dir.mkdir()
    db_file = stage / ".build.duckdb"
    log(f"Preparing v4 account split from {split_root}")
    con = duckdb.connect(str(db_file))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory={quote(temp_dir)}")
    con.execute("SET max_temp_directory_size='20GB'")
    con.execute("SET preserve_insertion_order=false")
    try:
        con.execute(f"CREATE TEMP TABLE input_apps AS SELECT * FROM read_parquet({quote(split_root/'catalog/input_apps.parquet')})")
        con.execute(f"CREATE TEMP TABLE linux_candidates AS SELECT * FROM read_parquet({quote(split_root/'catalog/linux_candidates.parquet')})")
        log(f"Loading {len(prior_files)} locally known prior-evaluation account lists")
        build_prior_account_table(con, prior_files)
        prior_rows = con.execute("SELECT count(*) FROM prior_evaluation_accounts").fetchone()[0]
        log("Counting eligible old-train users in the frozen input catalog")
        build_user_stats(con, split_root)
        prior_overlap = con.execute("""SELECT count(DISTINCT s.user_key)
            FROM user_stats s JOIN prior_evaluation_accounts p USING(user_key)""").fetchone()[0]
        diagnostic_row = con.execute("""SELECT
            count(*) AS train_users_with_positive_linux_target,
            count(*) FILTER (WHERE n_total_before_hide = 1) AS no_total_input_after_hide,
            count(*) FILTER (WHERE n_positive_before_hide = 1) AS no_positive_input_after_hide,
            count(*) FILTER (WHERE n_total_before_hide >= 2 AND n_positive_before_hide = 2) AS one_positive_input_after_hide,
            count(*) FILTER (WHERE n_total_before_hide >= 2 AND n_positive_before_hide >= 2) AS cohort_eligible_before_prior_exclusions,
            count(*) FILTER (WHERE n_total_before_hide >= 2 AND n_positive_before_hide >= 2 AND n_total_before_hide - 1 <= 2) AS eligible_stratum_1_2,
            count(*) FILTER (WHERE n_total_before_hide >= 2 AND n_positive_before_hide >= 2 AND n_total_before_hide - 1 BETWEEN 3 AND 4) AS eligible_stratum_3_4,
            count(*) FILTER (WHERE n_total_before_hide >= 2 AND n_positive_before_hide >= 2 AND n_total_before_hide - 1 >= 5) AS eligible_stratum_5_plus
            FROM user_stats""").fetchone()
        diagnostic_names = [field[0] for field in con.description]
        diagnostics = {name: int(value) for name, value in zip(diagnostic_names, diagnostic_row)}
        diagnostics["prior_evaluation_account_ids_union"] = int(prior_rows)
        diagnostics["old_train_users_overlapping_prior_evaluation_ids"] = int(prior_overlap)
        build_role_assignments(con, seed, quotas)
        role_counts = con.execute("""SELECT role, input_stratum, count(*) AS users
            FROM role_assignments GROUP BY role, input_stratum ORDER BY role, input_stratum""").fetchall()
        diagnostics["reserved_counts_by_role_and_stratum"] = {
            role: {stratum: int(count) for stratum, count in []} for role in ROLE_NAMES
        }
        diagnostics["reserved_counts_by_role_and_stratum"] = {
            role: {stratum: 0 for stratum in STRATA} for role in ROLE_NAMES
        }
        for role, stratum, count in role_counts:
            diagnostics["reserved_counts_by_role_and_stratum"][role][stratum] = int(count)
        reserved_ids = np.asarray(
            con.execute("SELECT user_key FROM role_assignments ORDER BY user_key").fetchnumpy()["user_key"],
            dtype=np.uint64,
        )
        if len(reserved_ids) != len(np.unique(reserved_ids)):
            raise ValueError("An account was assigned to more than one v4 role")
        reserved_table = pa.table({"user_key": pa.array(reserved_ids, type=pa.uint64())})
        pq.write_table(reserved_table, stage / "reserved_accounts.parquet", compression="zstd")
        copy_catalog = stage / "catalog"
        copy_catalog.mkdir()
        shutil.copy2(split_root / "catalog/input_apps.parquet", copy_catalog / "input_apps.parquet")
        shutil.copy2(split_root / "catalog/linux_candidates.parquet", copy_catalog / "linux_candidates.parquet")

        log("Building selected-account histories and one hidden positive target per user")
        build_selected_history(con, split_root)
        cohort_checks = {}
        for role in ROLE_NAMES:
            create_role_cohort_tables(con, role, seed)
            cohort_checks[role] = validate_role_cohort(con, role, quotas)
        intersections = con.execute("""SELECT
            count(*) - count(DISTINCT user_key) AS duplicate_role_account_rows,
            count(*) FILTER (WHERE role NOT IN ('fusion_fit','selection','confirmation')) AS invalid_roles
            FROM role_assignments""").fetchone()
        if any(intersections):
            raise ValueError(f"Role assignment checks failed: {intersections}")
        if con.execute("SELECT count(*) FROM role_assignments a JOIN prior_evaluation_accounts p USING(user_key)").fetchone()[0]:
            raise ValueError("A role account overlaps a known prior evaluation cohort")

        for role in ROLE_NAMES:
            role_root = stage / role
            role_root.mkdir()
            export_table(con, f"{role}_users", role_root / "users.parquet", "user_key")
            export_table(con, f"{role}_query", role_root / "query.parquet", "user_key, app_id")
            export_table(con, f"{role}_targets", role_root / "targets.parquet", "user_key, target_order")
            export_table(con, f"{role}_exclusions", role_root / "exclusions.parquet", "user_key, app_id")

        log("Writing new base-training positives with every v4 role account removed")
        train_pairs, train_users = filter_training_positive(
            split_root / "train/positive_catalog.parquet",
            stage / "train/positive_catalog.parquet",
            reserved_ids,
        )
        diagnostics["base_train_positive_pairs"] = int(train_pairs)
        diagnostics["base_train_positive_users"] = int(train_users)
        con.register("reserved_arrow", reserved_table)
        try:
            con.execute("CREATE TEMP TABLE reserved_v4_accounts AS SELECT * FROM reserved_arrow")
        finally:
            con.unregister("reserved_arrow")
        base_leak = con.execute(f"""SELECT count(*) FROM read_parquet({quote(stage/'train/positive_catalog.parquet')}) p
            JOIN reserved_v4_accounts r USING(user_key)""").fetchone()[0]
        if base_leak:
            raise ValueError(f"New base-training file contains {base_leak} reserved positive pairs")
        diagnostics["base_train_reserved_pair_leaks"] = int(base_leak)

        files = file_inventory(stage)
        source_manifest_hash = digest(split_manifest_path)
        manifest = {
            "version": VERSION,
            "status": "complete",
            "seed": int(seed),
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": "Fresh v4 account reservation, cohort construction, and base-training projection only; no models fitted, predictions generated, settings selected, or confirmation results inspected.",
            "source": {
                "split_root": str(split_root),
                "split_manifest_sha256": source_manifest_hash,
                "split_version": split_manifest["version"],
                "input_catalog_apps": 37_610,
                "linux_candidate_apps": 6_630,
                "training_positive_catalog": split_manifest["files"]["train/positive_catalog.parquet"],
            },
            "historical_exposure": {
                "statement": "These accounts were sampled from split_v1 TRAIN and their interactions were available to v1-v3 training runs. This v4 builder excludes them from the new base-training file and from locally known prior evaluation account lists, but they are not untouched by all historical development.",
                "forge_identity_crosscheck": "Forgemaster source account identities are not present in the local workspace, so a Kieran-to-Forgemaster identity cross-check could not be performed.",
                "confirmation_label": "A reserved v4 confirmation cohort after account exclusion; not a pristine, never-seen historical final test.",
            },
            "prior_evaluation_account_sources": source_info,
            "rules": {
                "eligible_source": "split_v1/train/interactions.parquet intersected with the frozen 37,610-app input catalog; one positive Linux target must be available.",
                "post_hide_eligibility": "At least one total input and one positive input remain after hiding one positive Linux target.",
                "post_hide_strata": {"1-2": "1 or 2 remaining catalog games", "3-4": "3 or 4 remaining catalog games", "5+": "5 or more remaining catalog games"},
                "target": "One positive Linux candidate sampled uniformly without replacement from each user's eligible candidates via SHA256 counter-mode rejection sampling.",
                "role_assignment": "Within each post-hide stratum, SHA256 rank users by seed and exact uint64 user_key; assign contiguous nonoverlapping rank ranges to fusion_fit, selection, and confirmation in that order.",
                "quotas": expected_quota_counts(quotas),
                "train_reservation": "All accounts assigned to any v4 role are removed from train/positive_catalog.parquet before new model fitting.",
                "no_input_users": "Users with a positive Linux target but zero remaining total inputs are counted in eligibility diagnostics and excluded from all model-selection cohorts.",
                "primary_stratum": "5+ remaining inputs. Shorter strata are reserved for separate diagnostics and do not replace the primary >=5 evaluation.",
            },
            "eligibility_diagnostics": diagnostics,
            "validation": {
                "passed": True,
                "checks": {
                    "role_cohorts": cohort_checks,
                    "role_account_overlap": 0,
                    "prior_evaluation_account_overlap": 0,
                    "base_train_reserved_pair_leaks": 0,
                },
            },
            "runtime": {"python": platform.python_version(), "duckdb": duckdb.__version__, "memory_limit": memory_limit, "threads": threads},
            "files": files,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        con.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        db_file.unlink(missing_ok=True)
        stage.rename(output)
        log(f"Completed v4 split at {output}")
        return manifest
    except Exception:
        con.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--memory-limit", choices=("1GB", "2GB"), default=MEMORY_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_builder(split_root=args.split, output=args.output, seed=args.seed, memory_limit=args.memory_limit)


if __name__ == "__main__":
    main()
