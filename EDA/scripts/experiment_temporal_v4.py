"""Prepare an interaction-time-controlled v4 evaluation supplement.

This builder freezes the temporal protocol, creates T0-visible histories and
future targets for the already reserved v4 account roles, and materializes a
fresh T0 training-pair file with every v4 role removed. It does not fit a
model or score any cohort. In particular, confirmation targets are written
only as a sealed cohort input and are never opened for scoring here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EDA = Path(__file__).resolve().parents[1]
MODELING = EDA / "data/processed/modeling"
EXPERIMENTS = MODELING / "experiments_v4"
V4_SPLIT = EXPERIMENTS / "split"
OUTPUT = EXPERIMENTS / "temporal_split"
WORK = EDA / "data/processed/kieran_original/eda_work.duckdb"
SOURCE_MANIFEST = EDA / "data/source_manifest.json"
V1_SPLIT = MODELING / "split_v1"
TRAIN_LINEAGE = V1_SPLIT / "train/review_lineage.parquet"
INPUT_APPS = V1_SPLIT / "catalog/input_apps.parquet"
LINUX_CANDIDATES = V1_SPLIT / "catalog/linux_candidates.parquet"
REFERENCE_MANIFEST = EDA / "data/reference/manifest.json"
RELEASE_SNAPSHOT = EDA / "legacy/data/processed/C_games_2025.parquet"
TEMPORAL_V3 = MODELING / "improvement_v3/temporal_cutoff2022"
TEMPORAL_PROTOCOL_PATH = EDA / "data/reference/temporal_protocol_v4.json"

T0 = 1_672_531_199  # 2022-12-31 23:59:59 UTC
T1 = 1_688_169_599  # 2023-06-30 23:59:59 UTC
T2 = 1_699_142_399  # 2023-11-04 23:59:59 UTC
T0_DATE = "2022-12-31"
SEED = 20261108
TARGET_NAMESPACE = "steam-temporal-v4-hidden-positive-linux"
ROLE_NAMES = ("fusion_fit", "selection", "confirmation")
MEMORY_LIMIT = "2GB"
THREADS = 2
TRAIN_BUCKETS = 64
ROW_GROUP_SIZE = 122_880
VERSION = "temporal-v4-cohort-preparation-v1"

PAIR_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()), ("positive", pa.bool_()),
])
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
    ("positive", pa.bool_()), ("linux", pa.bool_()),
    ("target_order", pa.int64()),
    ("timestamp_created", pa.int64()), ("timestamp_updated", pa.int64()),
])
EXCLUSION_SCHEMA = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64())])

TEMPORAL_PROTOCOL = {
    "version": "temporal-evaluation-protocol-v4",
    "status": "fixed_before_temporal_metrics",
    "created_before_temporal_metrics": True,
    "cutoffs": {
        "T0": {"unix_seconds": T0, "utc": "2022-12-31T23:59:59+00:00"},
        "T1": {"unix_seconds": T1, "utc": "2023-06-30T23:59:59+00:00"},
        "T2": {"unix_seconds": T2, "utc": "2023-11-04T23:59:59+00:00"},
        "comparisons": "T0 history includes created <= T0 AND updated <= T0; targets use open-left, closed-right horizons.",
    },
    "sources": {
        "accounts": "The fusion_fit, selection, and confirmation rosters already reserved by experiments_v4/split; random-holdout target files are not used for this supplement.",
        "accepted_reviews": "split_v1/train/review_lineage.parquet joined by exact user, App ID, review ID, and label to valid_raw in the audited Kieran DuckDB cache.",
        "training": "Reuse the audited temporal_cutoff2022 accepted positive-pair artifact only if its source, protocol, code, output-file hashes and row counts verify; otherwise derive from accepted train lineage joined to valid_raw.",
        "catalog": "Keep the fixed 37,610 input App IDs and 6,630 Linux candidate App IDs; score arrays retain the complete shapes and apply an availability mask.",
        "release_dates": {
            "snapshot": "legacy/data/processed/C_games_2025.parquet",
            "sha256": "e65e452f5a5e5ec95a79d5a5ad8811739b3ff31384c4ff0a7926747829a82efc",
            "availability": "A Linux candidate is available only when its current parsed release_date is known and <= 2022-12-31. Unknown and later dates are masked; no candidate is dropped from the saved 6,630-item arrays.",
        },
        "metadata_limit": "Current Linux support and metadata do not reconstruct historical platform support, releases, or metadata versions.",
    },
    "training": {
        "cutoff": "Use accepted positive TRAIN review pairs with timestamp_created and timestamp_updated both non-null and <= T0, within the fixed input catalog.",
        "v4_exclusion": "Remove every v4 reserved account (fusion_fit, selection, and confirmation) from the temporal training pairs before any fresh fit.",
        "freshness_limit": "The reserved accounts were part of historical v1-v3 training data; this is a new account-disjoint v4 refit, not an untouched historical development set.",
    },
    "cohorts": {
        "history": "Accepted review pairs in the same reserved role account whose created and updated timestamps are both known and <= T0; positives and negatives are visible. Deduplicate to one account/App pair.",
        "target": "One positive current-Linux candidate per account, uniformly sampled without replacement from distinct eligible Apps released by T0 and absent from that account's T0 history.",
        "fusion_fit": {"target_created": f"({T0},{T1}]", "target_updated": f"<= {T1}"},
        "selection": {"target_created": f"({T0},{T1}]", "target_updated": f"<= {T1}"},
        "confirmation": {"target_created": f"({T1},{T2}]", "target_updated": f"<= {T2}", "scoring": "Do not score or inspect confirmation outcomes until the separate selection lock names frozen finalists."},
        "minimum_history": "No minimum history is used to select a temporal cohort; target-eligible accounts with zero or short T0 histories remain in the cohort. Report 1-2, 3-4, 5+ input groups and zero-input counts separately.",
        "sampling": f"SHA256 counter-mode rejection sampling over ascending eligible App IDs; namespace={TARGET_NAMESPACE}; seed={SEED}; exact uint64 user key and temporal role.",
    },
    "reporting": {
        "primary_population": "T0 histories with at least 5 remaining input games and at least 1 positive; align the v4 primary population.",
        "supplemental_population": "All target-eligible accounts; show 1-2, 3-4, 5+, zero-input, and zero-positive-input groups separately. Do not pool the deliberately stratified v4 role samples as a natural population.",
        "metrics": "Use the already frozen v4 ranking and uncertainty rules; this protocol fixes temporal data construction and availability only.",
    },
    "limitations": [
        "The experiment controls accepted interaction creation and update timestamps but does not reconstruct historical catalog, metadata, Linux support, or recommendation impressions.",
        "Candidate availability is approximated with a current 2025 release-date snapshot; unknown release dates are excluded conservatively.",
        "Current Linux candidate flags may reflect changes after T0; they are retained because historical platform support is unavailable.",
        "The reserved v4 accounts were included in historical v1-v3 training and development, so the temporal confirmation is not a pristine never-seen test.",
        "Confirmation target files are materialized for a later locked evaluation and are not scored by this builder.",
    ],
}


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze_protocol(path: Path = TEMPORAL_PROTOCOL_PATH) -> tuple[Path, str]:
    """Write once or require the already frozen temporal rules to match exactly."""
    path = path.resolve()
    if path.exists():
        frozen = json.loads(path.read_text("utf-8"))
        if frozen != TEMPORAL_PROTOCOL:
            raise ValueError(f"Frozen temporal protocol differs from this builder: {path}")
    else:
        write_json(path, TEMPORAL_PROTOCOL)
    return path, digest(path)


def uniform_index(user_key: int, count: int, seed: int = SEED, role: str = "selection") -> int:
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("count must be a positive integer")
    limit = (1 << 256) - (1 << 256) % int(count)
    counter = 0
    while True:
        payload = f"{TARGET_NAMESPACE}:{seed}:{role}:{int(user_key)}:{counter}"
        value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        if value < limit:
            return value % int(count)
        counter += 1


def choose_target_app(user_key: int, app_ids: list[int], seed: int = SEED, role: str = "selection") -> int:
    apps = [int(app_id) for app_id in app_ids]
    if not apps or apps != sorted(apps) or len(apps) != len(set(apps)) or apps[0] <= 0:
        raise ValueError("Eligible target App IDs must be positive, ascending, unique, and nonempty")
    return apps[uniform_index(user_key, len(apps), seed, role)]


def history_visible(created: int | None, updated: int | None, cutoff: int = T0) -> bool:
    return created is not None and updated is not None and created <= cutoff and updated <= cutoff


def target_in_horizon(
    created: int | None, updated: int | None, start_exclusive: int, end_inclusive: int,
) -> bool:
    return created is not None and updated is not None and start_exclusive < created <= end_inclusive and updated <= end_inclusive


def release_available(release_date: str | None, cutoff_date: str = T0_DATE) -> bool:
    """Conservatively mask unknown/malformed release dates and post-T0 releases."""
    if not release_date:
        return False
    try:
        return datetime.strptime(release_date[:10], "%Y-%m-%d").date() <= datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False


def source_path(eda_root: Path, source: dict) -> Path:
    relative = Path(source["file"])
    path = (eda_root / relative).resolve()
    if eda_root.resolve() not in path.parents:
        raise ValueError(f"Source path escapes EDA root: {relative}")
    return path


def validate_old_temporal_training_artifact(
    temporal_root: Path = TEMPORAL_V3,
    eda_root: Path = EDA,
    expected_builder: Path | None = None,
    check_source_hashes: bool = True,
) -> tuple[bool, list[str], Path | None]:
    """Check the old temporal artifact's protocol, report, outputs, and sources.

    The old model itself is never reused. Only its audited T0 training positive
    pair file may be used as a data source, and only if every recorded hash and
    row count passes. A failure routes the builder to accepted-lineage fallback.
    """
    temporal_root = temporal_root.resolve()
    result = temporal_root / "result"
    protocol_path = temporal_root / "protocol.json"
    model_manifest_path = result / "model/manifest.json"
    report_path = result / "report.json"
    train_path = result / "training/positive_catalog.parquet"
    reasons: list[str] = []
    try:
        old_protocol = json.loads(protocol_path.read_text("utf-8"))
        model = json.loads(model_manifest_path.read_text("utf-8"))
        report = json.loads(report_path.read_text("utf-8"))
    except Exception as error:
        return False, [f"missing_or_invalid_old_temporal_receipt:{type(error).__name__}"], None

    protocol_hash = digest(protocol_path)
    if model.get("status") != "complete" or model.get("cutoff_unix") != T0:
        reasons.append("old_model_manifest_not_complete_at_T0")
    if model.get("protocol_sha256") != protocol_hash:
        reasons.append("old_protocol_hash_mismatch")
    if report.get("status") != "complete" or report.get("protocol", {}).get("sha256") != protocol_hash:
        reasons.append("old_report_or_protocol_hash_mismatch")
    if report.get("source_fingerprints") != model.get("source_fingerprints"):
        reasons.append("old_source_fingerprint_receipts_disagree")

    expected_builder = expected_builder or EDA / "scripts/experiment_temporal_v3.py"
    builder_hash = digest(expected_builder) if expected_builder.is_file() else None
    if not builder_hash or model.get("builder_sha256") != builder_hash or report.get("builder_sha256") != builder_hash:
        reasons.append("old_builder_hash_mismatch")
    snapshot = temporal_root / "prediction_replay/temporal_builder_snapshot.py"
    if not snapshot.is_file() or digest(snapshot) != builder_hash:
        reasons.append("old_builder_snapshot_hash_mismatch")

    files = report.get("files", {})
    if "training/positive_catalog.parquet" not in files:
        reasons.append("old_report_missing_training_file_receipt")
    output_receipts_valid = True
    for relative, expected in files.items():
        path = result / relative
        if not path.is_file():
            reasons.append(f"old_output_missing:{relative}")
            output_receipts_valid = False
            continue
        if path.stat().st_size != expected.get("bytes") or digest(path) != expected.get("sha256"):
            reasons.append(f"old_output_hash_or_size_mismatch:{relative}")
            output_receipts_valid = False
            continue
        if path.suffix == ".parquet" and expected.get("rows") is not None:
            output_pf = pq.ParquetFile(path)
            output_rows = output_pf.metadata.num_rows
            output_pf.close()
            if output_rows != expected.get("rows"):
                reasons.append(f"old_output_row_count_mismatch:{relative}")
                output_receipts_valid = False
    if not output_receipts_valid:
        return False, reasons, None
    train_receipt = files.get("training/positive_catalog.parquet", {})
    if train_path.is_file():
        train_pf = pq.ParquetFile(train_path)
        if train_pf.schema_arrow != PAIR_SCHEMA:
            reasons.append("old_training_schema_mismatch")
        expected_rows = int(model.get("train_support", {}).get("positive_pairs", -1))
        if train_pf.metadata.num_rows != expected_rows or train_receipt.get("rows") != expected_rows:
            reasons.append("old_training_row_count_mismatch")
        train_pf.close()
        if int(report.get("training", {}).get("positive_pairs", -2)) != expected_rows:
            reasons.append("old_training_report_count_mismatch")

    if check_source_hashes:
        for label, source in model.get("source_fingerprints", {}).items():
            try:
                path = source_path(eda_root, source)
                if not path.is_file():
                    reasons.append(f"old_source_missing:{label}")
                    continue
                if path.stat().st_size != source.get("bytes") or digest(path) != source.get("sha256"):
                    reasons.append(f"old_source_hash_or_size_mismatch:{label}")
            except Exception as error:
                reasons.append(f"old_source_invalid:{label}:{type(error).__name__}")
    if old_protocol.get("cutoff", {}).get("unix_seconds") != T0:
        reasons.append("old_protocol_cutoff_mismatch")
    return not reasons, reasons, train_path if not reasons else None


def verify_release_snapshot() -> dict:
    if not RELEASE_SNAPSHOT.is_file():
        raise FileNotFoundError(f"Release-date snapshot missing: {RELEASE_SNAPSHOT}")
    reference = json.loads(REFERENCE_MANIFEST.read_text("utf-8"))
    expected = next((item for item in reference.get("former_catalog_inputs", [])
                     if item.get("file") == RELEASE_SNAPSHOT.name), None)
    if expected is None or digest(RELEASE_SNAPSHOT) != expected.get("sha256"):
        raise ValueError("Current release-date snapshot differs from the frozen reference manifest")
    split_manifest = json.loads((V1_SPLIT / "manifest.json").read_text("utf-8"))
    if split_manifest.get("sources", {}).get("reference_manifest_sha256") != digest(REFERENCE_MANIFEST):
        raise ValueError("Reference manifest no longer matches split_v1")
    return {
        "path": RELEASE_SNAPSHOT.relative_to(EDA).as_posix(),
        "bytes": RELEASE_SNAPSHOT.stat().st_size,
        "sha256": digest(RELEASE_SNAPSHOT),
        "reference_manifest_sha256": digest(REFERENCE_MANIFEST),
        "reference_file_sha256": expected["sha256"],
    }


def verify_v4_split() -> tuple[dict, dict[str, np.ndarray]]:
    manifest_path = V4_SPLIT / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("validation", {}).get("passed"):
        raise ValueError("The v4 account split is not complete and verified")
    if manifest.get("source", {}).get("split_manifest_sha256") != digest(V1_SPLIT / "manifest.json"):
        raise ValueError("The v4 account split is pinned to a different v1 split")
    roles: dict[str, np.ndarray] = {}
    for role in ROLE_NAMES:
        path = V4_SPLIT / role / "users.parquet"
        relative = path.relative_to(V4_SPLIT).as_posix()
        receipt = manifest.get("files", {}).get(relative)
        if not receipt or receipt.get("sha256") != digest(path):
            raise ValueError(f"Reserved v4 role roster hash changed: {relative}")
        users = np.asarray(pq.read_table(path, columns=["user_key"]).column("user_key").to_numpy(), dtype=np.uint64)
        if len(users) != 20_000 or np.any(users[1:] <= users[:-1]):
            raise ValueError(f"Invalid reserved v4 account roster: {role}")
        roles[role] = users
    combined = np.concatenate([roles[role] for role in ROLE_NAMES])
    if len(np.unique(combined)) != len(combined):
        raise ValueError("V4 temporal roles are not account-disjoint")
    reserved_path = V4_SPLIT / "reserved_accounts.parquet"
    receipt = manifest.get("files", {}).get("reserved_accounts.parquet")
    if not receipt or receipt.get("sha256") != digest(reserved_path):
        raise ValueError("Reserved v4 account union hash changed")
    reserved = np.asarray(pq.read_table(reserved_path, columns=["user_key"]).column("user_key").to_numpy(), dtype=np.uint64)
    if len(reserved) != 60_000 or not np.array_equal(reserved, np.sort(combined)):
        raise ValueError("Reserved v4 account union differs from the three role rosters")
    return manifest, roles


def create_release_availability(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    metadata = RELEASE_SNAPSHOT
    con.execute(f"""CREATE TEMP TABLE release_rows AS
        SELECT CAST(appid AS BIGINT) AS app_id,
               CAST(release_date AS VARCHAR) AS release_date_snapshot,
               try_cast(release_date AS DATE) AS release_date
        FROM read_parquet({quote(metadata)})""")
    duplicate_count = con.execute("""SELECT count(*) FROM (
        SELECT app_id FROM release_rows GROUP BY app_id HAVING count(*) > 1
    )""").fetchone()[0]
    if duplicate_count:
        raise ValueError(f"Release snapshot has {duplicate_count} duplicate App IDs")
    con.execute(f"""CREATE TEMP TABLE candidate_availability AS
        SELECT c.app_id, r.release_date_snapshot, r.release_date,
               (r.release_date IS NOT NULL AND r.release_date <= DATE '{T0_DATE}') AS available_at_t0
        FROM read_parquet({quote(LINUX_CANDIDATES)}) c
        LEFT JOIN release_rows r USING(app_id)""")
    rows = con.execute("""SELECT count(*) AS candidates,
        count(*) FILTER(WHERE release_date IS NOT NULL) AS known_release_date,
        count(*) FILTER(WHERE release_date IS NULL) AS unknown_release_date,
        count(*) FILTER(WHERE release_date IS NOT NULL AND release_date > DATE '2022-12-31') AS after_t0,
        count(*) FILTER(WHERE available_at_t0) AS available_at_t0
        FROM candidate_availability""").fetchone()
    names = [item[0] for item in con.description]
    result = {key: int(value) for key, value in zip(names, rows)}
    if result["candidates"] != 6_630:
        raise ValueError("Candidate availability mask must retain all 6,630 Linux candidate rows")
    return result


def create_role_account_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""CREATE TEMP TABLE v4_role_accounts AS
        SELECT user_key, 'fusion_fit'::VARCHAR AS role
        FROM read_parquet({quote(V4_SPLIT/'fusion_fit/users.parquet')})
        UNION ALL
        SELECT user_key, 'selection'::VARCHAR AS role
        FROM read_parquet({quote(V4_SPLIT/'selection/users.parquet')})
        UNION ALL
        SELECT user_key, 'confirmation'::VARCHAR AS role
        FROM read_parquet({quote(V4_SPLIT/'confirmation/users.parquet')})""")
    total, distinct_users, invalid_roles = con.execute("""SELECT count(*),count(DISTINCT user_key),
        count(*) FILTER(WHERE role NOT IN ('fusion_fit','selection','confirmation'))
        FROM v4_role_accounts""").fetchone()
    if int(total) != 60_000 or int(distinct_users) != 60_000 or int(invalid_roles):
        raise ValueError("Invalid v4 temporal role account table")


def materialize_role_events(con: duckdb.DuckDBPyConnection, split: Path = V1_SPLIT) -> dict[str, int]:
    """Join only already reserved TRAIN lineage to valid_raw by exact review key."""
    con.execute(f"""CREATE TEMP TABLE v4_role_lineage AS
        SELECT l.user_key,l.app_id,l.review_key,l.positive
        FROM read_parquet({quote(split/'train/review_lineage.parquet')}) l
        SEMI JOIN v4_role_accounts a USING(user_key)
        JOIN read_parquet({quote(INPUT_APPS)}) c USING(app_id)""")
    try:
        con.execute("SELECT timestamp_created,timestamp_updated FROM valid_raw LIMIT 0")
    except Exception as error:
        raise ValueError("The audited Kieran cache must expose valid_raw review timestamps") from error
    con.execute("""CREATE TEMP TABLE role_events AS
        SELECT l.user_key,l.app_id,l.review_key,l.positive,a.role,
               CAST(r.timestamp_created AS BIGINT) AS timestamp_created,
               CAST(r.timestamp_updated AS BIGINT) AS timestamp_updated,
               i.linux
        FROM v4_role_lineage l
        JOIN v4_role_accounts a USING(user_key)
        JOIN valid_raw r USING(user_key,app_id,review_key,positive)
        JOIN input_apps i USING(app_id)""")
    rows, accounts = con.execute("SELECT count(*),count(DISTINCT user_key) FROM role_events").fetchone()
    if int(accounts) > 60_000:
        raise ValueError("Temporal lineage join returned accounts outside v4 role rosters")
    return {"accepted_events": int(rows), "accounts_with_accepted_events": int(accounts)}


def create_temporal_cohort_tables(
    con: duckdb.DuckDBPyConnection,
    role: str,
    seed: int = SEED,
    events_table: str = "role_events",
    availability_table: str = "candidate_availability",
) -> dict[str, int]:
    """Create one as-of query and one future target for a reserved role."""
    if role not in ROLE_NAMES or events_table != "role_events" or availability_table != "candidate_availability":
        raise ValueError("Unknown temporal role or internal source table")
    lower, upper = (T1, T2) if role == "confirmation" else (T0, T1)
    prefix = f"temporal_{role}"
    con.execute(f"""CREATE TEMP TABLE {prefix}_history AS
        SELECT user_key,app_id,bool_or(positive) AS positive,bool_or(linux) AS linux
        FROM role_events
        WHERE role='{role}'
          AND timestamp_created IS NOT NULL AND timestamp_updated IS NOT NULL
          AND timestamp_created <= {T0} AND timestamp_updated <= {T0}
        GROUP BY user_key,app_id""")
    conflicts = con.execute(f"""SELECT count(*) FROM (
        SELECT user_key,app_id FROM role_events
        WHERE role='{role}' AND timestamp_created IS NOT NULL AND timestamp_updated IS NOT NULL
          AND timestamp_created <= {T0} AND timestamp_updated <= {T0}
        GROUP BY user_key,app_id HAVING count(DISTINCT positive) > 1
    )""").fetchone()[0]
    if conflicts:
        raise ValueError(f"Accepted lineage has {conflicts} label-conflicting T0 account/App pairs")

    con.execute(f"""CREATE TEMP TABLE {prefix}_target_candidates AS
        SELECT e.user_key,e.app_id,true AS positive,true AS linux,
               e.timestamp_created::BIGINT AS timestamp_created,
               e.timestamp_updated::BIGINT AS timestamp_updated
        FROM role_events e
        JOIN candidate_availability a USING(app_id)
        ANTI JOIN {prefix}_history h USING(user_key,app_id)
        WHERE e.role='{role}' AND e.positive AND e.linux AND a.available_at_t0
          AND e.timestamp_created IS NOT NULL AND e.timestamp_updated IS NOT NULL
          AND e.timestamp_created > {lower} AND e.timestamp_created <= {upper}
          AND e.timestamp_updated <= {upper}
        QUALIFY row_number() OVER (
            PARTITION BY e.user_key,e.app_id
            ORDER BY e.timestamp_created,e.timestamp_updated,e.review_key
        ) = 1""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_target_users AS
        SELECT user_key,count(*)::BIGINT AS n_eligible_targets,
               list(app_id ORDER BY app_id) AS target_apps
        FROM {prefix}_target_candidates GROUP BY user_key""")
    function_name = f"_temporal_v4_target_{role}"
    con.create_function(
        function_name,
        lambda user_key, app_ids: choose_target_app(int(user_key), app_ids, seed, role),
        [duckdb.sqltype("UBIGINT"), duckdb.list_type(duckdb.sqltype("BIGINT"))],
        duckdb.sqltype("BIGINT"),
    )
    try:
        con.execute(f"""CREATE TEMP TABLE {prefix}_targets AS
            SELECT c.user_key,{function_name}(c.user_key,c.target_apps) AS app_id,
                   true AS positive,true AS linux,1::BIGINT AS target_order
            FROM {prefix}_target_users c""")
    finally:
        con.remove_function(function_name)
    con.execute(f"""CREATE TEMP TABLE {prefix}_selected_targets AS
        SELECT t.user_key,t.app_id,t.positive,t.linux,t.target_order,
               c.timestamp_created,c.timestamp_updated
        FROM {prefix}_targets t JOIN {prefix}_target_candidates c USING(user_key,app_id)""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_users AS
        SELECT u.user_key,coalesce(count(h.app_id),0)::BIGINT AS n_total_inputs,
               coalesce(count(h.app_id) FILTER(WHERE h.positive),0)::BIGINT AS n_positive_inputs,
               u.n_eligible_targets::BIGINT AS n_eligible_targets
        FROM {prefix}_target_users u
        LEFT JOIN {prefix}_history h USING(user_key)
        GROUP BY u.user_key,u.n_eligible_targets""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_query AS
        SELECT h.user_key,h.app_id,h.positive,h.linux
        FROM {prefix}_history h SEMI JOIN {prefix}_users u USING(user_key)""")
    con.execute(f"CREATE TEMP TABLE {prefix}_exclusions AS SELECT user_key,app_id FROM {prefix}_query")

    checks = con.execute(f"""SELECT
        (SELECT count(*) FROM {prefix}_targets t JOIN {prefix}_query q USING(user_key,app_id)) target_query_overlap,
        (SELECT count(*) FROM {prefix}_targets t LEFT JOIN {availability_table} a USING(app_id)
            WHERE a.available_at_t0 IS DISTINCT FROM TRUE) target_not_available,
        (SELECT count(*) FROM {prefix}_selected_targets t
            WHERE t.timestamp_created <= {lower} OR t.timestamp_created > {upper} OR t.timestamp_updated > {upper}) bad_target_times,
        (SELECT count(*) FROM (
            SELECT u.user_key FROM {prefix}_users u LEFT JOIN {prefix}_query q USING(user_key)
            GROUP BY u.user_key,u.n_total_inputs,u.n_positive_inputs
            HAVING count(q.app_id)<>u.n_total_inputs OR
                   count(q.app_id) FILTER(WHERE q.positive)<>u.n_positive_inputs
        )) bad_query_counts,
        (SELECT count(*) FROM (
            (SELECT user_key,app_id FROM {prefix}_query EXCEPT SELECT user_key,app_id FROM {prefix}_exclusions)
            UNION ALL
            (SELECT user_key,app_id FROM {prefix}_exclusions EXCEPT SELECT user_key,app_id FROM {prefix}_query)
        )) bad_exclusions,
        (SELECT count(*) FROM {prefix}_history h JOIN {prefix}_targets t USING(user_key,app_id)) history_target_overlap""").fetchone()
    if any(int(value) for value in checks):
        raise ValueError(f"Temporal cohort checks failed for {role}: {checks}")
    counts = con.execute(f"""SELECT
        (SELECT count(*) FROM {prefix}_users) AS users,
        (SELECT count(*) FROM {prefix}_target_candidates) AS eligible_target_pairs,
        (SELECT count(*) FROM {prefix}_query) AS query_pairs,
        (SELECT count(*) FROM {prefix}_users WHERE n_total_inputs=0) AS zero_input_users,
        (SELECT count(*) FROM {prefix}_users WHERE n_positive_inputs=0) AS zero_positive_input_users,
        (SELECT count(*) FROM {prefix}_users WHERE n_total_inputs>=5 AND n_positive_inputs>=1) AS primary_users,
        (SELECT count(*) FROM {prefix}_users WHERE n_total_inputs BETWEEN 1 AND 2) AS history_1_2_users,
        (SELECT count(*) FROM {prefix}_users WHERE n_total_inputs BETWEEN 3 AND 4) AS history_3_4_users,
        (SELECT count(*) FROM {prefix}_users WHERE n_total_inputs>=5) AS history_5_plus_users""").fetchone()
    keys = [item[0] for item in con.description]
    summary = {key: int(value) for key, value in zip(keys, counts)}
    summary["target_query_overlap"] = int(checks[0])
    summary["bad_target_times"] = int(checks[2])
    summary["lower_open"] = int(lower)
    summary["upper_closed"] = int(upper)
    return summary


def export_table(con: duckdb.DuckDBPyConnection, table: str, path: Path, order: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM {table} ORDER BY {order}) TO {quote(path)} "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})")


def filter_reserved_training_pairs(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    output: Path,
    reserved_table: str = "v4_role_accounts",
) -> dict[str, int]:
    """Fresh sorted T0 positives, removing every account in the v4 role roster."""
    if reserved_table != "v4_role_accounts":
        raise ValueError("Only the fixed v4 role reservation table is supported")
    source_pf = pq.ParquetFile(source)
    if source_pf.schema_arrow != PAIR_SCHEMA:
        raise ValueError("T0 training source must match (uint64 user, int64 app, bool positive)")
    output.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (
        SELECT p.user_key,p.app_id,p.positive
        FROM read_parquet({quote(source)}) p
        ANTI JOIN v4_role_accounts r USING(user_key)
        WHERE p.positive
        ORDER BY p.user_key,p.app_id
    ) TO {quote(output)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})""")
    pf = pq.ParquetFile(output)
    if pf.schema_arrow != PAIR_SCHEMA:
        raise ValueError("Filtered T0 training schema changed")
    rows = int(pf.metadata.num_rows)
    users = int(con.execute(f"SELECT count(DISTINCT user_key) FROM read_parquet({quote(output)})").fetchone()[0])
    overlap = int(con.execute(f"""SELECT count(*) FROM read_parquet({quote(output)}) p
        JOIN v4_role_accounts r USING(user_key)""").fetchone()[0])
    if overlap:
        raise ValueError(f"T0 training file contains {overlap} pairs from reserved v4 accounts")
    return {"positive_pairs": rows, "positive_users": users, "reserved_pair_overlap": overlap}


def materialize_t0_training_from_lineage(
    con: duckdb.DuckDBPyConnection,
    stage: Path,
    destination: Path,
    split: Path = V1_SPLIT,
) -> dict[str, int]:
    """Safe bounded-memory fallback when the audited v3 T0 pair artifact fails."""
    temp_root = stage / "lineage_fallback"
    lineage_parts = temp_root / "lineage"
    raw_parts = temp_root / "raw"
    pair_parts = temp_root / "pairs"
    for path in (lineage_parts, raw_parts, pair_parts):
        path.mkdir(parents=True, exist_ok=False)
    con.execute("SET max_temp_directory_size='30GB'")
    bucket_n = TRAIN_BUCKETS
    input_sql = f"""SELECT l.user_key,l.app_id,l.review_key,l.positive,
             CAST(hash(l.user_key) % {bucket_n} AS INTEGER) AS bucket
        FROM read_parquet({quote(split/'train/review_lineage.parquet')}) l
        JOIN input_apps i USING(app_id)
        WHERE l.positive"""
    raw_sql = f"""SELECT r.user_key,r.app_id,r.review_key,r.positive,
             CAST(hash(r.user_key) % {bucket_n} AS INTEGER) AS bucket
        FROM valid_raw r
        JOIN input_apps i USING(app_id)
        WHERE r.positive AND r.timestamp_created IS NOT NULL AND r.timestamp_updated IS NOT NULL
          AND r.timestamp_created <= {T0} AND r.timestamp_updated <= {T0}"""
    con.execute(f"COPY ({input_sql}) TO {quote(lineage_parts)} (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY(bucket), ROW_GROUP_SIZE {ROW_GROUP_SIZE})")
    con.execute(f"COPY ({raw_sql}) TO {quote(raw_parts)} (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY(bucket), ROW_GROUP_SIZE {ROW_GROUP_SIZE})")
    event_rows = 0
    for bucket in range(bucket_n):
        lpath = lineage_parts / f"bucket={bucket}" / "*.parquet"
        rpath = raw_parts / f"bucket={bucket}" / "*.parquet"
        if not lpath.parent.is_dir() or not rpath.parent.is_dir():
            continue
        output = pair_parts / f"pairs_{bucket:02d}.parquet"
        con.execute(f"""COPY (
            SELECT DISTINCT l.user_key,l.app_id,true AS positive
            FROM read_parquet({quote(lpath)},hive_partitioning=true) l
            JOIN read_parquet({quote(rpath)},hive_partitioning=true) r
              USING(user_key,app_id,review_key,positive)
            ORDER BY l.user_key,l.app_id
        ) TO {quote(output)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})""")
        event_rows += int(pq.ParquetFile(output).metadata.num_rows)
    parts = sorted(pair_parts.glob("pairs_*.parquet"))
    if not parts:
        raise ValueError("No T0 positive pair could be derived from accepted lineage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (
        SELECT user_key,app_id,positive FROM read_parquet({quote(pair_parts/'pairs_*.parquet')})
        ANTI JOIN v4_role_accounts USING(user_key)
        ORDER BY user_key,app_id
    ) TO {quote(destination)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})""")
    rows = int(pq.ParquetFile(destination).metadata.num_rows)
    users = int(con.execute(f"SELECT count(DISTINCT user_key) FROM read_parquet({quote(destination)})").fetchone()[0])
    shutil.rmtree(temp_root)
    return {"positive_pairs": rows, "positive_users": users, "source_positive_pairs_before_v4_exclusion": event_rows}


def file_inventory(root: Path) -> dict[str, dict]:
    inventory: dict[str, dict] = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root).as_posix()
        pf = pq.ParquetFile(path)
        inventory[relative] = {
            "rows": int(pf.metadata.num_rows),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "schema": [{"name": field.name, "type": str(field.type)} for field in pf.schema_arrow],
        }
    return inventory


def build(output: Path = OUTPUT, protocol_path: Path = TEMPORAL_PROTOCOL_PATH) -> dict:
    """Freeze the data rules and create temporal artifacts only; no fitting or scoring."""
    protocol_path, protocol_sha = freeze_protocol(protocol_path)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite temporal v4 artifacts: {output}")
    v4_manifest, roles = verify_v4_split()
    old_ok, old_reasons, old_training_path = validate_old_temporal_training_artifact()
    release_info = verify_release_snapshot()
    source_manifest = json.loads(SOURCE_MANIFEST.read_text("utf-8"))
    v1_manifest_path = V1_SPLIT / "manifest.json"
    v1_manifest = json.loads(v1_manifest_path.read_text("utf-8"))
    if digest(v1_manifest_path) != source_manifest["split_artifacts"]["manifest_sha256"]:
        raise ValueError("split_v1 manifest differs from source_manifest.json")
    if digest(WORK) != v1_manifest.get("sources", {}).get("database_sha256"):
        raise ValueError("Kieran cache differs from the pinned split_v1 source fingerprint")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.building-{uuid.uuid4().hex[:10]}")
    stage.mkdir(parents=True, exist_ok=False)
    temp_dir = stage / "duckdb_temp"
    temp_dir.mkdir()
    connection = duckdb.connect(str(WORK), read_only=True)
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    connection.execute(f"SET threads={THREADS}")
    connection.execute(f"SET temp_directory={quote(temp_dir)}")
    connection.execute("SET max_temp_directory_size='30GB'")
    connection.execute("SET preserve_insertion_order=false")
    try:
        connection.execute(f"CREATE TEMP TABLE input_apps AS SELECT * FROM read_parquet({quote(INPUT_APPS)})")
        connection.execute(f"CREATE TEMP TABLE linux_candidates AS SELECT * FROM read_parquet({quote(LINUX_CANDIDATES)})")
        create_role_account_table(connection)
        log("Creating the T0 release-date mask over the complete Linux candidate array")
        availability_counts = create_release_availability(connection)
        log("Joining reserved account lineage to valid_raw review timestamps")
        event_counts = materialize_role_events(connection)
        cohort_summaries = {}
        for role in ROLE_NAMES:
            log(f"Preparing temporal {role} cohort; no model scores are computed")
            cohort_summaries[role] = create_temporal_cohort_tables(connection, role)

        catalog_dir = stage / "catalog"
        catalog_dir.mkdir()
        shutil.copy2(INPUT_APPS, catalog_dir / "input_apps.parquet")
        shutil.copy2(LINUX_CANDIDATES, catalog_dir / "linux_candidates.parquet")
        export_table(connection, "candidate_availability", catalog_dir / "candidate_availability.parquet", "app_id")
        # Save the protocol with the data artifact so temporal rules travel with the cohort files.
        shutil.copy2(protocol_path, stage / "protocol.json")

        training_dir = stage / "train"
        training_dir.mkdir()
        if old_ok and old_training_path is not None:
            training_source_kind = "reused_audited_temporal_v3_training_pairs_after_v4_account_exclusion"
            training_source_path = old_training_path
            training_source_sha256 = digest(old_training_path)
            log("Old temporal2022 receipts passed; filtering all v4 role accounts from its T0 positive pairs")
            training_counts = filter_reserved_training_pairs(
                connection, training_source_path, training_dir / "positive_catalog.parquet")
        else:
            training_source_kind = "derived_from_accepted_train_lineage_and_valid_raw_fallback"
            training_source_path = None
            training_source_sha256 = None
            log("Old temporal2022 receipt failed; deriving T0 positives from accepted train lineage")
            training_counts = materialize_t0_training_from_lineage(
                connection, stage, training_dir / "positive_catalog.parquet")
        training_counts["source_kind"] = training_source_kind
        training_counts["source_sha256"] = training_source_sha256
        training_counts["old_artifact_validation_passed"] = bool(old_ok)
        training_counts["old_artifact_validation_reasons"] = old_reasons

        # Validate the new training source has no account-level overlap with any temporal role.
        leaks = connection.execute(f"""SELECT count(*) FROM read_parquet({quote(training_dir/'positive_catalog.parquet')}) t
            JOIN v4_role_accounts r USING(user_key)""").fetchone()[0]
        if int(leaks):
            raise ValueError(f"Temporal training has {leaks} reserved v4 account pairs")
        training_counts["reserved_account_pair_overlap"] = int(leaks)

        for role in ROLE_NAMES:
            role_dir = stage / role
            role_dir.mkdir()
            prefix = f"temporal_{role}"
            export_table(connection, f"{prefix}_users", role_dir / "users.parquet", "user_key")
            export_table(connection, f"{prefix}_query", role_dir / "query.parquet", "user_key,app_id")
            export_table(connection, f"{prefix}_selected_targets", role_dir / "targets.parquet", "user_key,target_order")
            export_table(connection, f"{prefix}_exclusions", role_dir / "exclusions.parquet", "user_key,app_id")

        files = file_inventory(stage)
        manifest = {
            "version": VERSION,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": "Temporal cohort and T0 training-pair preparation only; no model fit or score was generated.",
            "protocol": {"path": TEMPORAL_PROTOCOL_PATH.relative_to(EDA).as_posix(), "sha256": protocol_sha},
            "sources": {
                "v4_split_manifest_sha256": digest(V4_SPLIT / "manifest.json"),
                "v1_split_manifest_sha256": digest(v1_manifest_path),
                "v1_reference_manifest_sha256": digest(REFERENCE_MANIFEST),
                "source_manifest_sha256": digest(SOURCE_MANIFEST),
                "accepted_train_lineage": {
                    "path": TRAIN_LINEAGE.relative_to(EDA).as_posix(),
                    "bytes": TRAIN_LINEAGE.stat().st_size,
                    "sha256": digest(TRAIN_LINEAGE),
                    "rows": pq.ParquetFile(TRAIN_LINEAGE).metadata.num_rows,
                },
                "kieran_cache": {
                    "path": WORK.relative_to(EDA).as_posix(),
                    "bytes": WORK.stat().st_size,
                    "sha256": v1_manifest["sources"]["database_sha256"],
                },
                "release_date_snapshot": release_info,
                "old_temporal_training_artifact": {
                    "validation_passed": bool(old_ok),
                    "reasons": old_reasons,
                    "training_pairs_path": str(old_training_path) if old_training_path else None,
                    "training_pairs_sha256": training_source_sha256,
                },
            },
            "historical_exposure": {
                "statement": "All three temporal role rosters were sampled from old split_v1 TRAIN accounts. Their interactions were available to v1-v3 training; this builder excludes them from the fresh T0 training pairs and does not claim a pristine historical independent test.",
                "forgemaster": "Forgemaster identity rows are absent locally, so no Kieran-to-Forgemaster cross-check was possible.",
            },
            "availability": availability_counts,
            "role_events": event_counts,
            "cohorts": cohort_summaries,
            "training": training_counts,
            "confirmation": {
                "target_files_materialized": True,
                "scored": False,
                "opened_by_builder": False,
                "lock_required_before_evaluation": True,
            },
            "limitations": TEMPORAL_PROTOCOL["limitations"],
            "validation": {
                "passed": True,
                "checks": {
                    "role_rosters_disjoint": True,
                    "t0_training_reserved_user_leaks": 0,
                    "temporal_target_query_overlaps": {role: cohort_summaries[role]["target_query_overlap"] for role in ROLE_NAMES},
                    "invalid_temporal_target_timestamps": {role: cohort_summaries[role]["bad_target_times"] for role in ROLE_NAMES},
                },
            },
            "runtime": {"python": platform.python_version(), "duckdb": duckdb.__version__, "memory_limit": MEMORY_LIMIT, "threads": THREADS},
            "files": files,
        }
        write_json(stage / "manifest.json", manifest)
        connection.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        stage.rename(output)
        log(f"Published temporal v4 cohort artifacts at {output}; no model fit or scoring was run")
        return manifest
    except Exception:
        connection.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--protocol", type=Path, default=TEMPORAL_PROTOCOL_PATH)
    args = parser.parse_args()
    build(args.output, args.protocol)


if __name__ == "__main__":
    main()
