"""Materialize the confirmed Kieran -> frank split without fitting a model.

Large and account-level artifacts remain under ignored data/processed/modeling.
The audited EDA database is read-only; all working tables are connection-local.
"""

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
from summarize_kieran_frank import verify_kieran


VERSION = "kieran-frank-split-v1"
DEFAULT_OUTPUT = EDA / "data/processed/modeling/split_v1"
WORK = EDA / "data/processed/kieran_original/eda_work.duckdb"
DEFAULT_SEED = 20261107
HOLDOUT_NAMESPACE = "steam-hidden-positive-linux-v1"
VALIDATION_NAMESPACE = "steam-eda-validation-v1"


def uniform_index(user_key, n, seed, split):
    """A reproducible unbiased index into sorted candidate App IDs.

    SHA256 counter-mode rejection sampling avoids modulo bias. Each account and
    split has its own deterministic stream, independent of input row order.
    """
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


def create_holdout(con, history, prefix, seed, min_total=5, min_positive=1):
    """history has unique user_key/app_id pairs, positive and linux columns.

    Materialize TEMP {prefix}_cohort/targets/query/exclusions. Thresholds count
    games AFTER removal of exactly one positive Linux target.
    """
    for name in (history, prefix):
        if not name.replace("_", "").isalnum() or not name[0].isalpha():
            raise ValueError("Unsafe SQL identifier")
    if min_total < 1 or min_positive < 1 or min_positive > min_total:
        raise ValueError("Invalid minimum query counts")
    rows = con.execute(f"""SELECT user_key,count(*)-1 n_total_inputs,
        count(*) FILTER(WHERE positive)-1 n_positive_inputs,
        list(app_id ORDER BY app_id) FILTER(WHERE positive AND linux) target_apps
        FROM {history} GROUP BY user_key
        HAVING count(*)>={int(min_total)+1}
            AND count(*) FILTER(WHERE positive)>={int(min_positive)+1}
            AND count(*) FILTER(WHERE positive AND linux)>0
        ORDER BY user_key""").fetchall()
    users, targets = [], []
    for user_key, n_total, n_positive, apps in rows:
        target = apps[uniform_index(user_key, len(apps), seed, prefix)]
        users.append((user_key, n_total, n_positive, len(apps)))
        targets.append((user_key, target, True, True))
    con.execute(f"""CREATE TEMP TABLE {prefix}_cohort(
        user_key UBIGINT,n_total_inputs BIGINT,n_positive_inputs BIGINT,
        n_eligible_targets BIGINT)""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_targets(
        user_key UBIGINT,app_id BIGINT,positive BOOLEAN,linux BOOLEAN)""")
    for name, values, schema in [
        (f"{prefix}_cohort", users,
         pa.schema([("user_key", pa.uint64()), ("n_total_inputs", pa.int64()),
                    ("n_positive_inputs", pa.int64()), ("n_eligible_targets", pa.int64())])),
        (f"{prefix}_targets", targets,
         pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64()),
                    ("positive", pa.bool_()), ("linux", pa.bool_())])),
    ]:
        data = pa.Table.from_arrays(
            [pa.array([row[i] for row in values], type=field.type)
             for i, field in enumerate(schema)], schema=schema)
        con.register("_holdout_batch", data)
        con.execute(f"INSERT INTO {name} SELECT * FROM _holdout_batch")
        con.unregister("_holdout_batch")
    con.execute(f"""CREATE TEMP TABLE {prefix}_query AS
        SELECT h.* FROM {history} h SEMI JOIN {prefix}_cohort USING(user_key)
        ANTI JOIN {prefix}_targets USING(user_key,app_id)""")
    con.execute(f"""CREATE TEMP TABLE {prefix}_exclusions AS
        SELECT user_key,app_id FROM {prefix}_query""")


def validate_holdout(con, prefix, min_total=5, min_positive=1):
    checks = one(con, f"""SELECT
        (SELECT count(*) FROM {prefix}_targets t JOIN {prefix}_query q
            USING(user_key,app_id)) target_query_overlap,
        (SELECT count(*) FROM {prefix}_targets t JOIN {prefix}_exclusions x
            USING(user_key,app_id)) masked_targets,
        (SELECT count(*) FROM {prefix}_targets WHERE NOT positive OR NOT linux
            OR positive IS NULL OR linux IS NULL) invalid_targets,
        (SELECT count(*) FROM (SELECT user_key,count(*) n FROM {prefix}_targets
            GROUP BY user_key HAVING n<>1)) non_single_targets,
        (SELECT count(*) FROM {prefix}_cohort WHERE n_total_inputs<{int(min_total)}
            OR n_positive_inputs<{int(min_positive)}) ineligible_users,
        (SELECT count(*) FROM (SELECT u.user_key FROM {prefix}_cohort u
            LEFT JOIN {prefix}_query q USING(user_key)
            GROUP BY u.user_key,u.n_total_inputs,u.n_positive_inputs
            HAVING count(q.app_id)<>u.n_total_inputs OR
            count(q.app_id) FILTER(WHERE q.positive)<>u.n_positive_inputs)) bad_query_counts,
        (SELECT count(*) FROM ((SELECT user_key,app_id FROM {prefix}_query
            EXCEPT SELECT * FROM {prefix}_exclusions) UNION ALL
            (SELECT * FROM {prefix}_exclusions EXCEPT
            SELECT user_key,app_id FROM {prefix}_query))) mask_differences""")
    if any(checks.values()):
        raise ValueError(f"Holdout validation failed: {prefix}: {checks}")
    return checks


def write_parquet(con, sql, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO {quote(path)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")
    log(f"Wrote {path.parent.name}/{path.name}: {pq.ParquetFile(path).metadata.num_rows:,} rows")


def prepare_histories(con):
    con.execute("""CREATE TEMP TABLE _ambiguous_frank_pairs AS
        SELECT DISTINCT f.user_key,f.app_id FROM frank_raw f
        JOIN _frozen_quarantine USING(review_key)""")
    con.execute("""CREATE TEMP TABLE test_hist AS
        SELECT f.user_key,f.app_id,f.positive,c.linux
        FROM frank f SEMI JOIN test_users USING(user_key)
        JOIN catalog c USING(app_id) ANTI JOIN _ambiguous_frank_pairs USING(user_key,app_id)
        WHERE c.reviewed""")
    con.execute("""CREATE TEMP TABLE validation_hist AS
        SELECT c.user_key,c.app_id,c.positive,a.linux
        FROM clean c JOIN user_stats u USING(user_key) JOIN catalog a USING(app_id)
        WHERE NOT u.is_test AND u.is_validation AND a.reviewed""")


def source_checks(con):
    differences = one(con, """SELECT
        (SELECT count(*) FROM ((SELECT * FROM catalog EXCEPT SELECT * FROM _frozen_catalog)
            UNION ALL (SELECT * FROM _frozen_catalog EXCEPT SELECT * FROM catalog))) catalog_differences,
        (SELECT count(*) FROM ((SELECT * FROM frank_raw EXCEPT ALL SELECT * FROM _pinned_frank_raw)
            UNION ALL (SELECT * FROM _pinned_frank_raw EXCEPT ALL SELECT * FROM frank_raw))) frank_differences,
        (SELECT count(*) FROM user_stats u LEFT JOIN test_users t USING(user_key)
            WHERE u.is_test IS DISTINCT FROM (t.user_key IS NOT NULL)) reservation_flag_errors,
        (SELECT count(*) FROM ((SELECT user_key FROM test_users EXCEPT
            SELECT f.user_key FROM frank f JOIN catalog c USING(app_id) WHERE c.reviewed
            GROUP BY f.user_key HAVING count(*)>=2 AND count(*) FILTER(WHERE f.positive AND c.linux)>0)
            UNION ALL (SELECT f.user_key FROM frank f JOIN catalog c USING(app_id) WHERE c.reviewed
            GROUP BY f.user_key HAVING count(*)>=2 AND count(*) FILTER(WHERE f.positive AND c.linux)>0
            EXCEPT SELECT user_key FROM test_users))) reservation_set_differences""")
    if any(differences.values()):
        raise ValueError(f"Audited cache differs from pinned source/reference: {differences}")
    expected = {
        "clean_pairs": 113195034, "train_pairs": 98857701,
        "train_positive_all_apps": 84791317, "validation_pairs": 10977308,
        "reserved_accounts": 87573, "input_apps": 37610, "linux_candidates": 6630,
        "catalog_train_positive": 78315268,
    }
    measured = one(con, """SELECT
        (SELECT count(*) FROM clean) clean_pairs,
        (SELECT sum(n_reviews) FROM user_stats WHERE NOT is_test AND NOT is_validation) train_pairs,
        (SELECT sum(n_positive) FROM user_stats WHERE NOT is_test AND NOT is_validation) train_positive_all_apps,
        (SELECT sum(n_reviews) FROM user_stats WHERE NOT is_test AND is_validation) validation_pairs,
        (SELECT count(*) FROM test_users) reserved_accounts,
        (SELECT count(*) FROM catalog WHERE reviewed) input_apps,
        (SELECT count(*) FROM catalog WHERE reviewed AND linux) linux_candidates,
        (SELECT sum(a.final_train_positive) FROM app_stats a JOIN catalog c USING(app_id)
            WHERE c.reviewed) catalog_train_positive""")
    if measured != expected:
        raise ValueError(f"Source differs from audited counts: {measured}")
    return measured


def export_all(con, root):
    for split, condition in [
        ("train", "NOT u.is_test AND NOT u.is_validation"),
        ("validation", "NOT u.is_test AND u.is_validation"),
    ]:
        write_parquet(con, f"SELECT c.* FROM clean c JOIN user_stats u USING(user_key) WHERE {condition}",
                      root / split / "interactions.parquet")
    # Derive the actual model rows from the stored train partition, never from
    # validation or test histories. Keep singleton support in this projection.
    write_parquet(con, f"""SELECT t.* FROM read_parquet({quote(root/'train/interactions.parquet')}) t
        JOIN catalog c USING(app_id) WHERE c.reviewed AND t.positive""",
        root / "train/positive_catalog.parquet")
    write_parquet(con, "SELECT user_key FROM test_users ORDER BY user_key",
                  root / "reserved_test_accounts.parquet")
    write_parquet(con, "SELECT app_id,linux FROM catalog WHERE reviewed ORDER BY app_id",
                  root / "catalog/input_apps.parquet")
    write_parquet(con, "SELECT app_id FROM catalog WHERE reviewed AND linux ORDER BY app_id",
                  root / "catalog/linux_candidates.parquet")
    for split in ("validation", "test"):
        for name in ("users", "targets", "query", "exclusions"):
            order = "user_key" if name == "users" else "user_key,app_id"
            table = f"{split}_cohort" if name == "users" else f"{split}_{name}"
            write_parquet(con, f"SELECT * FROM {table} ORDER BY {order}",
                          root / split / f"{name}.parquet")
    # A provenance sidecar preserves every accepted raw review occurrence.
    # Exact repeated rows may repeat an ID; interactions stay deduplicated.
    con.execute("""CREATE TEMP TABLE _bad_original_ids AS SELECT review_key FROM review_keys
        WHERE min_user<>max_user OR min_app<>max_app""")
    con.execute("""CREATE TEMP TABLE _bad_label_pairs AS SELECT user_key,app_id FROM pair_groups
        WHERE min_label<>max_label""")
    for split, condition in [
        ("train", "NOT u.is_test AND NOT u.is_validation"),
        ("validation", "NOT u.is_test AND u.is_validation"),
    ]:
        write_parquet(con, f"""SELECT r.user_key,r.app_id,r.review_key,r.positive
            FROM valid_raw r ANTI JOIN _bad_original_ids USING(review_key)
            ANTI JOIN cross_review_conflicts USING(review_key)
            ANTI JOIN _bad_label_pairs USING(user_key,app_id)
            JOIN user_stats u USING(user_key) WHERE {condition}""",
            root / split / "review_lineage.parquet")
    write_parquet(con, """SELECT f.user_key,f.app_id,f.review_key,f.positive
        FROM frank_raw f SEMI JOIN test_hist USING(user_key,app_id)
        SEMI JOIN test_users USING(user_key)
        SEMI JOIN test_cohort USING(user_key)""", root / "test/review_lineage.parquet")


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        result[path.relative_to(root).as_posix()] = {
            "rows": parquet.metadata.num_rows, "bytes": path.stat().st_size,
            "sha256": digest(path),
            "schema": [{"name": f.name, "type": str(f.type)} for f in parquet.schema_arrow],
        }
    return result


def build(output, seed, memory="4GB", threads=2):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing split directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.building-{uuid.uuid4().hex[:10]}"
    stage.mkdir()
    manifest = {"version": VERSION, "status": "building", "seed": seed,
                "started_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = stage / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    save()
    try:
        metrics_path = EDA / "reports/kieran_original_frank_eda_metrics.json"
        metrics = json.loads(metrics_path.read_text("utf-8"))
        log("Verifying canonical shard, ingestion and frozen source fingerprints")
        verified = verify_kieran(metrics)
        ref = EDA / "data/reference"
        reference = json.loads((ref / "manifest.json").read_text("utf-8"))
        for name, info in reference["files"].items():
            if digest(ref / name) != info["sha256"]:
                raise ValueError(f"Frozen reference changed: {name}")
        frank_path = EDA / "data/processed/frank/reviews_identity_columns.parquet"
        if digest(frank_path) != metrics["frank_provenance"]["projection_sha256"]:
            raise ValueError("Frozen frank review projection changed")
        manifest["sources"] = {
            "verification": verified, "eda_metrics_sha256": digest(metrics_path),
            "database_sha256": digest(WORK), "database_opened_read_only": True,
            "reference_manifest_sha256": digest(ref / "manifest.json"),
            "frank_commit": metrics["frank_provenance"]["commit"],
            "frank_projection_sha256": digest(frank_path),
            "canonical_shards": metrics["canonical_inputs"],
            "cache_policy": "Use materialized audited tables only; legacy-path raw/frank_input views are not evaluated. Current canonical/projection fingerprints verified; cache database fingerprint pinned for this split.",
        }
        manifest["runtime"] = {"python": platform.python_version(),
            "duckdb": duckdb.__version__, "pyarrow": pa.__version__,
            "threads": threads, "memory_limit": memory}
        manifest["rules"] = {
            "validation_account_rule": f"first 15 MD5 hex digits of {VALIDATION_NAMESPACE}:<exact SteamID> modulo 10 == 0; test reservation takes priority",
            "holdout_namespace": HOLDOUT_NAMESPACE,
            "target_selection": "SHA256 counter-mode rejection sampling into ascending eligible positive Linux App IDs; payload namespace:seed:split:exact-user-key:counter",
            "minimum_total_inputs_after_holdout": 5,
            "minimum_positive_inputs_after_holdout": 1,
            "query_aggregation": "equal_weight_mean_of_positive_inputs",
            "training_scope": "positive_catalog.parquet: fixed 37610-app catalog; positive singleton support retained",
            "interaction_archives": "interactions.parquet retains all clean apps and both labels; not direct model input",
            "quarantine_policy": "keep_existing",
            "recommendation_fallback": "none; up to 10 strictly positive personalized scores; empty slots stay empty",
            "exclusions": "all remaining query games, both labels; hidden target absent from query and mask",
            "no_temporal_restriction": True,
            "review_lineage": "all accepted raw review occurrences; exact repeats may repeat review_key",
            "models_trained": False,
        }
        save()
        with tempfile.TemporaryDirectory(prefix="split_v1_") as temp:
            con = duckdb.connect(str(WORK), read_only=True)
            try:
                con.execute(f"SET threads={int(threads)}")
                con.execute(f"SET memory_limit={quote(memory)}")
                con.execute("SET preserve_insertion_order=false")
                con.execute(f"SET temp_directory={quote(temp)}")
                con.execute(f"CREATE TEMP TABLE _frozen_catalog AS SELECT * FROM read_parquet({quote(ref/'catalog.parquet')})")
                con.execute(f"""CREATE TEMP TABLE _pinned_frank_raw AS SELECT
                    cast(author_steamid AS UBIGINT) user_key,cast(appid AS BIGINT) app_id,
                    cast(recommendationid AS UBIGINT) review_key,voted_up positive,
                    timestamp_created,timestamp_updated FROM read_parquet({quote(frank_path)})""")
                manifest["audited_source_counts"] = source_checks(con)
                con.execute(f"CREATE TEMP TABLE _frozen_quarantine AS SELECT * FROM read_parquet({quote(ref/'review_app_quarantine.parquet')})")
                prepare_histories(con)
                for split in ("validation", "test"):
                    log(f"Selecting and freezing {split} targets, seed={seed}")
                    create_holdout(con, f"{split}_hist", split, seed)
                    validate_holdout(con, split)
                export_all(con, stage)
            finally:
                con.close()
        manifest["files"] = inventory(stage)
        save()
        # Import here so tests can exercise the sampler without reading sources.
        from verify_modeling_splits import verify_split
        manifest["validation"] = verify_split(stage, verify_hashes=True)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["builder_sha256"] = digest(__file__)
        manifest["verifier_sha256"] = digest(Path(__file__).with_name("verify_modeling_splits.py"))
        save()
        stage.rename(output)
        log(f"Published verified split: {output}")
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--memory", default="4GB")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    manifest = build(args.output, args.seed, args.memory, args.threads)
    print(json.dumps({"version": manifest["version"], "status": manifest["status"],
                      "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
