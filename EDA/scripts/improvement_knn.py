"""Build train-only KNN diagnostics and bounded/shrunk v2 indexes.

The only fitted input is the pinned split_v1 training positive catalog. Holdout
files are opened only by ``--diagnose`` to explain the already saved v1 result;
the diagnostics do not evaluate or select any v2 model.
"""

import argparse
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
import scipy
from scipy import sparse

from build_item_knn import DEFAULT_SPLIT, read_sources
from eda_large_review_frank import EDA, digest, log, quote
from item_knn import DEFAULT_INDEX, ItemKnnCatalog, _build_user_item_csr


DEFAULT_OUTPUT = EDA / "data/processed/modeling/improvement_v2/knn"
DEFAULT_REPORT = EDA / "reports/item_knn_failure_analysis_v2.json"
DEFAULT_BASELINE = EDA / "data/processed/modeling/baseline_v1"
KS = (100, 200, 500)
SHRINKAGES = (0, 10, 50)
BLOCK_SIZE = 64
SORT_BATCH_SIZE = 131072
DUCKDB_MEMORY_LIMIT = "3GB"
DUCKDB_THREADS = 4
MODEL_VERSION = "binary-item-knn-improvement-v2"


def rank_neighbors(candidate_col, cooc_cols, cooc_counts, app_ids, support,
                   neighbors, shrinkage=0):
    """Rank all raw neighbors by shrunk binary cosine, returning CSR order.

    Counts and supports are promoted to int64 before any score arithmetic.
    Ranking uses similarity descending and App ID ascending; returned entries
    are then sorted by input column, matching the saved CSR storage convention.
    """
    if isinstance(neighbors, bool) or int(neighbors) != neighbors or int(neighbors) < 0:
        raise ValueError("neighbors must be a nonnegative integer")
    if isinstance(shrinkage, bool) or int(shrinkage) != shrinkage or int(shrinkage) < 0:
        raise ValueError("shrinkage must be a nonnegative integer")
    neighbors, shrinkage = int(neighbors), int(shrinkage)
    if neighbors == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    cols = np.asarray(cooc_cols, dtype=np.int64)
    counts = np.asarray(cooc_counts, dtype=np.int64)
    app_ids = np.asarray(app_ids, dtype=np.int64)
    support = np.asarray(support, dtype=np.int64)
    candidate_support = int(support[candidate_col])
    if candidate_support <= 0 or not len(cols):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    if np.any(cols < 0) or np.any(cols >= len(app_ids)):
        raise ValueError("co-occurrence column outside the fixed input catalog")
    keep = ((cols != candidate_col) & (counts > 0) & (support[cols] > 0))
    cols = cols[keep]
    counts = counts[keep]
    if not len(cols):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    cosine = counts.astype(np.float64) / np.sqrt(
        float(candidate_support) * support[cols].astype(np.float64))
    if shrinkage:
        cosine *= counts.astype(np.float64) / (counts.astype(np.float64) + shrinkage)
    ranked = np.lexsort((app_ids[cols], -cosine))[:neighbors]
    selected_cols = cols[ranked].astype(np.int32, copy=False)
    selected_values = cosine[ranked]
    storage = np.argsort(selected_cols, kind="stable")
    return selected_cols[storage], selected_values[storage]


def diagnostic_category(target_support, train_supported_positive_inputs,
                        raw_connection_count, retained_connection_count,
                        target_rank_v1):
    """Return one mutually exclusive K0 failure category for a user."""
    if target_support == 0:
        return "target_support0"
    if train_supported_positive_inputs == 0:
        return "all_positive_inputs_support0"
    if raw_connection_count == 0:
        return "raw_connection0"
    if retained_connection_count == 0:
        return "pruned"
    if target_rank_v1 > 0:
        return "hit"
    return "rank_failure"


def _sort_train_pairs(source_path, sorted_path, temp_dir):
    """External-sort the pinned positives with bounded DuckDB resources."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        connection.execute(f"SET threads={DUCKDB_THREADS}")
        connection.execute(f"SET temp_directory={quote(temp_dir)}")
        connection.execute("SET max_temp_directory_size='20GB'")
        connection.execute(f"""COPY (
            SELECT user_key,app_id
            FROM read_parquet({quote(source_path)})
            WHERE positive
            ORDER BY user_key,app_id
        ) TO {quote(sorted_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)""")
    finally:
        connection.close()


def _write_catalogs(folder, app_ids, linux, candidate_app_ids, support):
    pq.write_table(pa.table({
        "row_id": pa.array(np.arange(len(app_ids), dtype=np.int32)),
        "app_id": pa.array(app_ids, type=pa.int64()),
        "linux": pa.array(linux, type=pa.bool_()),
        "positive_user_support": pa.array(support, type=pa.int64()),
    }), folder / "game_rows.parquet", compression="zstd")
    candidate_rows = np.searchsorted(app_ids, candidate_app_ids).astype(np.int32)
    pq.write_table(pa.table({
        "row_id": pa.array(np.arange(len(candidate_app_ids), dtype=np.int32)),
        "app_id": pa.array(candidate_app_ids, type=pa.int64()),
        "input_row_id": pa.array(candidate_rows, type=pa.int32()),
    }), folder / "candidate_rows.parquet", compression="zstd")


def _count_singleton_users(indptr, chunk_size=1_000_000):
    count = 0
    row_count = len(indptr) - 1
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        count += int(np.count_nonzero(
            indptr[start + 1:stop + 1] - indptr[start:stop] == 1))
    return count


def _inventory(folder):
    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        info = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            info["rows"] = parquet.metadata.num_rows
            info["schema"] = [{"name": field.name, "type": str(field.type)}
                              for field in parquet.schema_arrow]
        files[path.name] = info
    return files


def _code_hashes():
    scripts = Path(__file__).parent
    return {
        "builder": digest(Path(__file__)),
        "frozen_v1_builder_helpers": digest(scripts / "build_item_knn.py"),
        "item_knn_catalog_and_retrieval": digest(scripts / "item_knn.py"),
    }


def _runtime():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pyarrow": pa.__version__,
            "duckdb": duckdb.__version__}


def _refresh_build_manifest(output_root, report_path):
    """Attach diagnostics without rewriting already published model manifests."""
    build_manifest_path = Path(output_root) / "manifest.json"
    if not build_manifest_path.is_file():
        return
    build_manifest = json.loads(build_manifest_path.read_text("utf-8"))
    if build_manifest.get("version") != MODEL_VERSION or build_manifest.get("status") != "complete":
        raise ValueError("Cannot attach diagnostics to an incomplete v2 model build")
    report = json.loads(Path(report_path).read_text("utf-8"))
    current_code = _code_hashes()
    model_hashes = {}
    for k in KS:
        for shrinkage in SHRINKAGES:
            name = _variant_name(k, shrinkage)
            model_manifest_path = Path(output_root) / name / "manifest.json"
            model_hashes[name] = digest(model_manifest_path)
    build_manifest["models"] = model_hashes
    build_manifest["diagnostics_code_sha256"] = current_code
    build_manifest["support_summary"] = report["training_support_summary"]
    build_manifest["resource_limits"] = {
        "duckdb_memory_limit": DUCKDB_MEMORY_LIMIT,
        "duckdb_threads": DUCKDB_THREADS,
        "candidate_block_size": build_manifest.get("resource_limits", {}).get(
            "candidate_block_size", BLOCK_SIZE),
    }
    build_manifest["diagnostics"] = {
        "report_file": Path(report_path).relative_to(EDA).as_posix()
        if Path(report_path).is_relative_to(EDA) else str(Path(report_path)),
        "report_sha256": digest(report_path),
        "user_diagnostic_files": {
            name: info["user_diagnostics_file"]
            for name, info in report["cohorts"].items()
        },
        "frank_test_used_for_model_selection": False,
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _variant_name(k, shrinkage):
    return f"k{k}_l{shrinkage}"


def build_models(output_root=DEFAULT_OUTPUT, split=DEFAULT_SPLIT,
                 block_size=BLOCK_SIZE):
    """Build exact raw candidate co-occurrences and nine reselected indexes."""
    if isinstance(block_size, bool) or int(block_size) != block_size or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite v2 KNN artifacts: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.building-{uuid.uuid4().hex[:10]}"
    stage.mkdir()
    work = stage / "_work"
    work.mkdir()
    sorted_path = work / "_train_positive_sorted.parquet"
    temp_dir = work / "_duckdb_temp"
    raw_path = stage / "raw_cooccurrence.parquet"
    code_hashes = _code_hashes()
    manifest_path = stage / "manifest.json"
    manifest = {"version": MODEL_VERSION, "status": "building",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "fit_scope": "pinned split_v1 train positive catalog only",
                "ranking_metrics_computed": False}

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

    save_manifest()
    try:
        log("Verifying pinned train positives and fixed input/candidate catalogs")
        paths, provenance, app_ids, linux, candidate_app_ids, expected_counts = read_sources(split)
        if expected_counts["train_positive_pairs"] >= np.iinfo(np.int32).max:
            raise ValueError("Positive user count exceeds safe sparse co-occurrence integer range")
        manifest.update({"sources": provenance, "source_counts": expected_counts})
        save_manifest()

        log("Externally sorting the pinned positives with a 3GB/four-thread DuckDB limit")
        _sort_train_pairs(paths["train_positive_catalog"], sorted_path, temp_dir)
        sorted_pf = pq.ParquetFile(sorted_path)
        if sorted_pf.metadata.num_rows != expected_counts["train_positive_pairs"]:
            raise ValueError("Sorted positive row count differs from the pinned split")
        log("Streaming sorted positives into the complete binary sparse user-item matrix")
        user_items, support, user_count = _build_user_item_csr(
            sorted_pf.iter_batches(batch_size=SORT_BATCH_SIZE, columns=["user_key", "app_id"]),
            app_ids, expected_pairs=expected_counts["train_positive_pairs"],
            expected_users=expected_counts["train_positive_users"])
        sorted_pf.close()
        if user_count > np.iinfo(np.int32).max:
            raise ValueError("Positive user count exceeds safe sparse co-occurrence integer range")
        singleton_users = _count_singleton_users(user_items.indptr)
        if singleton_users != expected_counts["positive_singleton_users_reported"]:
            raise ValueError("Measured positive singleton users differ from the pinned split count")
        log(f"Built binary CSR for {user_count:,} train-positive users and "
            f"{user_items.nnz:,} pairs; singleton users remain included")
        user_items = user_items.tocsc()
        sorted_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

        candidate_columns = np.searchsorted(app_ids, candidate_app_ids).astype(np.int32)
        outputs = {}
        for k in KS:
            for shrinkage in SHRINKAGES:
                capacity = len(candidate_app_ids) * min(k, len(app_ids) - 1)
                outputs[(k, shrinkage)] = {
                    "indices": np.empty(capacity, dtype=np.int32),
                    "data": np.empty(capacity, dtype=np.float64),
                    "indptr": np.zeros(len(candidate_app_ids) + 1, dtype=np.int32),
                    "nnz": 0,
                }

        raw_schema = pa.schema([
            ("candidate_app_id", pa.int64()),
            ("neighbor_app_id", pa.int64()),
            ("cooccurrence_count", pa.int64()),
            ("raw_cosine", pa.float64()),
            ("raw_cosine_rank", pa.int32()),
        ])
        raw_edges = 0
        with pq.ParquetWriter(raw_path, raw_schema, compression="zstd") as raw_writer:
            for block_start in range(0, len(candidate_app_ids), block_size):
                block_stop = min(block_start + block_size, len(candidate_app_ids))
                candidate_by_user = user_items[:, candidate_columns[block_start:block_stop]].T.tocsr()
                candidate_by_user = candidate_by_user.astype(np.int32, copy=False)
                cooccurrence = (candidate_by_user @ user_items).tocsr()
                cooccurrence.sum_duplicates()
                cooccurrence.sort_indices()
                if cooccurrence.nnz and int(cooccurrence.data.max()) > user_count:
                    raise ValueError("Co-occurrence count exceeds the number of train-positive users")

                block_candidate_ids = []
                block_neighbor_ids = []
                block_counts = []
                block_cosines = []
                block_ranks = []
                for local_row in range(block_stop - block_start):
                    global_row = block_start + local_row
                    row_start, row_end = cooccurrence.indptr[local_row:local_row + 2]
                    cols = cooccurrence.indices[row_start:row_end].astype(np.int64, copy=False)
                    counts = cooccurrence.data[row_start:row_end].astype(np.int64, copy=False)
                    keep = ((cols != candidate_columns[global_row]) & (counts > 0)
                            & (support[cols] > 0))
                    cols = cols[keep]
                    counts = counts[keep]
                    if len(cols):
                        raw_cosines = counts.astype(np.float64) / np.sqrt(
                            float(support[candidate_columns[global_row]])
                            * support[cols].astype(np.float64))
                        ranking = np.lexsort((app_ids[cols], -raw_cosines))
                        raw_ranks = np.empty(len(cols), dtype=np.int32)
                        raw_ranks[ranking] = np.arange(1, len(cols) + 1, dtype=np.int32)
                        block_candidate_ids.append(np.full(len(cols), candidate_app_ids[global_row], dtype=np.int64))
                        block_neighbor_ids.append(app_ids[cols].astype(np.int64, copy=False))
                        block_counts.append(counts.astype(np.int64, copy=False))
                        block_cosines.append(raw_cosines)
                        block_ranks.append(raw_ranks)

                    for (k, shrinkage), out in outputs.items():
                        selected_cols, selected_values = rank_neighbors(
                            int(candidate_columns[global_row]),
                            cooccurrence.indices[row_start:row_end],
                            cooccurrence.data[row_start:row_end],
                            app_ids, support, k, shrinkage)
                        count = len(selected_cols)
                        end = out["nnz"] + count
                        if end > len(out["indices"]):
                            raise ValueError(f"Top-{k} output exceeded allocated capacity")
                        out["indices"][out["nnz"]:end] = selected_cols
                        out["data"][out["nnz"]:end] = selected_values
                        out["nnz"] = end
                        out["indptr"][global_row + 1] = end

                if block_candidate_ids:
                    n_edges = sum(len(row) for row in block_counts)
                    raw_edges += n_edges
                    raw_writer.write_table(pa.table({
                        "candidate_app_id": pa.array(np.concatenate(block_candidate_ids), type=pa.int64()),
                        "neighbor_app_id": pa.array(np.concatenate(block_neighbor_ids), type=pa.int64()),
                        "cooccurrence_count": pa.array(np.concatenate(block_counts), type=pa.int64()),
                        "raw_cosine": pa.array(np.concatenate(block_cosines), type=pa.float64()),
                        "raw_cosine_rank": pa.array(np.concatenate(block_ranks), type=pa.int32()),
                    }))
                log(f"Processed exact raw co-occurrence for Linux candidate rows "
                    f"{block_stop:,}/{len(candidate_app_ids):,}; saved {raw_edges:,} edges")
                del candidate_by_user, cooccurrence

        if pq.ParquetFile(raw_path).metadata.num_rows != raw_edges:
            raise ValueError("Raw co-occurrence parquet row count differs from streamed edges")
        _write_catalogs(stage, app_ids, linux, candidate_app_ids, support)

        v1 = ItemKnnCatalog.load(DEFAULT_INDEX, verify_hashes=True)
        v1_manifest = json.loads((DEFAULT_INDEX / "manifest.json").read_text("utf-8"))
        if v1_manifest["sources"]["sources"]["train_positive_catalog"]["sha256"] != provenance["sources"]["train_positive_catalog"]["sha256"]:
            raise ValueError("Frozen v1 index was built from a different pinned train-positive catalog")
        base_out = outputs[(100, 0)]
        exact = sparse.csr_matrix((base_out["data"][:base_out["nnz"]],
                                   base_out["indices"][:base_out["nnz"]],
                                   base_out["indptr"]),
                                  shape=(len(candidate_app_ids), len(app_ids)))
        exact.sort_indices()
        checks = {
            "shape_equal": exact.shape == v1.similarities.shape,
            "indptr_exact": np.array_equal(exact.indptr, v1.similarities.indptr),
            "indices_exact": np.array_equal(exact.indices, v1.similarities.indices),
            "float64_data_exact": exact.data.dtype == np.dtype("float64")
            and np.array_equal(exact.data, v1.similarities.data),
        }
        if not all(checks.values()):
            max_difference = None
            if exact.data.shape == v1.similarities.data.shape and len(exact.data):
                max_difference = float(np.max(np.abs(exact.data - v1.similarities.data)))
            raise ValueError(f"K100/lambda0 failed exact frozen v1 equivalence: {checks}; "
                             f"max_float64_difference={max_difference}")
        if not (np.array_equal(v1.app_ids, app_ids)
                and np.array_equal(v1.candidate_app_ids, candidate_app_ids)
                and np.array_equal(v1.support, support)):
            raise ValueError("Pinned training mapping/support differs from the frozen v1 index")

        variant_manifests = {}
        for (k, shrinkage), out in outputs.items():
            folder = stage / _variant_name(k, shrinkage)
            folder.mkdir()
            matrix = sparse.csr_matrix((out["data"][:out["nnz"]],
                                        out["indices"][:out["nnz"]], out["indptr"]),
                                       shape=(len(candidate_app_ids), len(app_ids)))
            matrix.sort_indices()
            sparse.save_npz(folder / "similarities_csr.npz", matrix, compressed=True)
            _write_catalogs(folder, app_ids, linux, candidate_app_ids, support)
            parameters = {
                "version": MODEL_VERSION,
                "neighbors": k,
                "minimum_cooccurrence": 1,
                "shrinkage_lambda": shrinkage,
                "shrinkage_formula": "cosine * cooccurrence / (cooccurrence + lambda)",
                "binary_positive_interactions": True,
                "support_counts_include_positive_singleton_users": True,
                "similarity": "cooccurrence/sqrt(full_train_positive_support_i*full_train_positive_support_j), diagonal zero",
                "neighbor_selection": "reselect from every raw positive co-occurrence for this K/lambda",
                "neighbor_order": "shrunk similarity descending, then App ID ascending",
                "query_score": "same as frozen v1; mean over every remaining positive query input",
                "fit_scope": "pinned split_v1 train positive catalog only",
            }
            (folder / "parameters.json").write_text(
                json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            item_manifest = {
                "version": MODEL_VERSION, "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "fit_scope": "pinned split_v1 train positive catalog only",
                "sources": provenance, "parameters": parameters,
                "index": {"format": "SciPy CSR float64 NPZ",
                          "shape": list(matrix.shape), "nnz": int(matrix.nnz)},
                "support_summary": {
                    "input_games_with_positive_support": int(np.count_nonzero(support)),
                    "input_games_with_zero_support": int(np.count_nonzero(support == 0)),
                    "minimum_positive_game_support": int(support[support > 0].min())
                    if np.any(support > 0) else None,
                    "maximum_positive_game_support": int(support.max()) if len(support) else 0,
                    "singleton_positive_users_measured": singleton_users,
                    "singleton_positive_users_reported_by_split":
                        expected_counts["positive_singleton_users_reported"],
                },
                "raw_cooccurrence_sha256": digest(raw_path),
                "code_sha256": code_hashes, "runtime": _runtime(),
                "ranking_metrics_computed": False,
                "files": _inventory(folder),
            }
            (folder / "manifest.json").write_text(
                json.dumps(item_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            variant_manifests[_variant_name(k, shrinkage)] = digest(folder / "manifest.json")
            log(f"Saved {_variant_name(k, shrinkage)}: {matrix.nnz:,} edges")

        if _code_hashes() != code_hashes:
            raise ValueError("Builder or retrieval code changed during the training build")
        manifest.update({
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "fit_scope": "pinned split_v1 train positive catalog only",
            "training_positive_users_including_singletons": user_count,
            "training_positive_pairs": int(user_items.nnz),
            "support_summary": {
                "input_games_with_positive_support": int(np.count_nonzero(support)),
                "input_games_with_zero_support": int(np.count_nonzero(support == 0)),
                "minimum_positive_game_support": int(support[support > 0].min())
                if np.any(support > 0) else None,
                "maximum_positive_game_support": int(support.max()) if len(support) else 0,
                "singleton_positive_users_measured": singleton_users,
                "singleton_positive_users_reported_by_split":
                    expected_counts["positive_singleton_users_reported"],
            },
            "resource_limits": {"duckdb_memory_limit": DUCKDB_MEMORY_LIMIT,
                                "duckdb_threads": DUCKDB_THREADS,
                                "candidate_block_size": block_size},
            "raw_cooccurrence": {"file": "raw_cooccurrence.parquet",
                                 "rows": raw_edges, "sha256": digest(raw_path),
                                 "columns": ["candidate_app_id", "neighbor_app_id",
                                             "cooccurrence_count", "raw_cosine",
                                             "raw_cosine_rank"]},
            "models": variant_manifests,
            "k100_lambda0_exact_v1_equivalence": checks,
            "code_sha256": code_hashes, "runtime": _runtime(),
            "ranking_metrics_computed": False,
        })
        save_manifest()
        stage.rename(output_root)
        log(f"Published train-only KNN v2 artifacts: {output_root}")
        return output_root
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_manifest()
        raise
    finally:
        sorted_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _verify_diagnostic_input(path, expected_file, split_root, split_manifest):
    relative = f"{path.parent.name}/{path.name}"
    expected = split_manifest["files"].get(relative)
    if expected is None or digest(path) != expected["sha256"]:
        raise ValueError(f"Pinned split diagnostic source changed: {relative}")
    return {"file": path.relative_to(EDA).as_posix(),
            "sha256": expected["sha256"], "bytes": path.stat().st_size,
            "rows": expected["rows"]}


def _categorize_diagnostic_rows(con):
    """Use train support, independently of whether v1 retained an input edge."""
    con.execute("""CREATE OR REPLACE TEMP TABLE diag_categorized AS
        SELECT *,CASE
            WHEN target_support=0 THEN 'target_support0'
            WHEN train_supported_positive_inputs=0 THEN 'all_positive_inputs_support0'
            WHEN raw_connection_count=0 THEN 'raw_connection0'
            WHEN retained_connection_count=0 THEN 'pruned'
            WHEN target_rank_v1>0 THEN 'hit'
            ELSE 'rank_failure' END category
        FROM diag_user_metrics""")


def _diagnose_split(con, split_root, split_manifest, split_name, v1,
                    retained_arrow, raw_path, baseline_root, output_folder):
    split_files = {}
    for filename in ("users.parquet", "query.parquet", "targets.parquet"):
        path = split_root / split_name / filename
        split_files[filename] = _verify_diagnostic_input(path, None, split_root, split_manifest)
    result_manifest_path = baseline_root / split_name / "manifest.json"
    per_user_path = baseline_root / split_name / "item_knn/per_user.parquet"
    result_manifest = json.loads(result_manifest_path.read_text("utf-8"))
    baseline_file = result_manifest.get("files", {}).get("item_knn/per_user.parquet")
    if (result_manifest.get("status") != "complete"
            or result_manifest.get("cohort") != split_name
            or not baseline_file or digest(per_user_path) != baseline_file["sha256"]):
        raise ValueError(f"Frozen v1 per-user diagnostics are incomplete or changed: {split_name}")

    query_path = split_root / split_name / "query.parquet"
    targets_path = split_root / split_name / "targets.parquet"
    users_path = split_root / split_name / "users.parquet"
    per_user_info = pq.ParquetFile(per_user_path)
    expected_users = pq.ParquetFile(users_path).metadata.num_rows
    if per_user_info.metadata.num_rows != expected_users:
        raise ValueError(f"Saved v1 per-user rows differ from {split_name} cohort")

    con.execute(f"CREATE OR REPLACE VIEW diag_users AS SELECT * FROM read_parquet({quote(users_path)})")
    con.execute(f"CREATE OR REPLACE VIEW diag_query AS SELECT * FROM read_parquet({quote(query_path)}) WHERE positive")
    con.execute(f"CREATE OR REPLACE VIEW diag_targets AS SELECT user_key,app_id target_app_id FROM read_parquet({quote(targets_path)})")
    con.execute(f"""CREATE OR REPLACE VIEW diag_v1 AS SELECT user_key,target_app_id,
        target_rank target_rank_v1,target_score target_score_v1
        FROM read_parquet({quote(per_user_path)})""")
    con.execute("""CREATE OR REPLACE TEMP TABLE diag_input_support AS
        SELECT app_id,positive_train_support,has_v1_neighbors FROM _v2_input_support""")
    con.execute(f"CREATE OR REPLACE VIEW diag_raw_edges AS SELECT * FROM read_parquet({quote(raw_path)})")
    con.register("_v2_retained_edges", retained_arrow)
    con.execute("""CREATE OR REPLACE TEMP VIEW diag_retained_edges AS
        SELECT * FROM _v2_retained_edges""")
    con.execute("""CREATE OR REPLACE TEMP TABLE diag_user_metrics AS
        WITH input_stats AS (
            SELECT q.user_key,count(*) n_positive_inputs,
                count(*) FILTER (WHERE coalesce(s.has_v1_neighbors,false)) supported_positive_inputs,
                count(*) FILTER (WHERE coalesce(s.positive_train_support,0)>0)
                    train_supported_positive_inputs
            FROM diag_query q LEFT JOIN diag_input_support s USING(app_id)
            GROUP BY q.user_key
        ), raw_stats AS (
            SELECT t.user_key,count(r.neighbor_app_id) raw_connection_count,
                min(r.raw_cosine_rank) min_raw_cosine_neighbor_rank,
                max(r.cooccurrence_count) max_raw_cooccurrence
            FROM diag_targets t JOIN diag_query q USING(user_key)
            LEFT JOIN diag_raw_edges r ON r.candidate_app_id=t.target_app_id
                AND r.neighbor_app_id=q.app_id
            GROUP BY t.user_key
        ), retained_stats AS (
            SELECT t.user_key,count(r.neighbor_app_id) retained_connection_count
            FROM diag_targets t JOIN diag_query q USING(user_key)
            LEFT JOIN diag_retained_edges r ON r.candidate_app_id=t.target_app_id
                AND r.neighbor_app_id=q.app_id
            GROUP BY t.user_key
        )
        SELECT t.user_key,t.target_app_id,coalesce(ts.positive_train_support,0)::BIGINT target_support,
            coalesce(i.n_positive_inputs,0)::BIGINT n_positive_inputs,
            coalesce(i.supported_positive_inputs,0)::BIGINT supported_positive_inputs,
            coalesce(i.train_supported_positive_inputs,0)::BIGINT train_supported_positive_inputs,
            coalesce(r.raw_connection_count,0)::BIGINT raw_connection_count,
            r.min_raw_cosine_neighbor_rank::INTEGER min_raw_cosine_neighbor_rank,
            coalesce(r.max_raw_cooccurrence,0)::BIGINT max_raw_cooccurrence,
            coalesce(k.retained_connection_count,0)::BIGINT retained_connection_count,
            v.target_rank_v1::INTEGER target_rank_v1,
            v.target_score_v1::DOUBLE target_score_v1,
            (v.target_score_v1=0) baseline_zero
        FROM diag_targets t
        JOIN diag_v1 v USING(user_key)
        LEFT JOIN diag_input_support ts ON ts.app_id=t.target_app_id
        LEFT JOIN input_stats i USING(user_key)
        LEFT JOIN raw_stats r USING(user_key)
        LEFT JOIN retained_stats k USING(user_key)
        WHERE t.target_app_id=v.target_app_id""")

    mismatch = con.execute("""SELECT count(*) FROM diag_user_metrics d
        JOIN diag_users u USING(user_key)
        WHERE d.n_positive_inputs<>u.n_positive_inputs""").fetchone()[0]
    expected_rows = con.execute("SELECT count(*) FROM diag_user_metrics").fetchone()[0]
    if expected_rows != expected_users or mismatch:
        raise ValueError(f"Diagnostic cohort mismatch for {split_name}: rows={expected_rows}, "
                         f"expected={expected_users}, positive_input_mismatches={mismatch}")
    impossible = con.execute("""SELECT count(*) FROM diag_user_metrics
        WHERE (train_supported_positive_inputs=0 AND raw_connection_count>0)
           OR (target_support=0 AND raw_connection_count>0)
           OR retained_connection_count>raw_connection_count""").fetchone()[0]
    if impossible:
        raise ValueError(f"Raw/train-support/retained connection inconsistency: {impossible}")
    _categorize_diagnostic_rows(con)
    output_folder.mkdir(parents=True, exist_ok=True)
    per_user_output = output_folder / f"{split_name}_user_diagnostics.parquet"
    per_user_output.unlink(missing_ok=True)
    con.execute(f"COPY (SELECT * FROM diag_categorized ORDER BY user_key) TO {quote(per_user_output)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")

    categories = ["target_support0", "all_positive_inputs_support0", "raw_connection0",
                  "pruned", "rank_failure", "hit"]
    counts = {row[0]: int(row[1]) for row in con.execute(
        "SELECT category,count(*) FROM diag_categorized GROUP BY category").fetchall()}
    counts = {category: counts.get(category, 0) for category in categories}
    if sum(counts.values()) != expected_users:
        raise ValueError("K0 categories are not mutually exclusive and exhaustive")
    zero_total, zero_pruned, zero_raw_none, zero_rank_failure = con.execute("""SELECT
        count(*) FILTER (WHERE baseline_zero),
        count(*) FILTER (WHERE baseline_zero AND raw_connection_count>0 AND retained_connection_count=0),
        count(*) FILTER (WHERE baseline_zero AND raw_connection_count=0),
        count(*) FILTER (WHERE baseline_zero AND retained_connection_count>0)
        FROM diag_categorized""").fetchone()
    unlimited, at_least_5, reachable_denom = con.execute("""SELECT
        count(*) FILTER (WHERE baseline_zero AND raw_connection_count>0),
        count(*) FILTER (WHERE baseline_zero AND max_raw_cooccurrence>=5),
        count(*) FILTER (WHERE baseline_zero AND raw_connection_count>0)
        FROM diag_categorized""").fetchone()
    if split_name == "validation" and expected_users != 325639:
        raise ValueError(f"Expected 325,639 validation users, got {expected_users}")
    if split_name == "test" and expected_users != 24342:
        raise ValueError(f"Expected 24,342 frank diagnostic users, got {expected_users}")
    expected_zero = {"validation": 55671, "test": 17819}[split_name]
    if zero_total != expected_zero:
        raise ValueError(f"v1 {split_name} zero-target-score count changed: {zero_total}")
    expected_pruned = {"validation": 55487, "test": 13448}[split_name]
    if zero_pruned != expected_pruned:
        raise ValueError(f"v1 {split_name} pruned-zero count changed: {zero_pruned}")
    expected_raw_none = {"validation": 184, "test": 4371}[split_name]
    if zero_raw_none != expected_raw_none:
        raise ValueError(f"v1 {split_name} raw-none zero-score count changed: {zero_raw_none}")
    if zero_raw_none + zero_pruned != zero_total or zero_rank_failure:
        raise ValueError(f"v1 {split_name} zero-score partition is incomplete or includes retained links")

    raw_none_zero = int(zero_raw_none)
    zero_rank_failure = int(zero_rank_failure)
    summary = {
        "users": int(expected_users),
        "diagnostic_role": "descriptive K0 cause analysis; not used for model selection or tuning",
        "categories": {category: {"users": value,
                                   "share_of_cohort": value / expected_users}
                       for category, value in counts.items()},
        "v1_zero_user_partition": {
            "users": int(zero_total),
            "pruned": {"users": int(zero_pruned), "denominator_users": int(zero_total),
                       "share": float(zero_pruned / zero_total) if zero_total else None},
            "raw_none": {"users": raw_none_zero, "denominator_users": int(zero_total),
                                "share": float(raw_none_zero / zero_total) if zero_total else None},
            "rank_failure": {"users": zero_rank_failure, "denominator_users": int(zero_total),
                             "share": float(zero_rank_failure / zero_total) if zero_total else None},
        },
        "unlimited_raw_reach_among_v1_zero_users": {
            "newly_reached_users": int(unlimited), "denominator_all_v1_zero_users": int(zero_total),
            "share_of_v1_zero_users": float(unlimited / zero_total) if zero_total else None,
        },
        "cooccurrence_at_least_5_among_v1_zero_users": {
            "newly_reached_users": int(at_least_5), "denominator_all_v1_zero_users": int(zero_total),
            "share_of_v1_zero_users": float(at_least_5 / zero_total) if zero_total else None,
            "denominator_unlimited_raw_reachable_zero_users": int(reachable_denom),
            "share_of_unlimited_raw_reachable_zero_users":
                float(at_least_5 / reachable_denom) if reachable_denom else None,
        },
        "v1_per_user_sha256": digest(per_user_path),
        "baseline_manifest_sha256": digest(result_manifest_path),
        "diagnostic_sources": {"split_files": split_files,
                               "per_user_file": {"file": per_user_path.relative_to(EDA).as_posix(),
                                                 "sha256": digest(per_user_path),
                                                 "bytes": per_user_path.stat().st_size},
                               "baseline_manifest": {"file": result_manifest_path.relative_to(EDA).as_posix(),
                                                     "sha256": digest(result_manifest_path)}},
        "user_diagnostics_file": {"file": per_user_output.relative_to(EDA).as_posix(),
                                  "sha256": digest(per_user_output),
                                  "rows": pq.ParquetFile(per_user_output).metadata.num_rows},
    }
    return summary


def diagnose(raw_path=DEFAULT_OUTPUT / "raw_cooccurrence.parquet",
             split=DEFAULT_SPLIT, output_root=DEFAULT_OUTPUT,
             baseline_root=DEFAULT_BASELINE, report_path=DEFAULT_REPORT):
    """Describe frozen v1 coverage categories, without changed ranking metrics."""
    raw_path, split, output_root = Path(raw_path), Path(split), Path(output_root)
    baseline_root, report_path = Path(baseline_root), Path(report_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw train co-occurrence cache is required: {raw_path}")
    source_manifest = json.loads((EDA / "data/source_manifest.json").read_text("utf-8"))
    expected_split = (EDA / source_manifest["split_artifacts"]["directory"]).resolve()
    if split.resolve() != expected_split:
        raise ValueError(f"Only pinned split_v1 is valid: {expected_split}")
    split_manifest_path = split / "manifest.json"
    split_config = source_manifest["split_artifacts"]
    if digest(split_manifest_path) != split_config["manifest_sha256"]:
        raise ValueError("Pinned split manifest changed before diagnostics")
    split_manifest = json.loads(split_manifest_path.read_text("utf-8"))
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("A complete verified pinned split is required for diagnostics")
    build_manifest_path = output_root / "manifest.json"
    if build_manifest_path.is_file():
        cached_manifest = json.loads(build_manifest_path.read_text("utf-8"))
        cached_raw = cached_manifest.get("raw_cooccurrence", {})
        if (cached_manifest.get("status") != "complete"
                or cached_manifest.get("fit_scope") != "pinned split_v1 train positive catalog only"
                or cached_raw.get("sha256") != digest(raw_path)
                or cached_raw.get("rows") != pq.ParquetFile(raw_path).metadata.num_rows):
            raise ValueError("Cached raw co-occurrence file differs from its completed train-only build manifest")
    v1 = ItemKnnCatalog.load(DEFAULT_INDEX, verify_hashes=True)
    v1_manifest_path = DEFAULT_INDEX / "manifest.json"
    v1_manifest = json.loads(v1_manifest_path.read_text("utf-8"))
    _, source_provenance, _, _, _, _ = read_sources(split)
    train_sha256 = source_provenance["sources"]["train_positive_catalog"]["sha256"]
    if v1_manifest["sources"]["sources"]["train_positive_catalog"]["sha256"] != train_sha256:
        raise ValueError("Frozen v1 and raw co-occurrence diagnostics use different train positives")
    if build_manifest_path.is_file() and (
            cached_manifest["sources"]["sources"]["train_positive_catalog"]["sha256"]
            != train_sha256):
        raise ValueError("Cached raw co-occurrence and pinned split train-positive source differ")
    support_summary = v1_manifest["support_summary"]
    if (support_summary["singleton_positive_users_measured"]
            != support_summary["singleton_positive_users_reported_by_split"]):
        raise ValueError("Frozen v1 singleton support check does not match the pinned split")
    candidate_rows = np.searchsorted(v1.app_ids, v1.candidate_app_ids)
    if (np.any(candidate_rows >= len(v1.app_ids))
            or not np.array_equal(v1.app_ids[candidate_rows], v1.candidate_app_ids)):
        raise ValueError("Frozen v1 candidate/input mapping is invalid")

    retained_rows, retained_cols = v1.similarities.nonzero()
    retained_arrow = pa.table({
        "candidate_app_id": pa.array(v1.candidate_app_ids[retained_rows], type=pa.int64()),
        "neighbor_app_id": pa.array(v1.app_ids[retained_cols], type=pa.int64()),
    })
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostic_folder = output_root / "diagnostics"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        connection.execute(f"SET threads={DUCKDB_THREADS}")
        connection.register("_v2_input_support", pa.table({
            "app_id": v1.app_ids,
            "positive_train_support": v1.support,
            "has_v1_neighbors": np.asarray(v1.similarities.getnnz(axis=0) > 0, dtype=bool),
        }))
        summaries = {}
        for split_name in ("validation", "test"):
            summaries[split_name] = _diagnose_split(
                connection, split, split_manifest, split_name, v1, retained_arrow,
                raw_path, baseline_root, diagnostic_folder)
    finally:
        connection.close()
    raw_file_sha = digest(raw_path)
    report = {
        "version": "item-knn-failure-analysis-v2",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reproducible train-only KNN reachability diagnosis; frank test is descriptive only",
        "fit_scope": "Pinned split_v1 train positive catalog only",
        "ranking_metrics_computed": False,
        "model_selection_or_tuning_used": False,
        "frank_test_user_count": summaries["test"]["users"],
        "frank_test_used_for_model_selection": False,
        "training_support_summary": support_summary,
        "training_positive_source_sha256": train_sha256,
        "category_order": ["target_support0", "all_positive_inputs_support0",
                           "raw_connection0", "pruned", "rank_failure", "hit"],
        "category_definitions": {
            "target_support0": "Hidden target has zero positive train support.",
            "all_positive_inputs_support0": "Every positive query input has zero positive train support; retained K100 coverage is a separate diagnostic.",
            "raw_connection0": "Target has no raw train co-occurrence with any positive query input.",
            "pruned": "At least one raw target/input link exists, but none survive v1 candidate top100.",
            "rank_failure": "At least one target/input link survives v1 top100, but saved v1 target rank is zero.",
            "hit": "Saved v1 target rank is between 1 and 10.",
        },
        "cooccurrence_semantics": "All positive train users, including singleton positive users; raw edges exclude self-pairs.",
        "raw_cooccurrence_file": {"file": raw_path.relative_to(EDA).as_posix()
                                  if raw_path.is_relative_to(EDA) else str(raw_path),
                                  "sha256": raw_file_sha,
                                  "rows": pq.ParquetFile(raw_path).metadata.num_rows},
        "split_manifest_sha256": digest(split_manifest_path),
        "v1_model_manifest_sha256": digest(v1_manifest_path),
        "cohorts": summaries,
        "runtime": _runtime(),
        "code_sha256": _code_hashes(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _refresh_build_manifest(output_root, report_path)
    log(f"Saved K0 failure diagnostics: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-models", action="store_true",
                        help="Build raw train co-occurrences and all nine v2 indexes")
    parser.add_argument("--diagnose", action="store_true",
                        help="Write K0 diagnostics from cached raw edges and frozen v1 outcomes")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = parser.parse_args()
    if not args.build_models and not args.diagnose:
        parser.error("choose --build-models, --diagnose, or both")
    if args.build_models:
        output_root = build_models(args.output_root, DEFAULT_SPLIT, args.block_size)
        raw_path = output_root / "raw_cooccurrence.parquet"
    else:
        output_root = args.output_root
        raw_path = output_root / "raw_cooccurrence.parquet"
    report = diagnose(raw_path, DEFAULT_SPLIT, output_root, DEFAULT_BASELINE, args.report) if args.diagnose else None
    result = {"status": "complete", "output_root": str(output_root),
              "raw_cooccurrence_sha256": digest(raw_path) if raw_path.is_file() else None,
              "model_variants": [_variant_name(k, shrinkage) for k in KS for shrinkage in SHRINKAGES]
              if args.build_models else [],
              "diagnostic_report": str(args.report) if report is not None else None}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
