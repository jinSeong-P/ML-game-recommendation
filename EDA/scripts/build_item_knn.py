"""Build a frozen sparse binary Item-KNN index from split_v1 train positives."""

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

from eda_large_review_frank import EDA, digest, log, quote
from item_knn import (DEFAULT_INDEX, VERSION, ItemKnnCatalog,
                      _build_user_item_csr, _top_cosine_neighbors)


DEFAULT_SPLIT = EDA / "data/processed/modeling/split_v1"
SORT_BATCH_SIZE = 131072
SORT_MEMORY_LIMIT = "1GB"


def read_sources(split=DEFAULT_SPLIT):
    """Verify the frozen source split and return its pinned training/catalog data."""
    source_manifest_path = EDA / "data/source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text("utf-8"))
    split_config = source_manifest["split_artifacts"]
    expected_split = (EDA / split_config["directory"]).resolve()
    split = Path(split).resolve()
    if split != expected_split:
        raise ValueError(f"Only the pinned modeling split may be used: {expected_split}")

    split_manifest_path = split / "manifest.json"
    if digest(split_manifest_path) != split_config["manifest_sha256"]:
        raise ValueError("Frozen split manifest changed")
    split_manifest = json.loads(split_manifest_path.read_text("utf-8"))
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("A verified complete fixed split is required")

    paths = {
        "train_positive_catalog": split / "train/positive_catalog.parquet",
        "input_apps": split / "catalog/input_apps.parquet",
        "linux_candidates": split / "catalog/linux_candidates.parquet",
    }
    provenance = {
        "source_manifest_sha256": digest(source_manifest_path),
        "split_manifest_sha256": digest(split_manifest_path),
        "train_positive_reviews_loaded_by_builder": True,
        "validation_or_test_reviews_or_ranking_metrics_loaded": False,
        "sources": {},
    }
    for name, path in paths.items():
        relative = path.relative_to(split).as_posix()
        expected = split_manifest["files"][relative]
        actual = digest(path)
        if actual != expected["sha256"]:
            raise ValueError(f"Pinned split source fingerprint changed: {relative}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Pinned split source size changed: {relative}")
        provenance["sources"][name] = {
            "file": path.relative_to(EDA).as_posix(),
            "sha256": actual,
            "bytes": path.stat().st_size,
            "rows": expected["rows"],
        }

    expected_counts = split_config["counts"]
    if (expected_counts["input_apps"], expected_counts["linux_candidates"],
            expected_counts["train_positive_catalog_pairs"],
            expected_counts["positive_train_accounts"]) != (37610, 6630, 78315268, 29836980):
        raise ValueError("Pinned Item-KNN input counts differ from the frozen baseline contract")

    input_pf = pq.ParquetFile(paths["input_apps"])
    if input_pf.schema_arrow.field("app_id").type != pa.int64() or input_pf.schema_arrow.field("linux").type != pa.bool_():
        raise ValueError("Unexpected fixed input catalog schema")
    input_table = pq.read_table(paths["input_apps"])
    app_ids = np.asarray(input_table.column("app_id").to_numpy(), dtype=np.int64)
    linux = np.asarray(input_table.column("linux").to_numpy(), dtype=bool)
    if len(app_ids) != expected_counts["input_apps"] or len(app_ids) != 37610:
        raise ValueError("Fixed input catalog row count changed")
    if np.any(app_ids <= 0) or np.any(app_ids[1:] <= app_ids[:-1]):
        raise ValueError("Fixed input App IDs must be unique and ascending")
    if len(linux) != len(app_ids):
        raise ValueError("Fixed input Linux flags have an invalid shape")

    candidate_pf = pq.ParquetFile(paths["linux_candidates"])
    if candidate_pf.schema_arrow.field("app_id").type != pa.int64():
        raise ValueError("Unexpected Linux candidate catalog schema")
    candidate_app_ids = np.asarray(
        pq.read_table(paths["linux_candidates"]).column("app_id").to_numpy(), dtype=np.int64)
    if len(candidate_app_ids) != expected_counts["linux_candidates"] or len(candidate_app_ids) != 6630:
        raise ValueError("Fixed Linux candidate row count changed")
    if np.any(candidate_app_ids <= 0) or np.any(candidate_app_ids[1:] <= candidate_app_ids[:-1]):
        raise ValueError("Linux candidate App IDs must be unique and ascending")
    if not np.array_equal(candidate_app_ids, app_ids[linux]):
        raise ValueError("Linux candidates differ from Linux-marked input catalog rows")

    train_pf = pq.ParquetFile(paths["train_positive_catalog"])
    expected_schema = {"user_key": pa.uint64(), "app_id": pa.int64(), "positive": pa.bool_()}
    for field, expected_type in expected_schema.items():
        if train_pf.schema_arrow.field(field).type != expected_type:
            raise ValueError(f"Unexpected training positive catalog type for {field}")
    if train_pf.metadata.num_rows != expected_counts["train_positive_catalog_pairs"]:
        raise ValueError("Training positive pair count changed")

    counts = {
        "input_apps": len(app_ids),
        "linux_candidates": len(candidate_app_ids),
        "train_positive_pairs": expected_counts["train_positive_catalog_pairs"],
        "train_positive_users": expected_counts["positive_train_accounts"],
        "positive_singleton_users_reported": expected_counts["positive_singleton_accounts"],
    }
    return paths, provenance, app_ids, linux, candidate_app_ids, counts


def file_inventory(root):
    files = {}
    for path in sorted(root.iterdir()):
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


def _sort_train_pairs(source_path, sorted_path, temp_dir):
    """External sort by full-precision user ID before CSR streaming."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit='{SORT_MEMORY_LIMIT}'")
        connection.execute("SET threads=2")
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


def _write_parquet(records, path, schema):
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path,
                   compression="zstd")


def _count_singleton_rows(indptr, chunk_size=1_000_000):
    count = 0
    row_count = len(indptr) - 1
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        count += int(np.count_nonzero(
            indptr[start + 1:stop + 1] - indptr[start:stop] == 1))
    return count


def build(output=DEFAULT_INDEX, split=DEFAULT_SPLIT, neighbors=100, block_size=64):
    """Build a staging Item-KNN artifact and publish only after hash/readback checks."""
    if isinstance(neighbors, bool) or int(neighbors) != neighbors or int(neighbors) < 1:
        raise ValueError("neighbors must be a positive integer")
    if isinstance(block_size, bool) or int(block_size) != block_size or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    neighbors, block_size = int(neighbors), int(block_size)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Item-KNN artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.building-{uuid.uuid4().hex[:10]}"
    stage.mkdir()
    sorted_path = stage / "_train_positive_sorted.parquet"
    temp_dir = stage / "_duckdb_temp"
    manifest = {"version": VERSION, "status": "building",
                "started_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = stage / "manifest.json"
    code_fingerprints = {"builder": digest(Path(__file__)),
                         "model_and_retrieval": digest(Path(__file__).with_name("item_knn.py"))}

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

    save_manifest()
    try:
        log("Verifying pinned train positives and fixed input/candidate catalogs")
        (paths, provenance, app_ids, linux, candidate_app_ids,
         expected_counts) = read_sources(split)
        manifest.update({"sources": provenance, "source_counts": expected_counts})
        save_manifest()

        log("Externally sorting train positives by uint64 user key and App ID")
        _sort_train_pairs(paths["train_positive_catalog"], sorted_path, temp_dir)
        sorted_pf = pq.ParquetFile(sorted_path)
        if sorted_pf.metadata.num_rows != expected_counts["train_positive_pairs"]:
            raise ValueError("Filtered training positive row count differs from the pinned split")

        log("Streaming sorted positives into the binary sparse user-item matrix")
        sorted_batches = sorted_pf.iter_batches(
            batch_size=SORT_BATCH_SIZE, columns=["user_key", "app_id"])
        user_items, support, user_count = _build_user_item_csr(
            sorted_batches, app_ids,
            expected_pairs=expected_counts["train_positive_pairs"],
            expected_users=expected_counts["train_positive_users"])
        sorted_pf.close()
        singleton_users = _count_singleton_rows(user_items.indptr)
        log(f"Built binary CSR: {user_count:,} users x {len(app_ids):,} games, "
            f"{user_items.nnz:,} positives; {singleton_users:,} singleton users retained")
        # CSC keeps item columns compact. Candidate column slices then need
        # only block-sized pointers instead of a 30M-user CSR indptr per block.
        user_items = user_items.tocsc()
        sorted_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

        candidate_columns = np.searchsorted(app_ids, candidate_app_ids).astype(np.int32)
        max_row_neighbors = min(neighbors, max(0, len(app_ids) - 1))
        output_indices = np.empty(len(candidate_app_ids) * max_row_neighbors, dtype=np.int32)
        output_values = np.empty(len(candidate_app_ids) * max_row_neighbors, dtype=np.float64)
        output_indptr = np.zeros(len(candidate_app_ids) + 1, dtype=np.int32)
        output_nnz = 0
        log(f"Computing exact candidate-by-input co-occurrence in blocks of {block_size}")
        for block_start in range(0, len(candidate_app_ids), block_size):
            block_stop = min(block_start + block_size, len(candidate_app_ids))
            candidate_by_user = user_items[:, candidate_columns[block_start:block_stop]].T.tocsr()
            candidate_by_user = candidate_by_user.astype(np.int32, copy=False)
            cooccurrence = (candidate_by_user @ user_items).tocsr()
            cooccurrence.sum_duplicates()
            for local_row in range(block_stop - block_start):
                global_row = block_start + local_row
                start, end = cooccurrence.indptr[local_row:local_row + 2]
                columns, values = _top_cosine_neighbors(
                    int(candidate_columns[global_row]),
                    cooccurrence.indices[start:end], cooccurrence.data[start:end],
                    app_ids, support, neighbors)
                count = len(columns)
                if output_nnz + count > len(output_indices):
                    raise ValueError("Neighbor output exceeded its allocated top-k capacity")
                output_indices[output_nnz:output_nnz + count] = columns
                output_values[output_nnz:output_nnz + count] = values
                output_nnz += count
                output_indptr[global_row + 1] = output_nnz
            log(f"Processed Linux candidate rows {block_stop:,}/{len(candidate_app_ids):,}")
            del candidate_by_user, cooccurrence

        similarities = sparse.csr_matrix(
            (output_values[:output_nnz], output_indices[:output_nnz], output_indptr),
            shape=(len(candidate_app_ids), len(app_ids)), copy=False)
        similarities.sort_indices()

        input_rows = [{"row_id": i, "app_id": int(app), "linux": bool(linux[i]),
                       "positive_user_support": int(support[i])}
                      for i, app in enumerate(app_ids)]
        candidate_input_rows = np.searchsorted(app_ids, candidate_app_ids)
        candidate_rows = [{"row_id": row, "app_id": int(app),
                           "input_row_id": int(candidate_input_rows[row])}
                          for row, app in enumerate(candidate_app_ids)]
        _write_parquet(input_rows, stage / "game_rows.parquet", pa.schema([
            ("row_id", pa.int32()), ("app_id", pa.int64()), ("linux", pa.bool_()),
            ("positive_user_support", pa.int64()),
        ]))
        _write_parquet(candidate_rows, stage / "candidate_rows.parquet", pa.schema([
            ("row_id", pa.int32()), ("app_id", pa.int64()), ("input_row_id", pa.int32()),
        ]))
        sparse.save_npz(stage / "similarities_csr.npz", similarities, compressed=True)

        parameters = {
            "version": VERSION,
            "neighbors": neighbors,
            "candidate_block_size": block_size,
            "minimum_cooccurrence": 1,
            "shrinkage": None,
            "binary_positive_interactions": True,
            "support_counts_include_positive_singleton_users": True,
            "similarity": "count(i,j)/sqrt(count(i)*count(j)); diagonal forced to zero",
            "neighbor_order": "similarity descending, then App ID ascending",
            "query_score": "sum(candidate similarity to each positive query item)/all positive query count",
            "recommendation_policy": "positive scores only; exclude supplied observed games and positives; no fill; max 10",
            "sparse_index": "candidate-by-input CSR; exact sparse co-occurrence products in candidate blocks",
        }
        (stage / "parameters.json").write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        manifest.update({
            "parameters": parameters,
            "index": {"format": "SciPy CSR float64 NPZ",
                      "shape": list(similarities.shape), "nnz": int(similarities.nnz)},
            "support_summary": {
                "input_games_with_positive_support": int(np.count_nonzero(support)),
                "input_games_with_zero_support": int(np.count_nonzero(support == 0)),
                "candidate_games_with_neighbors": int(np.count_nonzero(np.diff(similarities.indptr))),
                "candidate_games_without_neighbors": int(np.count_nonzero(np.diff(similarities.indptr) == 0)),
                "minimum_positive_game_support": int(support[support > 0].min()) if np.any(support > 0) else None,
                "maximum_positive_game_support": int(support.max()) if len(support) else 0,
                "singleton_positive_users_measured": singleton_users,
                "singleton_positive_users_reported_by_split": expected_counts["positive_singleton_users_reported"],
            },
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pyarrow": pa.__version__,
                        "duckdb": duckdb.__version__},
            "code_sha256": code_fingerprints,
            "ranking_metrics_computed": False,
            "files": file_inventory(stage),
        })
        if (digest(Path(__file__)) != code_fingerprints["builder"]
                or digest(Path(__file__).with_name("item_knn.py")) != code_fingerprints["model_and_retrieval"]):
            raise ValueError("Model source code changed during training; rebuild from a fixed version")
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()

        log("Reloading staged index and verifying all saved hashes and mappings")
        loaded = ItemKnnCatalog.load(stage, verify_hashes=True)
        if (not np.array_equal(loaded.app_ids, app_ids)
                or not np.array_equal(loaded.candidate_app_ids, candidate_app_ids)
                or not np.array_equal(loaded.support, support)
                or loaded.similarities.shape != similarities.shape
                or loaded.similarities.nnz != similarities.nnz):
            raise ValueError("Staged Item-KNN readback differs from built arrays")
        manifest["verification"] = {
            "passed": True,
            "all_saved_file_hashes_verified": True,
            "input_mapping_rows": len(loaded.app_ids),
            "candidate_mapping_rows": len(loaded.candidate_app_ids),
            "support_values_checked": len(loaded.support),
            "similarity_shape": list(loaded.similarities.shape),
            "similarity_nonzeros": int(loaded.similarities.nnz),
            "self_similarity_zero": True,
            "top_k_row_limit_checked": True,
        }
        save_manifest()
        stage.rename(output)
        log(f"Published verified Item-KNN index: {output}")
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_manifest()
        raise
    finally:
        sorted_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()
    result = build(args.output, args.split, args.neighbors, args.block_size)
    print(json.dumps({"status": result["status"], "index": result["index"],
                      "support_summary": result["support_summary"],
                      "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
