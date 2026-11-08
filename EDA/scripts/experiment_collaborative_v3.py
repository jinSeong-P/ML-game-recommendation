"""Train-only collaborative model variants for the v3 frozen experiment suite.

Artifacts contain fixed candidate-by-input item weights or fixed item factors.
No validation/test reviews, targets, rankings, or metrics are read here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
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
from item_knn import _build_user_item_csr


MODEL_ROOT = EDA / "data/processed/modeling/improvement_v3/models"
SPLIT_ROOT = EDA / "data/processed/modeling/split_v1"
V2_KNN_ROOT = EDA / "data/processed/modeling/improvement_v2/knn"
V2_RAW = V2_KNN_ROOT / "raw_cooccurrence.parquet"
V1_ROOT = EDA / "data/processed/modeling/item_knn_v1"
POSITIVE_TRAIN = SPLIT_ROOT / "train/positive_catalog.parquet"
INTERACTIONS_TRAIN = SPLIT_ROOT / "train/interactions.parquet"
LINEAGE_TRAIN = SPLIT_ROOT / "train/review_lineage.parquet"
CANONICAL_ROOT = EDA / "data/processed/kieran_original"
METADATA_ROOT = EDA / "data/processed/modeling/improvement_v2/tfidf/metadata_channel"
VERSION = "collaborative-models-v3"
BLOCK_SIZE = 48
SORT_BATCH_SIZE = 131_072
DUCKDB_MEMORY_LIMIT = "3GB"
FEATURE_DUCKDB_MEMORY_LIMIT = "5GB"
DUCKDB_THREADS = 2
MAX_BUILD_GB = 5.0
TRAIN_USER_COUNT = 29_836_980


def _json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _inventory(folder: Path) -> dict:
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return files


def _source_info(path: Path) -> dict:
    return {"file": path.relative_to(EDA).as_posix(),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def _assert_under_model_root(path: Path) -> Path:
    root = MODEL_ROOT.resolve()
    resolved = Path(path).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Build path escapes model root: {resolved}")
    return resolved


def _safe_remove_build_stage(path: Path) -> None:
    resolved = _assert_under_model_root(path)
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_manifest() -> dict:
    split_manifest_path = SPLIT_ROOT / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text("utf-8"))
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("Only a verified complete split_v1 can train v3 models")
    if digest(POSITIVE_TRAIN) != split_manifest["files"]["train/positive_catalog.parquet"]["sha256"]:
        raise ValueError("Pinned positive training source hash changed")
    if digest(INTERACTIONS_TRAIN) != split_manifest["files"]["train/interactions.parquet"]["sha256"]:
        raise ValueError("Pinned training interactions hash changed")
    if digest(LINEAGE_TRAIN) != split_manifest["files"]["train/review_lineage.parquet"]["sha256"]:
        raise ValueError("Pinned training lineage hash changed")
    v2_manifest_path = V2_KNN_ROOT / "manifest.json"
    v2 = json.loads(v2_manifest_path.read_text("utf-8"))
    if v2.get("status") != "complete" or v2.get("ranking_metrics_computed") is not False:
        raise ValueError("Raw graph source must be a completed train-only v2 artifact")
    if digest(V2_RAW) != v2["raw_cooccurrence"]["sha256"]:
        raise ValueError("Pinned raw co-occurrence source hash changed")
    return {
        "split_manifest": _source_info(split_manifest_path),
        "train_positive_catalog": _source_info(POSITIVE_TRAIN),
        "train_interactions": _source_info(INTERACTIONS_TRAIN),
        "train_review_lineage": _source_info(LINEAGE_TRAIN),
        "v2_raw_cooccurrence": _source_info(V2_RAW),
        "v2_manifest": _source_info(v2_manifest_path),
        "fixed_catalog": _source_info(SPLIT_ROOT / "catalog/input_apps.parquet"),
        "candidate_catalog": _source_info(SPLIT_ROOT / "catalog/linux_candidates.parquet"),
        "v1_game_rows": _source_info(V1_ROOT / "game_rows.parquet"),
        "v1_candidate_rows": _source_info(V1_ROOT / "candidate_rows.parquet"),
    }


def _catalog() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = pq.read_table(V1_ROOT / "game_rows.parquet").to_pydict()
    apps = np.asarray(rows["app_id"], dtype=np.int64)
    support = np.asarray(rows["positive_user_support"], dtype=np.int64)
    linux = np.asarray(rows["linux"], dtype=bool)
    candidates = apps[linux]
    if len(apps) != 37_610 or len(candidates) != 6_630:
        raise ValueError("Frozen input/candidate catalog dimensions changed")
    if np.any(apps[1:] <= apps[:-1]) or np.any(candidates[1:] <= candidates[:-1]):
        raise ValueError("Frozen App ID mappings must be ascending and unique")
    return apps, candidates, linux, support


def _variant_roster() -> dict:
    """Pinned candidate set; registry is written before any model fitting."""
    variants = {}
    def add(name, kind, parameters, mode="knn", status="planned"):
        variants[name] = {"artifact_type": kind, "scorer_mode": mode,
                          "parameters": parameters, "status": status}

    for minimum in (2, 3, 5):
        add(f"min_cooccurrence_{minimum}_cosine_k100", "item_similarity_csr",
            {"minimum_cooccurrence": minimum, "neighbors": 100,
             "neighbor_selection": "raw cosine after minimum co-occurrence filter",
             "stored_score": "binary cosine"})
    add("candidate_oriented_cosine_k100", "item_similarity_csr",
        {"orientation": "candidate-row top100", "neighbors": 100, "stored_score": "binary cosine"})
    add("input_oriented_cosine_k100", "item_similarity_csr",
        {"orientation": "input-column top100 Linux candidates", "neighbors": 100,
         "stored_score": "binary cosine"})
    add("candidate_input_union_cosine_k100", "item_similarity_csr",
        {"orientation": "union of candidate-row top100 and input-column top100",
         "neighbors_per_orientation": 100, "stored_score": "binary cosine"})
    add("unpruned_cosine", "item_similarity_csr",
        {"neighbor_pruning": "none", "stored_score": "binary cosine"})
    add("cooccurrence_selected_cosine_scored_k100", "item_similarity_csr",
        {"selection": "co-occurrence count descending, then cosine descending, then App ID",
         "neighbors": 100, "stored_score": "binary cosine"})
    add("lift_selected_cosine_scored_k100", "item_similarity_csr",
        {"selection": "lift descending, then co-occurrence descending, then App ID",
         "lift": "cooccurrence * positive_train_users / (candidate_support * input_support)",
         "neighbors": 100, "stored_score": "binary cosine"})
    for alpha in (.25, .75):
        for locality in (1, 2):
            label = f"{alpha:.2f}".replace(".", "")
            add(f"asymmetric_cosine_alpha{label}_locality{locality}_k100", "item_similarity_csr",
                {"candidate_support_exponent": alpha,
                 "input_support_exponent": 1.0-alpha,
                 "formula": "cooccurrence / (candidate_support**alpha * input_support**(1-alpha))",
                 "locality_exponent": locality, "locality_formula": "base_score**locality",
                 "neighbors": 100})
    for method in ("inverse_sqrt_degree", "inverse_degree"):
        add(f"activity_{method}_cosine_k100", "item_similarity_csr",
            {"reviewer_weight": "1/sqrt(train positive input degree)" if method == "inverse_sqrt_degree"
             else "1/train positive input degree",
             "normalization": "weighted cosine using weighted item marginals",
             "neighbors": 100})
    for beta in (.3, .6):
        add(f"rp3beta_alpha1_beta{int(beta*10):02d}_k100", "item_similarity_csr",
            {"algorithm": "RP3beta", "length": 3, "alpha": 1.0, "beta": beta,
             "formula": "sum_u 1/(item_support_input * user_degree_u) / candidate_support**beta",
             "orientation": "input item to candidate item", "neighbors": 100})
    for regularization in (100, 1000):
        add(f"ease_lambda{regularization}_exact", "item_similarity_csr",
            {"algorithm": "EASE", "regularization": regularization,
             "catalog_items": 37_610, "solver": "exact full-catalog float32 Cholesky",
             "score_formula": "-P_inverse[input,candidate] / P_inverse[candidate,candidate]; diagonal zero"})
    for factors in (32, 64):
        add(f"ials_rank{factors}_reg0_1_conf20_iter10_seed20261108", "item_factors",
            {"algorithm": "implicit ALS", "factors": factors, "regularization": .1,
             "confidence_scale": 20, "alpha": 20, "iterations": 10, "seed": 20261108,
             "fit_users": "train positive catalog degree >=2"},
            mode="factors")
        for blend in (0.0, 0.5, 1.0):
            blend_label = str(blend).replace(".", "")
            add(f"metadata_ridge_alpha100_rank{factors}_blend{blend_label}", "item_factors",
                {"algorithm": "metadata-to-collaborative embedding ridge",
                 "ridge_alpha": 100, "target": f"iALS rank-{factors} item factors",
                 "feature_source": "v2 tags and genres TF-IDF",
                 "fit_rows": "train-supported items only",
                 "blend_supported_prediction": blend,
                 "regularization": .1, "alpha": 20},
                mode="factors")
    for beta in (.1, .3):
        add(f"positive_negative_relation_beta{int(beta*100):02d}", "signed_item_similarity",
            {"signed_similarity_formula": "(P.T@P - P.T@N - N.T@P) / sqrt(observed_support_i * observed_support_j)",
             "negative_negative": "excluded",
             "neighbor_selection": "signed weight absolute value descending, then App ID; top 100",
             "negative_query_coefficient": beta,
             "query_score_formula": "(G@positive_query - beta*G@negative_query) / positive_input_count",
             "train_game_positivity_prior": "add-one smoothed positive rate"})
    add("time_decay_3year_cosine_k100", "item_similarity_csr",
        {"weighting": "exponential review-age decay",
         "half_life_days": 1095, "reference": "latest accepted TRAIN review timestamp",
         "weighted_cosine": "cooccurrence of per-interaction weighted vectors",
         "neighbors": 100})
    add("playtime_percentile_cosine_k100", "item_similarity_csr",
        {"weighting": "0.5 + per-game percentile rank of log1p(playtime_forever)",
         "playtime_source": "canonical playtime_forever snapshot",
         "missing_playtime_weight": .5, "weighted_cosine": "weighted interaction vectors",
         "neighbors": 100})
    return variants


def initialize_registry(force=False) -> dict:
    """Write exact model names and parameter sets before any model fit."""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    path = MODEL_ROOT / "registry.json"
    sources = _source_manifest()
    roster = _variant_roster()
    current = {
        "version": VERSION, "status": "building",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_scope": "split_v1 training rows only",
        "holdout_reviews_or_metrics_loaded": False,
        "source_fingerprints": sources,
        "resource_policy": {"duckdb_memory_limit": DUCKDB_MEMORY_LIMIT,
                            "feature_duckdb_memory_limit": FEATURE_DUCKDB_MEMORY_LIMIT,
                            "duckdb_threads": DUCKDB_THREADS, "max_build_gb": MAX_BUILD_GB},
        "models": roster,
    }
    if path.exists() and not force:
        old = json.loads(path.read_text("utf-8"))
        old_models = old.get("models", {})
        for name in list(old_models):
            artifact_exists = (MODEL_ROOT / name / "manifest.json").is_file()
            if name not in roster:
                if old_models[name].get("status") == "complete" or artifact_exists:
                    raise ValueError(f"Cannot remove completed model from registry: {name}")
                del old_models[name]
                continue
            previous = {k: v for k, v in old_models[name].items()
                        if k not in {"status", "directory", "detail"}}
            desired = {k: v for k, v in roster[name].items()
                       if k not in {"status", "directory", "detail"}}
            if previous != desired:
                if old_models[name].get("status") == "complete" or artifact_exists:
                    raise ValueError(f"Cannot change completed model parameters: {name}")
                old_models[name] = roster[name]
        for name in set(roster).difference(old_models):
            old_models[name] = roster[name]
        old["models"] = old_models
        old["source_fingerprints"] = sources
        return old
    _json_write(path, current)
    return current


def _write_model(name: str, matrix, parameters: dict, sources: dict,
                 extra_sparse: dict | None = None, extra_arrays: dict | None = None,
                 artifact_type="item_similarity_csr", scorer_mode="knn",
                 compressed=True, fit_summary: dict | None = None) -> Path:
    folder = MODEL_ROOT / name
    _assert_under_model_root(folder)
    if folder.exists():
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"Refusing to overwrite incomplete model directory: {folder}")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if manifest.get("parameters") != parameters or manifest.get("status") != "complete":
            raise ValueError(f"Existing model manifest differs: {name}")
        for relative, info in manifest.get("files", {}).items():
            if digest(folder / relative) != info["sha256"]:
                raise ValueError(f"Existing model file changed: {name}/{relative}")
        return folder
    stage = folder.with_name(f".{folder.name}.building-{os.getpid()}")
    if stage.exists():
        _safe_remove_build_stage(stage)
    stage.mkdir(parents=True)
    try:
        if matrix is not None:
            if sparse.issparse(matrix):
                matrix = matrix.tocsr()
                matrix.sort_indices()
                if not np.all(np.isfinite(matrix.data)):
                    raise ValueError(f"Nonfinite score in {name}")
                sparse.save_npz(stage / "similarities_csr.npz", matrix, compressed=compressed)
            else:
                factor_values = np.asarray(matrix, dtype=np.float32)
                app_ids, _, _, _ = _catalog()
                if factor_values.ndim != 2 or factor_values.shape[0] != len(app_ids):
                    raise ValueError(f"Item factor rows do not match fixed catalog: {name}")
                np.save(stage / "item_factors.npy", factor_values, allow_pickle=False)
                pq.write_table(pa.table({
                    "row_id": pa.array(np.arange(len(app_ids), dtype=np.int32)),
                    "app_id": pa.array(app_ids, type=pa.int64()),
                }), stage / "item_rows.parquet", compression="zstd")
        for key, value in (extra_sparse or {}).items():
            value = value.tocsr()
            value.sort_indices()
            if not np.all(np.isfinite(value.data)):
                raise ValueError(f"Nonfinite score in {name}/{key}")
            sparse.save_npz(stage / key, value, compressed=True)
        for key, value in (extra_arrays or {}).items():
            np.save(stage / key, np.asarray(value), allow_pickle=False)
        builder_hash = digest(Path(__file__))
        snapshot_name = "builder_snapshot.py"
        shutil.copy2(Path(__file__), stage / snapshot_name)
        index = None
        if matrix is not None and sparse.issparse(matrix):
            index = {"format": "SciPy CSR NPZ", "shape": list(matrix.shape),
                     "nnz": int(matrix.nnz), "dtype": str(matrix.dtype)}
        elif matrix is not None:
            index = {"format": "NumPy NPY", "shape": list(np.shape(matrix)),
                     "dtype": str(np.asarray(matrix).dtype)}
        manifest = {
            "version": VERSION, "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "family": "collaborative", "artifact_type": artifact_type,
            "scorer_mode": scorer_mode,
            "fit_scope": "split_v1 train only; validation and test not read",
            "matrix_orientation": "Linux candidate rows x fixed input-catalog columns"
            if scorer_mode in {"knn", "signed_knn"} else "fixed input-catalog rows x latent factors",
            "parameters": parameters, "sources": sources, "index": index,
            "builder_sha256": builder_hash, "builder_snapshot": snapshot_name,
            "fit_summary": fit_summary,
            "ranking_metrics_computed": False,
            "files": _inventory(stage),
        }
        _json_write(stage / "manifest.json", manifest)
        _assert_under_model_root(stage)
        _assert_under_model_root(folder)
        stage.replace(folder)
    except Exception:
        if stage.exists():
            _safe_remove_build_stage(stage)
        raise
    return folder


def _update_registry(name: str, status: str, directory: str | None = None,
                     detail: str | None = None) -> None:
    path = MODEL_ROOT / "registry.json"
    registry = json.loads(path.read_text("utf-8"))
    entry = registry["models"][name]
    entry["status"] = status
    if directory:
        entry["directory"] = (Path("data/processed/modeling/improvement_v3/models") /
                              directory).as_posix()
    if detail:
        entry["detail"] = detail
    _json_write(path, registry)


def _edge_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    apps, candidates, _, support = _catalog()
    table = pq.read_table(V2_RAW, columns=[
        "candidate_app_id", "neighbor_app_id", "cooccurrence_count", "raw_cosine"])
    candidate_app = np.asarray(table["candidate_app_id"], dtype=np.int64)
    neighbor_app = np.asarray(table["neighbor_app_id"], dtype=np.int64)
    counts = np.asarray(table["cooccurrence_count"], dtype=np.int64)
    cosines = np.asarray(table["raw_cosine"], dtype=np.float64)
    candidate_rows = np.searchsorted(candidates, candidate_app).astype(np.int32)
    input_cols = np.searchsorted(apps, neighbor_app).astype(np.int32)
    if (np.any(candidate_rows >= len(candidates)) or
            not np.array_equal(candidates[candidate_rows], candidate_app) or
            np.any(input_cols >= len(apps)) or
            not np.array_equal(apps[input_cols], neighbor_app)):
        raise ValueError("Raw co-occurrence App IDs differ from fixed mappings")
    if np.any(counts <= 0) or np.any(~np.isfinite(cosines)) or np.any(cosines <= 0):
        raise ValueError("Raw train co-occurrence contains invalid counts or cosine values")
    if np.any(candidate_rows[1:] < candidate_rows[:-1]):
        raise ValueError("Raw co-occurrence is not grouped by ascending candidate ID")
    starts = np.flatnonzero(np.r_[True, candidate_rows[1:] != candidate_rows[:-1]])
    ends = np.r_[starts[1:], len(candidate_rows)]
    for start, end in zip(starts, ends):
        if np.any(input_cols[start+1:end] <= input_cols[start:end-1]):
            raise ValueError("Raw co-occurrence row is not unique and App-ID sorted")
    del table, candidate_app, neighbor_app
    gc.collect()
    return candidate_rows, input_cols, counts, cosines


def _csr_from_rows(rows: list[np.ndarray], vals: list[np.ndarray], shape,
                   dtype=np.float32):
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    if rows:
        lengths = np.fromiter((len(x) for x in rows), count=len(rows), dtype=np.int64)
        indptr[1:] = np.cumsum(lengths, dtype=np.int64)
        if indptr[-1]:
            indices = np.concatenate(rows).astype(np.int32, copy=False)
            data = np.concatenate(vals).astype(dtype, copy=False)
        else:
            indices = np.empty(0, dtype=np.int32)
            data = np.empty(0, dtype=dtype)
    else:
        indices, data = np.empty(0, dtype=np.int32), np.empty(0, dtype=dtype)
    out = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=dtype)
    out.sort_indices()
    return out


def _select_one_row(cols, counts, cosines, app_ids, *, limit=100, min_count=1,
                    selection="cosine", selection_scores=None, stored_scores=None):
    keep = (counts >= min_count) & (cols >= 0)
    if not np.any(keep):
        return np.empty(0, np.int32), np.empty(0, np.float32)
    c = cols[keep]
    co = counts[keep]
    cos = cosines[keep]
    stored = cos if stored_scores is None else stored_scores[keep]
    if selection_scores is None:
        if selection == "cosine":
            key = cos
        elif selection == "cooccurrence":
            key = co
        else:
            key = selection_scores
    else:
        key = selection_scores[keep]
    if selection == "cooccurrence":
        order = np.lexsort((app_ids[c], -cos, -co))
    else:
        order = np.lexsort((app_ids[c], -key))
    chosen = order[:limit]
    storage = np.argsort(c[chosen], kind="stable")
    chosen = chosen[storage]
    return c[chosen].astype(np.int32, copy=False), stored[chosen].astype(np.float32, copy=False)


def _candidate_oriented_matrices(candidate_rows, input_cols, counts, cosines,
                                 apps, candidates, support):
    n_edges = len(candidate_rows)
    boundaries = np.flatnonzero(np.r_[True, candidate_rows[1:] != candidate_rows[:-1]])
    ends = np.r_[boundaries[1:], n_edges]
    row_groups = {int(r): (int(s), int(e)) for r, s, e in
                  zip(candidate_rows[boundaries], boundaries, ends)}
    output = {}
    min_specs = {
        f"min_cooccurrence_{minimum}_cosine_k100": (minimum, "cosine", None)
        for minimum in (2, 3, 5)
    }
    min_specs["candidate_oriented_cosine_k100"] = (1, "cosine", None)
    min_specs["cooccurrence_selected_cosine_scored_k100"] = (1, "cooccurrence", None)
    for alpha in (.25, .75):
        for locality in (1, 2):
            key = f"asymmetric_cosine_alpha{f'{alpha:.2f}'.replace('.', '')}_locality{locality}_k100"
            min_specs[key] = (1, "custom", (alpha, locality))

    row_data: dict[str, list[np.ndarray]] = {key: [] for key in min_specs}
    row_indices: dict[str, list[np.ndarray]] = {key: [] for key in min_specs}
    nnz_all = []
    idx_all = []
    indptr_all = np.zeros(len(candidates) + 1, dtype=np.int64)
    for row in range(len(candidates)):
        span = row_groups.get(row)
        if span is None:
            cols = np.empty(0, dtype=np.int32)
            co = np.empty(0, dtype=np.int64)
            cos = np.empty(0, dtype=np.float64)
        else:
            s, e = span
            cols, co, cos = input_cols[s:e], counts[s:e], cosines[s:e]
        self_col = int(np.searchsorted(apps, candidates[row]))
        nonself = cols != self_col
        cols, co, cos = cols[nonself], co[nonself], cos[nonself]
        for name, (minimum, selection, custom) in min_specs.items():
            if custom is None:
                out_cols, out_values = _select_one_row(
                    cols, co, cos, apps, min_count=minimum, selection=selection)
            else:
                alpha, locality = custom
                score = co.astype(np.float64) / (
                    float(support[self_col]) ** alpha
                    * support[cols].astype(np.float64) ** (1.0-alpha))
                score = np.power(score, locality)
                out_cols, out_values = _select_one_row(
                    cols, co, cos, apps, selection="custom", selection_scores=score,
                    stored_scores=score)
            row_indices[name].append(out_cols)
            row_data[name].append(out_values)
        # Lift selects using full positive-catalog user count and supports.
        if len(cols):
            lift = co.astype(np.float64) * TRAIN_USER_COUNT / (
                float(support[self_col]) * support[cols].astype(np.float64))
            out_cols, out_values = _select_one_row(
                cols, co, cos, apps, selection="lift", selection_scores=lift)
        else:
            out_cols, out_values = np.empty(0, np.int32), np.empty(0, np.float32)
        row_indices.setdefault("lift_selected_cosine_scored_k100", []).append(out_cols)
        row_data.setdefault("lift_selected_cosine_scored_k100", []).append(out_values)
        # Preserve every nonzero raw co-occurrence for the unpruned cosine matrix.
        if len(cols):
            order = np.argsort(cols, kind="stable")
            nnz_all.append(cols[order].astype(np.int32, copy=False))
            idx_all.append(cos[order].astype(np.float32, copy=False))
        indptr_all[row+1] = indptr_all[row] + (len(cols))
    shape = (len(candidates), len(apps))
    for name in row_indices:
        output[name] = _csr_from_rows(row_indices[name], row_data[name], shape)
    # The unpruned representation is exact over all observed candidate/input pairs.
    unpruned_indices = np.concatenate(nnz_all) if nnz_all else np.empty(0, np.int32)
    unpruned_data = np.concatenate(idx_all) if idx_all else np.empty(0, np.float32)
    output["unpruned_cosine"] = sparse.csr_matrix(
        (unpruned_data, unpruned_indices, indptr_all), shape=shape)
    output["unpruned_cosine"].sort_indices()
    return output


def _input_oriented_matrix(candidate_rows, input_cols, cosines, apps, candidates,
                           shape, neighbors=100):
    nonself = apps[input_cols] != candidates[candidate_rows]
    candidate_rows = candidate_rows[nonself]
    input_cols = input_cols[nonself]
    cosines = cosines[nonself]
    # Rank each input column by cosine, then candidate App ID. The matrix stays
    # in candidate x input orientation for the common scorer.
    order = np.lexsort((candidates[candidate_rows], -cosines, input_cols))
    sorted_cols = input_cols[order]
    starts = np.flatnonzero(np.r_[True, sorted_cols[1:] != sorted_cols[:-1]])
    ends = np.r_[starts[1:], len(order)]
    take = np.zeros(len(order), dtype=bool)
    for start, end in zip(starts, ends):
        take[order[start:min(start+neighbors, end)]] = True
    chosen = np.flatnonzero(take)
    out = sparse.coo_matrix(
        (cosines[chosen].astype(np.float32),
         (candidate_rows[chosen], input_cols[chosen])), shape=shape).tocsr()
    out.sum_duplicates()
    out.sort_indices()
    return out


def build_graph_models() -> list[str]:
    initialize_registry()
    sources = _source_manifest()
    apps, candidates, _, support = _catalog()
    candidate_rows, input_cols, counts, cosines = _edge_arrays()
    outputs = _candidate_oriented_matrices(
        candidate_rows, input_cols, counts, cosines, apps, candidates, support)
    lift_name = "lift_selected_cosine_scored_k100"
    # Lift is selected independently, then scored by cosine.
    # _candidate_oriented_matrices stores cosine for this selection.
    input_matrix = _input_oriented_matrix(
        candidate_rows, input_cols, cosines, apps, candidates,
        (len(candidates), len(apps)))
    outputs["input_oriented_cosine_k100"] = input_matrix
    outputs["candidate_input_union_cosine_k100"] = (
        outputs["candidate_oriented_cosine_k100"].maximum(input_matrix)).tocsr()
    outputs["candidate_input_union_cosine_k100"].sort_indices()
    names = []
    roster = _variant_roster()
    for name, matrix in outputs.items():
        params = roster[name]["parameters"]
        folder = _write_model(name, matrix, params, sources)
        _update_registry(name, "complete", name)
        log(f"Saved train-only graph model {name}: {matrix.shape}, {matrix.nnz:,} edges")
        names.append(name)
    del outputs, candidate_rows, input_cols, counts, cosines
    gc.collect()
    return names


def _available_memory_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().available) / (1024**3)
    except Exception:
        return float("inf")


def _require_memory(stage: str, reserve_gb: float = MAX_BUILD_GB) -> None:
    available = _available_memory_gb()
    if available < reserve_gb:
        raise MemoryError(f"{stage} requires a resource window; available RAM is {available:.2f} GB "
                          f"(minimum {reserve_gb:.2f} GB)")


def _sorted_train_positives(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    sorted_path = work_dir / "positive_train_sorted.parquet"
    if sorted_path.exists():
        return sorted_path
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")
        con.execute(f"SET temp_directory={quote(work_dir / 'duckdb_temp')}")
        con.execute("SET max_temp_directory_size='30GB'")
        con.execute(f"""COPY (
            SELECT user_key,app_id FROM read_parquet({quote(POSITIVE_TRAIN)})
            WHERE positive ORDER BY user_key,app_id
        ) TO {quote(sorted_path)}
        (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)""")
    finally:
        con.close()
    return sorted_path


def _positive_csr(work_dir: Path):
    """Build a full train-positive CSR, retaining all singleton users."""
    _require_memory("train-positive sparse matrix", 2.5)
    apps, candidates, _, expected_support = _catalog()
    sorted_path = _sorted_train_positives(work_dir)
    split_manifest = json.loads((SPLIT_ROOT / "manifest.json").read_text("utf-8"))
    counts = split_manifest["validation"]["counts"]
    pf = pq.ParquetFile(sorted_path)
    matrix, support, users = _build_user_item_csr(
        pf.iter_batches(batch_size=SORT_BATCH_SIZE, columns=["user_key", "app_id"]),
        apps, counts["train_positive_catalog_pairs"], counts["positive_train_accounts"])
    pf.close()
    if users != TRAIN_USER_COUNT or not np.array_equal(support, expected_support):
        raise ValueError("Train-positive CSR does not reproduce frozen v1 support")
    return matrix, support, users, apps, candidates


def _build_weighted_user_item_csr(sorted_path: Path, apps: np.ndarray,
                                  expected_pairs: int, expected_users: int,
                                  weight_column: str):
    """Build a sorted weighted CSR without a Python user-ID dictionary."""
    pf = pq.ParquetFile(sorted_path)
    if pf.metadata.num_rows != expected_pairs:
        raise ValueError(f"Weighted train pair count differs: {weight_column}")
    indices = np.empty(expected_pairs, dtype=np.int32)
    values = np.empty(expected_pairs, dtype=np.float32)
    indptr = np.zeros(expected_users + 1, dtype=np.int32)
    pair_count = user_count = 0
    pending_user = None
    previous_user = previous_app = None
    for batch in pf.iter_batches(batch_size=SORT_BATCH_SIZE,
                                 columns=["user_key", "app_id", weight_column]):
        users = np.asarray(batch.column("user_key").to_numpy(zero_copy_only=False))
        app_values = np.asarray(batch.column("app_id").to_numpy(zero_copy_only=False))
        weights = np.asarray(batch.column(weight_column).to_numpy(zero_copy_only=False),
                             dtype=np.float32)
        if users.dtype != np.dtype("uint64") or app_values.dtype.kind != "i":
            raise ValueError("Weighted training IDs have invalid types")
        if np.any(users[1:] < users[:-1]):
            raise ValueError("Weighted train batches are not sorted by user key")
        if previous_user is not None:
            if users[0] < previous_user:
                raise ValueError("Weighted train batches are not globally sorted")
            if users[0] == previous_user and app_values[0] <= previous_app:
                raise ValueError("Weighted training has duplicate or unsorted pairs")
        same_user = users[1:] == users[:-1]
        if np.any(app_values[1:][same_user] <= app_values[:-1][same_user]):
            raise ValueError("Weighted training has duplicate user/App ID pairs")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Every weighted positive train pair must have positive finite weight")
        item_rows = np.searchsorted(apps, app_values)
        if np.any(item_rows >= len(apps)) or not np.array_equal(apps[item_rows], app_values):
            raise ValueError("Weighted training pair falls outside fixed input catalog")
        end = pair_count + batch.num_rows
        if end > expected_pairs:
            raise ValueError("Weighted training pair count exceeds pinned split")
        indices[pair_count:end] = item_rows.astype(np.int32, copy=False)
        values[pair_count:end] = weights
        group_ends = np.flatnonzero(users[1:] != users[:-1]) + 1
        group_ends = np.append(group_ends, batch.num_rows)
        if pending_user is not None and users[0] != pending_user:
            indptr[user_count + 1] = pair_count
            user_count += 1
        completed_groups = len(group_ends) - 1
        if user_count + completed_groups > expected_users:
            raise ValueError("Weighted training user count exceeds pinned split")
        if completed_groups:
            indptr[user_count + 1:user_count + completed_groups + 1] = (
                pair_count + group_ends[:completed_groups])
            user_count += completed_groups
        pending_user = int(users[-1])
        previous_user, previous_app = users[-1], app_values[-1]
        pair_count = end
    pf.close()
    if pending_user is not None:
        if user_count >= expected_users:
            raise ValueError("Weighted training user count exceeds pinned split")
        indptr[user_count + 1] = pair_count
        user_count += 1
    if pair_count != expected_pairs or user_count != expected_users:
        raise ValueError(f"Weighted training CSR count mismatch: {pair_count:,} pairs, "
                         f"{user_count:,} users")
    matrix = sparse.csr_matrix((values, indices, indptr),
                               shape=(expected_users, len(apps)), copy=False)
    if matrix.nnz != expected_pairs or not matrix.has_canonical_format:
        raise ValueError("Weighted CSR integrity check failed")
    return matrix


def _write_weighted_pair_source(work_dir: Path) -> tuple[Path, dict]:
    """Hydrate only accepted TRAIN lineage from canonical Kieran columns."""
    work_dir.mkdir(parents=True, exist_ok=True)
    weighted_path = work_dir / "accepted_train_positive_pair_weights.parquet"
    canonical_manifest_path = CANONICAL_ROOT / "ingestion_manifest.json"
    canonical_manifest = json.loads(canonical_manifest_path.read_text("utf-8"))
    if canonical_manifest.get("status") != "complete":
        raise ValueError("Canonical Kieran source is not complete")
    shard_names = canonical_manifest["published_parquet_shards"]
    shard_paths = [CANONICAL_ROOT / name for name in shard_names]
    if len(shard_paths) != 46 or any(not path.is_file() for path in shard_paths):
        raise ValueError("Canonical Kieran shard inventory differs from pinned ingestion manifest")
    sources = {
        "canonical_ingestion_manifest": _source_info(canonical_manifest_path),
        "canonical_shards": [_source_info(path) for path in shard_paths],
    }
    if weighted_path.exists():
        try:
            if pq.ParquetFile(weighted_path).metadata.num_rows == 78_315_268:
                sources["accepted_train_pair_weights"] = _source_info(weighted_path)
                return weighted_path, sources
        except Exception:
            pass
        _assert_under_model_root(weighted_path)
        weighted_path.unlink(missing_ok=True)
    work_temp = work_dir / "duckdb_weight_temp"
    work_temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{FEATURE_DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory={quote(work_temp)}")
        con.execute("SET max_temp_directory_size='30GB'")
        raw_paths = "[" + ",".join(quote(path) for path in shard_paths) + "]"
        con.execute(f"""CREATE TEMP VIEW accepted_train_lineage AS
            SELECT l.user_key,l.app_id,l.review_key
            FROM read_parquet({quote(LINEAGE_TRAIN)}) l
            JOIN read_parquet({quote(SPLIT_ROOT / 'catalog/input_apps.parquet')}) a USING(app_id)
            WHERE l.positive""")
        con.execute(f"""CREATE TEMP TABLE pair_events AS
            SELECT l.user_key,l.app_id,
                max(try_cast(r.timestamp_created AS BIGINT)) AS timestamp_created,
                max(try_cast(r.playtime_forever AS DOUBLE)) AS playtime_forever
            FROM accepted_train_lineage l
            JOIN read_parquet({raw_paths}, union_by_name=true) r
              ON try_cast(r.user_id AS UBIGINT)=l.user_key
             AND try_cast(r.review_id AS UBIGINT)=l.review_key
             AND r.app_id=l.app_id
            GROUP BY l.user_key,l.app_id""")
        pair_count, user_count, missing_time = con.execute("""SELECT count(*),
            count(DISTINCT user_key),
            count(*) FILTER(WHERE timestamp_created IS NULL OR timestamp_created<0)
            FROM pair_events""").fetchone()
        if pair_count != 78_315_268 or user_count != TRAIN_USER_COUNT:
            raise ValueError(f"Canonical train lineage join produced {pair_count:,} pairs/"
                             f"{user_count:,} users; expected 78,315,268/{TRAIN_USER_COUNT:,}")
        if missing_time:
            raise ValueError(f"Train time weights cannot fit {missing_time:,} missing/invalid timestamps")
        latest_time = con.execute("SELECT max(timestamp_created) FROM pair_events").fetchone()[0]
        half_life_seconds = 3 * 365 * 24 * 60 * 60
        con.execute(f"""COPY (
            SELECT user_key,app_id,
                cast(pow(0.5,({int(latest_time)} - timestamp_created)::DOUBLE/{half_life_seconds})
                    AS REAL) AS time_weight,
                cast(CASE WHEN playtime_forever IS NULL THEN 0.5
                    ELSE 0.5 + percent_rank() OVER(
                        PARTITION BY app_id
                        ORDER BY ln(1+greatest(playtime_forever,0)))
                    END AS REAL) AS playtime_weight
            FROM pair_events
            ORDER BY user_key,app_id
        ) TO {quote(weighted_path)}
        (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)""")
    finally:
        con.close()
    if pq.ParquetFile(weighted_path).metadata.num_rows != 78_315_268:
        raise ValueError("Accepted train pair weight artifact has the wrong row count")
    sources["accepted_train_pair_weights"] = _source_info(weighted_path)
    return weighted_path, sources


def _weighted_candidate_cosine(x_csc: sparse.csc_matrix, apps: np.ndarray,
                               candidates: np.ndarray, neighbors=100) -> sparse.csr_matrix:
    """Compute exact candidate-by-input cosine for weighted interaction vectors."""
    _require_memory("weighted train item graph", 2.0)
    xt = x_csc.T.tocsr()
    norms = np.asarray(x_csc.power(2).sum(axis=0)).ravel().astype(np.float64)
    candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
    out_indices, out_values = [], []
    indptr = np.zeros(len(candidates)+1, dtype=np.int64)
    for block_start in range(0, len(candidates), BLOCK_SIZE):
        block_stop = min(block_start + BLOCK_SIZE, len(candidates))
        counts = (xt[candidate_cols[block_start:block_stop], :] @ x_csc).tocsr()
        counts.sum_duplicates()
        for local in range(block_stop - block_start):
            row = block_start + local
            start, end = counts.indptr[local:local+2]
            cols = counts.indices[start:end].astype(np.int32, copy=False)
            vals = counts.data[start:end].astype(np.float64, copy=False)
            candidate_col = int(candidate_cols[row])
            keep = (cols != candidate_col) & (vals > 0) & (norms[cols] > 0)
            cols, vals = cols[keep], vals[keep]
            score = vals / np.sqrt(norms[candidate_col] * norms[cols])
            out_cols, out_vals = _topk_scores_row(cols, score, apps, candidate_col, neighbors)
            out_indices.append(out_cols)
            out_values.append(out_vals)
            indptr[row+1] = indptr[row] + len(out_cols)
        del counts
    indices = np.concatenate(out_indices) if out_indices else np.empty(0, np.int32)
    data = np.concatenate(out_values) if out_values else np.empty(0, np.float32)
    out = sparse.csr_matrix((data, indices, indptr),
                            shape=(len(candidates), len(apps)), dtype=np.float32)
    out.sort_indices()
    return out


def build_temporal_and_playtime_models() -> list[str]:
    initialize_registry()
    _require_memory("train-only time/playtime weighting", 3.0)
    work_dir = MODEL_ROOT / "_collaborative_work"
    weighted_path, feature_sources = _write_weighted_pair_source(work_dir)
    sources = _source_manifest()
    sources.update(feature_sources)
    apps, candidates, _, _ = _catalog()
    registry = _variant_roster()
    complete = []
    for column, name in (
            ("time_weight", "time_decay_3year_cosine_k100"),
            ("playtime_weight", "playtime_percentile_cosine_k100")):
        matrix_csr = _build_weighted_user_item_csr(
            weighted_path, apps, 78_315_268, TRAIN_USER_COUNT, column)
        matrix = _weighted_candidate_cosine(matrix_csr.tocsc(), apps, candidates)
        del matrix_csr
        gc.collect()
        _write_model(name, matrix, registry[name]["parameters"], sources)
        _update_registry(name, "complete", name)
        log(f"Saved accepted train-only weighted model {name}: {matrix.nnz:,} edges")
        complete.append(name)
        del matrix
        gc.collect()
    return complete


def _topk_scores_row(cols: np.ndarray, values: np.ndarray, apps: np.ndarray,
                     candidate_col: int, neighbors=100):
    keep = (cols != candidate_col) & np.isfinite(values) & (values > 0)
    cols, values = cols[keep], values[keep]
    if not len(cols):
        return np.empty(0, np.int32), np.empty(0, np.float32)
    order = np.lexsort((apps[cols], -values))[:neighbors]
    order = order[np.argsort(cols[order], kind="stable")]
    return cols[order].astype(np.int32), values[order].astype(np.float32)


def _weighted_model_blocked(x_csc: sparse.csc_matrix, weights_by_user: np.ndarray,
                            apps: np.ndarray, candidates: np.ndarray,
                            kind: str, betas=()) -> dict:
    """Weighted train-only candidate/input cosine and standard RP3beta matrices."""
    _require_memory(kind, 2.0)
    weighted = x_csc.astype(np.float32, copy=True)
    # Square-root edge scaling yields a pair contribution of w_u and a
    # weighted-cosine marginal of sum_u(w_u), matching the declared reviewer
    # activity weight.
    weighted.data = np.sqrt(np.asarray(weights_by_user[weighted.indices], dtype=np.float32))
    if kind == "activity_inverse_sqrt":
        weighted_support = np.asarray(weighted.power(2).sum(axis=0)).ravel().astype(np.float64)
    else:
        weighted_support = np.asarray(weighted.power(2).sum(axis=0)).ravel().astype(np.float64)
    positive_support = np.asarray(x_csc.sum(axis=0)).ravel().astype(np.float64)
    weighted_xt = weighted.T.tocsr()
    candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
    shape = (len(candidates), len(apps))
    names = []
    if kind == "activity_inverse_sqrt":
        names = [("activity_inverse_sqrt_degree_cosine_k100", None)]
    elif kind == "activity_inverse_degree":
        names = [("activity_inverse_degree_cosine_k100", None)] + [
            (f"rp3beta_alpha1_beta{int(beta*10):02d}_k100", beta) for beta in betas]
    rows_by_name = {name: [] for name, _ in names}
    values_by_name = {name: [] for name, _ in names}
    for block_start in range(0, len(candidates), BLOCK_SIZE):
        block_stop = min(block_start + BLOCK_SIZE, len(candidates))
        rows = weighted_xt[candidate_cols[block_start:block_stop], :]
        cooc = (rows @ weighted).tocsr()
        cooc.sum_duplicates()
        for local in range(block_stop - block_start):
            global_row = block_start + local
            start, end = cooc.indptr[local:local+2]
            cols = cooc.indices[start:end].astype(np.int32, copy=False)
            weighted_counts = cooc.data[start:end].astype(np.float64, copy=False)
            candidate_col = int(candidate_cols[global_row])
            valid = ((cols != candidate_col) & (weighted_counts > 0)
                     & (weighted_support[cols] > 0))
            cols = cols[valid]
            weighted_counts = weighted_counts[valid]
            for name, beta in names:
                if beta is None:
                    score = weighted_counts / np.sqrt(
                        weighted_support[candidate_col] * weighted_support[cols])
                else:
                    # RP3beta: item->user probability 1/support_source; user->item
                    # probability 1/user_degree (alpha=1); penalize destination popularity.
                    score = weighted_counts / (
                        positive_support[cols] * positive_support[candidate_col] ** beta)
                out_cols, out_values = _topk_scores_row(
                    cols, score, apps, candidate_col, neighbors=100)
                rows_by_name[name].append(out_cols)
                values_by_name[name].append(out_values)
        del rows, cooc
    result = {}
    for name, beta in names:
        matrix = _csr_from_rows(rows_by_name[name], values_by_name[name], shape)
        result[name] = matrix
    del weighted, weighted_xt
    gc.collect()
    return result


def build_activity_and_rp3_models() -> list[str]:
    initialize_registry()
    _require_memory("activity-weighted item graphs", 3.0)
    work_dir = MODEL_ROOT / "_collaborative_work"
    x_csr, support, users, apps, candidates = _positive_csr(work_dir)
    degree = np.diff(x_csr.indptr).astype(np.float64)
    if len(degree) != users or np.any(degree <= 0):
        raise ValueError("Positive user degrees must include every singleton train account")
    x_csc = x_csr.tocsc()
    del x_csr
    gc.collect()
    sources = _source_manifest()
    outputs = {}
    roster = _variant_roster()
    sources = _source_manifest()
    sqrt_outputs = _weighted_model_blocked(
        x_csc, 1.0 / np.sqrt(degree), apps, candidates, "activity_inverse_sqrt")
    for name, matrix in sqrt_outputs.items():
        _write_model(name, matrix, roster[name]["parameters"], sources)
        _update_registry(name, "complete", name)
        log(f"Saved train-only weighted graph {name}: {matrix.nnz:,} edges")
    del sqrt_outputs
    gc.collect()
    outputs = _weighted_model_blocked(
        x_csc, 1.0 / degree, apps, candidates, "activity_inverse_degree",
        betas=(.3,.6))
    completed = list(outputs)
    for name, matrix in outputs.items():
        _write_model(name, matrix, roster[name]["parameters"], sources)
        _update_registry(name, "complete", name)
        log(f"Saved train-only weighted graph {name}: {matrix.nnz:,} edges")
    del outputs, x_csc, degree
    gc.collect()
    return completed


def _signed_train_matrices(work_dir: Path):
    """Build aligned positive/negative matrices on train-positive user rows only."""
    work_dir.mkdir(parents=True, exist_ok=True)
    users_path = work_dir / "positive_train_user_ids.parquet"
    negative_path = work_dir / "negative_train_positive_users.parquet"
    negative_support_path = work_dir / "negative_train_item_support.parquet"
    input_path = SPLIT_ROOT / "catalog/input_apps.parquet"
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")
        temp_dir = work_dir / "duckdb_signed_temp"
        temp_dir.mkdir(exist_ok=True)
        con.execute(f"SET temp_directory={quote(temp_dir)}")
        con.execute("SET max_temp_directory_size='30GB'")
        if not users_path.exists():
            con.execute(f"""COPY (
                SELECT DISTINCT user_key FROM read_parquet({quote(POSITIVE_TRAIN)})
                ORDER BY user_key
            ) TO {quote(users_path)}
            (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)""")
        if not negative_path.exists():
            con.execute(f"""COPY (
                SELECT t.user_key,t.app_id
                FROM read_parquet({quote(INTERACTIONS_TRAIN)}) t
                JOIN read_parquet({quote(input_path)}) a USING(app_id)
                SEMI JOIN read_parquet({quote(users_path)}) u USING(user_key)
                WHERE NOT t.positive
                ORDER BY t.user_key,t.app_id
            ) TO {quote(negative_path)}
            (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)""")
        if not negative_support_path.exists():
            con.execute(f"""COPY (
                SELECT t.app_id,count(*)::BIGINT AS negative_support
                FROM read_parquet({quote(INTERACTIONS_TRAIN)}) t
                JOIN read_parquet({quote(input_path)}) a USING(app_id)
                WHERE NOT t.positive
                GROUP BY t.app_id ORDER BY t.app_id
            ) TO {quote(negative_support_path)}
            (FORMAT PARQUET,COMPRESSION ZSTD)""")
        negative_pairs = pq.ParquetFile(negative_path).metadata.num_rows
        user_ids = np.asarray(pq.read_table(users_path)["user_key"].to_numpy(),
                              dtype=np.uint64)
        if len(user_ids) != TRAIN_USER_COUNT or np.any(user_ids[1:] <= user_ids[:-1]):
            raise ValueError("Positive-user row mapping differs from train-positive split")
        neg_table = pq.read_table(negative_path)
        neg_users = np.asarray(neg_table["user_key"].to_numpy(), dtype=np.uint64)
        neg_apps = np.asarray(neg_table["app_id"].to_numpy(), dtype=np.int64)
        apps, candidates, _, positive_support = _catalog()
        user_rows = np.searchsorted(user_ids, neg_users).astype(np.int32)
        if (np.any(user_rows >= len(user_ids))
                or not np.array_equal(user_ids[user_rows], neg_users)):
            raise ValueError("Negative interactions could not be aligned to positive train users")
        item_cols = np.searchsorted(apps, neg_apps).astype(np.int32)
        if (np.any(item_cols >= len(apps))
                or not np.array_equal(apps[item_cols], neg_apps)):
            raise ValueError("Negative train interaction falls outside fixed input catalog")
        negative_csr = sparse.coo_matrix(
            (np.ones(negative_pairs, dtype=np.uint8), (user_rows, item_cols)),
            shape=(TRAIN_USER_COUNT, len(apps))).tocsr()
        negative_csr.sum_duplicates()
        if negative_csr.nnz != negative_pairs or np.any(negative_csr.data != 1):
            raise ValueError("Train-negative interactions contain duplicate pairs")
        support_rows = pq.read_table(negative_support_path).to_pydict()
        negative_support = np.zeros(len(apps), dtype=np.int64)
        support_apps = np.asarray(support_rows["app_id"], dtype=np.int64)
        support_cols = np.searchsorted(apps, support_apps)
        if np.any(support_cols >= len(apps)) or not np.array_equal(apps[support_cols], support_apps):
            raise ValueError("Negative support artifact contains a non-catalog App ID")
        negative_support[support_cols] = np.asarray(
            support_rows["negative_support"], dtype=np.int64)
    finally:
        con.close()
    del neg_table, neg_users, neg_apps, user_rows, item_cols
    gc.collect()
    return negative_csr, user_ids, negative_support, positive_support, apps, candidates


def _select_signed_row(cols: np.ndarray, values: np.ndarray,
                       apps: np.ndarray, candidate_col: int, neighbors=100):
    keep = (cols != candidate_col) & np.isfinite(values) & (values != 0)
    cols, values = cols[keep], values[keep]
    if not len(cols):
        return np.empty(0, np.int32), np.empty(0, np.float32)
    order = np.lexsort((apps[cols], -np.abs(values)))[:neighbors]
    order = order[np.argsort(cols[order], kind="stable")]
    return cols[order].astype(np.int32), values[order].astype(np.float32)


def _signed_item_matrix(positive_csr: sparse.csr_matrix,
                        negative_csr: sparse.csr_matrix,
                        apps: np.ndarray, candidates: np.ndarray,
                        observed_support: np.ndarray, neighbors=100):
    """Build G=(P'P-P'N-N'P)/sqrt(s_i*s_j), excluding N'N."""
    p_csc = positive_csr.astype(np.float32).tocsc()
    n_csc = negative_csr.astype(np.float32).tocsc()
    pt = p_csc.T.tocsr()
    nt = n_csc.T.tocsr()
    candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
    out_rows, out_values = [], []
    shape = (len(candidates), len(apps))
    indptr = np.zeros(len(candidates)+1, dtype=np.int64)
    for block_start in range(0, len(candidates), BLOCK_SIZE):
        block_stop = min(block_start+BLOCK_SIZE, len(candidates))
        item_rows = candidate_cols[block_start:block_stop]
        pp = (pt[item_rows, :] @ p_csc).tocsr()
        pn = (pt[item_rows, :] @ n_csc).tocsr()
        np_relation = (nt[item_rows, :] @ p_csc).tocsr()
        signed = pp - pn - np_relation
        signed.sum_duplicates()
        signed.eliminate_zeros()
        for local in range(block_stop-block_start):
            global_row = block_start+local
            start,end = signed.indptr[local:local+2]
            cols = signed.indices[start:end].astype(np.int32,copy=False)
            counts = signed.data[start:end].astype(np.float64,copy=False)
            candidate_col = int(candidate_cols[global_row])
            keep = ((cols != candidate_col) & (observed_support[cols] > 0)
                    & (observed_support[candidate_col] > 0))
            cols,counts=cols[keep],counts[keep]
            values=counts/np.sqrt(
                float(observed_support[candidate_col])*observed_support[cols])
            out_cols,out_vals=_select_signed_row(
                cols,values,apps,candidate_col,neighbors)
            out_rows.append(out_cols);out_values.append(out_vals)
            indptr[global_row+1]=indptr[global_row]+len(out_cols)
        del pp,pn,np_relation,signed
    indices=np.concatenate(out_rows) if out_rows else np.empty(0,np.int32)
    data=np.concatenate(out_values) if out_values else np.empty(0,np.float32)
    result=sparse.csr_matrix((data,indices,indptr),shape=shape,dtype=np.float32)
    result.sort_indices()
    return result


def build_signed_relation_models() -> list[str]:
    """Fit the no-negative-negative signed item relation and train prior."""
    initialize_registry()
    _require_memory("positive/negative train relations", 3.0)
    work_dir = MODEL_ROOT / "_collaborative_work"
    negative_csr,user_ids,negative_support,positive_support,apps,candidates = (
        _signed_train_matrices(work_dir))
    positive_sorted = _sorted_train_positives(work_dir)
    split_manifest = json.loads((SPLIT_ROOT / "manifest.json").read_text("utf-8"))
    counts = split_manifest["validation"]["counts"]
    positive_csr,verified_support,users = _build_user_item_csr(
        pq.ParquetFile(positive_sorted).iter_batches(
            batch_size=SORT_BATCH_SIZE,columns=["user_key","app_id"]),
        apps,counts["train_positive_catalog_pairs"],counts["positive_train_accounts"])
    if users != TRAIN_USER_COUNT or not np.array_equal(verified_support,positive_support):
        raise ValueError("Signed relation positive matrix differs from frozen train positives")
    observed_support=positive_support+negative_support
    prior=(positive_support.astype(np.float64)+1.)/(observed_support.astype(np.float64)+2.)
    matrix=_signed_item_matrix(positive_csr,negative_csr,apps,candidates,observed_support)
    sources=_source_manifest()
    users_path=work_dir/"positive_train_user_ids.parquet"
    negative_path=work_dir/"negative_train_positive_users.parquet"
    negative_support_path=work_dir/"negative_train_item_support.parquet"
    sources.update({
        "positive_user_row_mapping":_source_info(users_path),
        "negative_train_input_pairs":_source_info(negative_path),
        "negative_train_item_support":_source_info(negative_support_path),
    })
    roster=_variant_roster()
    names=[]
    for beta in (.1,.3):
        name=f"positive_negative_relation_beta{int(beta*100):02d}"
        _write_model(name,matrix,roster[name]["parameters"],sources,
                     extra_arrays={"train_positive_rate.npy":prior.astype(np.float32)},
                     artifact_type="signed_item_similarity",scorer_mode="signed_knn")
        _update_registry(name,"complete",name)
        names.append(name)
    del positive_csr,negative_csr,matrix
    gc.collect()
    return names


def _write_pending_models(registry: dict) -> None:
    for name, entry in registry["models"].items():
        if entry["status"] != "complete":
            model_dir = MODEL_ROOT / name
            if not model_dir.exists():
                entry["status"] = "pending_resource_window" if (
                    name.startswith("ease_") or name.startswith("ials_")
                    or name.startswith("metadata_ridge_")) else "planned"
    _json_write(MODEL_ROOT / "registry.json", registry)


def build_ials_models() -> list[str]:
    """Fit full train-positive iALS for degree>=2 users; save item factors only."""
    initialize_registry()
    _require_memory("implicit ALS rank 32/64", 5.0)
    try:
        from implicit.cpu.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits
    except Exception as error:
        raise RuntimeError(f"implicit CPU ALS is unavailable: {type(error).__name__}: {error}") from error
    work_dir = MODEL_ROOT / "_collaborative_work"
    x_csr, support, users, apps, candidates = _positive_csr(work_dir)
    degree = np.diff(x_csr.indptr)
    fit_rows = degree >= 2
    fit_users = int(fit_rows.sum())
    fit_matrix = x_csr[fit_rows].astype(np.float32).tocsr()
    fit_matrix.sort_indices()
    train_pair_count = int(x_csr.nnz)
    fit_pair_count = int(fit_matrix.nnz)
    singleton_user_count = int(np.count_nonzero(degree == 1))
    del x_csr, support, candidates, degree, fit_rows
    gc.collect()
    if fit_users != 12_794_791:
        log(f"iALS train-user count differs from the backlog estimate: {fit_users:,}")
    sources = _source_manifest()
    roster = _variant_roster()
    names = []
    for factors in (32, 64):
        name = f"ials_rank{factors}_reg0_1_conf20_iter10_seed20261108"
        existing_folder = MODEL_ROOT / name
        if existing_folder.exists():
            manifest_path = existing_folder / "manifest.json"
            if not manifest_path.is_file():
                raise FileExistsError(f"Refusing to overwrite incomplete iALS directory: {name}")
            existing_manifest = json.loads(manifest_path.read_text("utf-8"))
            if (existing_manifest.get("status") != "complete"
                    or existing_manifest.get("parameters") != roster[name]["parameters"]):
                raise ValueError(f"Existing iALS model differs from frozen parameters: {name}")
            for relative, info in existing_manifest.get("files", {}).items():
                if digest(existing_folder / relative) != info["sha256"]:
                    raise ValueError(f"Existing iALS artifact fingerprint changed: {name}/{relative}")
            names.append(name)
            log(f"Reused fingerprint-verified train-only iALS rank {factors}")
            continue
        if factors == 64:
            _require_memory("implicit ALS rank 64", 5.0)
        params = roster[name]["parameters"]
        model = AlternatingLeastSquares(
            factors=factors, regularization=.1, alpha=20, iterations=10,
            random_state=20261108, num_threads=2)
        log(f"Fitting iALS rank {factors} on {fit_users:,} degree>=2 train users")
        # implicit uses the model's two OpenMP workers. Keep BLAS serial to
        # avoid nested OpenMP/BLAS pools during the ALS updates.
        with threadpool_limits(limits={"blas": 1, "openmp": 2}):
            model.fit(fit_matrix, show_progress=True)
        item_factors = np.asarray(model.item_factors, dtype=np.float32)
        if item_factors.shape != (len(apps), factors) or not np.all(np.isfinite(item_factors)):
            raise ValueError(f"iALS rank-{factors} returned invalid train item factors")
        fit_summary = {
            "positive_train_users_in_catalog": int(users),
            "positive_train_users_degree_at_least_2": fit_users,
            "positive_train_pairs_catalog": train_pair_count,
            "positive_train_pairs_fit": fit_pair_count,
            "singleton_users_retained_for_item_support": singleton_user_count,
        }
        _write_model(name, item_factors, params, sources,
                     artifact_type="item_factors", scorer_mode="factors",
                     fit_summary=fit_summary)
        _update_registry(name, "complete", name)
        log(f"Saved train-only iALS rank {factors} item factors")
        names.append(name)
        del item_factors, model
        gc.collect()
    del fit_matrix
    gc.collect()
    return names


def build_metadata_ridge_models() -> list[str]:
    """Regress public tag/genre TF-IDF features to train-fitted iALS factors."""
    initialize_registry()
    from sklearn.linear_model import Ridge
    apps, _, _, support = _catalog()
    metadata_manifest = json.loads((METADATA_ROOT / "manifest.json").read_text("utf-8"))
    if metadata_manifest.get("status") != "complete":
        raise ValueError("The v2 metadata feature catalog is not complete")
    metadata_rows = pq.read_table(METADATA_ROOT / "game_rows.parquet").to_pydict()
    metadata_apps = np.asarray(metadata_rows["app_id"], dtype=np.int64)
    if not np.array_equal(metadata_apps, apps):
        raise ValueError("Metadata feature rows differ from the frozen input game map")
    features = sparse.load_npz(METADATA_ROOT / "tfidf_csr.npz").tocsr()
    if features.shape[0] != len(apps):
        raise ValueError("Metadata feature matrix has the wrong fixed-catalog row count")
    metadata_sources = {
        "metadata_manifest": _source_info(METADATA_ROOT / "manifest.json"),
        "metadata_rows": _source_info(METADATA_ROOT / "game_rows.parquet"),
        "metadata_tfidf": _source_info(METADATA_ROOT / "tfidf_csr.npz"),
    }
    sources = _source_manifest()
    sources.update(metadata_sources)
    roster = _variant_roster()
    names = []
    train_rows = np.flatnonzero(support > 0)
    for factors in (32, 64):
        als_name = f"ials_rank{factors}_reg0_1_conf20_iter10_seed20261108"
        als_folder = MODEL_ROOT / als_name
        als_manifest = json.loads((als_folder / "manifest.json").read_text("utf-8"))
        if als_manifest.get("status") != "complete":
            raise ValueError(f"Metadata ridge requires completed train-only factors: {als_name}")
        als_factor_path = als_folder / "item_factors.npy"
        if digest(als_factor_path) != als_manifest["files"]["item_factors.npy"]["sha256"]:
            raise ValueError(f"iALS item-factor fingerprint changed: {als_name}")
        targets = np.load(als_factor_path, mmap_mode="r")
        if targets.shape != (len(apps), factors):
            raise ValueError(f"iALS item factor mapping differs for rank {factors}")
        ridge = Ridge(alpha=100, fit_intercept=True, solver="lsqr",
                      tol=1e-4, max_iter=1000)
        ridge.fit(features[train_rows], np.asarray(targets[train_rows], dtype=np.float32))
        predicted = np.asarray(ridge.predict(features), dtype=np.float32)
        for blend in (0.0, 0.5, 1.0):
            name = f"metadata_ridge_alpha100_rank{factors}_blend{str(blend).replace('.', '')}"
            factors_out = predicted.copy()
            factors_out[train_rows] = (
                (1.0-blend)*np.asarray(targets[train_rows], dtype=np.float32)
                + blend*predicted[train_rows])
            model_sources = dict(sources)
            model_sources["target_iALS_item_factors"] = _source_info(als_factor_path)
            model_sources["target_iALS_manifest"] = _source_info(als_folder / "manifest.json")
            _write_model(
                name, factors_out, roster[name]["parameters"], model_sources,
                extra_arrays={
                    "ridge_coefficients.npy": np.asarray(ridge.coef_, dtype=np.float32),
                    "ridge_intercept.npy": np.asarray(ridge.intercept_, dtype=np.float32),
                },
                artifact_type="item_factors", scorer_mode="factors",
                fit_summary={"ridge_train_items": int(len(train_rows)),
                             "ridge_feature_count": int(features.shape[1]),
                             "ridge_prediction_rows": int(len(apps)),
                             "supported_item_blend": blend})
            _update_registry(name, "complete", name)
            names.append(name)
            log(f"Saved metadata-to-iALS ridge rank {factors}, supported blend {blend:g}")
            del factors_out
        del targets, ridge, predicted
        gc.collect()
    return names


def build_ease_models() -> list[str]:
    """Fit exact full-catalog EASE, then solve only Linux candidate inverse columns."""
    initialize_registry()
    available = _available_memory_gb()
    if available < 10.0:
        detail = f"requires >=10.00 GB available RAM; observed {available:.2f} GB"
        for lam in (100, 1000):
            _update_registry(f"ease_lambda{lam}_exact","pending_resource_window",
                             detail=detail)
        raise MemoryError("Exact EASE was not started: " + detail)
    work_dir = MODEL_ROOT / "_collaborative_work" / "ease"
    work_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_dir).free < 13 * (1024**3):
        detail = "requires at least 13 GiB free disk for the exact Gram, factor and output"
        for lam in (100, 1000):
            _update_registry(f"ease_lambda{lam}_exact","pending_resource_window",
                             detail=detail)
        raise OSError("Exact EASE was not started: " + detail)
    from scipy.linalg.lapack import get_lapack_funcs
    x_csr,support,users,apps,candidates=_positive_csr(
        MODEL_ROOT/"_collaborative_work")
    n_items=len(apps)
    gram_path=work_dir/"full_catalog_gram_f32.dat"
    candidate_cols=np.searchsorted(apps,candidates).astype(np.int32)
    gram_receipt_path=work_dir/"full_catalog_gram_f32.complete.json"
    expected_gram_bytes=n_items*n_items*np.dtype(np.float32).itemsize
    gram_receipt=None
    if (gram_path.is_file() and gram_path.stat().st_size==expected_gram_bytes
            and gram_receipt_path.is_file()):
        try:
            candidate_receipt=json.loads(gram_receipt_path.read_text("utf-8"))
            if (candidate_receipt.get("shape")==[n_items,n_items]
                    and candidate_receipt.get("train_positive_pairs")==int(support.sum())
                    and candidate_receipt.get("complete") is True
                    and digest(gram_path)==candidate_receipt.get("sha256")):
                gram_receipt=candidate_receipt
        except (OSError,ValueError,KeyError):
            gram_receipt=None
    if gram_receipt is not None:
        gram=np.memmap(gram_path,dtype=np.float32,mode="r",
                       shape=(n_items,n_items),order="C")
        del x_csr
        gc.collect()
        log(f"Reusing verified full-catalog EASE Gram: {gram_receipt['sha256']}")
    else:
        x_csc=x_csr.astype(np.float32).tocsc()
        del x_csr
        gc.collect()
        item_rows=x_csc.T.tocsr()
        gram=np.memmap(gram_path,dtype=np.float32,mode="w+",
                       shape=(n_items,n_items),order="C")
        for block_start in range(0,n_items,BLOCK_SIZE):
            block_stop=min(block_start+BLOCK_SIZE,n_items)
            block=(item_rows[block_start:block_stop,:]@x_csc).tocsr()
            dense=block.toarray().astype(np.float32,copy=False)
            gram[block_start:block_stop,:]=dense
            del block,dense
            if block_start==0 or block_stop==n_items or block_start%(BLOCK_SIZE*100)==0:
                log(f"EASE Gram rows {block_stop:,}/{n_items:,}")
        gram.flush()
        del item_rows,x_csc
        gc.collect()
        gram_receipt={
            "complete":True,"shape":[n_items,n_items],
            "train_positive_pairs":int(support.sum()),
            "sha256":digest(gram_path),
        }
        _json_write(gram_receipt_path,gram_receipt)
    source=_source_manifest()
    source["ease_solver"]={
        "gram_scope":"all fixed-catalog items and all positive train users including singletons",
        "gram_precision":"float32",
        "gram_sha256":gram_receipt["sha256"],
    }
    roster=_variant_roster()
    names=[]
    n_candidates=len(candidates)
    indices_per_row=n_items-1
    total_nnz=n_candidates*indices_per_row
    indptr=np.arange(0,total_nnz+1,indices_per_row,dtype=np.int64)
    all_item_cols=np.arange(n_items,dtype=np.int32)
    for regularization in (100,1000):
        name=f"ease_lambda{regularization}_exact"
        factor_path=work_dir/f"ease_lambda{regularization}_cholesky_f32.dat"
        factor=None
        try:
            if gram is None:
                gram=np.memmap(gram_path,dtype=np.float32,mode="r",
                               shape=(n_items,n_items),order="C")
            factor=np.memmap(factor_path,dtype=np.float32,mode="w+",
                             shape=(n_items,n_items),order="F")
            for row_start in range(0,n_items,BLOCK_SIZE):
                row_stop=min(row_start+BLOCK_SIZE,n_items)
                factor[row_start:row_stop,:]=gram[row_start:row_stop,:]
            diag=np.arange(n_items)
            factor[diag,diag]+=np.float32(regularization)
            factor.flush()
            gram._mmap.close()
            del gram
            gram=None
            gc.collect()
            _require_memory(f"EASE lambda={regularization} Cholesky/solve",5.0)
            potrf=get_lapack_funcs("potrf",(factor,))
            factorized,info=potrf(factor,lower=1,overwrite_a=1,clean=0)
            if info!=0:
                raise np.linalg.LinAlgError(f"EASE Cholesky failed at pivot {info}")
            if not np.shares_memory(factor,factorized):
                raise MemoryError("LAPACK copied the full EASE factor matrix")
            factor=factorized
            del factorized
            factor.flush()
            potrs=get_lapack_funcs("potrs",(factor,))
            indices=np.empty(total_nnz,dtype=np.int32)
            data=np.empty(total_nnz,dtype=np.float32)
            candidate_batch=8
            for begin in range(0,n_candidates,candidate_batch):
                end=min(begin+candidate_batch,n_candidates)
                rhs=np.zeros((n_items,end-begin),dtype=np.float32,order="F")
                target_cols=candidate_cols[begin:end]
                rhs[target_cols,np.arange(end-begin)]=1.
                solved,solve_info=potrs(factor,rhs,lower=1,overwrite_b=1)
                if solve_info!=0:
                    raise np.linalg.LinAlgError(f"EASE candidate solve failed: {solve_info}")
                for local,global_row in enumerate(range(begin,end)):
                    target=int(target_cols[local])
                    diagonal=float(solved[target,local])
                    if not np.isfinite(diagonal) or diagonal<=0:
                        raise np.linalg.LinAlgError(
                            "EASE inverse diagonal is not finite and positive")
                    scores=-solved[:,local]/diagonal
                    start=global_row*indices_per_row
                    indices[start:start+target]=all_item_cols[:target]
                    data[start:start+target]=scores[:target]
                    right_len=n_items-target-1
                    if right_len:
                        indices[start+target:start+indices_per_row]=all_item_cols[target+1:]
                        data[start+target:start+indices_per_row]=scores[target+1:]
                del rhs,solved
            matrix=sparse.csr_matrix(
                (data,indices,indptr),shape=(n_candidates,n_items),copy=False)
            matrix.sort_indices()
            params=roster[name]["parameters"]
            _write_model(
                name,matrix,params,source,compressed=False,
                fit_summary={"catalog_items":n_items,"train_positive_users":int(users),
                             "train_positive_pairs":int(support.sum()),
                             "candidate_inverse_columns_solved":n_candidates,
                             "full_inverse_materialized":False})
            _update_registry(name,"complete",name)
            log(f"Saved exact full-catalog EASE lambda {regularization}")
            names.append(name)
            del matrix,data,indices
            gc.collect()
        except Exception as error:
            _update_registry(name,"failed",
                             detail=f"{type(error).__name__}: {error}")
            raise
        finally:
            if factor is not None:
                try:
                    factor.flush()
                except Exception:
                    pass
                del factor
            gc.collect()
            _assert_under_model_root(factor_path)
            factor_path.unlink(missing_ok=True)
    if gram is not None:
        gram._mmap.close()
        del gram
    del indptr,all_item_cols
    gc.collect()
    _assert_under_model_root(gram_path)
    gram_path.unlink(missing_ok=True)
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("registry","graphs","activity","ease","ials",
                        "signed","temporal","ridge"), default="registry")
    parser.add_argument("--force-registry", action="store_true")
    args = parser.parse_args(argv)
    registry = initialize_registry(force=args.force_registry)
    _write_pending_models(registry)
    if args.stage == "registry":
        log(f"Wrote frozen collaborative model registry: {MODEL_ROOT / 'registry.json'}")
        return 0
    funcs = {
        "graphs": build_graph_models,
        "activity": build_activity_and_rp3_models,
        "ease": build_ease_models,
        "ials": build_ials_models,
        "signed": build_signed_relation_models,
        "temporal": build_temporal_and_playtime_models,
        "ridge": build_metadata_ridge_models,
    }
    funcs[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
