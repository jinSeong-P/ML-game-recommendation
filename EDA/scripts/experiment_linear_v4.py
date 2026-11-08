"""Train-only v4 collaborative and linear model builders.

The builders only open the frozen v4 train positives and catalog mappings.
They never read validation/test interactions or targets and never calculate
ranking metrics. Each command builds one small, explicit parameter group so
the full catalog Cholesky factor and candidate weights are not held in RAM at
the same time as multiple other fits.

Author formula references were checked at these repository commits:
* DAN: https://github.com/psm1206/DAN/tree/fc28a9542f141d13f495d5529aa3f2c486961356
  (`code/model.py`, EASE_DAN)
* RLAE/RDLAE: https://github.com/jaewan7599/RDLAE_SIGIR2023/tree/12a89623660cb8514220d146aba40b778d7594f2
  (`strong/code/model.py`, RLAE and RDLAE)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

from eda_large_review_frank import EDA, digest, log, quote
from item_knn import _build_user_item_csr, _top_cosine_neighbors


VERSION = "linear-models-v4"
SPLIT_ROOT = EDA / "data/processed/modeling/experiments_v4/split"
MODEL_ROOT = EDA / "data/processed/modeling/experiments_v4/models"
POSITIVE_TRAIN = SPLIT_ROOT / "train/positive_catalog.parquet"
INPUT_CATALOG = SPLIT_ROOT / "catalog/input_apps.parquet"
CANDIDATE_CATALOG = SPLIT_ROOT / "catalog/linux_candidates.parquet"
SPLIT_MANIFEST = SPLIT_ROOT / "manifest.json"
BLOCK_SIZE = 48
CANDIDATE_SOLVE_BATCH = 8
CUDA_GPU_MEMORY_RESERVE_GIB = 1.0
SORT_BATCH_SIZE = 131_072
THREADS = 2
DUCKDB_MEMORY_LIMIT = "2GB"
SUPPORTED_DAN_ALPHA = (0.0, 0.1, 0.2, 0.3)
SUPPORTED_DAN_BETA = (0.0, 0.5, 1.0)
SUPPORTED_DROP_P = (0.0, 0.1, 0.3)
SUPPORTED_REGULARIZATION = (100, 1000)
RLAE_XI = (0.0, 0.25, 0.5, 0.75, 1.0)
RDLAE_XI = 0.5
SUPPORTED_RHO = (0.0, 0.25, 0.5, 1.0)

DAN_REFERENCE = {
    "repository": "https://github.com/psm1206/DAN",
    "commit": "fc28a9542f141d13f495d5529aa3f2c486961356",
    "file": "code/model.py",
    "git_blob_sha": "0bdeab8fa839235a5dfca0e813d80a4bf529b6ae",
    "file_sha256": "8367bc8ae1c504fdb517dc2798539917097ab272de73d83bf2dac9b20ae25bf4",
    "class": "EASE_DAN",
    "equations": {
        "gram": "X.T @ diag(user_degree ** -beta) @ X",
        "diagonal_penalty": "lambda + p / (1-p) * train_item_support",
        "base_weights": "I - inverse(gram + diag(penalty)) / diag(inverse(...))",
        "popularity": "W[input,candidate] *= support[input]**alpha / support[candidate]**alpha",
    },
}
RDLAE_REFERENCE = {
    "repository": "https://github.com/jaewan7599/RDLAE_SIGIR2023",
    "commit": "12a89623660cb8514220d146aba40b778d7594f2",
    "file": "strong/code/model.py",
    "git_blob_sha": "61a1847189d32fd9dae24ce9e2dbb74a52579ead",
    "file_sha256": "f6123ca0adcf9e55963f3cf2a4d8c248cdb3dfa92969b9e2ffb07adc568ff8b4",
    "classes": ["RLAE", "RDLAE"],
    "equations": {
        "rlae_multiplier": "max(lambda * inverse_diagonal[candidate], 1 - xi)",
        "rdlae_diagonal_penalty": "lambda + support * p / (1-p)",
        "rdlae_xi": "fixed at 0.5 for the p sweep; author implementation requires xi",
    },
}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _name_num(value: float, width: int = 3) -> str:
    return f"{int(round(float(value) * 1000)):0{width}d}"


def _variant_roster() -> dict:
    """Return the predeclared variant grid; no score results enter this list."""
    variants: dict[str, dict] = {}

    def add(name, family, parameters, artifact_type, scorer_mode):
        variants[name] = {
            "family": family,
            "parameters": parameters,
            "artifact_type": artifact_type,
            "scorer_mode": scorer_mode,
            "status": "planned",
        }

    for lam in SUPPORTED_REGULARIZATION:
        add(f"ease_lambda{lam}_exact", "ease",
            {"regularization": lam, "solver": "full-catalog float32 Cholesky",
             "normalization": "divide each candidate column by its inverse diagonal"},
            "dense_candidate_weights", "linear")
        for xi in RLAE_XI:
            add(f"rlae_lambda{lam}_xi{_name_num(xi)}", "rlae",
                {"regularization": lam, "xi": xi,
                 "candidate_multiplier": "max(lambda * P_jj, 1-xi)",
                 "factorization_reused_across_xi": True},
                "dense_candidate_weights", "linear")
        for drop_p in SUPPORTED_DROP_P:
            add(f"rdlae_lambda{lam}_p{_name_num(drop_p)}_xi{_name_num(RDLAE_XI)}", "rdlae",
                {"regularization": lam, "drop_p": drop_p, "xi": RDLAE_XI,
                 "diagonal_penalty": "lambda + support * p/(1-p)",
                 "xi_policy": "fixed 0.5 for the p sweep"},
                "dense_candidate_weights", "linear")
        for beta in SUPPORTED_DAN_BETA:
            for drop_p in (0.0, 0.3):
                for scale_mode in ("raw", "scale_matched"):
                    for alpha in SUPPORTED_DAN_ALPHA:
                        name = (f"dan_ease_lambda{lam}_a{_name_num(alpha)}"
                                f"_b{_name_num(beta)}_p{_name_num(drop_p)}_{scale_mode}")
                        add(name, "dan_ease",
                            {"regularization": lam, "alpha": alpha, "beta": beta,
                             "drop_p": drop_p, "lambda_scale_mode": scale_mode,
                             "scale_match": "weighted Gram trace / unweighted Gram trace",
                             "popularity_zero_support": "zero candidate coefficients; zero-support items cannot be inputs"},
                            "dense_candidate_weights", "linear")

    add("binary_cosine_k100", "item_knn",
        {"similarity": "binary cosine", "neighbors": 100},
        "item_similarity_csr", "knn")
    add("activity_inverse_degree_cosine_k100", "item_knn",
        {"reviewer_weight": "1 / positive input degree",
         "similarity": "weighted cosine", "neighbors": 100},
        "item_similarity_csr", "knn")
    for rho in SUPPORTED_RHO:
        for selection in ("fixed", "reselected"):
            add(f"account_pair_rho{_name_num(rho)}_{selection}_k100", "account_pair_contribution",
                {"rho": rho, "user_pair_weight": "(1-rho)/d + rho/(d*max(1,d-1))",
                 "neighbors": 100, "neighbor_policy": selection,
                 "fixed_neighbor_source": "activity_inverse_degree_cosine_k100"},
                "item_similarity_csr", "knn")

    for neighbors in (200, 500):
        add(f"fslim_neighbor_cosine_k{neighbors}", "fslim_neighbor_baseline",
            {"similarity": "binary cosine", "neighbors": neighbors},
            "item_similarity_csr", "knn")
        for l2 in (100, 1000):
            for l1_relative in (0.0, 0.001, 0.01):
                name = (f"fslim_m{neighbors}_l2{l2}_l1r"
                        f"{_name_num(l1_relative)}")
                add(name, "fslim",
                    {"neighbors": neighbors, "l2": l2,
                     "l1_lambda_max_fraction": l1_relative,
                     "objective": "0.5*||y-Xw||^2 + l1*||w||_1 + 0.5*l2*||w||^2",
                     "constraint": "w >= 0", "solver": "candidate-wise cyclic coordinate descent",
                     "tolerance": 1e-5, "max_iterations": 1000},
                    "item_similarity_csr", "linear")
    return variants


def freeze_roster() -> Path:
    """Pin the complete v4 linear roster before any builder is allowed to fit."""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    path = MODEL_ROOT / "roster.json"
    roster = _variant_roster()
    canonical = json.dumps(roster, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected_sha = __import__("hashlib").sha256(canonical.encode("utf-8")).hexdigest()
    current = {
        "version": VERSION,
        "status": "frozen",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "roster_sha256": expected_sha,
        "split_root": SPLIT_ROOT.relative_to(EDA).as_posix(),
        "variants": roster,
        "formula_references": {"dan": DAN_REFERENCE, "rdlae": RDLAE_REFERENCE},
        "fit_scope": "v4 train positives only; validation and test inputs are never read",
    }
    if path.exists():
        old = json.loads(path.read_text("utf-8"))
        old_sha = old.get("roster_sha256")
        if old_sha != expected_sha or old.get("variants") != roster:
            raise ValueError("The frozen v4 linear roster differs from this builder; refusing to replace it")
        return path
    _atomic_json(path, current)
    return path


def _strict_ids(values, label: str) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids <= 0) or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1])):
        raise ValueError(f"{label} must contain unique ascending positive App IDs")
    return ids


def _read_app_ids(path: Path, label: str) -> np.ndarray:
    table = pq.read_table(path, columns=["app_id"])
    return _strict_ids(table["app_id"].to_numpy(zero_copy_only=False), label)


def _catalog() -> tuple[np.ndarray, np.ndarray]:
    apps = _read_app_ids(INPUT_CATALOG, "v4 input catalog")
    candidates = _read_app_ids(CANDIDATE_CATALOG, "v4 Linux candidates")
    if len(apps) != 37_610 or len(candidates) != 6_630:
        raise ValueError(f"Frozen v4 catalog dimensions changed: {len(apps)} inputs, {len(candidates)} candidates")
    candidate_cols = np.searchsorted(apps, candidates)
    if np.any(candidate_cols >= len(apps)) or not np.array_equal(apps[candidate_cols], candidates):
        raise ValueError("The frozen Linux candidate catalog is not a subset of input_apps")
    return apps, candidates


def _manifest_value(value, keys: set[str]):
    if isinstance(value, dict):
        for key in keys:
            if key in value and isinstance(value[key], (int, float)):
                return int(value[key])
        for child in value.values():
            result = _manifest_value(child, keys)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _manifest_value(child, keys)
            if result is not None:
                return result
    return None


def _source_manifest() -> dict:
    for path in (SPLIT_MANIFEST, POSITIVE_TRAIN, INPUT_CATALOG, CANDIDATE_CATALOG):
        if not path.is_file():
            raise FileNotFoundError(path)
    split = json.loads(SPLIT_MANIFEST.read_text("utf-8"))
    if split.get("status") != "complete":
        raise ValueError("The v4 split must be complete before linear models are fit")
    expected_files = split.get("files", {})
    sources = {}
    for relative, path in (("train/positive_catalog.parquet", POSITIVE_TRAIN),
                           ("catalog/input_apps.parquet", INPUT_CATALOG),
                           ("catalog/linux_candidates.parquet", CANDIDATE_CATALOG)):
        actual = digest(path)
        expected = expected_files.get(relative, {})
        if isinstance(expected, dict) and expected.get("sha256") and expected["sha256"] != actual:
            raise ValueError(f"Pinned v4 split checksum mismatch: {relative}")
        sources[relative] = {"file": path.relative_to(EDA).as_posix(),
                             "bytes": path.stat().st_size, "sha256": actual}
    sources["split_manifest"] = {
        "file": SPLIT_MANIFEST.relative_to(EDA).as_posix(),
        "bytes": SPLIT_MANIFEST.stat().st_size,
        "sha256": digest(SPLIT_MANIFEST),
    }
    return sources


def _count_train_users() -> int:
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={THREADS}")
        return int(con.execute(
            f"SELECT count(DISTINCT user_key) FROM read_parquet({quote(POSITIVE_TRAIN)})"
        ).fetchone()[0])
    finally:
        con.close()


def _positive_csr() -> tuple[sparse.csr_matrix, np.ndarray, int, np.ndarray, np.ndarray]:
    """Load only v4 positive train pairs; preserve all singleton users."""
    apps, candidates = _catalog()
    pf = pq.ParquetFile(POSITIVE_TRAIN)
    expected_pairs = int(pf.metadata.num_rows)
    split = json.loads(SPLIT_MANIFEST.read_text("utf-8"))
    expected_users = _manifest_value(split, {
        "positive_train_accounts", "positive_train_users", "train_positive_users",
        "base_train_positive_users",
    })
    if expected_users is None:
        expected_users = _count_train_users()
    if expected_pairs <= 0 or expected_users <= 0:
        raise ValueError("The v4 positive train split is empty")
    x, support, users = _build_user_item_csr(
        pf.iter_batches(batch_size=SORT_BATCH_SIZE, columns=["user_key", "app_id"]),
        apps, expected_pairs, expected_users)
    pf.close()
    if not sparse.isspmatrix_csr(x) or x.nnz != expected_pairs or users != expected_users:
        raise ValueError("The v4 positive user-item CSR differs from the frozen split counts")
    degree = np.diff(x.indptr)
    if np.any(degree <= 0):
        raise ValueError("Every positive train user, including singletons, must have nonzero degree")
    return x, support.astype(np.int64, copy=False), int(users), apps, candidates


def _pair_contribution_weights(degree, rho: float) -> np.ndarray:
    """Interpolate inverse-degree and equal-total-pair account weights."""
    if not np.isfinite(rho) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    d = np.asarray(degree, dtype=np.float64)
    if d.ndim != 1 or np.any(~np.isfinite(d)) or np.any(d < 1):
        raise ValueError("Every train positive-user degree must be a positive integer")
    weights = ((1.0 - float(rho)) / d
               + float(rho) / (d * np.maximum(1.0, d - 1.0)))
    return weights.astype(np.float32)


def _topk_by_score(cols, values, app_ids, limit: int):
    cols = np.asarray(cols, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    if cols.shape != values.shape or limit < 0:
        raise ValueError("Neighbor columns/scores have inconsistent shapes")
    keep = np.isfinite(values) & (values > 0)
    cols, values = cols[keep], values[keep]
    if not len(cols) or limit == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    ranked = np.lexsort((app_ids[cols], -values))[:limit]
    storage = np.argsort(cols[ranked], kind="stable")
    ranked = ranked[storage]
    return cols[ranked], values[ranked].astype(np.float32)


def _csr_from_row_lists(rows: list[np.ndarray], values: list[np.ndarray], shape,
                        dtype=np.float32) -> sparse.csr_matrix:
    if len(rows) != shape[0] or len(values) != shape[0]:
        raise ValueError("CSR row count differs from the candidate mapping")
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    if np.any(lengths != np.fromiter((len(v) for v in values), dtype=np.int64, count=len(values))):
        raise ValueError("CSR row index/value lengths differ")
    indptr = np.empty(shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    indices = np.concatenate(rows).astype(np.int32, copy=False) if indptr[-1] else np.empty(0, np.int32)
    data = np.concatenate(values).astype(dtype, copy=False) if indptr[-1] else np.empty(0, dtype)
    result = sparse.csr_matrix((data, indices, indptr), shape=shape)
    result.sort_indices()
    return result


def _manifest_file_info(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def _safe_under_model_root(path: Path) -> Path:
    root = MODEL_ROOT.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Unsafe artifact path outside model root: {resolved}")
    return resolved


def _verify_existing(folder: Path, name: str, parameters: dict, sources: dict) -> bool:
    manifest_path = folder / "manifest.json"
    if not folder.exists():
        return False
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing to overwrite incomplete v4 model directory: {folder}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("version") != VERSION
            or manifest.get("model_name") != name or manifest.get("parameters") != parameters
            or manifest.get("sources") != sources):
        raise ValueError(f"Existing v4 model differs from current roster/sources: {name}")
    for relative, info in manifest.get("files", {}).items():
        path = (folder / relative).resolve()
        if folder.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Unsafe or missing v4 model artifact: {name}/{relative}")
        if digest(path) != info.get("sha256"):
            raise ValueError(f"Existing v4 model artifact checksum changed: {name}/{relative}")
    return True


def _new_stage(name: str, parameters: dict, sources: dict) -> Path | None:
    folder = MODEL_ROOT / name
    if _verify_existing(folder, name, parameters, sources):
        return None
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    stage = MODEL_ROOT / f".{name}.building-{os.getpid()}-{time.time_ns()}"
    _safe_under_model_root(stage)
    stage.mkdir()
    return stage


def _finish_artifact(stage: Path, name: str, family: str, artifact_type: str,
                     scorer_mode: str, parameters: dict, sources: dict,
                     support: np.ndarray, apps: np.ndarray, candidates: np.ndarray,
                     fit_summary: dict) -> Path:
    np.save(stage / "support.npy", np.asarray(support, dtype=np.int64), allow_pickle=False)
    np.save(stage / "input_app_ids.npy", np.asarray(apps, dtype=np.int64), allow_pickle=False)
    np.save(stage / "candidate_app_ids.npy", np.asarray(candidates, dtype=np.int64), allow_pickle=False)
    diagonal_path = stage / "candidate_inverse_diagonal.npy"
    if not diagonal_path.is_file() and fit_summary.get("candidate_inverse_diagonal_sha256"):
        expected_diagonal_sha = fit_summary["candidate_inverse_diagonal_sha256"]
        # Derivation builders can reuse an already solved alpha=0 base. Carry
        # its matching candidate diagonal into the derived artifact by hash.
        for source_diagonal in MODEL_ROOT.glob("*/candidate_inverse_diagonal.npy"):
            if source_diagonal.is_file() and digest(source_diagonal) == expected_diagonal_sha:
                shutil.copy2(source_diagonal, diagonal_path)
                break
    if diagonal_path.is_file():
        candidate_diagonal = np.load(diagonal_path, mmap_mode="r", allow_pickle=False)
        if (candidate_diagonal.shape != (len(candidates),)
                or candidate_diagonal.dtype != np.dtype(np.float32)
                or np.any(~np.isfinite(candidate_diagonal))
                or np.any(candidate_diagonal <= 0)):
            raise ValueError(f"Candidate inverse diagonals are invalid: {name}")
        diagonal_sha = digest(diagonal_path)
        _close_memmap(candidate_diagonal)
        fit_summary = dict(fit_summary,
                           candidate_inverse_diagonal_file="candidate_inverse_diagonal.npy",
                           candidate_inverse_diagonal_sha256=diagonal_sha)
    shutil.copy2(Path(__file__), stage / "builder_snapshot.py")
    files = {path.name: _manifest_file_info(path) for path in sorted(stage.iterdir()) if path.is_file()}
    weight_shape = None
    weight_dtype = None
    if (stage / "weights.npy").is_file():
        mapped = np.load(stage / "weights.npy", mmap_mode="r", allow_pickle=False)
        weight_shape = list(mapped.shape)
        weight_dtype = str(mapped.dtype)
        if weight_shape != [len(candidates), len(apps)] or mapped.dtype != np.dtype(np.float32):
            raise ValueError("Dense candidate weights differ from frozen v4 mappings or precision")
        candidate_cols = np.searchsorted(apps, candidates)
        for start in range(0, len(candidates), 128):
            stop = min(start + 128, len(candidates))
            block = np.asarray(mapped[start:stop])
            if np.any(~np.isfinite(block)):
                raise ValueError(f"Dense v4 candidate weights contain nonfinite values: {name}")
            if np.any(block[np.arange(stop-start), candidate_cols[start:stop]] != 0):
                raise ValueError(f"Dense v4 candidate weights contain nonzero self coefficients: {name}")
        _close_memmap(mapped)
        del mapped
    elif (stage / "similarities_csr.npz").is_file():
        matrix = sparse.load_npz(stage / "similarities_csr.npz")
        if not sparse.isspmatrix_csr(matrix) or not matrix.has_sorted_indices:
            raise ValueError("Saved v4 graph matrix must be sorted CSR")
        if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data <= 0):
            raise ValueError("Stored v4 graph edges must be finite positive values")
        weight_shape = list(matrix.shape)
        weight_dtype = str(matrix.dtype)
        fit_summary = dict(fit_summary, nnz=int(matrix.nnz))
        del matrix
    builder_path = Path(__file__)
    manifest = {
        "version": VERSION,
        "status": "complete",
        "model_name": name,
        "family": family,
        "artifact_type": artifact_type,
        "scorer_mode": scorer_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_scope": "experiments_v4 split train/positive_catalog.parquet only",
        "validation_test_interactions_read": False,
        "ranking_metrics_computed": False,
        "matrix_orientation": "Linux candidate rows x fixed input-catalog columns",
        "parameters": parameters,
        "sources": sources,
        "source_hashes": {key: value["sha256"] for key, value in sources.items()},
        "formula_references": ({"dan": DAN_REFERENCE} if family == "dan_ease"
                                else {"rdlae": RDLAE_REFERENCE} if family in {"rlae", "rdlae"}
                                else {}),
        "index": {"shape": weight_shape, "dtype": weight_dtype,
                  "format": "NumPy NPY dense" if (stage / "weights.npy").is_file()
                  else "SciPy CSR NPZ"},
        "fit_summary": fit_summary,
        "builder_sha256": digest(builder_path),
        "files": files,
    }
    _atomic_json(stage / "manifest.json", manifest)
    folder = MODEL_ROOT / name
    _safe_under_model_root(stage)
    _safe_under_model_root(folder)
    if folder.exists():
        raise FileExistsError(f"v4 output appeared while building: {name}")
    stage.replace(folder)
    return folder


def _save_sparse_artifact(name: str, matrix: sparse.csr_matrix, family: str,
                          parameters: dict, sources: dict, support: np.ndarray,
                          apps: np.ndarray, candidates: np.ndarray,
                          fit_summary: dict) -> Path | None:
    freeze_roster()
    stage = _new_stage(name, parameters, sources)
    if stage is None:
        return None
    try:
        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.shape != (len(candidates), len(apps)):
            raise ValueError("Candidate/input model shape differs from frozen v4 mappings")
        if matrix.nnz and (np.any(~np.isfinite(matrix.data)) or np.any(matrix.data <= 0)):
            raise ValueError("Sparse similarity models must contain finite positive edges")
        sparse.save_npz(stage / "similarities_csr.npz", matrix, compressed=True)
        return _finish_artifact(stage, name, family, "item_similarity_csr", "knn",
                                parameters, sources, support, apps, candidates, fit_summary)
    except Exception:
        if stage.exists():
            _safe_under_model_root(stage)
            shutil.rmtree(stage)
        raise


def _build_binary_knn(x: sparse.csr_matrix, support: np.ndarray,
                      apps: np.ndarray, candidates: np.ndarray,
                      neighbors: int = 100):
    """Exact binary cosine top-k on v4 train data; integer Gram avoids count rounding."""
    x_csc = x.astype(np.int32).tocsc()
    item_rows = x_csc.T.tocsr()
    candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, len(candidates), BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, len(candidates))
        cooc = (item_rows[candidate_cols[start:stop], :] @ x_csc).tocsr()
        cooc.sum_duplicates()
        cooc.sort_indices()
        for local in range(stop - start):
            row = start + local
            ccol = int(candidate_cols[row])
            begin, end = cooc.indptr[local:local+2]
            cols, counts = cooc.indices[begin:end], cooc.data[begin:end].astype(np.int64, copy=False)
            out_cols, out_values = _top_cosine_neighbors(
                ccol, cols, counts, apps, support, neighbors)
            rows.append(out_cols)
            values.append(out_values.astype(np.float32, copy=False))
        del cooc
    del item_rows, x_csc
    gc.collect()
    return _csr_from_row_lists(rows, values, (len(candidates), len(apps)))


def _weighted_cosine_rows(x: sparse.csr_matrix, apps: np.ndarray,
                          candidates: np.ndarray, pair_weights: np.ndarray,
                          neighbors: int, fixed_neighbor_rows: list[np.ndarray] | None = None):
    """Build weighted-cosine top-k and optionally rescore a fixed neighbor set."""
    x_csc = x.astype(np.float32).tocsc()
    weighted = x_csc.copy()
    weighted.data = np.sqrt(np.asarray(pair_weights[weighted.indices], dtype=np.float32))
    item_weight = np.zeros(len(apps), dtype=np.float64)
    for col in range(len(apps)):
        begin, end = weighted.indptr[col:col+2]
        if end > begin:
            data = weighted.data[begin:end].astype(np.float64, copy=False)
            item_weight[col] = float(np.dot(data, data))
    item_rows = weighted.T.tocsr()
    candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
    selected_rows: list[np.ndarray] = []
    selected_values: list[np.ndarray] = []
    fixed_rows: list[np.ndarray] = []
    fixed_values: list[np.ndarray] = []
    if fixed_neighbor_rows is not None and len(fixed_neighbor_rows) != len(candidates):
        raise ValueError("Fixed neighbor mapping does not match the candidate catalog")
    for start in range(0, len(candidates), BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, len(candidates))
        cooc = (item_rows[candidate_cols[start:stop], :] @ weighted).tocsr()
        cooc.sum_duplicates()
        cooc.sort_indices()
        for local in range(stop-start):
            row = start + local
            ccol = int(candidate_cols[row])
            begin, end = cooc.indptr[local:local+2]
            cols = cooc.indices[begin:end].astype(np.int32, copy=False)
            counts = cooc.data[begin:end].astype(np.float64, copy=False)
            valid = ((cols != ccol) & (counts > 0) & (item_weight[cols] > 0)
                     & (item_weight[ccol] > 0))
            cols, counts = cols[valid], counts[valid]
            denom = np.sqrt(item_weight[ccol] * item_weight[cols])
            cos = np.minimum(counts / denom, 1.0)
            out_cols, out_values = _topk_by_score(cols, cos, apps, neighbors)
            selected_rows.append(out_cols)
            selected_values.append(out_values)
            if fixed_neighbor_rows is not None:
                fixed_cols = np.asarray(fixed_neighbor_rows[row], dtype=np.int32)
                if len(fixed_cols):
                    positions = np.searchsorted(cols, fixed_cols)
                    if (np.any(positions >= len(cols))
                            or not np.array_equal(cols[positions], fixed_cols)):
                        raise ValueError("A rho-weighted fit lost a positive base-neighbor co-occurrence")
                    fixed_scores = counts[positions] / np.sqrt(
                        item_weight[ccol] * item_weight[fixed_cols])
                    order = np.argsort(fixed_cols, kind="stable")
                    fixed_rows.append(fixed_cols[order])
                    fixed_values.append(fixed_scores[order].astype(np.float32))
                else:
                    fixed_rows.append(np.empty(0, np.int32))
                    fixed_values.append(np.empty(0, np.float32))
        del cooc
    del x_csc, weighted, item_rows
    gc.collect()
    top = _csr_from_row_lists(selected_rows, selected_values, (len(candidates), len(apps)))
    fixed = (_csr_from_row_lists(fixed_rows, fixed_values, (len(candidates), len(apps)))
             if fixed_neighbor_rows is not None else None)
    return top, fixed, selected_rows


def _build_knn_and_pair_models():
    freeze_roster()
    sources = _source_manifest()
    x, support, users, apps, candidates = _positive_csr()
    degree = np.diff(x.indptr).astype(np.float64)
    if len(degree) != users:
        raise ValueError("Positive user degree vector differs from train user count")
    if _available_memory_gb() < 3.0:
        raise MemoryError("KNN/pair-contribution builders require >=3 GiB available RAM")
    summary = {"train_positive_users": users, "train_positive_pairs": int(x.nnz),
               "singleton_positive_users": int(np.count_nonzero(degree == 1)),
               "catalog_items": int(len(apps)), "linux_candidates": int(len(candidates))}

    binary = _build_binary_knn(x, support, apps, candidates, 100)
    params = _variant_roster()["binary_cosine_k100"]["parameters"]
    _save_sparse_artifact("binary_cosine_k100", binary, "item_knn", params, sources,
                          support, apps, candidates, summary)
    del binary
    gc.collect()

    rho0 = _pair_contribution_weights(degree, 0.0)
    activity, _, base_neighbors = _weighted_cosine_rows(
        x, apps, candidates, rho0, 100)
    base_summary = dict(summary, reviewer_weight="1 / positive user degree")
    _save_sparse_artifact("activity_inverse_degree_cosine_k100", activity, "item_knn",
                          _variant_roster()["activity_inverse_degree_cosine_k100"]["parameters"],
                          sources, support, apps, candidates, base_summary)
    roster = _variant_roster()
    # The selected matrix is already the rho=0 reselected matrix. Save the
    # separately named fixed-neighbor variant from identical values, making
    # their identity at rho=0 explicit in the frozen roster.
    _save_sparse_artifact("account_pair_rho000_reselected_k100", activity, "account_pair_contribution",
                          roster["account_pair_rho000_reselected_k100"]["parameters"],
                          sources, support, apps, candidates, dict(summary, rho=0.0))
    _save_sparse_artifact("account_pair_rho000_fixed_k100", activity, "account_pair_contribution",
                          roster["account_pair_rho000_fixed_k100"]["parameters"],
                          sources, support, apps, candidates, dict(summary, rho=0.0))
    del activity
    gc.collect()

    for rho in SUPPORTED_RHO[1:]:
        pair_weight = _pair_contribution_weights(degree, rho)
        reselected, fixed, _ = _weighted_cosine_rows(
            x, apps, candidates, pair_weight, 100, fixed_neighbor_rows=base_neighbors)
        for selection, matrix in (("reselected", reselected), ("fixed", fixed)):
            name = f"account_pair_rho{_name_num(rho)}_{selection}_k100"
            _save_sparse_artifact(name, matrix, "account_pair_contribution",
                                  roster[name]["parameters"], sources, support, apps,
                                  candidates, dict(summary, rho=rho,
                                                   fixed_neighbor_source="rho=0 activity inverse degree"))
        del pair_weight, reselected, fixed
        gc.collect()
    del base_neighbors, rho0, x
    gc.collect()
    return ["binary_cosine_k100", "activity_inverse_degree_cosine_k100"] + [
        f"account_pair_rho{_name_num(rho)}_{selection}_k100"
        for rho in SUPPORTED_RHO for selection in ("fixed", "reselected")]


def _available_memory_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1024**3
    except Exception:
        return float("inf")


def _require_linear_resources():
    """Refuse to launch exact full-catalog factorizations outside a safe window."""
    available = _available_memory_gb()
    disk = shutil.disk_usage(MODEL_ROOT.parent if MODEL_ROOT.parent.exists() else EDA)
    disk_gb = disk.free / 1024**3
    # 37,610 squared float32 is 5.27 GiB; factor, blocked Gram and a 0.93 GiB
    # dense candidate output coexist transiently. The conservative gate avoids
    # starting the fit if the host cannot honor memory-mapped LAPACK work.
    if available < 10.0:
        raise MemoryError(f"Exact v4 linear fit needs >=10 GiB available RAM; found {available:.2f} GiB")
    if disk_gb < 13.0:
        raise OSError(f"Exact v4 linear fit needs >=13 GiB free disk; found {disk_gb:.2f} GiB")


def _require_linear_memory(stage: str, minimum_gib: float = 10.0):
    available = _available_memory_gb()
    if available < minimum_gib:
        raise MemoryError(f"{stage} paused below the v4 {minimum_gib:g} GiB memory gate; available RAM is {available:.2f} GiB")


def _gram_cache_path(beta: float) -> Path:
    return MODEL_ROOT / "_linear_work" / f"gram_beta{_name_num(beta)}.f32.dat"


def _gram_cache_receipt(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _weighted_support(x_csc: sparse.csc_matrix) -> np.ndarray:
    values = np.zeros(x_csc.shape[1], dtype=np.float64)
    for col in range(x_csc.shape[1]):
        start, end = x_csc.indptr[col:col+2]
        if end > start:
            data = x_csc.data[start:end].astype(np.float64, copy=False)
            values[col] = float(np.dot(data, data))
    return values


def _build_gram_memmap(x: sparse.csr_matrix, support: np.ndarray,
                       beta: float, source_hash: str, work_dir: Path):
    """Create X' D_user X in float32 row blocks, keeping one full Gram on disk."""
    n_items = x.shape[1]
    expected_bytes = n_items * n_items * np.dtype(np.float32).itemsize
    work_dir.mkdir(parents=True, exist_ok=True)
    gram_path = _gram_cache_path(beta)
    receipt_path = _gram_cache_receipt(gram_path)
    if gram_path.is_file() and gram_path.stat().st_size == expected_bytes and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if (receipt.get("complete") is True and receipt.get("shape") == [n_items, n_items]
                    and receipt.get("beta") == beta and receipt.get("train_sha256") == source_hash
                    and receipt.get("sha256") == digest(gram_path)):
                gram = np.memmap(gram_path, dtype=np.float32, mode="r", shape=(n_items, n_items), order="C")
                if beta == 0.0:
                    item_support = support.astype(np.float64)
                else:
                    x_tmp = x.astype(np.float32).tocsc()
                    degree = np.diff(x.indptr).astype(np.float64)
                    weights = np.power(degree, -beta).astype(np.float32)
                    x_tmp.data = np.sqrt(weights[x_tmp.indices])
                    item_support = _weighted_support(x_tmp)
                    del x_tmp, degree, weights
                    gc.collect()
                return gram, item_support, gram_path, receipt
        except (OSError, ValueError, KeyError):
            pass

    temp_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    if gram_path.exists():
        _safe_under_model_root(gram_path)
        gram_path.unlink()
    temp_receipt.unlink(missing_ok=True)
    degree = np.diff(x.indptr).astype(np.float64)
    if len(degree) == 0 or np.any(degree <= 0):
        raise ValueError("Cannot form a degree-weighted Gram with empty user rows")
    weights = np.power(degree, -beta).astype(np.float32)
    x_csc = x.astype(np.float32).tocsc()
    if beta == 0.0:
        x_csc.data.fill(1.0)
    else:
        x_csc.data = np.sqrt(weights[x_csc.indices])
    # The 10 GiB fit admission check ran before this CSR/CSC/Gram allocation.
    # After those controlled arrays exist, reserve 5 GiB for transient sparse
    # products and let the OS evict the mapped Gram pages.
    _require_linear_memory("v4 Gram construction", minimum_gib=5.0)
    item_support = _weighted_support(x_csc)
    item_rows = x_csc.T.tocsr()
    gram = np.memmap(gram_path, dtype=np.float32, mode="w+", shape=(n_items, n_items), order="C")
    for start in range(0, n_items, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, n_items)
        block = (item_rows[start:stop, :] @ x_csc).tocsr()
        dense = block.toarray().astype(np.float32, copy=False)
        gram[start:stop, :] = dense
        del block, dense
        if start == 0 or stop == n_items or start % (BLOCK_SIZE * 100) == 0:
            log(f"v4 Gram beta={beta:g}: {stop:,}/{n_items:,} item rows")
        if stop == n_items or stop % (BLOCK_SIZE * 25) == 0:
            _require_linear_memory("v4 Gram construction", minimum_gib=5.0)
    gram.flush()
    receipt = {"complete": True, "shape": [n_items, n_items], "beta": beta,
               "train_sha256": source_hash, "train_positive_pairs": int(x.nnz),
               "sha256": digest(gram_path)}
    _atomic_json(receipt_path, receipt)
    del x_csc, item_rows, degree, weights
    gc.collect()
    return gram, item_support, gram_path, receipt


def _close_memmap(array):
    if isinstance(array, np.memmap):
        try:
            array.flush()
            array._mmap.close()
        except Exception:
            pass


class _CudaCholeskySolver:
    """Hold one float32 CUDA Cholesky factor and solve candidate columns."""

    def __init__(self, torch, factor, batch_size: int, settings: dict):
        self.torch = torch
        self.factor = factor
        self.batch_size = int(batch_size)
        self.settings = settings

    def solve_columns(self, candidate_cols: np.ndarray) -> np.ndarray:
        torch = self.torch
        cols = np.asarray(candidate_cols, dtype=np.int64)
        n_items = self.factor.shape[0]
        if cols.ndim != 1 or np.any(cols < 0) or np.any(cols >= n_items):
            raise ValueError("CUDA candidate columns do not align with the Cholesky factor")
        solved = np.empty((n_items, len(cols)), dtype=np.float32, order="F")
        device = self.factor.device
        with torch.inference_mode():
            for start in range(0, len(cols), self.batch_size):
                stop = min(start + self.batch_size, len(cols))
                batch_cols = cols[start:stop]
                rhs = torch.zeros((stop - start, n_items), device=device,
                                  dtype=torch.float32).T
                rows = torch.as_tensor(batch_cols, device=device, dtype=torch.long)
                rhs[rows, torch.arange(stop - start, device=device)] = 1.0
                torch.linalg.solve_triangular(self.factor, rhs, upper=False, out=rhs)
                torch.linalg.solve_triangular(self.factor.T, rhs, upper=True, out=rhs)
                solved[:, start:stop] = rhs.cpu().numpy()
                del rhs, rows
        torch.cuda.synchronize(device)
        return solved

    def close(self):
        torch = self.torch
        if self.factor is not None:
            try:
                torch.cuda.synchronize(self.factor.device)
            except Exception:
                pass
            self.factor = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _factorize_cuda(gram: np.memmap, gamma: np.ndarray, work_dir: Path,
                    *, batch_size: int = CANDIDATE_SOLVE_BATCH):
    """Factor the regularized full Gram on CUDA and return the live solver."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA linear backend requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA linear backend requested but CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    n_items = gram.shape[0]
    if gamma.shape != (n_items,) or batch_size < 1:
        raise ValueError("CUDA Cholesky penalty or solve batch shape is invalid")
    matrix_bytes = n_items * n_items * np.dtype(np.float32).itemsize
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    minimum_free = matrix_bytes + int(CUDA_GPU_MEMORY_RESERVE_GIB * 1024**3)
    if free_bytes < minimum_free:
        raise MemoryError(
            "CUDA Cholesky needs one full float32 factor plus "
            f"{CUDA_GPU_MEMORY_RESERVE_GIB:g} GiB reserve; "
            f"found {free_bytes / 1024**3:.2f} GiB free"
        )
    memory_fraction = 0.90
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    gram_path = Path(gram.filename)
    gram_dtype, gram_shape = gram.dtype, gram.shape
    factor = torch.empty((n_items, n_items), device=device, dtype=torch.float32).T
    try:
        with torch.inference_mode():
            for start in range(0, n_items, 256):
                stop = min(start + 256, n_items)
                factor[start:stop].copy_(torch.from_numpy(np.array(gram[start:stop], copy=True)))
            factor.diagonal().add_(torch.as_tensor(gamma, device=device, dtype=torch.float32))
            # The Gram mapping is closed before factorization so its cached
            # pages do not compete with the device-backed full factor.
            _close_memmap(gram)
            gc.collect()
            _require_linear_memory("v4 CUDA Cholesky factorization", minimum_gib=5.0)
            info = torch.empty((), device=device, dtype=torch.int32)
            torch.linalg.cholesky_ex(factor, upper=False, check_errors=True, out=(factor, info))
            torch.cuda.synchronize(device)
            if int(info.item()) != 0:
                raise np.linalg.LinAlgError(f"CUDA v4 Cholesky failed at pivot {int(info.item())}")
            del info
    except Exception:
        del factor
        gc.collect()
        torch.cuda.empty_cache()
        raise
    gram = np.memmap(gram_path, dtype=gram_dtype, mode="r", shape=gram_shape, order="C")
    settings = {
        "backend": "cuda",
        "library": "PyTorch",
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_index": int(device.index),
        "dtype": "float32",
        "tf32": False,
        "memory_fraction": memory_fraction,
        "available_gpu_bytes_before_factorization": int(free_bytes),
        "total_gpu_bytes": int(total_bytes),
        "minimum_free_gpu_bytes": int(minimum_free),
        "cholesky": "torch.linalg.cholesky_ex, lower triangle, in-place",
        "candidate_solve": "two torch.linalg.solve_triangular calls per RHS batch",
        "candidate_solve_batch_columns": int(batch_size),
    }
    return _CudaCholeskySolver(torch, factor, batch_size, settings), gram


def _factorize(gram: np.memmap, gamma: np.ndarray, work_dir: Path):
    """Exact float32 lower Cholesky in a column-major disk-backed array."""
    from scipy.linalg.lapack import get_lapack_funcs

    n_items = gram.shape[0]
    gram_path = Path(gram.filename)
    gram_dtype = gram.dtype
    gram_shape = gram.shape
    factor_path = work_dir / f"factor_work_{os.getpid()}.f32.dat"
    if factor_path.exists():
        _safe_under_model_root(factor_path)
        factor_path.unlink()
    factor = np.memmap(factor_path, dtype=np.float32, mode="w+",
                       shape=(n_items, n_items), order="F")
    for start in range(0, n_items, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, n_items)
        factor[start:stop, :] = gram[start:stop, :]
    diag = np.arange(n_items)
    factor[diag, diag] += np.asarray(gamma, dtype=np.float32)
    factor.flush()
    # The 10 GiB check is made before creating/touching the factor. Once the
    # 5.27 GiB factor memmap is populated, a 5 GiB available-RAM floor is the
    # scratch gate; retaining the original Gram mapping here can needlessly
    # pin its pages beside the factor. Reopen it only for residual validation.
    _close_memmap(gram)
    gc.collect()
    _require_linear_memory("v4 Cholesky factorization", minimum_gib=5.0)
    potrf = get_lapack_funcs("potrf", (factor,))
    factorized, info = potrf(factor, lower=1, overwrite_a=1, clean=0)
    if info != 0:
        _close_memmap(factor)
        factor_path.unlink(missing_ok=True)
        raise np.linalg.LinAlgError(f"v4 Cholesky failed at pivot {info}")
    if not np.shares_memory(factor, factorized):
        _close_memmap(factor)
        factor_path.unlink(missing_ok=True)
        raise MemoryError("LAPACK copied the full v4 Cholesky factor instead of using its memmap")
    factor.flush()
    gram = np.memmap(gram_path, dtype=gram_dtype, mode="r", shape=gram_shape, order="C")
    return factor, factor_path, get_lapack_funcs("potrs", (factor,)), gram


def _sample_solve_residual(gram, gamma: np.ndarray, factor, potrs,
                           candidate_cols, *, sample_limit: int = 3,
                           tolerance: float = 1e-4) -> dict:
    """Check sampled candidate solves against the original full Gram equation."""
    n_items = gram.shape[0]
    gamma = np.asarray(gamma, dtype=np.float64)
    candidates = np.asarray(candidate_cols, dtype=np.int64)
    if (gram.shape != (n_items, n_items) or gamma.shape != (n_items,)
            or not len(candidates) or np.any(candidates < 0) or np.any(candidates >= n_items)):
        raise ValueError("Sampled solve residual inputs do not align")
    stride = max(1, len(candidates) // max(1, sample_limit))
    samples = np.unique(candidates[::stride][:sample_limit])
    rhs = np.zeros((n_items, len(samples)), dtype=np.float32, order="F")
    rhs[samples, np.arange(len(samples))] = 1.0
    solved, info = potrs(factor, rhs.copy(order="F"), lower=1, overwrite_b=1)
    if info != 0 or np.any(~np.isfinite(solved)):
        raise np.linalg.LinAlgError(f"Sampled v4 candidate solve failed: {info}")
    return _sample_solve_residual_from_solutions(
        gram, gamma, samples, solved, tolerance=tolerance)


def _sample_solve_residual_from_solutions(gram, gamma: np.ndarray,
                                         samples: np.ndarray, solved: np.ndarray,
                                         *, tolerance: float = 1e-4) -> dict:
    """Check a sampled solve matrix against the original full Gram equation."""
    n_items = gram.shape[0]
    gamma = np.asarray(gamma, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.int64)
    solved = np.asarray(solved, dtype=np.float32)
    if (gram.shape != (n_items, n_items) or gamma.shape != (n_items,)
            or solved.shape != (n_items, len(samples)) or not len(samples)
            or np.any(samples < 0) or np.any(samples >= n_items)
            or np.any(~np.isfinite(solved))):
        raise ValueError("Sampled v4 solution matrix does not align with the full Gram")
    rhs = np.zeros((n_items, len(samples)), dtype=np.float64)
    rhs[samples, np.arange(len(samples))] = 1.0
    product = np.asarray(gram @ solved, dtype=np.float64)
    product += gamma[:, None] * solved.astype(np.float64)
    residual = product - rhs
    max_row_sum = 0.0
    for start in range(0, n_items, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, n_items)
        block = np.asarray(gram[start:stop, :], dtype=np.float64)
        if np.any(~np.isfinite(block)) or np.any(block < -1e-5):
            raise ValueError("Original v4 Gram has nonfinite or negative values")
        row_sum = block.sum(axis=1, dtype=np.float64) + gamma[start:stop]
        max_row_sum = max(max_row_sum, float(np.max(row_sum)))
    if not np.isfinite(max_row_sum) or max_row_sum <= 0:
        raise ValueError("Original v4 Gram/penalty has an invalid infinity norm")
    max_backward_error = 0.0
    for local in range(len(samples)):
        denominator = max_row_sum * float(np.max(np.abs(solved[:, local]))) + 1.0
        backward_error = float(np.max(np.abs(residual[:, local]))) / denominator
        max_backward_error = max(max_backward_error, backward_error)
    result = {
        "sample_candidate_columns": samples.astype(int).tolist(),
        "sampled_rhs_count": int(len(samples)),
        "max_absolute_residual": float(np.max(np.abs(residual))),
        "max_backward_error": float(max_backward_error),
        "tolerance": float(tolerance),
    }
    if max_backward_error > tolerance:
        raise np.linalg.LinAlgError(
            f"Sampled v4 linear solve backward error {max_backward_error:.3g} exceeds {tolerance:g}")
    return result


def _sample_cuda_solve_residual(gram, gamma: np.ndarray, solver: _CudaCholeskySolver,
                                candidate_cols, *, sample_limit: int = 3,
                                tolerance: float = 1e-4) -> dict:
    candidates = np.asarray(candidate_cols, dtype=np.int64)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("CUDA sampled residual needs candidate columns")
    stride = max(1, len(candidates) // max(1, sample_limit))
    samples = np.unique(candidates[::stride][:sample_limit])
    solved = solver.solve_columns(samples)
    return _sample_solve_residual_from_solutions(
        gram, gamma, samples, solved, tolerance=tolerance)


def _rlae_multiplier(gamma: np.ndarray, inverse_diagonal: np.ndarray,
                     xi: float) -> np.ndarray:
    """Candidate-wise multiplier equivalent to the author's RLAE lagrange update."""
    gamma = np.asarray(gamma, dtype=np.float64)
    inverse_diagonal = np.asarray(inverse_diagonal, dtype=np.float64)
    if gamma.shape != inverse_diagonal.shape or not 0 <= xi <= 1:
        raise ValueError("RLAE gamma/diagonal shape or xi is invalid")
    if np.any(~np.isfinite(gamma)) or np.any(~np.isfinite(inverse_diagonal)) or np.any(inverse_diagonal <= 0):
        raise ValueError("RLAE requires finite positive inverse diagonal entries")
    # The author code sets lag_j=((1-xi)/P_jj - gamma_j) when
    # 1-gamma_j*P_jj > xi; otherwise lag_j=0. Multiplying their coefficient
    # by the EASE coefficient gives max(gamma_j*P_jj, 1-xi).
    return np.maximum(gamma * inverse_diagonal, 1.0 - float(xi))


def _dan_popularity_factors(support: np.ndarray, alpha: float):
    """Return input and candidate factors in stored candidate-by-input orientation."""
    support = np.asarray(support, dtype=np.float64)
    if not 0 <= alpha <= 1 or np.any(support < 0):
        raise ValueError("DAN alpha/support is invalid")
    input_factor = np.zeros_like(support)
    positive = support > 0
    input_factor[positive] = np.power(support[positive], float(alpha))
    candidate_factor = np.zeros_like(support)
    candidate_factor[positive] = np.power(support[positive], -float(alpha))
    return input_factor, candidate_factor


def _dan_candidate_coefficients(ease_candidate_row: np.ndarray,
                                support: np.ndarray,
                                candidate_col: int,
                                alpha: float) -> np.ndarray:
    """Apply the DAN author's popularity factors to one candidate row."""
    base = np.asarray(ease_candidate_row, dtype=np.float64)
    support = np.asarray(support, dtype=np.int64)
    if base.shape != support.shape or not 0 <= int(candidate_col) < len(support):
        raise ValueError("DAN candidate coefficients do not align with input support")
    input_factor, candidate_factor = _dan_popularity_factors(support, alpha)
    result = base * input_factor * candidate_factor[int(candidate_col)]
    result[int(candidate_col)] = 0.0
    return result.astype(np.float32)


def _preserved_ease_reference_paths(work_dir: Path, regularization: int):
    stem = f"ease_reference_lambda{regularization}"
    return (work_dir / f"{stem}_candidate_weights.npy",
            work_dir / f"{stem}_inverse_diagonal.npy",
            work_dir / f"{stem}.json")


def _verify_preserved_ease_reference(work_dir: Path, regularization: int,
                                    train_sha256: str, n_candidates: int,
                                    n_inputs: int) -> dict:
    weights_path, diagonal_path, receipt_path = _preserved_ease_reference_paths(
        work_dir, regularization)
    if not all(path.is_file() for path in (weights_path, diagonal_path, receipt_path)):
        raise FileNotFoundError("RLAE artifacts require their preserved EASE coefficient/diagonal cache")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if (receipt.get("complete") is not True or receipt.get("regularization") != regularization
            or receipt.get("train_sha256") != train_sha256
            or receipt.get("candidate_count") != n_candidates
            or receipt.get("input_count") != n_inputs
            or receipt.get("ease_candidate_weights_sha256") != digest(weights_path)
            or receipt.get("candidate_inverse_diagonal_sha256") != digest(diagonal_path)):
        raise ValueError("Preserved RLAE EASE candidate references failed integrity checks")
    return receipt


def _build_rlae_from_ease_cache(regularization: int, named: list[tuple[str, dict]],
                                sources: dict) -> list[str]:
    """Reuse saved EASE candidate coefficients/diagonals for all requested xi."""
    work_dir = MODEL_ROOT / "_linear_work"
    source_hash = sources["train/positive_catalog.parquet"]["sha256"]
    receipt = _verify_preserved_ease_reference(
        work_dir, regularization, source_hash, 6_630, 37_610)
    ease_name = f"ease_lambda{regularization}_exact"
    ease_params = _variant_roster()[ease_name]["parameters"]
    ease_folder = MODEL_ROOT / ease_name
    if not _verify_existing(ease_folder, ease_name, ease_params, sources):
        raise FileNotFoundError(f"Build {ease_name} before requesting its RLAE xi sweep")
    free_gib = shutil.disk_usage(MODEL_ROOT).free / 1024**3
    missing_count = sum(not (MODEL_ROOT / name / "manifest.json").is_file() for name, _ in named)
    output_gib = missing_count * 6_630 * 37_610 * 4 / 1024**3
    if free_gib < output_gib + 1.0:
        raise OSError(f"RLAE xi sweep needs {output_gib + 1.0:.2f} GiB free disk; found {free_gib:.2f} GiB")
    apps = _strict_ids(np.load(ease_folder / "input_app_ids.npy", allow_pickle=False), "EASE input IDs")
    candidates = _strict_ids(np.load(ease_folder / "candidate_app_ids.npy", allow_pickle=False), "EASE candidate IDs")
    support = np.asarray(np.load(ease_folder / "support.npy", allow_pickle=False), dtype=np.int64)
    if (len(apps) != 37_610 or len(candidates) != 6_630 or support.shape != apps.shape
            or np.any(support < 0)):
        raise ValueError("Saved EASE support or App ID mappings differ from v4 catalogs")
    reference_weights, reference_diagonal, _ = _preserved_ease_reference_paths(work_dir, regularization)
    base = np.load(reference_weights, mmap_mode="r", allow_pickle=False)
    inverse_diagonal = np.load(reference_diagonal, mmap_mode="r", allow_pickle=False)
    if (base.shape != (len(candidates), len(apps)) or base.dtype != np.dtype(np.float32)
            or inverse_diagonal.shape != (len(candidates),)
            or inverse_diagonal.dtype != np.dtype(np.float32)):
        raise ValueError("Preserved EASE candidate coefficient or diagonal shape is invalid")
    candidate_cols = np.searchsorted(apps, candidates)
    if np.any(~np.isfinite(inverse_diagonal)) or np.any(inverse_diagonal <= 0):
        raise ValueError("Preserved EASE inverse candidate diagonals must be finite and positive")
    stage_by_name = {}
    output_by_name = {}
    xi_by_name = {name: float(params["xi"]) for name, params in named}
    try:
        for name, params in named:
            stage = _new_stage(name, params, sources)
            if stage is None:
                continue
            output_by_name[name] = np.lib.format.open_memmap(
                stage / "weights.npy", mode="w+", dtype=np.float32,
                shape=(len(candidates), len(apps)))
            stage_by_name[name] = stage
        if not output_by_name:
            return [name for name, _ in named]

        for start in range(0, len(candidates), 128):
            stop = min(start + 128, len(candidates))
            base_block = np.asarray(base[start:stop])
            if np.any(~np.isfinite(base_block)):
                raise ValueError("Preserved EASE candidate coefficients contain nonfinite values")
            diag_block = np.asarray(inverse_diagonal[start:stop], dtype=np.float64)
            for name in output_by_name:
                xi = xi_by_name[name]
                output_by_name[name][start:stop, :] = _rlae_weights_from_ease(
                    base_block, diag_block, regularization, xi,
                    candidate_cols[start:stop])
            if start == 0 or stop == len(candidates) or start % 1024 == 0:
                for array in output_by_name.values():
                    array.flush()

        ease_summary = json.loads((ease_folder / "manifest.json").read_text("utf-8")).get("fit_summary", {})
        for name, params in named:
            if name not in output_by_name:
                continue
            output = output_by_name.pop(name)
            output.flush()
            _close_memmap(output)
            del output
            diagonal_path = stage_by_name[name] / "candidate_inverse_diagonal.npy"
            np.save(diagonal_path, np.asarray(inverse_diagonal, dtype=np.float32), allow_pickle=False)
            diagonal_sha = digest(diagonal_path)
            summary = {
                "train_positive_users": ease_summary.get("train_positive_users"),
                "train_positive_pairs": ease_summary.get("train_positive_pairs"),
                "singleton_positive_users": ease_summary.get("singleton_positive_users"),
                "catalog_items": len(apps), "linux_candidates": len(candidates),
                "candidate_inverse_columns_solved": len(candidates),
                "full_inverse_materialized": False,
                "gram_sha256": ease_summary.get("gram_sha256"),
                "sampled_solve_residual": ease_summary.get("sampled_solve_residual"),
                "solver_settings": ease_summary.get("solver_settings"),
                "candidate_inverse_diagonal_file": diagonal_path.name,
                "candidate_inverse_diagonal_sha256": diagonal_sha,
                "solver_reused_from": ease_name,
                "factorization_reused": True,
                "preserved_ease_reference": {
                    "candidate_weights": reference_weights.relative_to(EDA).as_posix(),
                    "candidate_weights_sha256": receipt["ease_candidate_weights_sha256"],
                    "inverse_candidate_diagonal": reference_diagonal.relative_to(EDA).as_posix(),
                    "inverse_candidate_diagonal_sha256": receipt["candidate_inverse_diagonal_sha256"],
                },
            }
            _finish_artifact(stage_by_name[name], name, "rlae", "dense_candidate_weights",
                             "linear", params, sources, support, apps, candidates, summary)
    except Exception:
        for array in output_by_name.values():
            _close_memmap(array)
        for stage in stage_by_name.values():
            if stage.exists():
                _safe_under_model_root(stage)
                shutil.rmtree(stage)
        raise
    finally:
        del base, inverse_diagonal
        gc.collect()
    return [name for name, _ in named]


def _rlae_weights_from_ease(ease_weights, inverse_diagonal, regularization: float,
                            xi: float, candidate_cols) -> np.ndarray:
    """Apply the RLAE target multiplier to one block of EASE candidate rows."""
    weights = np.asarray(ease_weights, dtype=np.float32)
    diag = np.asarray(inverse_diagonal, dtype=np.float64)
    cols = np.asarray(candidate_cols, dtype=np.int64)
    if (weights.ndim != 2 or diag.shape != (weights.shape[0],)
            or cols.shape != diag.shape or np.any(cols < 0)
            or np.any(cols >= weights.shape[1]) or regularization <= 0 or not 0 <= xi <= 1
            or np.any(~np.isfinite(diag)) or np.any(diag <= 0)):
        raise ValueError("RLAE candidate row inputs are invalid")
    multiplier = np.maximum(float(regularization) * diag, 1.0 - float(xi))
    out = (weights * multiplier[:, None]).astype(np.float32, copy=True)
    out[np.arange(len(cols)), cols] = 0.0
    return out


def _rdlae_diagonal_penalty(support: np.ndarray, regularization: float,
                            drop_p: float) -> np.ndarray:
    if regularization <= 0 or not 0 <= drop_p < 1:
        raise ValueError("RDLAE regularization/drop_p is invalid")
    return float(regularization) + np.asarray(support, dtype=np.float64) * (float(drop_p) / (1.0 - float(drop_p)))


def _linear_names(family: str, regularization: int, *, alpha=None, beta=0.0,
                  drop_p=0.0, scale_mode="raw", xi=None) -> list[tuple[str, dict]]:
    roster = _variant_roster()
    if family == "ease":
        names = [f"ease_lambda{regularization}_exact"]
    elif family == "rlae":
        xis = RLAE_XI if xi is None else (float(xi),)
        names = [f"rlae_lambda{regularization}_xi{_name_num(v)}" for v in xis]
    elif family == "rdlae":
        names = [f"rdlae_lambda{regularization}_p{_name_num(drop_p)}_xi{_name_num(RDLAE_XI)}"]
    elif family == "dan":
        alphas = SUPPORTED_DAN_ALPHA if alpha is None else (float(alpha),)
        names = [f"dan_ease_lambda{regularization}_a{_name_num(a)}_b{_name_num(beta)}"
                 f"_p{_name_num(drop_p)}_{scale_mode}" for a in alphas]
    else:
        raise ValueError(f"Unsupported linear family: {family}")
    missing = [name for name in names if name not in roster]
    if missing:
        raise ValueError(f"Requested variant is not frozen in roster: {missing}")
    return [(name, roster[name]["parameters"]) for name in names]


def build_linear_group(family: str, regularization: int, *, alpha=None,
                       beta: float = 0.0, drop_p: float = 0.0,
                       scale_mode: str = "raw", xi=None,
                       solver_backend: str = "cpu"):
    """Build EASE, DAN-EASE, RLAE xi sweep, or one RDLAE p setting.

    DAN's `scale_matched` mode multiplies the base lambda by the weighted to
    binary Gram trace ratio. The DAN paper/code's item-specific p support term
    remains as stated in the implementation.
    """
    freeze_roster()
    if family == "rlae":
        if _available_memory_gb() < 3.0:
            raise MemoryError("RLAE coefficient reuse requires >=3 GiB available RAM")
    else:
        _require_linear_resources()
    if regularization not in SUPPORTED_REGULARIZATION:
        raise ValueError("regularization must be 100 or 1000 for the frozen v4 roster")
    if family == "dan":
        if beta not in SUPPORTED_DAN_BETA or drop_p not in (0.0, 0.3):
            raise ValueError("DAN beta/drop_p is outside the frozen grid")
        if scale_mode not in {"raw", "scale_matched"}:
            raise ValueError("DAN scale_mode must be raw or scale_matched")
        if alpha is not None and alpha not in SUPPORTED_DAN_ALPHA:
            raise ValueError("DAN alpha is outside the frozen grid")
        effective_xi = None
    elif family == "rdlae":
        if beta != 0 or drop_p not in SUPPORTED_DROP_P:
            raise ValueError("RDLAE p is outside the frozen grid")
        effective_xi = RDLAE_XI
    elif family in {"ease", "rlae"}:
        beta, drop_p, scale_mode, alpha = 0.0, 0.0, "raw", None
        effective_xi = xi
        if family == "ease" and xi is not None:
            raise ValueError("EASE does not accept xi")
        if family == "rlae" and xi is not None and xi not in RLAE_XI:
            raise ValueError("RLAE xi is outside the frozen grid")
    else:
        raise ValueError(f"Unsupported linear family: {family}")

    named = _linear_names(family, regularization, alpha=alpha, beta=beta,
                          drop_p=drop_p, scale_mode=scale_mode, xi=effective_xi)
    if solver_backend not in {"cpu", "cuda"}:
        raise ValueError("solver_backend must be 'cpu' or 'cuda'")
    sources = _source_manifest()
    if family == "rlae":
        return _build_rlae_from_ease_cache(regularization, named, sources)
    # Avoid loading the 78M-edge CSR or rebuilding a Gram if every requested
    # artifact is already fingerprint-verified. RLAE references are required
    # sidecars and are independently checked before a completed sweep is reused.
    if all(_verify_existing(MODEL_ROOT / name, name, params, sources)
           for name, params in named):
        if family in {"ease", "rlae"}:
            work_dir = MODEL_ROOT / "_linear_work"
            _verify_preserved_ease_reference(work_dir, regularization,
                                             sources["train/positive_catalog.parquet"]["sha256"],
                                             6_630, 37_610)
        return [name for name, _ in named]
    x, support, users, apps, candidates = _positive_csr()
    source_hash = sources["train/positive_catalog.parquet"]["sha256"]
    work_dir = MODEL_ROOT / "_linear_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    gram, gram_diagonal, gram_path, gram_receipt = _build_gram_memmap(
        x, support, beta if family == "dan" else 0.0, source_hash, work_dir)
    if family == "dan":
        gram_scale = float(np.sum(gram_diagonal, dtype=np.float64)
                           / max(1.0, float(np.sum(support, dtype=np.float64))))
        effective_lambda = float(regularization) * (gram_scale if scale_mode == "scale_matched" else 1.0)
        gamma = effective_lambda + support.astype(np.float64) * (drop_p / (1.0 - drop_p))
    elif family == "rdlae":
        gram_scale = 1.0
        effective_lambda = float(regularization)
        gamma = _rdlae_diagonal_penalty(support, regularization, drop_p)
    else:
        gram_scale = 1.0
        effective_lambda = float(regularization)
        gamma = np.full(len(apps), effective_lambda, dtype=np.float64)
    if gamma.shape != support.shape or np.any(~np.isfinite(gamma)) or np.any(gamma <= 0):
        raise ValueError("The v4 diagonal penalty is invalid")

    stage_by_name = {}
    dense_by_name = {}
    output_parameters = {}
    factor = None
    factor_path = None
    ease_reference_weights = None
    ease_reference_diagonal = None
    ease_reference_tmp = None
    ease_reference_info = None
    ease_reference_diag_tmp = None
    dan_popularity_by_name = {}
    cuda_solver = None
    inverse_diagonal_values = np.empty(len(candidates), dtype=np.float32)
    try:
        for name, params in named:
            stage = _new_stage(name, params, sources)
            if stage is None:
                continue
            weights = np.lib.format.open_memmap(
                stage / "weights.npy", mode="w+", dtype=np.float32,
                shape=(len(candidates), len(apps)))
            stage_by_name[name] = stage
            dense_by_name[name] = weights
            output_parameters[name] = params
        if not dense_by_name:
            return [name for name, _ in named]

        if family == "dan":
            dan_popularity_by_name = {
                name: _dan_popularity_factors(support, output_parameters[name]["alpha"])
                for name in dense_by_name
            }

        if family == "ease":
            ref_weights, ref_diagonal, _ = _preserved_ease_reference_paths(
                work_dir, regularization)
            # A reference is replaced atomically only after this complete
            # candidate solve. The EASE coefficients and P_jj values are kept
            # outside each xi-specific 1 GiB model artifact to avoid copies.
            ease_reference_tmp = work_dir / f".ease_reference_lambda{regularization}_{os.getpid()}.npy"
            ease_reference_diag_tmp = work_dir / f".ease_reference_diag_lambda{regularization}_{os.getpid()}.npy"
            ease_reference_weights = np.lib.format.open_memmap(
                ease_reference_tmp, mode="w+", dtype=np.float32,
                shape=(len(candidates), len(apps)))
            ease_reference_diagonal = np.lib.format.open_memmap(
                ease_reference_diag_tmp,
                mode="w+", dtype=np.float32, shape=(len(candidates),))

        if solver_backend == "cuda":
            cuda_solver, gram = _factorize_cuda(
                gram, gamma, work_dir, batch_size=CANDIDATE_SOLVE_BATCH)
            solver_settings = cuda_solver.settings
        else:
            factor, factor_path, potrs, gram = _factorize(gram, gamma, work_dir)
            solver_settings = {
                "backend": "cpu",
                "library": "SciPy LAPACK",
                "dtype": "float32",
                "cholesky": "LAPACK potrf, lower triangle, in-place disk-backed factor",
                "candidate_solve": "LAPACK potrs",
                "candidate_solve_batch_columns": int(CANDIDATE_SOLVE_BATCH),
            }
        candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
        if cuda_solver is not None:
            residual_check = _sample_cuda_solve_residual(
                gram, gamma, cuda_solver, candidate_cols, sample_limit=3, tolerance=1e-4)
        else:
            residual_check = _sample_solve_residual(
                gram, gamma, factor, potrs, candidate_cols, sample_limit=3, tolerance=1e-4)
        log(f"v4 {family} sampled solve backward error: {residual_check['max_backward_error']:.3g}")
        for start in range(0, len(candidates), CANDIDATE_SOLVE_BATCH):
            stop = min(start + CANDIDATE_SOLVE_BATCH, len(candidates))
            count = stop - start
            target_cols = candidate_cols[start:stop]
            if cuda_solver is not None:
                solved = cuda_solver.solve_columns(target_cols)
            else:
                rhs = np.zeros((len(apps), count), dtype=np.float32, order="F")
                rhs[target_cols, np.arange(count)] = 1.0
                solved, info = potrs(factor, rhs, lower=1, overwrite_b=1)
                if info != 0:
                    raise np.linalg.LinAlgError(f"v4 candidate triangular solve failed: {info}")
            for local, row in enumerate(range(start, stop)):
                target = int(target_cols[local])
                inverse_diagonal = float(solved[target, local])
                if not np.isfinite(inverse_diagonal) or inverse_diagonal <= 0:
                    raise np.linalg.LinAlgError("Candidate inverse diagonal must be finite and positive")
                inverse_diagonal_values[row] = np.float32(inverse_diagonal)
                inverse_column = solved[:, local]
                if family == "dan":
                    base = -inverse_column / inverse_diagonal
                    # alpha may vary inside a group; apply its factors per output.
                    for name in dense_by_name:
                        input_factor, candidate_factor = dan_popularity_by_name[name]
                        row_values = (base * input_factor * candidate_factor[target]).astype(np.float32, copy=False)
                        row_values[target] = 0.0
                        dense_by_name[name][row, :] = row_values
                elif family == "ease":
                    row_values = -inverse_column / inverse_diagonal
                    row_values[target] = 0.0
                    dense_by_name[named[0][0]][row, :] = row_values.astype(np.float32, copy=False)
                    ease_reference_weights[row, :] = row_values.astype(np.float32, copy=False)
                    ease_reference_diagonal[row] = np.float32(inverse_diagonal)
                elif family == "rlae":
                    base = -inverse_column / inverse_diagonal
                    base[target] = 0.0
                    ease_reference_weights[row, :] = base.astype(np.float32, copy=False)
                    ease_reference_diagonal[row] = np.float32(inverse_diagonal)
                    for name in dense_by_name:
                        xi_value = float(output_parameters[name]["xi"])
                        multiplier = float(_rlae_multiplier(
                            np.asarray([effective_lambda]), np.asarray([inverse_diagonal]), xi_value)[0])
                        row_values = base * multiplier
                        row_values[target] = 0.0
                        dense_by_name[name][row, :] = row_values.astype(np.float32, copy=False)
                else:  # RDLAE
                    multiplier = float(_rlae_multiplier(
                        np.asarray([gamma[target]]), np.asarray([inverse_diagonal]), RDLAE_XI)[0])
                    row_values = (-inverse_column / inverse_diagonal) * multiplier
                    row_values[target] = 0.0
                    dense_by_name[named[0][0]][row, :] = row_values.astype(np.float32, copy=False)
            if cuda_solver is None:
                del rhs
            del solved
            if start == 0 or stop == len(candidates) or start % 512 == 0:
                for array in dense_by_name.values():
                    array.flush()
                log(f"v4 {family} lambda={regularization}: solved {stop:,}/{len(candidates):,} candidate columns")

        if family == "ease":
            ref_weights, ref_diagonal, ref_receipt_path = _preserved_ease_reference_paths(
                work_dir, regularization)
            ease_reference_weights.flush()
            ease_reference_diagonal.flush()
            _close_memmap(ease_reference_weights)
            _close_memmap(ease_reference_diagonal)
            ease_reference_weights = ease_reference_diagonal = None
            os.replace(ease_reference_tmp, ref_weights)
            os.replace(ease_reference_diag_tmp, ref_diagonal)
            ease_reference_info = {
                "complete": True, "regularization": int(regularization),
                "train_sha256": source_hash,
                "input_app_ids_sha256": __import__("hashlib").sha256(apps.tobytes()).hexdigest(),
                "candidate_app_ids_sha256": __import__("hashlib").sha256(candidates.tobytes()).hexdigest(),
                "candidate_inverse_diagonal_sha256": digest(ref_diagonal),
                "ease_candidate_weights_sha256": digest(ref_weights),
                "candidate_count": len(candidates), "input_count": len(apps),
            }
            _atomic_json(ref_receipt_path, ease_reference_info)

        train_degree = np.diff(x.indptr)
        common_summary = {
            "train_positive_users": int(users),
            "train_positive_pairs": int(x.nnz),
            "singleton_positive_users": int(np.count_nonzero(train_degree == 1)),
            "catalog_items": int(len(apps)),
            "linux_candidates": int(len(candidates)),
            "candidate_inverse_columns_solved": int(len(candidates)),
            "full_inverse_materialized": False,
            "gram_sha256": gram_receipt["sha256"],
            "gram_dtype": "float32",
            "cholesky_dtype": "float32",
            "solver_settings": solver_settings,
            "estimated_dense_weights_gib_per_variant": round(len(candidates)*len(apps)*4/1024**3, 3),
            "available_ram_gate_gib": 10.0,
            "sampled_solve_residual": residual_check,
        }
        for name, params in named:
            if name not in dense_by_name:
                continue
            diagonal_path = stage_by_name[name] / "candidate_inverse_diagonal.npy"
            np.save(diagonal_path, inverse_diagonal_values, allow_pickle=False)
            diagonal_sha = digest(diagonal_path)
            array = dense_by_name.pop(name)
            array.flush()
            _close_memmap(array)
            del array
            fit_summary = dict(common_summary, regularization=int(regularization),
                               effective_lambda=float(effective_lambda),
                               gram_scale_match_ratio=float(gram_scale),
                               beta=float(beta), drop_p=float(drop_p),
                               xi=params.get("xi"), alpha=params.get("alpha"),
                               candidate_inverse_diagonal_file=diagonal_path.name,
                               candidate_inverse_diagonal_sha256=diagonal_sha,
                               gram_source_file=gram_path.name,
                               formula_reference=(DAN_REFERENCE.get("repository") if family == "dan"
                                                  else RDLAE_REFERENCE.get("repository") if family in {"rlae", "rdlae"}
                                                  else None))
            if family == "ease":
                ref_weights, ref_diagonal, ref_receipt_path = _preserved_ease_reference_paths(
                    work_dir, regularization)
                fit_summary["preserved_ease_reference"] = {
                    "candidate_weights": ref_weights.relative_to(EDA).as_posix(),
                    "inverse_candidate_diagonal": ref_diagonal.relative_to(EDA).as_posix(),
                    "receipt": ref_receipt_path.relative_to(EDA).as_posix(),
                    "candidate_weights_sha256": ease_reference_info["ease_candidate_weights_sha256"],
                    "inverse_candidate_diagonal_sha256": ease_reference_info["candidate_inverse_diagonal_sha256"],
                }
            _finish_artifact(stage_by_name[name], name,
                             "dan_ease" if family == "dan" else family,
                             "dense_candidate_weights", "linear", params, sources,
                             support, apps, candidates, fit_summary)
        if cuda_solver is None:
            _close_memmap(factor)
            factor_path.unlink(missing_ok=True)
            factor = None
    except Exception:
        _close_memmap(ease_reference_weights)
        _close_memmap(ease_reference_diagonal)
        if ease_reference_tmp is not None:
            ease_reference_tmp.unlink(missing_ok=True)
        if ease_reference_diag_tmp is not None:
            ease_reference_diag_tmp.unlink(missing_ok=True)
        if factor is not None:
            _close_memmap(factor)
        if factor_path is not None:
            factor_path.unlink(missing_ok=True)
        for array in dense_by_name.values():
            _close_memmap(array)
        for stage in stage_by_name.values():
            if stage.exists():
                _safe_under_model_root(stage)
                shutil.rmtree(stage)
        raise
    finally:
        if cuda_solver is not None:
            cuda_solver.close()
        _close_memmap(gram)
        del x
        gc.collect()
    return [name for name, _ in named]


def _coordinate_descent_nonnegative(gram, cross, l1: float, l2: float,
                                    tolerance: float = 1e-5, max_iterations: int = 1000):
    """Cyclic exact coordinate updates for nonnegative Elastic Net coefficients."""
    q = np.asarray(gram, dtype=np.float64)
    c = np.asarray(cross, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or c.shape != (q.shape[0],):
        raise ValueError("fSLIM Gram/cross shapes are inconsistent")
    if (np.any(~np.isfinite(q)) or np.any(~np.isfinite(c)) or l1 < 0 or l2 < 0
            or tolerance <= 0 or max_iterations < 1):
        raise ValueError("fSLIM coordinate-descent inputs are invalid")
    q = (q + q.T) * 0.5
    w = np.zeros(len(c), dtype=np.float64)
    residual = c.copy()  # c - Qw
    diagonal = np.diag(q)
    if np.any(diagonal < 0):
        raise ValueError("fSLIM Gram diagonal cannot be negative")
    scale = max(1.0, float(np.max(np.abs(c))) if len(c) else 1.0)
    if not len(c):
        return w, 0, True
    converged = False
    for iteration in range(1, int(max_iterations) + 1):
        max_change = 0.0
        for j in range(len(c)):
            denom = diagonal[j] + l2
            if denom <= 0:
                continue
            old = w[j]
            without_self = residual[j] + diagonal[j] * old
            new = max(0.0, (without_self - l1) / denom)
            change = new - old
            if change:
                w[j] = new
                residual -= change * q[:, j]
                max_change = max(max_change, abs(change))
        if max_change <= tolerance * max(1.0, float(np.max(w))):
            converged = True
            break
    return w.astype(np.float32), iteration, converged


def _build_fslim():
    """Fit nonnegative, neighbor-restricted fSLIM with candidate-wise Gram CD."""
    freeze_roster()
    # One sparse positive matrix plus one 500x500 dense Gram fits well below the
    # full-catalog Cholesky peak, but the iteration count is intentionally
    # conservative and the command remains opt-in.
    if _available_memory_gb() < 3.0:
        raise MemoryError("fSLIM requires >=3 GiB available RAM for the train CSR and candidate Grams")
    sources = _source_manifest()
    x, support, users, apps, candidates = _positive_csr()
    x_csc = x.astype(np.int32).tocsc()
    item_rows = x_csc.T.tocsr()
    candidate_cols = np.searchsorted(apps, candidates).astype(np.int32)
    roster = _variant_roster()
    names = [f"fslim_m{m}_l2{l2}_l1r{_name_num(r)}"
             for m in (200, 500) for l2 in (100, 1000) for r in (0.0, .001, .01)]
    neighbor_names = ["fslim_neighbor_cosine_k200", "fslim_neighbor_cosine_k500"]
    sparse_names = names + neighbor_names
    params = {name: roster[name]["parameters"] for name in sparse_names}
    active = [name for name in sparse_names
              if not _verify_existing(MODEL_ROOT / name, name, params[name], sources)]
    if not active:
        del item_rows, x_csc, x
        return sparse_names

    row_cols = {name: [] for name in active}
    row_data = {name: [] for name in active}
    candidate_zero_counts = {name: 0 for name in names if name in active}
    nonconverged = {name: 0 for name in names if name in active}
    iterations = {name: 0 for name in names if name in active}
    try:
        for start in range(0, len(candidates), BLOCK_SIZE):
            stop = min(start + BLOCK_SIZE, len(candidates))
            cooc = (item_rows[candidate_cols[start:stop], :] @ x_csc).tocsr()
            cooc.sum_duplicates()
            cooc.sort_indices()
            for local in range(stop - start):
                row = start + local
                ccol = int(candidate_cols[row])
                begin, end = cooc.indptr[local:local+2]
                all_cols = cooc.indices[begin:end].astype(np.int32, copy=False)
                all_counts = cooc.data[begin:end].astype(np.int64, copy=False)
                neighbor_sets = {}
                for m in (200, 500):
                    cols, cos = _top_cosine_neighbors(ccol, all_cols, all_counts,
                                                      apps, support, m)
                    valid = ((all_cols != ccol) & (all_counts > 0) & (support[all_cols] > 0))
                    ranked_cols = all_cols[valid]
                    ranked_counts = all_counts[valid].astype(np.float64)
                    ranked_scores = ranked_counts / np.sqrt(
                        float(support[ccol]) * support[ranked_cols].astype(np.float64))
                    rank_order = np.lexsort((apps[ranked_cols], -ranked_scores))[:m]
                    neighbor_sets[m] = ranked_cols[rank_order].astype(np.int32, copy=False)
                    baseline_name = f"fslim_neighbor_cosine_k{m}"
                    if baseline_name in active:
                        row_cols[baseline_name].append(cols)
                        row_data[baseline_name].append(cos.astype(np.float32, copy=False))

                selected500 = neighbor_sets[500]
                if not len(selected500):
                    for m in (200, 500):
                        chosen = neighbor_sets[m]
                        for l2 in (100, 1000):
                            for relative in (0.0, .001, .01):
                                name = f"fslim_m{m}_l2{l2}_l1r{_name_num(relative)}"
                                if name in active:
                                    row_cols[name].append(np.empty(0, np.int32))
                                    row_data[name].append(np.empty(0, np.float32))
                                    candidate_zero_counts[name] += 1
                    continue
                feature = x_csc[:, selected500]
                gram500 = (feature.T @ feature).toarray().astype(np.float64)
                loc = np.searchsorted(all_cols, selected500)
                if (np.any(loc >= len(all_cols)) or not np.array_equal(all_cols[loc], selected500)):
                    raise ValueError("Selected fSLIM neighbor lacks positive candidate co-occurrence")
                cross500 = all_counts[loc].astype(np.float64)
                del feature
                for m in (200, 500):
                    chosen = neighbor_sets[m]
                    width = len(chosen)
                    if not width:
                        continue
                    # Top-m ranking order is preserved by the first m columns
                    # of the top-500 feature list.
                    q = gram500[:width, :width]
                    cross = cross500[:width]
                    lambda_max = float(np.max(cross)) if width else 0.0
                    for l2 in (100, 1000):
                        for relative in (0.0, .001, .01):
                            name = f"fslim_m{m}_l2{l2}_l1r{_name_num(relative)}"
                            if name not in active:
                                continue
                            l1 = relative * lambda_max
                            coefficients, steps, converged = _coordinate_descent_nonnegative(
                                q, cross, l1, l2, tolerance=1e-5, max_iterations=1000)
                            nz = coefficients > 0
                            out_cols = chosen[nz]
                            out_values = coefficients[nz]
                            order = np.argsort(out_cols, kind="stable")
                            out_cols, out_values = out_cols[order], out_values[order]
                            row_cols[name].append(out_cols.astype(np.int32, copy=False))
                            row_data[name].append(out_values.astype(np.float32, copy=False))
                            candidate_zero_counts[name] += int(not np.any(nz))
                            nonconverged[name] += int(not converged)
                            iterations[name] += steps
                del gram500, cross500
            del cooc
            if start == 0 or stop == len(candidates) or start % 512 == 0:
                log(f"v4 fSLIM: solved candidate rows {stop:,}/{len(candidates):,}")

        summary = {"train_positive_users": int(users), "train_positive_pairs": int(x.nnz),
                   "singleton_positive_users": int(np.count_nonzero(np.diff(x.indptr) == 1)),
                   "candidate_wise_coordinate_descent": True,
                   "coefficient_zero_rate": {}, "nonconverged_candidate_fits": {},
                   "mean_iterations": {}}
        for name in active:
            matrix = _csr_from_row_lists(row_cols[name], row_data[name],
                                         (len(candidates), len(apps)))
            fit_summary = dict(summary)
            if name.startswith("fslim_m"):
                n_rows = max(1, len(candidates))
                fit_summary["candidate_zero_coefficient_rows"] = candidate_zero_counts[name]
                fit_summary["candidate_zero_coefficient_rate"] = candidate_zero_counts[name] / n_rows
                fit_summary["nonconverged_candidate_fits"] = nonconverged[name]
                fit_summary["mean_coordinate_descent_iterations"] = iterations[name] / n_rows
            _save_sparse_artifact(name, matrix,
                                  "fslim" if name.startswith("fslim_m") else "fslim_neighbor_baseline",
                                  params[name], sources, support, apps, candidates, fit_summary)
            del matrix
        return sparse_names
    except Exception:
        raise
    finally:
        del item_rows, x_csc, x
        gc.collect()


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-roster", help="write the fixed v4 variant roster before fitting")
    sub.add_parser("build-knn", help="build binary/activity KNN and account-pair rho sweeps")
    sub.add_parser("build-fslim", help="fit neighbor-restricted nonnegative fSLIM grids")
    linear = sub.add_parser("build-linear", help="fit one EASE/DAN/RLAE/RDLAE parameter group")
    linear.add_argument("--family", choices=("ease", "dan", "rlae", "rdlae"), required=True)
    linear.add_argument("--regularization", type=int, choices=SUPPORTED_REGULARIZATION, required=True)
    linear.add_argument("--alpha", type=float, choices=SUPPORTED_DAN_ALPHA)
    linear.add_argument("--beta", type=float, choices=SUPPORTED_DAN_BETA, default=0.0)
    linear.add_argument("--drop-p", type=float, choices=(0.0, 0.1, 0.3), default=0.0)
    linear.add_argument("--scale-mode", choices=("raw", "scale_matched"), default="raw")
    linear.add_argument("--xi", type=float, choices=RLAE_XI)
    linear.add_argument("--solver-backend", choices=("cpu", "cuda"), default="cpu",
                        help="use the disk-backed SciPy or verified CUDA Cholesky solver")
    args = parser.parse_args()
    if args.command == "freeze-roster":
        print(freeze_roster())
    elif args.command == "build-knn":
        _build_knn_and_pair_models()
    elif args.command == "build-fslim":
        _build_fslim()
    else:
        if args.family == "dan" and args.drop_p not in (0.0, 0.3):
            parser.error("DAN drop-p is only 0 or 0.3")
        if args.family != "dan" and args.alpha is not None:
            parser.error("--alpha applies only to DAN")
        if args.family not in {"dan"} and args.beta != 0.0:
            parser.error("--beta applies only to DAN")
        if args.family not in {"dan", "rdlae"} and args.drop_p != 0.0:
            parser.error("--drop-p applies only to DAN and RDLAE")
        if args.family in {"ease", "dan", "rdlae"} and args.xi is not None:
            parser.error("--xi is only configurable for RLAE; RDLAE uses the frozen xi=0.5")
        names = build_linear_group(args.family, args.regularization,
                                   alpha=args.alpha, beta=args.beta,
                                   drop_p=args.drop_p, scale_mode=args.scale_mode,
                                   xi=args.xi, solver_backend=args.solver_backend)
        for name in names:
            print(name)


if __name__ == "__main__":
    _cli()
