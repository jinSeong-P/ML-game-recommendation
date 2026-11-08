"""Fit and evaluate the predeclared temporal v4 recommendation controls.

Only T0-positive pairs from experiments_v4/temporal_split are used to fit the
fresh collaborative controls. Evaluation is limited to the temporal selection
role; confirmation is deliberately unavailable in this runner. Run the stages
explicitly, for example:

    python EDA/scripts/run_temporal_models_v4.py --stage train
    python EDA/scripts/run_temporal_models_v4.py --stage evaluate

The default output is kept separate from both the source split and the regular
v4 model/evaluation artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize

from eda_large_review_frank import EDA, digest, log
from evaluate_baselines import rank_positive
from improvement_common import LOG_LABELS, macro, paired_rank_ci, write_json
from item_knn import _build_user_item_csr


EXPERIMENTS = EDA / "data/processed/modeling/experiments_v4"
TEMPORAL = EXPERIMENTS / "temporal_split"
REFERENCE_PROTOCOL = EDA / "data/reference/temporal_protocol_v4.json"
OUTPUT = EXPERIMENTS / "temporal_controls_v4"
TRAINING = OUTPUT / "training"
MODELS = OUTPUT / "models"
EVALUATIONS = OUTPUT / "evaluations"
CONTENT_SOURCE = EDA / "data/processed/modeling/improvement_v3/content/metadata_extended_tfidf"

ROLE = "selection"
VARIANTS = ("binary_item_knn_k100", "activity_inverse_degree_cosine_k100", "fixed_tfidf_content")
K_NEIGHBORS = 100
PAIRED_REPS = 2_000
BOOTSTRAP_SEED = 20261108
BATCH_SIZE = 128
PAIR_SCHEMA = pa.schema([
    ("user_key", pa.uint64()), ("app_id", pa.int64()), ("positive", pa.bool_()),
])


def _strict_positions(values, reference, label: str) -> np.ndarray:
    values = np.asarray(values)
    reference = np.asarray(reference)
    positions = np.searchsorted(reference, values)
    if np.any(positions >= len(reference)) or np.any(reference[positions] != values):
        raise ValueError(f"{label} contains IDs outside its fixed catalog")
    return positions.astype(np.int64, copy=False)


def _read_verified_table(root: Path, relative: str, manifest: dict) -> pa.Table:
    path = root / relative
    receipt = manifest.get("files", {}).get(relative)
    if not receipt or not path.is_file():
        raise ValueError(f"Temporal split is missing its file receipt: {relative}")
    if path.stat().st_size != receipt.get("bytes") or digest(path) != receipt.get("sha256"):
        raise ValueError(f"Temporal split fingerprint changed: {relative}")
    table = pq.read_table(path)
    if table.num_rows != receipt.get("rows"):
        raise ValueError(f"Temporal split row count changed: {relative}")
    return table


def verify_temporal_split(root: Path = TEMPORAL) -> tuple[dict, str, str]:
    """Validate frozen temporal inputs before reading pairs or cohorts."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    protocol_path = root / "protocol.json"
    if not manifest_path.is_file() or not protocol_path.is_file():
        raise ValueError("Temporal v4 split is not materialized")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("validation", {}).get("passed"):
        raise ValueError("Temporal v4 split is incomplete or failed validation")
    protocol_hash = digest(protocol_path)
    reference_hash = digest(REFERENCE_PROTOCOL)
    if protocol_hash != reference_hash or manifest.get("protocol", {}).get("sha256") != protocol_hash:
        raise ValueError("Temporal protocol receipt differs from the frozen reference")
    if manifest.get("confirmation", {}).get("scored") is not False:
        raise ValueError("Temporal split confirmation must remain unscored")
    if int(manifest.get("training", {}).get("reserved_account_pair_overlap", -1)) != 0:
        raise ValueError("Temporal training contains reserved role accounts")
    return manifest, digest(manifest_path), protocol_hash


def load_temporal_catalog(root: Path = TEMPORAL) -> dict:
    manifest, manifest_hash, protocol_hash = verify_temporal_split(root)
    root = Path(root).resolve()
    apps_table = _read_verified_table(root, "catalog/input_apps.parquet", manifest)
    candidates_table = _read_verified_table(root, "catalog/linux_candidates.parquet", manifest)
    availability_table = _read_verified_table(root, "catalog/candidate_availability.parquet", manifest)
    apps_rows = apps_table.to_pydict()
    apps = np.asarray(apps_rows["app_id"], dtype=np.int64)
    linux = np.asarray(apps_rows["linux"], dtype=bool)
    candidates = np.asarray(candidates_table.column("app_id").to_numpy(), dtype=np.int64)
    if (len(apps) != 37_610 or len(candidates) != 6_630 or int(linux.sum()) != 6_630
            or np.any(apps <= 0) or np.any(np.diff(apps) <= 0)
            or np.any(np.diff(candidates) <= 0) or not np.array_equal(apps[linux], candidates)):
        raise ValueError("The fixed 37,610 input / 6,630 candidate catalog changed")
    availability = availability_table.to_pydict()
    available_ids = np.asarray(availability["app_id"], dtype=np.int64)
    available = np.asarray(availability["available_at_t0"], dtype=bool)
    if (len(available_ids) != 6_630 or not np.array_equal(available_ids, candidates)
            or int(available.sum()) != int(manifest.get("availability", {}).get("available_at_t0", -1))):
        raise ValueError("T0 availability mask does not preserve the complete candidate array")
    return {
        "apps": apps, "linux": linux, "candidate_ids": candidates,
        "candidate_input_cols": np.flatnonzero(linux).astype(np.int32),
        "available": available,
        "hashes": {
            "temporal_manifest_sha256": manifest_hash,
            "temporal_protocol_sha256": protocol_hash,
            "input_apps_sha256": digest(root / "catalog/input_apps.parquet"),
            "linux_candidates_sha256": digest(root / "catalog/linux_candidates.parquet"),
            "candidate_availability_sha256": digest(root / "catalog/candidate_availability.parquet"),
        },
    }


def load_selection_cohort(root: Path = TEMPORAL, catalog: dict | None = None) -> dict:
    """Load every target-eligible selection account, including short histories."""
    role = ROLE
    if role != "selection":  # Keep the role fixed even if edited by a caller.
        raise ValueError("Temporal v4 controls can evaluate selection only")
    manifest, manifest_hash, protocol_hash = verify_temporal_split(root)
    root = Path(root).resolve()
    catalog = catalog or load_temporal_catalog(root)
    users_t = _read_verified_table(root, "selection/users.parquet", manifest).to_pydict()
    query_t = _read_verified_table(root, "selection/query.parquet", manifest).to_pydict()
    targets_t = _read_verified_table(root, "selection/targets.parquet", manifest).to_pydict()
    exclusions_t = _read_verified_table(root, "selection/exclusions.parquet", manifest).to_pydict()

    apps = catalog["apps"]
    candidates = catalog["candidate_ids"]
    available = catalog["available"]
    user_ids = np.asarray(users_t["user_key"], dtype=np.uint64)
    if not len(user_ids) or np.any(user_ids[1:] <= user_ids[:-1]):
        raise ValueError("Selection users must be unique, ascending uint64 IDs")
    query_users = np.asarray(query_t["user_key"], dtype=np.uint64)
    query_apps = np.asarray(query_t["app_id"], dtype=np.int64)
    query_rows = _strict_positions(query_users, user_ids, "Query account")
    query_cols = _strict_positions(query_apps, apps, "Query App")
    query_positive = np.asarray(query_t["positive"], dtype=bool)
    query_linux = np.asarray(query_t["linux"], dtype=bool)
    if not np.array_equal(query_linux, catalog["linux"][query_cols]):
        raise ValueError("Temporal query Linux flags differ from the fixed input catalog")
    shape = (len(user_ids), len(apps))
    all_query = sparse.csr_matrix((np.ones(len(query_rows), dtype=np.uint8),
                                   (query_rows, query_cols)), shape=shape)
    if all_query.nnz != len(query_rows) or np.any(all_query.data != 1):
        raise ValueError("Temporal query has duplicate account/App pairs")
    positive = sparse.csr_matrix((np.ones(int(query_positive.sum()), dtype=np.float32),
                                  (query_rows[query_positive], query_cols[query_positive])), shape=shape)
    if positive.nnz != int(query_positive.sum()):
        raise ValueError("Temporal positive query has duplicate account/App pairs")

    n_total = np.asarray(all_query.sum(axis=1)).ravel().astype(np.int64)
    n_positive = np.asarray(positive.sum(axis=1)).ravel().astype(np.int64)
    if (not np.array_equal(n_total, np.asarray(users_t["n_total_inputs"], dtype=np.int64))
            or not np.array_equal(n_positive, np.asarray(users_t["n_positive_inputs"], dtype=np.int64))):
        raise ValueError("Selection query counts differ from the temporal user roster")

    target_users = np.asarray(targets_t["user_key"], dtype=np.uint64)
    target_ids = np.asarray(targets_t["app_id"], dtype=np.int64)
    if (not np.array_equal(target_users, user_ids)
            or not np.all(np.asarray(targets_t["positive"], dtype=bool))
            or not np.all(np.asarray(targets_t["linux"], dtype=bool))
            or not np.all(np.asarray(targets_t["target_order"], dtype=np.int64) == 1)):
        raise ValueError("Selection targets must be one positive Linux candidate per user")
    target_candidates = _strict_positions(target_ids, candidates, "Target candidate")
    if not np.all(available[target_candidates]):
        raise ValueError("Selection target includes a candidate unavailable at T0")
    target_cols = _strict_positions(target_ids, apps, "Target App")
    target_history = np.asarray(all_query[np.arange(len(user_ids)), target_cols]).ravel()
    if np.any(target_history):
        raise ValueError("A hidden target is visible in the temporal history")

    excluded_users = np.asarray(exclusions_t["user_key"], dtype=np.uint64)
    excluded_apps = np.asarray(exclusions_t["app_id"], dtype=np.int64)
    erows = _strict_positions(excluded_users, user_ids, "Exclusion account")
    ecols = _strict_positions(excluded_apps, apps, "Exclusion App")
    exclusions = sparse.csr_matrix((np.ones(len(erows), dtype=np.uint8), (erows, ecols)), shape=shape)
    if exclusions.nnz != all_query.nnz or (exclusions != all_query).nnz:
        raise ValueError("Temporal exclusions must equal all visible positive and negative inputs")

    return {
        "name": ROLE, "user_ids": user_ids, "positive": positive,
        "exclusions": exclusions[:, catalog["candidate_input_cols"]].tocsr(),
        "target_ids": target_ids, "target_candidates": target_candidates,
        "n_total": n_total, "n_positive": n_positive,
        "hashes": {
            **catalog["hashes"], "temporal_manifest_sha256": manifest_hash,
            "temporal_protocol_sha256": protocol_hash,
            "selection_users_sha256": digest(root / "selection/users.parquet"),
            "selection_query_sha256": digest(root / "selection/query.parquet"),
            "selection_targets_sha256": digest(root / "selection/targets.parquet"),
            "selection_exclusions_sha256": digest(root / "selection/exclusions.parquet"),
        },
    }


def primary_mask(cohort: dict) -> np.ndarray:
    """Frozen v4 primary eligibility: at least five inputs and one positive."""
    return (np.asarray(cohort["n_total"]) >= 5) & (np.asarray(cohort["n_positive"]) >= 1)


def activity_inverse_degree_row_scale(degree: np.ndarray) -> np.ndarray:
    """Square-root row multiplier for the frozen inverse-user-degree kernel."""
    degree = np.asarray(degree, dtype=np.float64)
    if degree.ndim != 1 or np.any(~np.isfinite(degree)) or np.any(degree <= 0):
        raise ValueError("Activity inverse-degree requires positive finite user degrees")
    return np.sqrt(1.0 / degree).astype(np.float32)


def _build_similarity(x: sparse.csr_matrix, apps: np.ndarray, candidate_cols: np.ndarray,
                      mode: str, neighbors: int = K_NEIGHBORS, block_size: int = 32) -> sparse.csr_matrix:
    """Build candidate-by-input top cosine rows from the supplied fresh CSR."""
    if mode not in ("binary", "activity_inverse_degree"):
        raise ValueError("Unknown temporal KNN weighting")
    if not sparse.isspmatrix_csr(x) or x.shape[1] != len(apps):
        raise ValueError("Training matrix must be CSR on the fixed input catalog")
    if neighbors < 1 or block_size < 1:
        raise ValueError("Neighbor and block sizes must be positive")
    degree = np.diff(x.indptr)
    if np.any(degree <= 0):
        raise ValueError("Temporal training CSR may not contain empty users")
    if mode == "binary":
        weighted = x.astype(np.float32, copy=False)
    else:
        row_scale = activity_inverse_degree_row_scale(degree)
        weighted = x.astype(np.float32, copy=False).multiply(row_scale[:, None]).tocsr()
    margins = np.asarray(weighted.power(2).sum(axis=0)).ravel().astype(np.float64)
    # Slice candidate rows repeatedly in CSR. Retaining CSC here would scan
    # the 26M-user column-pointer array for every 32-candidate block. This one
    # conversion trades a bounded temporary copy for fast row access.
    candidate_rows = weighted[:, candidate_cols].T.tocsr()
    indptr = [0]
    out_indices: list[int] = []
    out_values: list[float] = []
    for begin in range(0, len(candidate_cols), block_size):
        end = min(len(candidate_cols), begin + block_size)
        block = (candidate_rows[begin:end] @ weighted).tocsr()
        for local, own_col in enumerate(candidate_cols[begin:end]):
            start, stop = block.indptr[local:local + 2]
            cols = block.indices[start:stop]
            values = block.data[start:stop].astype(np.float64, copy=False)
            keep = (cols != own_col) & (values > 0) & (margins[cols] > 0) & (margins[own_col] > 0)
            cols, values = cols[keep], values[keep]
            values = values / np.sqrt(margins[own_col] * margins[cols])
            chosen = np.lexsort((apps[cols], -values))[:neighbors]
            cols, values = cols[chosen], values[chosen]
            storage_order = np.argsort(cols, kind="stable")
            out_indices.extend(cols[storage_order].astype(np.int32, copy=False).tolist())
            out_values.extend(values[storage_order].astype(np.float32, copy=False).tolist())
            indptr.append(len(out_indices))
        log(f"temporal {mode} KNN: {end:,}/{len(candidate_cols):,} candidates")
    matrix = sparse.csr_matrix((np.asarray(out_values, dtype=np.float32),
        np.asarray(out_indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=(len(candidate_cols), len(apps)))
    matrix.sort_indices()
    if np.any(np.diff(matrix.indptr) > neighbors):
        raise ValueError("Stored temporal KNN row exceeds the fixed neighbor count")
    for row, col in enumerate(candidate_cols):
        start, stop = matrix.indptr[row:row + 2]
        if np.any(matrix.indices[start:stop] == col):
            raise ValueError("Temporal KNN model contains a self neighbor")
    return matrix


def _training_source(root: Path = TEMPORAL) -> tuple[dict, dict, Path, int, int]:
    temporal_manifest, manifest_hash, protocol_hash = verify_temporal_split(root)
    root = Path(root).resolve()
    train_path = root / "train/positive_catalog.parquet"
    receipt = temporal_manifest.get("files", {}).get("train/positive_catalog.parquet")
    if not receipt or not train_path.is_file():
        raise ValueError("Temporal T0 positive training artifact is missing")
    if train_path.stat().st_size != receipt.get("bytes") or digest(train_path) != receipt.get("sha256"):
        raise ValueError("Temporal T0 positive training fingerprint changed")
    expected_pairs = int(temporal_manifest["training"]["positive_pairs"])
    expected_users = int(temporal_manifest["training"]["positive_users"])
    parquet = pq.ParquetFile(train_path)
    if parquet.metadata.num_rows != expected_pairs:
        raise ValueError("Temporal training pair count differs from its split manifest")
    schema = parquet.schema_arrow
    if schema != PAIR_SCHEMA:
        raise ValueError("Temporal training pair schema changed")
    return temporal_manifest, {
        "temporal_manifest_sha256": manifest_hash, "temporal_protocol_sha256": protocol_hash,
        "training_pairs_sha256": digest(train_path), "training_pairs": expected_pairs,
        "training_users": expected_users,
    }, train_path, expected_pairs, expected_users


def _positive_training_batches(train_path: Path, batch_size: int = 131_072):
    for batch in pq.ParquetFile(train_path).iter_batches(
            batch_size=batch_size, columns=["user_key", "app_id", "positive"]):
        positive = np.asarray(batch.column(batch.schema.get_field_index("positive")).to_numpy(zero_copy_only=False), dtype=bool)
        if not np.all(positive):
            raise ValueError("Temporal positive_catalog.parquet contains a negative training pair")
        yield batch


def training_matrix(root: Path = TEMPORAL, output: Path = TRAINING) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    temporal_manifest, source, train_path, expected_pairs, expected_users = _training_source(root)
    catalog = load_temporal_catalog(root)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix_path, support_path, receipt_path = output / "positive_csr.npz", output / "support.npy", output / "manifest.json"
    if receipt_path.exists():
        manifest = json.loads(receipt_path.read_text("utf-8"))
        if (manifest.get("source") != source or manifest.get("status") != "complete"
                or manifest.get("code_sha256") != digest(__file__)):
            raise ValueError("Existing temporal CSR cache belongs to different inputs/code")
        for path, key in ((matrix_path, "positive_csr_sha256"), (support_path, "support_sha256")):
            if not path.is_file() or digest(path) != manifest.get(key):
                raise ValueError(f"Existing temporal CSR cache changed: {path.name}")
        x = sparse.load_npz(matrix_path)
        support = np.load(support_path, mmap_mode="r")
        if x.shape != (expected_users, len(catalog["apps"])) or x.nnz != expected_pairs:
            raise ValueError("Existing temporal CSR cache has the wrong shape or pair count")
        return x, np.asarray(support), manifest
    if matrix_path.exists() or support_path.exists():
        raise ValueError("Partial temporal CSR cache exists without a receipt; inspect it before retrying")
    log(f"Building fresh temporal CSR: {expected_pairs:,} positive pairs / {expected_users:,} users")
    x, support, _ = _build_user_item_csr(
        _positive_training_batches(train_path), catalog["apps"], expected_pairs, expected_users)
    if x.shape != (expected_users, 37_610) or x.nnz != expected_pairs:
        raise ValueError("Fresh temporal CSR dimensions differ from the pinned training split")
    sparse.save_npz(matrix_path, x, compressed=True)
    np.save(support_path, support)
    manifest = {
        "status": "complete", "scope": "fresh T0 temporal positive train pairs only",
        "source": source, "shape": list(x.shape), "nnz": int(x.nnz),
        "positive_csr_sha256": digest(matrix_path), "support_sha256": digest(support_path),
        "code_sha256": digest(__file__),
    }
    write_json(receipt_path, manifest)
    return x, support, manifest


def fit_models(root: Path = TEMPORAL, output: Path = OUTPUT) -> dict:
    """Fit the two fresh temporal KNN models; never reads full-data CF models."""
    output = Path(output).resolve()
    training_dir, models_dir = output / "training", output / "models"
    x, support, train_receipt = training_matrix(root, training_dir)
    catalog = load_temporal_catalog(root)
    models_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, mode in ((VARIANTS[0], "binary"), (VARIANTS[1], "activity_inverse_degree")):
        folder = models_dir / name
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text("utf-8"))
            if (manifest.get("status") != "complete"
                    or manifest.get("training_sha256") != train_receipt["source"]["training_pairs_sha256"]
                    or manifest.get("training_receipt_sha256") != digest(training_dir / "manifest.json")
                    or manifest.get("code_sha256") != digest(__file__)):
                raise ValueError(f"Existing temporal model has incompatible inputs: {name}")
            matrix_path = folder / "similarities_csr.npz"
            if digest(matrix_path) != manifest.get("similarities_sha256"):
                raise ValueError(f"Existing temporal model changed: {name}")
            results[name] = manifest
            continue
        if folder.exists():
            raise ValueError(f"Partial temporal model folder exists without a receipt: {folder}")
        folder.mkdir(parents=True)
        started = time.perf_counter()
        matrix = _build_similarity(x, catalog["apps"], catalog["candidate_input_cols"], mode)
        matrix_path = folder / "similarities_csr.npz"
        sparse.save_npz(matrix_path, matrix, compressed=True)
        manifest = {
            "status": "complete", "variant": name, "weighting": mode,
            "neighbors": K_NEIGHBORS, "shape": list(matrix.shape), "nnz": int(matrix.nnz),
            "training_sha256": train_receipt["source"]["training_pairs_sha256"],
            "training_receipt_sha256": digest(training_dir / "manifest.json"),
            "temporal_manifest_sha256": catalog["hashes"]["temporal_manifest_sha256"],
            "protocol_sha256": catalog["hashes"]["temporal_protocol_sha256"],
            "input_catalog_sha256": catalog["hashes"]["input_apps_sha256"],
            "candidate_availability_sha256": catalog["hashes"]["candidate_availability_sha256"],
            "similarities_sha256": digest(matrix_path), "code_sha256": digest(__file__),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(manifest_path, manifest)
        results[name] = manifest
        del matrix
        gc.collect()
    return results


class KnnScorer:
    def __init__(self, matrix: sparse.csr_matrix, name: str, model_manifest: dict):
        self.matrix = matrix
        self.spec = {"kind": "fresh_temporal_item_knn", "variant": name,
                     "manifest_sha256": hashlib.sha256(json.dumps(model_manifest, sort_keys=True).encode()).hexdigest()}

    def score(self, q: sparse.csr_matrix) -> np.ndarray:
        scores = q @ self.matrix.T
        scores = scores.toarray() if sparse.issparse(scores) else np.asarray(scores)
        return scores / np.maximum(np.asarray(q.sum(axis=1)).ravel(), 1)[:, None]


class FixedContentScorer:
    """The existing fixed public TF-IDF content control; no CF artifacts load."""
    def __init__(self, source: Path = CONTENT_SOURCE, apps: np.ndarray | None = None, linux: np.ndarray | None = None):
        source = Path(source)
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Fixed TF-IDF content control is missing")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("Fixed TF-IDF content control is incomplete")
        for file_name in ("tfidf_csr.npz", "features_csr.npz"):
            expected = manifest.get("files", {}).get(file_name, {}).get("sha256")
            if not expected or digest(source / file_name) != expected:
                raise ValueError(f"Fixed TF-IDF content artifact changed: {file_name}")
        self.channels = [sparse.load_npz(source / file_name).tocsr()
                         for file_name in ("tfidf_csr.npz", "features_csr.npz")]
        if apps is not None and any(matrix.shape[0] != len(apps) for matrix in self.channels):
            raise ValueError("TF-IDF rows differ from the fixed temporal input catalog")
        if linux is None:
            raise ValueError("Candidate mask is required for the fixed content control")
        self.linux = np.asarray(linux, dtype=bool)
        if any(matrix.shape[0] != len(self.linux) for matrix in self.channels):
            raise ValueError("TF-IDF rows differ from the fixed temporal candidate array")
        self.spec = {"kind": "fixed_public_tfidf_content", "source_manifest_sha256": digest(manifest_path),
                     "tfidf_sha256": digest(source / "tfidf_csr.npz"),
                     "features_sha256": digest(source / "features_csr.npz")}

    def score(self, q: sparse.csr_matrix) -> np.ndarray:
        result = None
        for matrix in self.channels:
            profile = normalize(q @ matrix)
            scores = profile @ matrix[self.linux].T
            scores = scores.toarray() if sparse.issparse(scores) else np.asarray(scores)
            result = scores * 0.5 if result is None else result + scores * 0.5
        return result


def _load_trained_knn(name: str, models_dir: Path, expected_training_sha: str) -> KnnScorer:
    if name not in VARIANTS[:2]:
        raise ValueError("Only fresh temporal collaborative models may use this loader")
    folder = Path(models_dir) / name
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Temporal model is not trained: {name}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    matrix_path = folder / "similarities_csr.npz"
    if (manifest.get("status") != "complete" or manifest.get("variant") != name
            or manifest.get("training_sha256") != expected_training_sha
            or digest(matrix_path) != manifest.get("similarities_sha256")):
        raise ValueError(f"Temporal model receipt is incompatible or changed: {name}")
    matrix = sparse.load_npz(matrix_path)
    if matrix.shape != (6_630, 37_610) or not sparse.isspmatrix_csr(matrix):
        raise ValueError(f"Temporal model shape changed: {name}")
    scorer = KnnScorer(matrix, name, manifest)
    scorer.spec["manifest_file_sha256"] = digest(manifest_path)
    scorer.spec["similarities_sha256"] = manifest["similarities_sha256"]
    return scorer


def pin_content_control(models_dir: Path, catalog: dict, scorer: FixedContentScorer) -> dict:
    """Write a local hashed receipt for the immutable public content control."""
    folder = Path(models_dir) / VARIANTS[2]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    receipt = {
        "status": "complete", "variant": VARIANTS[2], "kind": "fixed_public_tfidf_content",
        "source_manifest_sha256": scorer.spec["source_manifest_sha256"],
        "tfidf_sha256": scorer.spec["tfidf_sha256"], "features_sha256": scorer.spec["features_sha256"],
        "input_catalog_sha256": catalog["hashes"]["input_apps_sha256"],
        "candidate_availability_sha256": catalog["hashes"]["candidate_availability_sha256"],
        "candidate_count": 6_630,
        "representation_shapes": [list(matrix.shape) for matrix in scorer.channels],
        "runner_sha256": digest(__file__),
    }
    if path.exists():
        existing = json.loads(path.read_text("utf-8"))
        if existing != receipt:
            raise ValueError("Existing fixed TF-IDF content receipt differs")
    else:
        write_json(path, receipt)
    scorer.spec["temporal_model_receipt_sha256"] = digest(path)
    return receipt


def _sanitize_scores(scores, exclusions: sparse.csr_matrix, available: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).copy()
    available = np.asarray(available, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(available) or exclusions.shape != values.shape:
        raise ValueError("Score array, exclusions, and availability mask shapes differ")
    if not np.all(np.isfinite(values)):
        raise ValueError("Model returned non-finite temporal scores")
    values[values <= 0] = 0
    values[:, ~available] = 0
    rows, cols = exclusions.nonzero()
    values[rows, cols] = 0
    return values


def _rank_metrics(ranks: np.ndarray, lengths: np.ndarray, cohort: dict,
                  target_support: np.ndarray) -> dict:
    from improvement_common import log_groups
    from experiment_v4_common import rank_values

    ranks = np.asarray(ranks, dtype=np.int8)
    lengths = np.asarray(lengths, dtype=np.int8)
    target_ids = np.asarray(cohort["target_ids"], dtype=np.int64)
    total = np.asarray(cohort["n_total"], dtype=np.int64)
    positive = np.asarray(cohort["n_positive"], dtype=np.int64)
    target_support = np.asarray(target_support, dtype=np.int64)
    if not (ranks.shape == lengths.shape == target_ids.shape == total.shape == positive.shape == target_support.shape):
        raise ValueError("Per-user temporal metric arrays differ in length")
    recall, ndcg = rank_values(ranks)
    primary = primary_mask(cohort)

    def group(mask):
        mask = np.asarray(mask, dtype=bool)
        n = int(mask.sum())
        return {"users": n, "target_games": int(len(np.unique(target_ids[mask]))),
                "recall": float(recall[mask].mean()) if n else None,
                "ndcg": float(ndcg[mask].mean()) if n else None}

    result = {
        "all_target_eligible": group(np.ones(len(ranks), dtype=bool)),
        "primary": group(primary),
        "primary_definition": "n_total_inputs >= 5 AND n_positive_inputs >= 1",
        "all_eligible_users": int(len(ranks)),
        "primary_users": int(primary.sum()),
        "primary_game_macro_recall": macro(recall[primary], target_ids[primary]) if primary.any() else None,
        "primary_game_macro_ndcg": macro(ndcg[primary], target_ids[primary]) if primary.any() else None,
        "primary_empty_users": int((lengths[primary] == 0).sum()),
        "by_total_history": {
            "zero-input": group(total == 0),
            "1-2": group((total >= 1) & (total <= 2)),
            "3-4": group((total >= 3) & (total <= 4)),
            "5+": group(total >= 5),
        },
        "zero-positive-input": group(positive == 0),
        "by_positive_history": {
            "zero": group(positive == 0),
            "1": group(positive == 1),
            "2": group(positive == 2),
            "3-4": group((positive >= 3) & (positive <= 4)),
            "5-9": group((positive >= 5) & (positive <= 9)),
            "10+": group(positive >= 10),
        },
        "by_support": {
            label: group(primary & (log_groups(target_support) == index))
            for index, label in enumerate(LOG_LABELS)
        },
        "tail": group(primary & (target_support > 0) & (target_support < 1000)),
    }
    return result


def paired_comparison(base: np.ndarray, candidate: np.ndarray, cohort: dict,
                      reps: int = PAIRED_REPS) -> dict:
    """Use the frozen v4 paired user and target-game bootstrap on primary users."""
    mask = primary_mask(cohort)
    a = np.asarray(base, dtype=np.int8)[mask]
    b = np.asarray(candidate, dtype=np.int8)[mask]
    if not len(a):
        raise ValueError("Paired temporal comparison has no primary users")
    out = paired_rank_ci(a, b, reps=reps, seed=BOOTSTRAP_SEED)
    from experiment_v4_common import rank_values
    ar, an = rank_values(a)
    br, bn = rank_values(b)
    targets = np.asarray(cohort["target_ids"], dtype=np.int64)[mask]
    _, inverse = np.unique(targets, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.zeros((reps, 2), dtype=np.float64)
    groups = int(inverse.max() + 1)
    for game in range(groups):
        selected = inverse == game
        pair_codes = a[selected].astype(np.int16) * 11 + b[selected].astype(np.int16)
        counts = np.bincount(pair_codes, minlength=121)
        active = np.flatnonzero(counts)
        ah, ad = rank_values(active // 11)
        bh, bd = rank_values(active % 11)
        samples = rng.multinomial(int(selected.sum()), counts[active] / selected.sum(), size=reps)
        boot[:, 0] += samples @ (bh - ah) / selected.sum() / groups
        boot[:, 1] += samples @ (bd - ad) / selected.sum() / groups
    out["game_macro"] = {
        key: {"delta": macro(values, targets), "ci95": np.quantile(boot[:, index], [0.025, 0.975]).tolist()}
        for index, (key, values) in enumerate((("Recall@10", br - ar), ("NDCG@10", bn - an)))
    }
    out["hit_overlap"] = {
        "both": int(((a > 0) & (b > 0)).sum()),
        "base_only": int(((a > 0) & (b == 0)).sum()),
        "candidate_only": int(((a == 0) & (b > 0)).sum()),
        "neither": int(((a == 0) & (b == 0)).sum()),
    }
    out["primary_users"] = int(len(a))
    return out


def _evaluate_model(name: str, model, cohort: dict, catalog: dict, target_support: np.ndarray,
                    folder: Path, batch_size: int = BATCH_SIZE) -> tuple[dict, np.ndarray]:
    if cohort.get("name") != ROLE:
        raise ValueError("Temporal v4 controls can evaluate selection only")
    if name not in VARIANTS:
        raise ValueError("Unknown temporal v4 control")
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    provenance = {
        "role": ROLE, "protocol_sha256": cohort["hashes"]["temporal_protocol_sha256"],
        "temporal_manifest_sha256": cohort["hashes"]["temporal_manifest_sha256"],
        "cohort_hashes": cohort["hashes"], "candidate_availability_sha256": catalog["hashes"]["candidate_availability_sha256"],
        "model": getattr(model, "spec", {}), "evaluator_sha256": digest(__file__),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if manifest.get("provenance") != provenance or manifest.get("status") != "complete":
            raise ValueError(f"Existing temporal evaluation uses different inputs: {name}")
        for file_name, expected in manifest.get("files", {}).items():
            if digest(folder / file_name) != expected:
                raise ValueError(f"Existing temporal evaluation changed: {file_name}")
        per = pq.read_table(folder / "per_user.parquet").to_pydict()
        return manifest, np.asarray(per["target_rank"], dtype=np.int8)

    n = len(cohort["user_ids"])
    ranks = np.zeros(n, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int8)
    candidate_ids = catalog["candidate_ids"]
    schema = pa.schema([
        ("user_key", pa.uint64()), ("app_id", pa.int64()),
        ("rank", pa.int8()), ("score", pa.float64()),
    ])
    started = time.perf_counter()
    with pq.ParquetWriter(folder / "recommendations.parquet", schema, compression="zstd") as writer:
        for begin in range(0, n, batch_size):
            end = min(n, begin + batch_size)
            raw = np.asarray(model.score(cohort["positive"][begin:end]), dtype=np.float64)
            scores = _sanitize_scores(raw, cohort["exclusions"][begin:end], catalog["available"])
            records = []
            for local, row in enumerate(range(begin, end)):
                chosen = rank_positive(scores[local], candidate_ids, 10)
                lengths[row] = len(chosen)
                hit = np.flatnonzero(chosen == cohort["target_candidates"][row])
                if len(hit):
                    ranks[row] = int(hit[0] + 1)
                records.extend((int(cohort["user_ids"][row]), int(candidate_ids[col]), rank + 1,
                                float(scores[local, col])) for rank, col in enumerate(chosen))
            table = pa.Table.from_pylist([dict(zip(schema.names, row)) for row in records], schema=schema)
            writer.write_table(table)
            if begin == 0 or end == n or begin % (batch_size * 100) == 0:
                log(f"temporal selection/{name}: {end:,}/{n:,} users")

    per_user = pa.table({
        "user_key": pa.array(cohort["user_ids"], type=pa.uint64()),
        "target_app_id": pa.array(cohort["target_ids"], type=pa.int64()),
        "target_rank": pa.array(ranks, type=pa.int8()),
        "recommendation_count": pa.array(lengths, type=pa.int8()),
        "n_total_inputs": pa.array(cohort["n_total"], type=pa.int64()),
        "n_positive_inputs": pa.array(cohort["n_positive"], type=pa.int64()),
        "target_train_support": pa.array(target_support, type=pa.int64()),
    })
    pq.write_table(per_user, folder / "per_user.parquet", compression="zstd")
    manifest = {
        "status": "complete", "variant": name, "role": ROLE,
        "provenance": provenance, "elapsed_seconds": time.perf_counter() - started,
        "metrics": _rank_metrics(ranks, lengths, cohort, target_support),
        "files": {file_name: digest(folder / file_name)
                  for file_name in ("per_user.parquet", "recommendations.parquet")},
    }
    write_json(manifest_path, manifest)
    return manifest, ranks


def evaluate_selection(root: Path = TEMPORAL, output: Path = OUTPUT,
                       batch_size: int = BATCH_SIZE, bootstrap_reps: int = PAIRED_REPS) -> dict:
    """Evaluate the two temporal refits and fixed TF-IDF content on selection."""
    temporal_manifest, _, _ = verify_temporal_split(root)
    train_source = temporal_manifest["files"]["train/positive_catalog.parquet"]["sha256"]
    models_dir = Path(output).resolve() / "models"
    training_receipt_path = Path(output).resolve() / "training/manifest.json"
    if not training_receipt_path.is_file():
        raise ValueError("Temporal controls must be trained before selection evaluation")
    train_receipt = json.loads(training_receipt_path.read_text("utf-8"))
    if train_receipt.get("source", {}).get("training_pairs_sha256") != train_source:
        raise ValueError("Temporal controls training receipt does not match this split")
    catalog = load_temporal_catalog(root)
    cohort = load_selection_cohort(root, catalog)
    x_support = np.load(Path(output).resolve() / "training/support.npy", mmap_mode="r")
    if len(x_support) != len(catalog["apps"]):
        raise ValueError("Temporal training support mapping has wrong shape")
    target_support = np.asarray(x_support)[_strict_positions(cohort["target_ids"], catalog["apps"], "Target App")]

    content_scorer = FixedContentScorer(CONTENT_SOURCE, catalog["apps"], catalog["linux"])
    pin_content_control(models_dir, catalog, content_scorer)
    models = {
        VARIANTS[0]: _load_trained_knn(VARIANTS[0], models_dir, train_source),
        VARIANTS[1]: _load_trained_knn(VARIANTS[1], models_dir, train_source),
        VARIANTS[2]: content_scorer,
    }
    results, ranks = {}, {}
    for name, model in models.items():
        folder = Path(output).resolve() / "evaluations/selection" / name
        result, model_ranks = _evaluate_model(name, model, cohort, catalog, target_support, folder, batch_size)
        results[name], ranks[name] = result, model_ranks

    comparisons = {
        f"{base}_vs_{candidate}": paired_comparison(ranks[base], ranks[candidate], cohort, bootstrap_reps)
        for base, candidate in ((VARIANTS[0], VARIANTS[1]), (VARIANTS[0], VARIANTS[2]))
    }
    paired_path = Path(output).resolve() / "evaluations/selection/paired_comparisons.json"
    paired_receipt = {
        "status": "complete", "role": ROLE,
        "protocol_sha256": cohort["hashes"]["temporal_protocol_sha256"],
        "primary_definition": "n_total_inputs >= 5 AND n_positive_inputs >= 1",
        "bootstrap_reps": bootstrap_reps, "bootstrap_seed": BOOTSTRAP_SEED,
        "cohort_hashes": cohort["hashes"], "comparisons": comparisons,
        "evaluation_receipts": {name: digest(Path(output).resolve() / "evaluations/selection" / name / "manifest.json")
                                 for name in VARIANTS},
    }
    if paired_path.exists():
        old = json.loads(paired_path.read_text("utf-8"))
        if old != paired_receipt:
            raise ValueError("Existing paired temporal comparison receipt differs")
    else:
        write_json(paired_path, paired_receipt)
    log("Temporal selection controls complete; confirmation was not opened")
    return {"evaluations": results, "paired_comparisons": paired_receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train", "evaluate"),
                        help="Run the fresh temporal refits or evaluate trained controls on selection")
    parser.add_argument("--temporal-root", type=Path, default=TEMPORAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.stage == "train":
        result = fit_models(args.temporal_root, args.output)
        log(f"Trained temporal controls: {', '.join(result)}")
    else:
        evaluate_selection(args.temporal_root, args.output, args.batch_size)


if __name__ == "__main__":
    main()
