"""Binary positive co-occurrence Item-KNN catalog and retrieval API.

The frozen index stores only the top cosine neighbors for Linux candidates.
Input-game supports include every positive train user, including users with a
single positive catalog game. Validation and test reviews are never loaded.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

from eda_large_review_frank import EDA, digest


DEFAULT_INDEX = EDA / "data/processed/modeling/item_knn_v1"
VERSION = "binary-item-knn-v1"


def _strict_ids(values, label):
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids <= 0) or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1])):
        raise ValueError(f"{label} must be unique positive App IDs in ascending order")
    return ids


def _top_cosine_neighbors(candidate_col, cooc_cols, cooc_counts, app_ids,
                          support, neighbors):
    """Rank one co-occurrence row by exact binary cosine, then App ID."""
    candidate_support = int(support[candidate_col])
    if candidate_support <= 0 or neighbors == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    cooc_cols = np.asarray(cooc_cols, dtype=np.int32)
    cooc_counts = np.asarray(cooc_counts, dtype=np.int64)
    keep = (cooc_cols != candidate_col) & (cooc_counts > 0) & (support[cooc_cols] > 0)
    cols = cooc_cols[keep]
    counts = cooc_counts[keep]
    if not len(cols):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    denominator = np.sqrt(float(candidate_support) * support[cols].astype(np.float64))
    values = counts.astype(np.float64) / denominator
    ranked = np.lexsort((app_ids[cols], -values))[:neighbors]
    selected_cols = cols[ranked]
    selected_values = values[ranked]

    # CSR storage is sorted by input row/App ID; ranking was already resolved.
    storage_order = np.argsort(selected_cols, kind="stable")
    return selected_cols[storage_order], selected_values[storage_order]


def _build_user_item_csr(sorted_batches, app_ids, expected_pairs, expected_users):
    """Stream sorted (user_key, app_id) batches into a binary user×item CSR.

    User IDs are checked and compared as uint64 integers; they are never
    converted to floating point. Sorted batch input permits constant-sized
    batch state and avoids a 30M-entry Python user-ID dictionary.
    """
    app_ids = _strict_ids(app_ids, "Input catalog")
    if expected_pairs < 0 or expected_users < 0:
        raise ValueError("Expected pair and user counts must be nonnegative")
    if expected_pairs >= np.iinfo(np.int32).max or expected_users >= np.iinfo(np.int32).max:
        raise ValueError("SciPy CSR index limits exceeded")

    indices = np.empty(expected_pairs, dtype=np.int32)
    values = np.ones(expected_pairs, dtype=np.uint8)
    indptr = np.zeros(expected_users + 1, dtype=np.int32)
    support = np.zeros(len(app_ids), dtype=np.int64)

    pair_count = 0
    user_count = 0
    pending_user = None
    previous_user = None
    previous_app = None

    for batch in sorted_batches:
        if batch.num_rows == 0:
            continue
        user_col = batch.schema.get_field_index("user_key")
        app_col = batch.schema.get_field_index("app_id")
        if user_col < 0 or app_col < 0:
            raise ValueError("Positive catalog batches require user_key and app_id columns")
        users = np.asarray(batch.column(user_col).to_numpy(zero_copy_only=False))
        apps = np.asarray(batch.column(app_col).to_numpy(zero_copy_only=False))
        if users.dtype != np.dtype("uint64"):
            raise ValueError("user_key must remain uint64 throughout the training build")
        if apps.dtype.kind != "i" or apps.dtype.itemsize != 8:
            raise ValueError("app_id must be an int64 value")
        if np.any(users[1:] < users[:-1]):
            raise ValueError("Training rows are not sorted by user_key")
        if previous_user is not None:
            if users[0] < previous_user:
                raise ValueError("Training batches are not sorted by user_key")
            if users[0] == previous_user and apps[0] <= previous_app:
                raise ValueError("Duplicate or unsorted user/App ID pair across batches")
        same_user = users[1:] == users[:-1]
        if np.any(apps[1:][same_user] <= apps[:-1][same_user]):
            raise ValueError("Duplicate or unsorted user/App ID pair")

        item_rows = np.searchsorted(app_ids, apps)
        if np.any(item_rows >= len(app_ids)) or np.any(app_ids[item_rows] != apps):
            raise ValueError("Training pair contains an App ID outside the fixed input catalog")
        end = pair_count + batch.num_rows
        if end > expected_pairs:
            raise ValueError("Training pair count exceeds the pinned split manifest")
        indices[pair_count:end] = item_rows.astype(np.int32, copy=False)
        support += np.bincount(item_rows, minlength=len(app_ids))

        group_ends = np.flatnonzero(users[1:] != users[:-1]) + 1
        group_ends = np.append(group_ends, batch.num_rows)

        # The final user in a batch may continue in the next batch.
        if pending_user is not None and users[0] != pending_user:
            if user_count >= expected_users:
                raise ValueError("Training user count exceeds the pinned split manifest")
            indptr[user_count + 1] = pair_count
            user_count += 1

        completed_groups = len(group_ends) - 1
        if user_count + completed_groups > expected_users:
            raise ValueError("Training user count exceeds the pinned split manifest")
        if completed_groups:
            indptr[user_count + 1:user_count + completed_groups + 1] = (
                pair_count + group_ends[:completed_groups]
            )
            user_count += completed_groups
        pending_user = int(users[-1])
        previous_user = users[-1]
        previous_app = apps[-1]
        pair_count = end

    if pair_count != expected_pairs:
        raise ValueError(f"Training pair count differs from the pinned split: {pair_count}")
    if pending_user is not None:
        if user_count >= expected_users:
            raise ValueError("Training user count exceeds the pinned split manifest")
        indptr[user_count + 1] = pair_count
        user_count += 1
    if user_count != expected_users:
        raise ValueError(f"Training user count differs from the pinned split: {user_count}")

    matrix = sparse.csr_matrix((values, indices, indptr),
                               shape=(expected_users, len(app_ids)), copy=False)
    if not matrix.has_canonical_format:
        matrix.sum_duplicates()
        matrix.sort_indices()
    if matrix.nnz != expected_pairs or np.any(matrix.indptr[1:] <= matrix.indptr[:-1]):
        raise ValueError("Binary user-item CSR integrity check failed")
    return matrix, support, user_count


class ItemKnnCatalog:
    """Top-neighbor binary Item-KNN catalog with mean-query scoring."""

    def __init__(self, app_ids, candidate_app_ids, similarities, support=None):
        self.app_ids = _strict_ids(app_ids, "Input catalog")
        self.candidate_app_ids = _strict_ids(candidate_app_ids, "Linux candidate catalog")
        self.similarities = sparse.csr_matrix(similarities, dtype=np.float64)
        if self.similarities.shape != (len(self.candidate_app_ids), len(self.app_ids)):
            raise ValueError("Similarity matrix shape differs from candidate/input mappings")
        if not np.all(np.isin(self.candidate_app_ids, self.app_ids)):
            raise ValueError("Linux candidate IDs must be contained in the input catalog")
        if self.similarities.nnz and (
                not np.all(np.isfinite(self.similarities.data))
                or np.any(self.similarities.data <= 0)
                or np.any(self.similarities.data > 1.0 + 1e-12)):
            raise ValueError("Stored similarities must be finite positive cosine values")
        self.similarities.sum_duplicates()
        self.similarities.sort_indices()
        self.support = None if support is None else np.asarray(support, dtype=np.int64)
        if self.support is not None and (
                self.support.shape != self.app_ids.shape or np.any(self.support < 0)):
            raise ValueError("Item support must be a nonnegative count per input App ID")
        self.row_by_app = {int(app): row for row, app in enumerate(self.app_ids)}

    @classmethod
    def load(cls, folder=DEFAULT_INDEX, verify_hashes=True):
        folder = Path(folder)
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if manifest.get("version") != VERSION or manifest.get("status") != "complete":
            raise ValueError("Refusing incomplete or incompatible Item-KNN artifact")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Item-KNN manifest has no pinned artifact files")
        required_files = {"game_rows.parquet", "candidate_rows.parquet",
                          "parameters.json", "similarities_csr.npz"}
        if not required_files.issubset(files):
            raise ValueError("Item-KNN manifest is missing a required pinned artifact")
        if verify_hashes:
            for relative, info in files.items():
                path = (folder / relative).resolve()
                if folder.resolve() not in path.parents:
                    raise ValueError("Unsafe path in Item-KNN manifest")
                if not path.is_file() or digest(path) != info.get("sha256"):
                    raise ValueError(f"Item-KNN artifact fingerprint changed: {relative}")

        rows = pq.read_table(folder / "game_rows.parquet").to_pydict()
        app_ids = _strict_ids(rows["app_id"], "Input catalog")
        if rows["row_id"] != list(range(len(app_ids))):
            raise ValueError("Input game row mapping is not contiguous")
        linux = np.asarray(rows["linux"], dtype=bool)
        support = np.asarray(rows["positive_user_support"], dtype=np.int64)
        if len(linux) != len(app_ids) or support.shape != app_ids.shape or np.any(support < 0):
            raise ValueError("Input catalog flags or support mapping has an invalid shape")

        candidate_rows = pq.read_table(folder / "candidate_rows.parquet").to_pydict()
        candidate_app_ids = _strict_ids(candidate_rows["app_id"], "Linux candidate catalog")
        if candidate_rows["row_id"] != list(range(len(candidate_app_ids))):
            raise ValueError("Candidate row mapping is not contiguous")
        candidate_input_rows = np.asarray(candidate_rows["input_row_id"], dtype=np.int64)
        if len(candidate_app_ids) and (
                np.any(candidate_input_rows < 0)
                or np.any(candidate_input_rows >= len(app_ids))):
            raise ValueError("Candidate input row mapping is out of range")
        if not np.array_equal(app_ids[candidate_input_rows], candidate_app_ids):
            raise ValueError("Candidate row mapping does not match the input catalog")
        if not np.array_equal(candidate_app_ids, app_ids[linux]):
            raise ValueError("Candidate catalog differs from Linux-marked input games")

        parameters = json.loads((folder / "parameters.json").read_text("utf-8"))
        neighbors = int(parameters["neighbors"])
        if neighbors < 1:
            raise ValueError("Invalid saved neighbor count")
        similarities = sparse.load_npz(folder / "similarities_csr.npz")
        if not sparse.isspmatrix_csr(similarities):
            raise ValueError("Saved similarities are not CSR")
        model = cls(app_ids, candidate_app_ids, similarities, support=support)
        expected_shape = [len(candidate_app_ids), len(app_ids)]
        if model.similarities.shape != tuple(expected_shape):
            raise ValueError("Saved similarity matrix shape differs from mappings")
        index_info = manifest.get("index", {})
        if index_info.get("shape") != expected_shape or index_info.get("nnz") != model.similarities.nnz:
            raise ValueError("Saved similarity matrix differs from the pinned manifest shape")
        if np.any(np.diff(model.similarities.indptr) > neighbors):
            raise ValueError("A candidate exceeds the frozen top-neighbor count")
        candidate_columns = np.searchsorted(model.app_ids, model.candidate_app_ids)
        for row, col in enumerate(candidate_columns):
            start, end = model.similarities.indptr[row:row + 2]
            if np.any(model.similarities.indices[start:end] == col):
                raise ValueError("Self-similarity must be zero")
        return model

    def recommend(self, positive_app_ids, exclude_app_ids=(), limit=10):
        """Recommend positive-score Linux items, averaging over every query.

        Query inputs must be unique members of the fixed input catalog.
        Excluded IDs may include non-catalog values; all positive inputs are
        also masked defensively. There is no score backfill.
        """
        if isinstance(limit, bool) or int(limit) != limit or not 0 <= int(limit) <= 10:
            raise ValueError("Baseline returns at most 10 recommendations")
        limit = int(limit)
        positives = [int(app) for app in positive_app_ids]
        if len(set(positives)) != len(positives):
            raise ValueError("Positive query App IDs must be unique")
        unknown = set(positives).difference(self.row_by_app)
        if unknown:
            raise ValueError("Positive query contains an App ID outside the fixed catalog")
        if not positives or limit == 0:
            return []

        query_cols = np.fromiter((self.row_by_app[app] for app in positives),
                                 dtype=np.int32, count=len(positives))
        # Sparse column slicing leaves absent links at zero, while division by
        # the full positive query length retains them in the denominator.
        summed = self.similarities[:, query_cols].sum(axis=1)
        scores = np.asarray(summed, dtype=np.float64).reshape(-1) / len(positives)
        excluded = set(int(app) for app in exclude_app_ids) | set(positives)
        allowed = (scores > 0) & np.fromiter(
            (int(app) not in excluded for app in self.candidate_app_ids),
            dtype=bool, count=len(self.candidate_app_ids))
        candidate_rows = np.flatnonzero(allowed)
        order = np.lexsort((self.candidate_app_ids[candidate_rows], -scores[candidate_rows]))[:limit]
        return [{"app_id": int(self.candidate_app_ids[candidate_rows[i]]),
                 "score": float(scores[candidate_rows[i]])} for i in order]
