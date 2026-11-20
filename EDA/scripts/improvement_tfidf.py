"""Build TF-IDF v2 diagnostics and feature artifacts from frozen split_v1 inputs.

This module reads only the fixed validation cohort for T0. It does not read the
frank cohort and does not rank any v2 experiment. T0 classifications come from
the already-saved v1 validation output; text-only variants and the separate
metadata channel are built for the parent evaluator to score.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import platform
import re
import shutil
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from eda_large_review_frank import EDA, digest, log, quote
from tfidf_content import (
    CLEANING_CONFIG,
    DEFAULT_INDEX,
    VECTORIZER_CONFIG,
    TfidfCatalog,
    tokenize_description,
)


VERSION = "tfidf-improvement-v2"
DEFAULT_SPLIT = EDA / "data/processed/modeling/split_v1"
DEFAULT_OUTPUT = EDA / "data/processed/modeling/improvement_v2/tfidf"
DEFAULT_REPORT = EDA / "reports/tfidf_ranking_analysis_v2.json"
DEFAULT_PROTOCOL = EDA / "data/reference/improvement_protocol_v2.json"
DEFAULT_RUN_LOCK = EDA / "data/reference/baseline_run_v1.json"
DEFAULT_BASELINE_RESULTS = EDA / "data/processed/modeling/baseline_v1/validation"
SAMPLE_SEED = 20261107
SAMPLES_PER_STRATUM = 5
TOP_K = 10
TOP4 = 4

INPUT_COUNT_BINS = (
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 9, "5-9"),
    (10, None, "10+"),
)

TEXT_VARIANTS = {
    "baseline_control": {
        "id": "v1_control",
        "sublinear_tf": False,
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
        "stop_words": None,
        "change": "No changes from the frozen v1 description TF-IDF settings.",
    },
    "T1_sublinear_tf": {
        "id": "sublinear_tf_only",
        "sublinear_tf": True,
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
        "stop_words": None,
        "change": "sublinear_tf=True only; unchanged text, tokenizer and remaining settings.",
    },
    "T2_unigram_bigram": {
        "id": "word_unigram_bigram_only",
        "sublinear_tf": False,
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 1.0,
        "stop_words": None,
        "change": "ngram_range=(1, 2) only; unchanged text, tokenizer and remaining settings.",
    },
}

# The source of the earlier one-off 21-word set was not retained. The gate is
# fixed before reading any v2 ranking output and uses a contribution share from
# sampled positive-rank failures. The 5% line is an operational evidence rule,
# not a claimed historical definition.
T3_GATE_MIN_HIGH_DF_SHARE = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def _file_inventory(folder: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p != folder / "manifest.json"):
        info: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix == ".parquet":
            pf = pq.ParquetFile(path)
            info["rows"] = int(pf.metadata.num_rows)
            info["schema"] = [{"name": f.name, "type": str(f.type)} for f in pf.schema_arrow]
        result[path.relative_to(folder).as_posix()] = info
    return result


def _verify_inventory(folder: Path, inventory: dict[str, dict[str, Any]]) -> None:
    for relative, expected in inventory.items():
        path = folder / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Frozen artifact fingerprint mismatch: {relative}")
        if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != expected["rows"]:
            raise ValueError(f"Frozen artifact row count mismatch: {relative}")


def _validate_fixed_sources(split: Path, v1: Path, protocol_path: Path, run_lock_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol.get("status") != "fixed_before_v2_validation_metrics":
        raise ValueError("v2 experiment protocol is not frozen")
    split_manifest_path = split / "manifest.json"
    split_manifest = read_json(split_manifest_path)
    if digest(split_manifest_path) != protocol["split_manifest_sha256"]:
        raise ValueError("split_v1 manifest differs from the frozen v2 protocol")
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("split_v1 is not verified")
    if split_manifest.get("seed") != SAMPLE_SEED:
        raise ValueError("split_v1 seed differs from the fixed validation seed")
    required_split_files = (
        "catalog/input_apps.parquet", "catalog/linux_candidates.parquet", "train/positive_catalog.parquet",
        "validation/users.parquet", "validation/query.parquet", "validation/targets.parquet",
        "validation/exclusions.parquet",
    )
    for relative in required_split_files:
        info = split_manifest.get("files", {}).get(relative)
        path = split / relative
        if info is None or not path.is_file() or digest(path) != info["sha256"]:
            raise ValueError(f"Frozen split file fingerprint mismatch: {relative}")

    run_lock = read_json(run_lock_path)
    if digest(run_lock_path) != protocol["baseline_run_lock_sha256"]:
        raise ValueError("baseline v1 run lock differs from the frozen v2 protocol")
    if run_lock.get("status") != "frozen_before_validation":
        raise ValueError("baseline v1 run lock is not frozen")

    v1_manifest_path = v1 / "manifest.json"
    v1_manifest = read_json(v1_manifest_path)
    if digest(v1_manifest_path) != run_lock["tfidf_manifest_sha256"] or v1_manifest.get("status") != "complete":
        raise ValueError("v1 TF-IDF artifact differs from the frozen baseline run")
    if v1_manifest["sources"]["split_manifest_sha256"] != digest(split_manifest_path):
        raise ValueError("v1 TF-IDF artifact was built from a different split")
    _verify_inventory(v1, v1_manifest["files"])

    val_root = DEFAULT_BASELINE_RESULTS
    val_manifest_path = val_root / "manifest.json"
    val_manifest = read_json(val_manifest_path)
    if (val_manifest.get("status") != "complete" or val_manifest.get("cohort") != "validation"
            or val_manifest.get("users") != protocol["cohort_users"]["validation"]
            or val_manifest.get("run_lock_sha256") != digest(run_lock_path)
            or val_manifest.get("split_manifest_sha256") != digest(split_manifest_path)
            or val_manifest.get("tfidf_manifest_sha256") != digest(v1_manifest_path)):
        raise ValueError("Saved baseline validation result differs from the frozen protocol")
    for name in ("tfidf/per_user.parquet", "tfidf/recommendations.parquet"):
        expected = val_manifest["files"].get(name)
        if expected is None or digest(val_root / name) != expected["sha256"]:
            raise ValueError(f"Saved baseline validation output changed: {name}")

    source_manifest_path = EDA / "data/source_manifest.json"
    reference_manifest_path = EDA / "data/reference/manifest.json"
    return {
        "split_manifest_sha256": digest(split_manifest_path),
        "v1_tfidf_manifest_sha256": digest(v1_manifest_path),
        "baseline_run_lock_sha256": digest(run_lock_path),
        "baseline_validation_manifest_sha256": digest(val_manifest_path),
        "improvement_protocol_sha256": digest(protocol_path),
        "source_manifest_sha256_at_build_start": digest(source_manifest_path),
        "reference_manifest_sha256": digest(reference_manifest_path),
        "files": {
            "split_manifest": {"path": split_manifest_path.relative_to(EDA).as_posix(), "sha256": digest(split_manifest_path)},
            "v1_tfidf_manifest": {"path": v1_manifest_path.relative_to(EDA).as_posix(), "sha256": digest(v1_manifest_path)},
            "baseline_run_lock": {"path": run_lock_path.relative_to(EDA).as_posix(), "sha256": digest(run_lock_path)},
            "baseline_validation_manifest": {"path": val_manifest_path.relative_to(EDA).as_posix(), "sha256": digest(val_manifest_path)},
            "source_manifest_snapshot": {"path": source_manifest_path.relative_to(EDA).as_posix(), "sha256": digest(source_manifest_path)},
            "reference_manifest": {"path": reference_manifest_path.relative_to(EDA).as_posix(), "sha256": digest(reference_manifest_path)},
        },
        "split_manifest": split_manifest,
        "v1_manifest": v1_manifest,
        "validation_manifest": val_manifest,
        "protocol": protocol,
    }


def make_text_vectorizer(*, sublinear_tf: bool = False, ngram_range: tuple[int, int] = (1, 1), max_df: float = 1.0) -> TfidfVectorizer:
    """Use the v1 tokenizer and defaults, changing only explicitly supplied fields."""
    return TfidfVectorizer(
        analyzer="word",
        tokenizer=tokenize_description,
        token_pattern=None,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=1,
        max_df=max_df,
        max_features=None,
        stop_words=None,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=sublinear_tf,
        binary=False,
        dtype=np.float32,
    )


def normalize_concept(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split())
    return normalized or None


def _literal_or_json(value: str) -> tuple[Any | None, str]:
    """Parse serialized metadata without eval; return a reasoned parse state."""
    text = value.strip()
    if not text:
        return None, "empty"
    try:
        return json.loads(text), "json"
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(text), "python_literal"
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None, "invalid"


def parse_metadata_labels(value: Any, expected: str) -> tuple[list[str], str, str | None]:
    """Parse A/C tags or C genres into unmodified phrase labels safely.

    C tags are serialized dictionaries whose keys are concepts and values are
    vote counts. Genres are serialized lists. A tags are already Arrow lists.
    Empty/missing and malformed data remain distinct in the returned status.
    """
    if value is None:
        return [], "missing", None
    parsed: Any = value
    parser = "native"
    if isinstance(value, str):
        parsed, parser = _literal_or_json(value)
        if parsed is None:
            return [], "invalid", parser
    if expected == "tags" and isinstance(parsed, dict):
        labels = list(parsed.keys())
        if any(not isinstance(k, str) for k in labels):
            return [], "invalid", parser
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in parsed.values()):
            return [], "invalid", parser
    elif isinstance(parsed, (list, tuple)):
        labels = list(parsed)
        if any(not isinstance(label, str) for label in labels):
            return [], "invalid", parser
    else:
        return [], "invalid", parser
    normalized = [label.strip() for label in labels if isinstance(label, str) and label.strip()]
    if not normalized:
        return [], "empty", parser
    return normalized, "valid", parser


def choose_tag_source(
    a_labels: list[str], a_status: str, c_labels: list[str], c_status: str,
) -> tuple[list[str], str]:
    """Apply the declared A-first, C-fallback tag source rule."""
    if a_status == "valid" and a_labels:
        return a_labels, "A"
    if c_status == "valid" and c_labels:
        return c_labels, "C"
    return [], "missing"


def _support_bucket(support: np.ndarray, edges: list[int], labels: list[str]) -> np.ndarray:
    values = np.asarray(support, dtype=np.int64)
    indices = np.searchsorted(np.asarray(edges, dtype=np.int64), values, side="right") - 1
    indices = np.clip(indices, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[indices]


def _input_count_bin(n: int) -> str:
    for low, high, label in INPUT_COUNT_BINS:
        if n >= low and (high is None or n <= high):
            return label
    raise ValueError(f"Positive input count outside supported bins: {n}")


def _selected_samples(
    user_keys: np.ndarray,
    outcome: np.ndarray,
    support_bucket: np.ndarray,
    input_counts: np.ndarray,
    top4_by_user: dict[int, list[tuple[int, float]]],
    slot_top_games: list[dict[str, Any]],
    seed: int = SAMPLE_SEED,
    per_stratum: int = SAMPLES_PER_STRATUM,
) -> tuple[dict[int, set[str]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: dict[int, set[str]] = defaultdict(set)
    strata: dict[str, int] = {}
    outcomes = ("success", "zero_score_failure", "positive_rank_failure")
    canonical_buckets = ("0", "1-9", "10-99", "100-999", "1000-9999", "10000-99999", "100000-999999", "1000000+")
    present_buckets = set(str(x) for x in support_bucket)
    buckets = [bucket for bucket in canonical_buckets if bucket in present_buckets]
    input_bin = np.asarray([_input_count_bin(int(n)) for n in input_counts], dtype=object)
    for label in outcomes:
        for bucket in buckets:
            for input_label in ("1", "2-4", "5-9", "10+"):
                mask = (outcome == label) & (support_bucket == bucket) & (input_bin == input_label)
                indices = np.flatnonzero(mask)
                key = f"{label}|support={bucket}|positive_inputs={input_label}"
                strata[key] = int(len(indices))
                if len(indices):
                    count = min(per_stratum, len(indices))
                    picks = rng.choice(indices, size=count, replace=False)
                    for i in picks:
                        selected[int(user_keys[i])].add("stratified_case")

    # Ensure the most frequently recommended games from the saved v1 validation
    # are represented among the example cases as well as in the concentration
    # table. A chosen user still retains its original stratification reason.
    special_users: dict[int, list[int]] = {}
    for game in slot_top_games:
        app_id = int(game["app_id"])
        eligible = [u for u, rows in top4_by_user.items() if any(a == app_id for a, _ in rows)]
        if eligible:
            user = int(rng.choice(np.asarray(eligible, dtype=np.uint64)))
            selected[user].add("top_slot_share_case")
            special_users.setdefault(user, []).append(app_id)
    return selected, {
        "seed": int(seed),
        "sampling_rule": "Up to five users without replacement in each outcome x protocol train-support log bucket x positive-input-count stratum; most frequent recommendation-slot games added when present in a user's saved top four.",
        "samples_per_stratum": int(per_stratum),
        "stratum_population": strata,
        "selected_users": len(selected),
        "selection_reason_counts": _aggregate_rows(reason for reasons in selected.values() for reason in reasons),
        "top_slot_share_example_users": len(special_users),
        "top_slot_share_game_app_ids": sorted({app for apps in special_users.values() for app in apps}),
    }


def _load_train_support(path: Path) -> dict[int, int]:
    con = duckdb.connect()
    try:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='1GB'")
        rows = con.execute(
            f"SELECT app_id, count(*) AS train_positive_user_support "
            f"FROM read_parquet({quote(path)}) GROUP BY app_id ORDER BY app_id"
        ).fetchall()
    finally:
        con.close()
    return {int(app): int(support) for app, support in rows}


def _read_validation_results(split: Path, baseline_results: Path, support: dict[int, int], protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    per_user_path = baseline_results / "tfidf/per_user.parquet"
    rec_path = baseline_results / "tfidf/recommendations.parquet"
    user = pq.read_table(per_user_path).to_pydict()
    user_keys = np.asarray(user["user_key"], dtype=np.uint64)
    if len(user_keys) != protocol["cohort_users"]["validation"] or np.any(user_keys[1:] <= user_keys[:-1]):
        raise ValueError("Saved validation per-user rows are not the fixed sorted cohort")
    n = len(user_keys)
    target_app = np.asarray(user["target_app_id"], dtype=np.int64)
    npos = np.asarray(user["n_positive_inputs"], dtype=np.int32)
    ntotal = np.asarray(user["n_total_inputs"], dtype=np.int32)
    rank = np.asarray(user["target_rank"], dtype=np.int8)
    score = np.asarray(user["target_score"], dtype=np.float64)
    train_support_saved = np.asarray(user["target_train_support"], dtype=np.int64)
    if np.any(npos < 1) or np.any(ntotal < 5) or np.any(~np.isfinite(score)):
        raise ValueError("Saved v1 validation outcomes violate fixed eligibility or finite-score rules")
    cohort_users = pq.read_table(split / "validation/users.parquet").to_pydict()
    cohort_targets = pq.read_table(split / "validation/targets.parquet").to_pydict()
    if (not np.array_equal(user_keys, np.asarray(cohort_users["user_key"], dtype=np.uint64))
            or not np.array_equal(npos, np.asarray(cohort_users["n_positive_inputs"], dtype=np.int32))
            or not np.array_equal(ntotal, np.asarray(cohort_users["n_total_inputs"], dtype=np.int32))
            or not np.array_equal(user_keys, np.asarray(cohort_targets["user_key"], dtype=np.uint64))
            or not np.array_equal(target_app, np.asarray(cohort_targets["app_id"], dtype=np.int64))):
        raise ValueError("Saved v1 validation outcomes differ from split_v1 users or targets")
    exact_support = np.asarray([support.get(int(app), 0) for app in target_app], dtype=np.int64)
    if not np.array_equal(exact_support, train_support_saved):
        raise ValueError("Stored target train support differs from train-only positive_catalog counts")
    outcomes = np.where(rank > 0, "success", np.where(score <= 0, "zero_score_failure", "positive_rank_failure"))
    if np.any((rank > 0) & (rank > TOP_K)):
        raise ValueError("Saved target rank exceeds the frozen Top 10")

    rec = pq.read_table(rec_path, columns=["user_key", "app_id", "rank", "score"]).to_pydict()
    rec_users = np.asarray(rec["user_key"], dtype=np.uint64)
    rec_apps = np.asarray(rec["app_id"], dtype=np.int64)
    rec_ranks = np.asarray(rec["rank"], dtype=np.int8)
    rec_scores = np.asarray(rec["score"], dtype=np.float64)
    top4: dict[int, list[tuple[int, float]]] = defaultdict(list)
    slots = Counter()
    top1_slots = Counter()
    last_user = None
    previous_rank = 0
    for u, app, r, s in zip(rec_users, rec_apps, rec_ranks, rec_scores):
        uid, aid, rk = int(u), int(app), int(r)
        if not math.isfinite(float(s)) or float(s) <= 0 or rk < 1 or rk > TOP_K:
            raise ValueError("Saved v1 recommendations contain a nonpositive score or invalid rank")
        if last_user != uid:
            last_user, previous_rank = uid, 0
        if rk != previous_rank + 1:
            raise ValueError("Saved v1 validation ranks are not contiguous and ordered")
        previous_rank = rk
        slots[aid] += 1
        if rk == 1:
            top1_slots[aid] += 1
        if rk <= TOP4:
            top4[uid].append((aid, float(s)))
    actual_by_group = Counter()
    for uid in rec_users:
        actual_by_group[int(uid)] += 1
    if any(actual_by_group.get(int(uid), 0) != int(count) for uid, count in zip(user_keys, user["recommendation_count"])):
        raise ValueError("Saved recommendation counts differ from per-user validation outcomes")
    fixed_user_set = set(int(uid) for uid in user_keys)
    if any(int(uid) not in fixed_user_set for uid in actual_by_group):
        raise ValueError("Saved recommendations contain a user outside the fixed validation cohort")

    expected = protocol["cohort_users"]["validation"]
    if n != expected:
        raise ValueError("Saved validation outcome count differs from the fixed v1 cohort")
    return {
        "users": user,
        "user_keys": user_keys,
        "target_app": target_app,
        "n_positive": npos,
        "n_total": ntotal,
        "rank": rank,
        "score": score,
        "train_support": exact_support,
        "outcome": outcomes,
        "top4": top4,
        "recommendations": {"users": rec_users, "apps": rec_apps, "ranks": rec_ranks, "scores": rec_scores},
        "slot_counts": slots,
        "top1_counts": top1_slots,
    }, {
        "validation_users": n,
        "outcomes": {label: int(np.count_nonzero(outcomes == label)) for label in ("success", "zero_score_failure", "positive_rank_failure")},
        "target_positive_score_users": int(np.count_nonzero(score > 0)),
        "target_zero_score_users": int(np.count_nonzero(score <= 0)),
        "saved_recommendation_rows": len(rec_apps),
        "saved_recommendation_slots": int(sum(slots.values())),
        "saved_empty_lists": int(np.count_nonzero(np.asarray(user["recommendation_count"]) == 0)),
        "saved_short_lists": int(np.count_nonzero(np.asarray(user["recommendation_count"]) < TOP_K)),
        "support_source": "train/positive_catalog.parquet, counted by distinct user-app rows with DuckDB (threads=2, memory_limit=1GB); matched against saved v1 target_train_support.",
    }


def _rows_for_terms(matrix: sparse.csr_matrix, profile_apps: list[int], app_to_row: dict[int, int]) -> sparse.csr_matrix:
    rows = [app_to_row[int(app)] for app in profile_apps]
    if not rows:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32)
    profile = matrix[rows].sum(axis=0)
    profile = sparse.csr_matrix(profile, dtype=np.float32) / np.float32(len(rows))
    return normalize(profile, norm="l2", copy=False).tocsr()


def contributions_for_pair(
    profile: sparse.csr_matrix,
    candidate: sparse.csr_matrix,
    terms: list[str] | np.ndarray,
    idf: np.ndarray,
    document_frequency: np.ndarray,
    n_documents: int,
) -> list[dict[str, Any]]:
    """Return exact per-term products whose sum is the profile/candidate cosine."""
    left, right = profile.tocsr(), candidate.tocsr()
    li, ri = left.indices, right.indices
    common, lpos, rpos = np.intersect1d(li, ri, assume_unique=True, return_indices=True)
    values = left.data[lpos].astype(np.float64) * right.data[rpos].astype(np.float64)
    output = [{
        "feature_index": int(i),
        "term": str(terms[int(i)]),
        "profile_weight": float(left.data[int(lp)]),
        "candidate_weight": float(right.data[int(rp)]),
        "contribution": float(value),
        "idf": float(idf[int(i)]),
        "document_frequency": int(document_frequency[int(i)]),
        "document_frequency_share": float(document_frequency[int(i)] / n_documents),
        "high_document_frequency_over_50_percent": bool(document_frequency[int(i)] / n_documents > 0.5),
    } for i, lp, rp, value in zip(common, lpos, rpos, values)]
    output.sort(key=lambda row: (-row["contribution"], row["term"]))
    return output


def _quantile_group(values: np.ndarray, percentile: float) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Use a strict upper-percentile cutoff; keep every boundary tie together below it."""
    values = np.asarray(values, dtype=np.float64)
    threshold = float(np.quantile(values, percentile, method="linear"))
    mask = values > threshold
    tie_mask = values == threshold
    return threshold, mask, {
        "method": "numpy.quantile(method='linear')",
        "percentile": float(percentile),
        "threshold_exclusive": True,
        "ties": "all games equal to the threshold remain below the upper group; never split a tie by App ID",
        "threshold": threshold,
        "games_at_threshold": int(np.count_nonzero(tie_mask)),
        "games_selected": int(np.count_nonzero(mask)),
        "candidate_count": int(len(values)),
        "selected_candidate_share": float(np.mean(mask)) if len(mask) else None,
    }


def _concentration_report(
    records: list[dict[str, Any]],
    candidate_apps: np.ndarray,
    recommendations: dict[str, np.ndarray],
    slot_counts: Counter,
    top1_counts: Counter,
    matrix: sparse.csr_matrix,
    feature_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_app = {int(row["app_id"]): row for row in records}
    token_count = np.asarray([int(by_app[int(app)]["token_count"]) for app in candidate_apps], dtype=np.int64)
    threshold, top_group, top_group_rule = _quantile_group(token_count, 0.8)
    rec_apps = recommendations["apps"]
    selected_ids = set(int(app) for app in candidate_apps[top_group])
    slot_share = (sum(1 for app in rec_apps if int(app) in selected_ids) / len(rec_apps)) if len(rec_apps) else None
    inclusive_ids = set(int(app) for app in candidate_apps[token_count >= threshold])
    inclusive_slot_share = (sum(1 for app in rec_apps if int(app) in inclusive_ids) / len(rec_apps)) if len(rec_apps) else None
    total_slots = int(len(rec_apps))
    top_slots = sorted(slot_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    top1 = sorted(top1_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    c_idx = {int(app): i for i, app in enumerate(np.asarray([r["app_id"] for r in records], dtype=np.int64))}
    recorded_stop_words = sorted(ENGLISH_STOP_WORDS)
    stop_set = set(recorded_stop_words)
    group_rows: list[dict[str, Any]] = []
    top_games = []
    for app, count in top_slots:
        row = by_app[int(app)]
        row_index = c_idx[int(app)]
        vector = matrix.getrow(row_index)
        square_total = float(np.dot(vector.data.astype(np.float64), vector.data.astype(np.float64)))
        stop_square = sum(float(value) ** 2 for j, value in zip(vector.indices, vector.data) if feature_names[int(j)] in stop_set)
        stop_share = stop_square / square_total if square_total else 0.0
        top_games.append({
            "app_id": int(app),
            "name": row.get("c_name") or row.get("a_name") or None,
            "recommendation_slots": int(count),
            "recommendation_slot_share": float(count / total_slots) if total_slots else None,
            "top1_slots": int(top1_counts.get(int(app), 0)),
            "token_count": int(row["token_count"]),
            "text_source": row["text_source"],
            "script_profile": row["script_profile"],
            "duplicate_text_group_size": int(row["duplicate_text_group_size"]),
            "recorded_sklearn_english_stop_words_squared_tfidf_mass_share": float(stop_share),
            "squared_weight_denominator": "sum_j(tfidf_weight_j ** 2) for this L2-normalized game vector; equals 1 when supported",
        })
        group_rows.append({
            "app_id": int(app),
            "name": top_games[-1]["name"],
            "recommendation_slots": int(count),
            "top1_slots": int(top1_counts.get(int(app), 0)),
            "token_count": int(row["token_count"]),
            "text_source": row["text_source"],
        })
    result = {
        "recommendation_slot_count": total_slots,
        "top_four_games_by_all_top10_slots": top_games,
        "top_four_games_by_rank1_count": [
            {"app_id": int(app), "name": by_app[int(app)].get("c_name") or by_app[int(app)].get("a_name") or None,
             "top1_slots": int(count), "top1_share": float(count / sum(top1_counts.values())) if sum(top1_counts.values()) else None}
            for app, count in top1
        ],
        "token_count_candidate_group": {
            **top_group_rule,
            "metric": "share of saved v1 Top 10 recommendation slots assigned to Linux candidates in the token-count upper group",
            "recommendation_slot_share": float(slot_share) if slot_share is not None else None,
            "comparison_to_prior_backlog_recording": {
                "recorded_share": 0.461,
                "computed_share": float(slot_share) if slot_share is not None else None,
                "difference_percentage_points": float((slot_share - 0.461) * 100) if slot_share is not None else None,
                "interpretation": "Strictly greater than the linear p80 cutoff with boundary ties kept below reproduces the previously recorded 46.1% to rounding; the inclusive-boundary sensitivity is reported separately.",
            },
            "boundary_ties_inclusive_sensitivity": {
                "games_selected": int(np.count_nonzero(token_count >= threshold)),
                "candidate_share": float(np.mean(token_count >= threshold)) if len(token_count) else None,
                "recommendation_slot_share": float(inclusive_slot_share) if inclusive_slot_share is not None else None,
            },
            "token_count_summary_linux_candidates": {
                "min": int(token_count.min()) if len(token_count) else None,
                "median": float(np.median(token_count)) if len(token_count) else None,
                "p80": threshold,
                "max": int(token_count.max()) if len(token_count) else None,
            },
        },
        "stopword_definition": {
            "historical_21_word_set": "not available in saved source artifacts; historical 21-word statistic is not claimed as reproduced",
            "new_defined_set": "sklearn.feature_extraction.text.ENGLISH_STOP_WORDS from the recorded runtime",
            "runtime_scikit_learn": sklearn.__version__,
            "word_count": len(recorded_stop_words),
            "terms": recorded_stop_words,
            "weight_share": "sum squared L2-normalized TF-IDF weights for listed words divided by total squared TF-IDF weight in each game vector",
            "comparison_population": "four most frequently recommended Linux candidate games across all saved v1 validation Top 10 slots",
        },
    }
    return result, group_rows


def _table_write(path: Path, rows: list[dict[str, Any]], schema: pa.Schema | None = None) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def _write_model_artifact(
    folder: Path,
    *,
    model_id: str,
    app_ids: np.ndarray,
    linux: np.ndarray,
    matrix: sparse.csr_matrix,
    terms: list[str],
    idf: np.ndarray,
    parameters: dict[str, Any],
    source_hashes: dict[str, str],
    code_hashes: dict[str, str],
    game_columns: dict[str, list[Any]] | None = None,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=False)
    matrix = matrix.tocsr().astype(np.float32)
    matrix.sort_indices()
    if matrix.shape[0] != len(app_ids) or matrix.shape[0] != len(linux) or matrix.shape[1] != len(terms) or len(idf) != len(terms):
        raise ValueError(f"Artifact dimensions do not align for {model_id}")
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError(f"Artifact has invalid TF-IDF values for {model_id}")
    row_counts = np.diff(matrix.indptr).astype(np.int32)
    base = {
        "row_id": np.arange(len(app_ids), dtype=np.int64).tolist(),
        "app_id": np.asarray(app_ids, dtype=np.int64).tolist(),
        "linux": np.asarray(linux, dtype=bool).tolist(),
        "feature_count": row_counts.tolist(),
    }
    if game_columns:
        base.update(game_columns)
    _table_write(folder / "game_rows.parquet", [{key: values[i] for key, values in base.items()} for i in range(len(app_ids))])
    df = np.asarray((matrix != 0).sum(axis=0)).ravel().astype(np.int32)
    _table_write(folder / "vocabulary_idf.parquet", [
        {"feature_index": int(i), "term": str(term), "idf": float(idf[i]), "document_frequency": int(df[i])}
        for i, term in enumerate(terms)
    ], pa.schema([
        ("feature_index", pa.int32()), ("term", pa.string()), ("idf", pa.float32()),
        ("document_frequency", pa.int32()),
    ]))
    sparse.save_npz(folder / "tfidf_csr.npz", matrix, compressed=True)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "complete",
        "model_id": model_id,
        "parameters": parameters,
        "source_file_hashes": source_hashes,
        "code_hashes": code_hashes,
        "index": {
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "nnz": int(matrix.nnz),
            "dtype": str(matrix.dtype),
            "document_frequency_denominator": int(matrix.shape[0]),
            "normalization": "row-wise L2; exact sparse cosine via inner product",
        },
        "conditional_gate": gate,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__, "pyarrow": pa.__version__},
    }
    manifest["files"] = _file_inventory(folder)
    write_json(folder / "manifest.json", manifest)
    return manifest


def _read_metadata_sources(
    split: Path,
    source_manifest: dict[str, Any],
    reference: dict[str, Any],
    descriptions: list[dict[str, Any]],
    source_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = split / "catalog/input_apps.parquet"
    a_path = EDA / "legacy/data/processed/A_games_metadata.parquet"
    c_path = EDA / "legacy/data/processed/C_games_2025.parquet"
    old_hashes = {row["file"]: row["sha256"] for row in reference["former_catalog_inputs"]}
    for name, path, expected in (
        ("input_apps", input_path, source_hashes["input_apps"]),
        ("A_games_metadata", a_path, old_hashes["A_games_metadata.parquet"]),
        ("C_games_2025", c_path, old_hashes["C_games_2025.parquet"]),
    ):
        if digest(path) != expected:
            raise ValueError(f"Pinned metadata input fingerprint changed: {name}")
    input_rows = pq.read_table(input_path).to_pylist()
    app_ids = [int(row["app_id"]) for row in input_rows]
    if app_ids != sorted(set(app_ids)):
        raise ValueError("Input app catalog must remain uniquely sorted")
    a_table = pq.read_table(a_path, columns=["app_id", "tags"]).to_pylist()
    c_table = pq.read_table(c_path, columns=["appid", "name", "tags", "genres"]).to_pylist()
    a_by_app: dict[int, Any] = {}
    for row in a_table:
        app = int(row["app_id"])
        if app in a_by_app:
            raise ValueError(f"Duplicate A metadata app id: {app}")
        a_by_app[app] = row["tags"]
    c_by_app: dict[int, dict[str, Any]] = {}
    for row in c_table:
        app = int(row["appid"])
        if app in c_by_app:
            raise ValueError(f"Duplicate C metadata app id: {app}")
        c_by_app[app] = row

    descriptions_by_app = {int(row["app_id"]): row for row in descriptions}
    audit: list[dict[str, Any]] = []
    parse_counts: Counter[tuple[str, str]] = Counter()
    a_available = c_tag_available = c_genre_available = unused_fallback = 0
    invalid_a = invalid_c_tags = invalid_c_genres = 0
    for catalog_row in input_rows:
        app = int(catalog_row["app_id"])
        a_labels, a_status, a_parser = parse_metadata_labels(a_by_app.get(app), "tags")
        c_row = c_by_app.get(app, {})
        c_tags, c_tag_status, c_tag_parser = parse_metadata_labels(c_row.get("tags"), "tags")
        genres, genre_status, genre_parser = parse_metadata_labels(c_row.get("genres"), "genres")
        parse_counts[("A_tags", a_status)] += 1
        parse_counts[("C_tags", c_tag_status)] += 1
        parse_counts[("C_genres", genre_status)] += 1
        if a_status == "invalid":
            invalid_a += 1
        if c_tag_status == "invalid":
            invalid_c_tags += 1
        if genre_status == "invalid":
            invalid_c_genres += 1
        chosen_tags, tag_source = choose_tag_source(a_labels, a_status, c_tags, c_tag_status)
        if a_status == "valid":
            a_available += 1
            if c_tag_status == "valid":
                unused_fallback += 1
        if c_tag_status == "valid":
            c_tag_available += 1
        if genre_status == "valid":
            c_genre_available += 1

        normalized_tags = sorted({value for label in chosen_tags if (value := normalize_concept(label))})
        normalized_genres = sorted({value for label in genres if (value := normalize_concept(label))})
        features = sorted({f"tag::{x}" for x in normalized_tags} | {f"genre::{x}" for x in normalized_genres})
        desc = descriptions_by_app[app]
        audit.append({
            "app_id": app,
            "linux": bool(catalog_row["linux"]),
            "name": c_row.get("name"),
            "a_tags_raw": a_by_app.get(app),
            "c_tags_raw": c_row.get("tags"),
            "c_genres_raw": c_row.get("genres"),
            "a_tags_status": a_status,
            "a_tags_parser": a_parser,
            "c_tags_status": c_tag_status,
            "c_tags_parser": c_tag_parser,
            "c_genres_status": genre_status,
            "c_genres_parser": genre_parser,
            "tag_source": tag_source,
            "a_tag_count": len(a_labels),
            "c_tag_count": len(c_tags),
            "chosen_tag_count": len(normalized_tags),
            "normalized_tags": normalized_tags,
            "genre_count": len(normalized_genres),
            "normalized_genres": normalized_genres,
            "metadata_features": features,
            "metadata_feature_count": len(features),
            "description_text_source": desc["text_source"],
            "description_token_count": int(desc["token_count"]),
        })
    summary = {
        "input_apps": len(audit),
        "A_valid_tag_rows": int(a_available),
        "C_valid_tag_rows": int(sum(r["c_tags_status"] == "valid" for r in audit)),
        "tag_rows_using_A": int(sum(r["tag_source"] == "A" for r in audit)),
        "tag_rows_using_C_fallback": int(sum(r["tag_source"] == "C" for r in audit)),
        "tag_rows_missing_or_invalid": int(sum(r["tag_source"] == "missing" for r in audit)),
        "unused_valid_C_tag_fallbacks_when_A_won": int(unused_fallback),
        "C_valid_genre_rows": int(c_genre_available),
        "A_tag_parser_invalid_rows": int(invalid_a),
        "C_tag_parser_invalid_rows": int(invalid_c_tags),
        "C_genre_parser_invalid_rows": int(invalid_c_genres),
        "parser_status_counts": {f"{field}:{status}": int(count) for (field, status), count in sorted(parse_counts.items())},
        "precedence": "A tags are chosen when they parse to a non-empty valid label list; C serialized tags are used only when A is missing, empty or invalid. C genres are a separate field.",
        "normalization": "Unicode NFKC, casefold, collapse and trim whitespace; preserve phrase text and punctuation; prefix tag:: or genre::; deduplicate concepts per game.",
    }
    return audit, summary


def _build_metadata_artifact(
    stage: Path,
    split: Path,
    descriptions: list[dict[str, Any]],
    app_ids: np.ndarray,
    linux: np.ndarray,
    source_hashes: dict[str, str],
    code_hashes: dict[str, str],
    val_target_app: np.ndarray,
    support: dict[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = read_json(EDA / "data/source_manifest.json")
    reference = read_json(EDA / "data/reference/manifest.json")
    audit, source_summary = _read_metadata_sources(split, source_manifest, reference, descriptions, source_hashes)
    docs = [row["metadata_features"] for row in audit]
    vec = TfidfVectorizer(analyzer=lambda doc: doc, binary=True, norm="l2", use_idf=True,
                         smooth_idf=True, sublinear_tf=False, dtype=np.float32)
    matrix = vec.fit_transform(docs).tocsr().astype(np.float32)
    matrix.sort_indices()
    names = vec.get_feature_names_out().tolist()
    idf = np.asarray(vec.idf_, dtype=np.float32)
    metadata_folder = stage / "metadata_channel"
    row_columns = {
        "tag_source": [row["tag_source"] for row in audit],
        "tag_available": [bool(row["chosen_tag_count"]) for row in audit],
        "genre_available": [bool(row["genre_count"]) for row in audit],
        "metadata_feature_count": [int(row["metadata_feature_count"]) for row in audit],
    }
    pars = {
        "channel": "tags_and_genres",
        "source_rule": "A valid non-empty tags list first; otherwise C tags dictionary keys; C genres list always separate.",
        "parser": "json.loads then ast.literal_eval for serialized strings; never eval.",
        "normalization": "Unicode NFKC + casefold + whitespace collapse; phrase-preserving; tag:: and genre:: prefixes.",
        "term_presence": "binary per game; duplicate concepts within a game collapsed",
        "tfidf": {"binary": True, "use_idf": True, "smooth_idf": True, "norm": "l2", "dtype": "float32"},
        "fit_scope": "all 37,610 fixed input catalog apps; metadata only, no reviews or validation labels",
        "source_audit": source_summary,
        "validation_target_support": {
            "targets": int(len(val_target_app)),
            "metadata_supported": int(sum(bool(audit[int(np.searchsorted(app_ids, app))]["metadata_feature_count"]) for app in val_target_app)),
            "metadata_unsupported": int(sum(not bool(audit[int(np.searchsorted(app_ids, app))]["metadata_feature_count"]) for app in val_target_app)),
            "tag_unsupported": int(sum(not bool(audit[int(np.searchsorted(app_ids, app))]["chosen_tag_count"]) for app in val_target_app)),
            "genre_unsupported": int(sum(not bool(audit[int(np.searchsorted(app_ids, app))]["genre_count"]) for app in val_target_app)),
        },
    }
    model_manifest = _write_model_artifact(
        metadata_folder,
        model_id="t4_separate_metadata_channel",
        app_ids=app_ids,
        linux=linux,
        matrix=matrix,
        terms=names,
        idf=idf,
        parameters=pars,
        source_hashes=source_hashes,
        code_hashes=code_hashes,
        game_columns=row_columns,
        gate=None,
    )
    # Raw and normalized metadata values remain available only in ignored output;
    # the tracked report records source provenance and aggregate parser results.
    _table_write(metadata_folder / "metadata_source_audit.parquet", audit)
    model_manifest["files"] = _file_inventory(metadata_folder)
    write_json(metadata_folder / "manifest.json", model_manifest)
    metadata_summary = {
        **source_summary,
        "metadata_vocab_size": len(names),
        "metadata_nnz": int(matrix.nnz),
        "metadata_featureless_apps": int(np.count_nonzero(np.diff(matrix.indptr) == 0)),
        "validation_target_support": pars["validation_target_support"],
        "validation_target_train_support_buckets": dict(Counter(
            str(_support_bucket(np.asarray([support.get(int(a), 0)]), [0, 1, 10, 100, 1000, 10000, 100000, 1000000],
                                ["0", "1-9", "10-99", "100-999", "1000-9999", "10000-99999", "100000-999999", "1000000+"])[0])
            for a in val_target_app
        )),
        "artifact_manifest_sha256": digest(metadata_folder / "manifest.json"),
    }
    return model_manifest, metadata_summary


def _serialize_case_rows(
    output_cases: list[dict[str, Any]],
    selected: dict[int, set[str]],
    per_user: dict[str, Any],
    query_inputs: dict[int, list[int]],
    records_by_app: dict[int, dict[str, Any]],
    top4_map: dict[int, list[tuple[int, float]]],
    support_bucket: np.ndarray,
    input_supported: np.ndarray,
    input_token_sum: np.ndarray,
    input_duplicate: np.ndarray,
    user_keys: np.ndarray,
    user_index: dict[int, int],
    score: np.ndarray,
    matrix: sparse.csr_matrix,
    app_ids: np.ndarray,
    feature_names: list[str],
    idf: np.ndarray,
    df: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    app_to_row = {int(app): i for i, app in enumerate(app_ids)}
    full_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    contribution_totals = {"high_df_wrong_top1": 0.0, "all_wrong_top1": 0.0, "high_df_target": 0.0, "all_target": 0.0}
    for uid in sorted(selected):
        i = user_index[uid]
        target = int(per_user["target_app_id"][i])
        target_record = records_by_app[target]
        recs = top4_map.get(uid, [])
        target_rank = int(per_user["target_rank"][i])
        wrong = next(((app, value) for rank_idx, (app, value) in enumerate(recs, start=1)
                      if app != target), None)
        positive_apps = query_inputs.get(uid, [])
        profile = _rows_for_terms(matrix, positive_apps, app_to_row)
        candidates: list[tuple[str, int, float | None, int | None]] = [("target", target, float(score[i]), target_rank or None)]
        if wrong is not None:
            candidates.append(("wrong_top_recommendation", int(wrong[0]), float(wrong[1]), next(j for j, pair in enumerate(recs, start=1) if pair[0] == wrong[0])))
        term_top = {}
        for role, candidate_app, stored_score, candidate_rank in candidates:
            candidate_row = app_to_row[candidate_app]
            contribs = contributions_for_pair(profile, matrix.getrow(candidate_row), feature_names, idf, df, len(app_ids))
            score_from_terms = float(sum(x["contribution"] for x in contribs))
            if stored_score is not None and abs(score_from_terms - stored_score) > 2e-5:
                raise ValueError(f"T0 term contributions do not sum to the saved score for user {uid} / {role}: {score_from_terms} vs {stored_score}")
            for term in contribs:
                term_rows.append({
                    "user_key": uid,
                    "outcome": str(per_user["outcome"][i]),
                    "support_log_bucket": str(support_bucket[i]),
                    "n_positive_inputs": int(per_user["n_positive_inputs"][i]),
                    "candidate_role": role,
                    "candidate_app_id": int(candidate_app),
                    "candidate_rank": candidate_rank,
                    "candidate_saved_score": stored_score,
                    **term,
                })
            term_top[role] = contribs[:12]
            outcome_name = str(per_user["outcome"][i])
            if role == "target":
                contribution_totals["all_target"] += score_from_terms
                contribution_totals["high_df_target"] += sum(c["contribution"] for c in contribs if c["high_document_frequency_over_50_percent"])
            elif outcome_name == "positive_rank_failure":
                contribution_totals["all_wrong_top1"] += score_from_terms
                contribution_totals["high_df_wrong_top1"] += sum(c["contribution"] for c in contribs if c["high_document_frequency_over_50_percent"])
        input_records = [records_by_app[int(app)] for app in positive_apps]
        wrong_app = int(wrong[0]) if wrong is not None else None
        wrong_record = records_by_app[wrong_app] if wrong_app is not None else None
        full_rows.append({
            "case_id": f"validation-{uid}",
            "user_key": uid,
            "outcome": str(per_user["outcome"][i]),
            "selection_reasons": sorted(selected[uid]),
            "support_log_bucket": str(support_bucket[i]),
            "target_train_support": int(per_user["target_train_support"][i]),
            "n_positive_inputs": int(per_user["n_positive_inputs"][i]),
            "positive_input_count_bin": _input_count_bin(int(per_user["n_positive_inputs"][i])),
            "n_total_inputs": int(per_user["n_total_inputs"][i]),
            "input_supported_count": int(input_supported[i]),
            "input_zero_feature_count": int(per_user["n_positive_inputs"][i] - input_supported[i]),
            "input_supported_share": float(input_supported[i] / per_user["n_positive_inputs"][i]),
            "input_mean_token_count": float(input_token_sum[i] / per_user["n_positive_inputs"][i]),
            "input_with_duplicate_text_count": int(input_duplicate[i]),
            "input_app_ids": [int(x) for x in positive_apps],
            "input_text_sources": [r["text_source"] for r in input_records],
            "input_script_profiles": [r["script_profile"] for r in input_records],
            "input_token_counts": [int(r["token_count"]) for r in input_records],
            "input_duplicate_text_group_sizes": [int(r["duplicate_text_group_size"]) for r in input_records],
            "target_app_id": target,
            "target_rank": target_rank,
            "target_score": float(score[i]),
            "target_text_source": target_record["text_source"],
            "target_text_status": target_record["text_status"],
            "target_script_profile": target_record["script_profile"],
            "target_token_count": int(target_record["token_count"]),
            "target_duplicate_text_group_size": int(target_record["duplicate_text_group_size"]),
            "target_supported": bool(target_record["feature_count"]),
            "top4_app_ids": [int(app) for app, _ in recs],
            "top4_scores": [float(value) for _, value in recs],
            "wrong_top_recommended_app_id": wrong_app,
            "wrong_top_recommended_score": float(wrong[1]) if wrong is not None else None,
            "wrong_top_recommended_rank": next((j for j, pair in enumerate(recs, start=1) if pair[0] == wrong_app), None) if wrong_app is not None else None,
            "wrong_top_text_source": wrong_record["text_source"] if wrong_record else None,
            "wrong_top_script_profile": wrong_record["script_profile"] if wrong_record else None,
            "wrong_top_token_count": int(wrong_record["token_count"]) if wrong_record else None,
            "wrong_top_duplicate_text_group_size": int(wrong_record["duplicate_text_group_size"]) if wrong_record else None,
            "target_top_terms": term_top.get("target", []),
            "wrong_top_terms": term_top.get("wrong_top_recommendation", []),
        })
    total_wrong = contribution_totals["all_wrong_top1"]
    gate_share = contribution_totals["high_df_wrong_top1"] / total_wrong if total_wrong > 0 else 0.0
    gate = {
        "decision_rule_fixed_before_v2_experiment_scoring": {
            "build_T3_when": "Aggregate contribution share from terms with document frequency >50% of the full fixed text catalog is at least 5% of all sampled wrong-top recommendation overlap in positive-rank-failure cases.",
            "threshold": T3_GATE_MIN_HIGH_DF_SHARE,
            "sample_basis": "fixed-seed T0 positive-rank-failure examples; same selected examples and exact profile-times-candidate term products as T0 report",
            "historical_stopword_21_list": "unavailable; this gate uses document frequency, not the unrecoverable historical stopword list",
        },
        "sampled_wrong_top1_overlap_contribution": float(total_wrong),
        "sampled_wrong_top1_high_df_contribution": float(contribution_totals["high_df_wrong_top1"]),
        "sampled_wrong_top1_high_df_contribution_share": float(gate_share),
        "sampled_target_overlap_contribution": float(contribution_totals["all_target"]),
        "sampled_target_high_df_contribution": float(contribution_totals["high_df_target"]),
        "t3_choice": "build" if gate_share >= T3_GATE_MIN_HIGH_DF_SHARE else "skip",
    }
    return full_rows, term_rows, gate


def _aggregate_rows(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def build_all(
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
    split: Path = DEFAULT_SPLIT,
    v1: Path = DEFAULT_INDEX,
    protocol_path: Path = DEFAULT_PROTOCOL,
    run_lock_path: Path = DEFAULT_RUN_LOCK,
    samples_per_stratum: int = SAMPLES_PER_STRATUM,
) -> dict[str, Any]:
    """Create all v2 TF-IDF artifacts and the validation-only T0 aggregate report."""
    started = time.perf_counter()
    output, report, split, v1 = map(lambda p: Path(p).resolve(), (output, report, split, v1))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing TF-IDF v2 artifacts: {output}")
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite existing T0 report: {report}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.building-{uuid.uuid4().hex[:10]}"
    stage.mkdir()
    try:
        log("Verifying frozen v1, split, protocol and validation-only outputs")
        pinned = _validate_fixed_sources(split, v1, Path(protocol_path).resolve(), Path(run_lock_path).resolve())
        protocol = pinned["protocol"]
        split_manifest = pinned["split_manifest"]
        v1_manifest = pinned["v1_manifest"]
        val_manifest = pinned["validation_manifest"]
        input_path = split / "catalog/input_apps.parquet"
        candidate_path = split / "catalog/linux_candidates.parquet"
        train_path = split / "train/positive_catalog.parquet"
        validation_dir = split / "validation"
        descriptions_path = v1 / "descriptions.parquet"
        v1_rows_path = v1 / "game_rows.parquet"
        v1_vocab_path = v1 / "vocabulary_idf.parquet"
        v1_matrix_path = v1 / "tfidf_csr.npz"
        code_hashes = {
            "improvement_tfidf": digest(Path(__file__).resolve()),
            "tfidf_content": digest(Path(__file__).with_name("tfidf_content.py")),
            "baseline_evaluator": digest(Path(__file__).with_name("evaluate_baselines.py")),
        }
        source_hashes = {
            "split_manifest": digest(split / "manifest.json"),
            "input_apps": digest(input_path),
            "linux_candidates": digest(candidate_path),
            "train_positive_catalog": digest(train_path),
            "v1_tfidf_manifest": digest(v1 / "manifest.json"),
            "v1_descriptions": digest(descriptions_path),
            "v1_game_rows": digest(v1_rows_path),
            "v1_vocabulary_idf": digest(v1_vocab_path),
            "v1_tfidf_csr": digest(v1_matrix_path),
            "validation_users": digest(validation_dir / "users.parquet"),
            "validation_query": digest(validation_dir / "query.parquet"),
            "validation_targets": digest(validation_dir / "targets.parquet"),
            "validation_exclusions": digest(validation_dir / "exclusions.parquet"),
            "baseline_validation_per_user": digest(DEFAULT_BASELINE_RESULTS / "tfidf/per_user.parquet"),
            "baseline_validation_recommendations": digest(DEFAULT_BASELINE_RESULTS / "tfidf/recommendations.parquet"),
            "A_games_metadata": digest(EDA / "legacy/data/processed/A_games_metadata.parquet"),
            "C_games_2025": digest(EDA / "legacy/data/processed/C_games_2025.parquet"),
            "source_manifest_snapshot": digest(EDA / "data/source_manifest.json"),
            "reference_manifest": digest(EDA / "data/reference/manifest.json"),
            "improvement_protocol": digest(Path(protocol_path).resolve()),
            "baseline_run_lock": digest(Path(run_lock_path).resolve()),
        }
        source_hashes["source_manifest_snapshot"] = pinned["source_manifest_sha256_at_build_start"]

        log("Reading fixed TF-IDF description catalog and saved validation outcomes")
        desc_table = pq.read_table(descriptions_path, columns=[
            "app_id", "linux", "text_source", "description_clean", "clean_sha256", "feature_count", "token_count", "unique_token_count",
            "text_status", "script_profile", "duplicate_text_group_size", "raw_char_count", "clean_char_count",
            "high_token_repetition",
        ])
        descriptions = desc_table.to_pylist()
        game_rows = pq.read_table(v1_rows_path).to_pydict()
        app_ids = np.asarray(game_rows["app_id"], dtype=np.int64)
        linux = np.asarray(game_rows["linux"], dtype=bool)
        if app_ids.tolist() != sorted(set(app_ids.tolist())) or app_ids.tolist() != [r["app_id"] for r in descriptions]:
            raise ValueError("TF-IDF descriptions and fixed game rows differ")
        if len(app_ids) != 37610 or int(linux.sum()) != 6630:
            raise ValueError("Fixed text catalog shape changed")
        if digest(EDA / "legacy/data/processed/A_games_metadata.parquet") != v1_manifest["sources"]["sources"]["A.description"]["sha256"]:
            raise ValueError("Pinned A metadata source differs from v1 text catalog")
        if digest(EDA / "legacy/data/processed/C_games_2025.parquet") != v1_manifest["sources"]["sources"]["C.short_description"]["sha256"]:
            raise ValueError("Pinned C metadata source differs from v1 text catalog")
        desc_by_app = {int(row["app_id"]): row for row in descriptions}
        candidate_apps = np.asarray(pq.read_table(candidate_path, columns=["app_id"]).column("app_id").to_numpy(), dtype=np.int64)

        log("Counting exact train supports from train positives only")
        train_support = _load_train_support(train_path)
        v1_outputs, t0_counts = _read_validation_results(split, DEFAULT_BASELINE_RESULTS, train_support, protocol)
        saved_tfidf = val_manifest["models"]["tfidf"]
        if (t0_counts["outcomes"]["success"] != saved_tfidf["hits"]
                or t0_counts["target_positive_score_users"] != saved_tfidf["target_positive_personalized_score_users"]
                or t0_counts["target_zero_score_users"] != saved_tfidf["target_zero_personalized_score_users"]):
            raise ValueError("T0 outcome groups do not reproduce the saved v1 validation aggregates")
        users = v1_outputs["users"]
        user_keys = v1_outputs["user_keys"]
        n_users = len(user_keys)
        support_labels = protocol["support_log_labels"]
        support_edges = protocol["support_log_edges"]
        support_buckets = _support_bucket(v1_outputs["train_support"], support_edges, support_labels)
        input_bins = np.asarray([_input_count_bin(int(n)) for n in v1_outputs["n_positive"]], dtype=object)

        recommendations = v1_outputs["recommendations"]
        top_game_counts = sorted(v1_outputs["slot_counts"].items(), key=lambda item: (-item[1], item[0]))[:4]
        top_games_force = [{"app_id": int(app), "slots": int(count)} for app, count in top_game_counts]
        top4_by_user = v1_outputs["top4"]
        selected, sampling = _selected_samples(
            user_keys, v1_outputs["outcome"], support_buckets, v1_outputs["n_positive"], top4_by_user,
            top_games_force, per_stratum=samples_per_stratum,
        )
        selected_ids = set(selected)
        selected_user_index = {int(uid): i for i, uid in enumerate(user_keys) if int(uid) in selected_ids}
        n_inputs_per_user = v1_outputs["n_positive"]
        input_supported = np.zeros(n_users, dtype=np.int32)
        input_total = np.zeros(n_users, dtype=np.int32)
        input_token_sum = np.zeros(n_users, dtype=np.int64)
        input_duplicate = np.zeros(n_users, dtype=np.int32)
        text_supported_by_row = np.asarray([row["text_status"] == "supported" for row in descriptions], dtype=np.int32)
        token_count_by_row = np.asarray([row["token_count"] for row in descriptions], dtype=np.int32)
        duplicate_by_row = np.asarray([row["duplicate_text_group_size"] > 1 for row in descriptions], dtype=np.int32)
        source_by_row = np.asarray([row["text_source"] for row in descriptions], dtype=object)
        script_by_row = np.asarray([row["script_profile"] for row in descriptions], dtype=object)
        outcome_labels = ("success", "zero_score_failure", "positive_rank_failure")
        input_source_counts = {label: Counter() for label in outcome_labels}
        input_script_counts = {label: Counter() for label in outcome_labels}
        input_duplicate_counts = Counter()
        input_supported_counts = Counter()
        input_token_values = {label: [] for label in outcome_labels}
        query_inputs: dict[int, list[int]] = defaultdict(list)
        app_to_row = {int(app): i for i, app in enumerate(app_ids)}
        validation_query_path = validation_dir / "query.parquet"
        query_pf = pq.ParquetFile(validation_query_path)
        for batch in query_pf.iter_batches(batch_size=250_000, columns=["user_key", "app_id", "positive"]):
            q = batch.to_pydict()
            q_user = np.asarray(q["user_key"], dtype=np.uint64)
            q_app = np.asarray(q["app_id"], dtype=np.int64)
            q_positive = np.asarray(q["positive"], dtype=bool)
            keep = np.flatnonzero(q_positive)
            if not len(keep):
                continue
            q_user, q_app = q_user[keep], q_app[keep]
            user_pos = np.searchsorted(user_keys, q_user)
            app_pos = np.searchsorted(app_ids, q_app)
            if np.any(user_pos >= n_users) or np.any(user_keys[user_pos] != q_user):
                raise ValueError("Validation query contains a non-cohort user")
            if np.any(app_pos >= len(app_ids)) or np.any(app_ids[app_pos] != q_app):
                raise ValueError("Validation query contains an app outside the fixed catalog")
            np.add.at(input_total, user_pos, 1)
            np.add.at(input_supported, user_pos, text_supported_by_row[app_pos])
            np.add.at(input_token_sum, user_pos, token_count_by_row[app_pos])
            np.add.at(input_duplicate, user_pos, duplicate_by_row[app_pos])
            chunk_outcomes = v1_outputs["outcome"][user_pos]
            for outcome_label in outcome_labels:
                outcome_mask = chunk_outcomes == outcome_label
                if not np.any(outcome_mask):
                    continue
                for counter, values in (
                    (input_source_counts[outcome_label], source_by_row[app_pos[outcome_mask]]),
                    (input_script_counts[outcome_label], script_by_row[app_pos[outcome_mask]]),
                ):
                    unique, counts = np.unique(values, return_counts=True)
                    counter.update({str(value): int(count) for value, count in zip(unique, counts)})
                input_duplicate_counts[outcome_label] += int(np.sum(duplicate_by_row[app_pos[outcome_mask]]))
                input_supported_counts[outcome_label] += int(np.sum(text_supported_by_row[app_pos[outcome_mask]]))
                input_token_values[outcome_label].append(token_count_by_row[app_pos[outcome_mask]])
            keep_selected = np.isin(user_pos, np.fromiter(selected_user_index.values(), dtype=np.int64))
            for row_idx, app in zip(user_pos[keep_selected], q_app[keep_selected]):
                uid = int(user_keys[row_idx])
                if uid in selected_ids:
                    query_inputs[uid].append(int(app))
        if not np.array_equal(input_total, n_inputs_per_user):
            raise ValueError("Fixed positive query row counts differ from saved validation users")
        if sum(sum(counter.values()) for counter in input_source_counts.values()) != int(n_inputs_per_user.sum()):
            raise ValueError("Positive input source aggregates do not cover the fixed validation query")
        if sum(input_supported_counts.values()) != int(input_supported.sum()):
            raise ValueError("Positive input text-support aggregates do not reproduce per-user support counts")
        if any(len(query_inputs.get(uid, [])) != int(n_inputs_per_user[i]) for uid, i in selected_user_index.items()):
            raise ValueError("Sampled validation positive inputs differ from the fixed query")

        log("Building v1 T0 cases and exact term contribution records")
        matrix = sparse.load_npz(v1_matrix_path).tocsr().astype(np.float32)
        vocab = pq.read_table(v1_vocab_path).to_pydict()
        feature_names = [str(x) for x in vocab["term"]]
        idf = np.asarray(vocab["idf"], dtype=np.float32)
        if matrix.shape != (len(app_ids), len(feature_names)) or matrix.nnz != v1_manifest["index"]["nnz"]:
            raise ValueError("Frozen v1 TF-IDF matrix differs from its manifest")
        document_frequency = np.asarray((matrix != 0).sum(axis=0)).ravel().astype(np.int32)
        expected_idf = (np.log((1.0 + matrix.shape[0]) / (1.0 + document_frequency)) + 1.0).astype(np.float32)
        if not np.allclose(expected_idf, idf, rtol=0.0, atol=2e-6):
            raise ValueError("Recomputed v1 document frequencies do not reproduce the smoothed IDF values")

        per_user = {
            "user_key": user_keys,
            "target_app_id": v1_outputs["target_app"],
            "target_rank": v1_outputs["rank"],
            "target_score": v1_outputs["score"],
            "target_train_support": v1_outputs["train_support"],
            "n_positive_inputs": v1_outputs["n_positive"],
            "n_total_inputs": v1_outputs["n_total"],
            "outcome": v1_outputs["outcome"],
        }
        records = descriptions
        records_by_app = desc_by_app
        per_user_for_cases = {
            **per_user,
            "outcome": v1_outputs["outcome"],
        }
        case_rows, term_rows, t3_gate = _serialize_case_rows(
            [], selected, per_user_for_cases, query_inputs, records_by_app, top4_by_user,
            support_buckets, input_supported, input_token_sum, input_duplicate,
            user_keys, selected_user_index, v1_outputs["score"], matrix, app_ids,
            feature_names, idf, document_frequency,
        )
        sampled_strata = Counter(
            f"{row['outcome']}|support={row['support_log_bucket']}|positive_inputs={row['positive_input_count_bin']}"
            for row in case_rows if "stratified_case" in row["selection_reasons"]
        )
        # T0 per-user rows contain exact fixed-cohort support and input counts.
        user_diag = []
        for i, uid_raw in enumerate(user_keys):
            uid = int(uid_raw)
            target = int(v1_outputs["target_app"][i])
            record = desc_by_app[target]
            top = top4_by_user.get(uid, [])
            user_diag.append({
                "user_key": uid,
                "outcome": str(v1_outputs["outcome"][i]),
                "target_app_id": target,
                "target_train_support": int(v1_outputs["train_support"][i]),
                "support_log_bucket": str(support_buckets[i]),
                "v1_target_rank": int(v1_outputs["rank"][i]),
                "v1_target_score": float(v1_outputs["score"][i]),
                "n_positive_inputs": int(v1_outputs["n_positive"][i]),
                "positive_input_count_bin": str(input_bins[i]),
                "n_total_inputs": int(v1_outputs["n_total"][i]),
                "supported_positive_inputs": int(input_supported[i]),
                "zero_feature_positive_inputs": int(v1_outputs["n_positive"][i] - input_supported[i]),
                "positive_input_supported_share": float(input_supported[i] / v1_outputs["n_positive"][i]),
                "positive_input_mean_token_count": float(input_token_sum[i] / v1_outputs["n_positive"][i]),
                "positive_input_duplicate_text_count": int(input_duplicate[i]),
                "target_text_supported": bool(record["text_status"] == "supported"),
                "target_text_source": record["text_source"],
                "target_text_status": record["text_status"],
                "target_script_profile": record["script_profile"],
                "target_token_count": int(record["token_count"]),
                "target_duplicate_text_group_size": int(record["duplicate_text_group_size"]),
                "top4_app_ids": [int(app) for app, _ in top],
                "top4_scores": [float(score) for _, score in top],
            })
        _table_write(stage / "T0_user_diagnostics.parquet", user_diag)
        _table_write(stage / "T0_cases.parquet", case_rows)
        _table_write(stage / "T0_term_contributions.parquet", term_rows)
        vocabulary_rows = [{
            "term": feature_names[i],
            "document_frequency": int(document_frequency[i]),
            "document_frequency_share": float(document_frequency[i] / len(app_ids)),
            "idf": float(idf[i]),
            "over_50_percent_document_frequency": bool(document_frequency[i] / len(app_ids) > 0.5),
        } for i in range(len(feature_names))]
        _table_write(stage / "T0_vocabulary_document_frequency.parquet", vocabulary_rows)

        # Summaries intentionally describe only the already-scored v1 baseline.
        outcomes = v1_outputs["outcome"]
        outcome_counts = {label: int(np.count_nonzero(outcomes == label)) for label in ("success", "zero_score_failure", "positive_rank_failure")}
        if sum(outcome_counts.values()) != n_users:
            raise ValueError("T0 outcome categories do not cover every validation user")
        support_input_aggregates = {}
        for outcome_name in ("success", "zero_score_failure", "positive_rank_failure"):
            support_input_aggregates[outcome_name] = {}
            for bucket in support_labels:
                support_input_aggregates[outcome_name][bucket] = {}
                for input_bin in ("1", "2-4", "5-9", "10+"):
                    mask = (outcomes == outcome_name) & (support_buckets == bucket) & (input_bins == input_bin)
                    ii = np.flatnonzero(mask)
                    support_input_aggregates[outcome_name][bucket][input_bin] = {
                        "users": int(len(ii)),
                        "target_text_supported": int(sum(desc_by_app[int(v1_outputs["target_app"][j])]["text_status"] == "supported" for j in ii)),
                        "target_text_unsupported": int(sum(desc_by_app[int(v1_outputs["target_app"][j])]["text_status"] != "supported" for j in ii)),
                        "positive_input_supported_count": _aggregate_rows(input_supported[ii].tolist()),
                        "mean_positive_input_supported_share": float(np.mean(input_supported[ii] / v1_outputs["n_positive"][ii])) if len(ii) else None,
                    }

        all_text_summary = {}
        linux_text_summary = {}
        for label, mask in (("all_input_catalog", np.ones(len(app_ids), dtype=bool)), ("linux_candidates", linux)):
            chosen = [row for row, keep in zip(descriptions, mask) if keep]
            target_app_idx = np.asarray([app_to_row[int(app)] for app in v1_outputs["target_app"]], dtype=np.int64)
            row_summ = {
                "games": len(chosen),
                "source_counts": _aggregate_rows(r["text_source"] for r in chosen),
                "status_counts": _aggregate_rows(r["text_status"] for r in chosen),
                "script_profile_counts": _aggregate_rows(r["script_profile"] for r in chosen),
                "token_count": {
                    "min": int(min((r["token_count"] for r in chosen), default=0)),
                    "median": float(np.median([r["token_count"] for r in chosen])) if chosen else None,
                    "p25": float(np.quantile([r["token_count"] for r in chosen], 0.25, method="linear")) if chosen else None,
                    "p75": float(np.quantile([r["token_count"] for r in chosen], 0.75, method="linear")) if chosen else None,
                    "p80": float(np.quantile([r["token_count"] for r in chosen], 0.80, method="linear")) if chosen else None,
                    "max": int(max((r["token_count"] for r in chosen), default=0)),
                    "high_token_repetition_games": int(sum(bool(r["high_token_repetition"]) for r in chosen)),
                    "duplicate_text_groups_with_more_than_one_game": int(sum(
                        count > 1 for count in Counter(r["clean_sha256"] for r in chosen if r["description_clean"]).values()
                    )),
                    "games_in_duplicate_text_groups": int(sum(
                        count for count in Counter(r["clean_sha256"] for r in chosen if r["description_clean"]).values() if count > 1
                    )),
                },
            }
            if label == "all_input_catalog":
                all_text_summary = row_summ
            else:
                linux_text_summary = row_summ
        target_text_by_outcome = {}
        input_text_by_outcome = {}
        for outcome_name in ("success", "zero_score_failure", "positive_rank_failure"):
            idx = np.flatnonzero(outcomes == outcome_name)
            target_records = [desc_by_app[int(v1_outputs["target_app"][i])] for i in idx]
            input_tokens = np.concatenate(input_token_values[outcome_name]) if input_token_values[outcome_name] else np.asarray([], dtype=np.int32)
            input_text_by_outcome[outcome_name] = {
                "positive_input_rows": int(input_tokens.size),
                "source_counts": {key: int(value) for key, value in sorted(input_source_counts[outcome_name].items())},
                "script_profile_counts": {key: int(value) for key, value in sorted(input_script_counts[outcome_name].items())},
                "token_count_quantiles_linear": {
                    str(q): float(np.quantile(input_tokens, q, method="linear")) if input_tokens.size else None
                    for q in (0, 0.25, 0.5, 0.75, 0.8, 1)
                },
                "positive_input_rows_in_duplicate_text_groups": int(input_duplicate_counts[outcome_name]),
                "positive_input_rows_with_supported_text": int(input_supported_counts[outcome_name]),
            }
            target_text_by_outcome[outcome_name] = {
                "users": int(len(idx)),
                "source_counts": _aggregate_rows(r["text_source"] for r in target_records),
                "status_counts": _aggregate_rows(r["text_status"] for r in target_records),
                "script_profile_counts": _aggregate_rows(r["script_profile"] for r in target_records),
                "token_count_quantiles_linear": {
                    str(q): float(np.quantile([r["token_count"] for r in target_records], q, method="linear")) if target_records else None
                    for q in (0, 0.25, 0.5, 0.75, 0.8, 1)
                },
                "targets_in_duplicate_text_groups": int(sum(r["duplicate_text_group_size"] > 1 for r in target_records)),
            }

        concentration, _ = _concentration_report(
            descriptions, candidate_apps, recommendations, v1_outputs["slot_counts"],
            v1_outputs["top1_counts"], matrix, feature_names,
        )
        # Attach pinned game titles for the selected top four cases.
        c_meta = pq.read_table(EDA / "legacy/data/processed/C_games_2025.parquet", columns=["appid", "name"])
        c_names = {int(a): str(n) if n is not None else None for a, n in zip(c_meta["appid"].to_pylist(), c_meta["name"].to_pylist())}
        for item in concentration["top_four_games_by_all_top10_slots"]:
            item["name"] = c_names.get(int(item["app_id"]))
        for item in concentration["top_four_games_by_rank1_count"]:
            item["name"] = c_names.get(int(item["app_id"]))
        # T3 is gated only from exact T0 sampled term overlap; no v2 rank is run.

        metadata_manifest, metadata_summary = _build_metadata_artifact(
            stage, split, descriptions, app_ids, linux, source_hashes, code_hashes,
            v1_outputs["target_app"], train_support,
        )

        log("Writing baseline control and isolated text-only feature variants")
        baseline_terms = feature_names
        baseline_params = {
            "variant": "v1_control",
            "cleaning": CLEANING_CONFIG,
            "tfidf": VECTORIZER_CONFIG,
            "experiment_scoring": False,
            "fit_scope": "Frozen v1 artifact fitted on all fixed input catalog cleaned descriptions; no reviews.",
        }
        _write_model_artifact(
            stage / "baseline_control", model_id="v1_control", app_ids=app_ids, linux=linux,
            matrix=matrix, terms=baseline_terms, idf=idf, parameters=baseline_params,
            source_hashes=source_hashes, code_hashes=code_hashes,
            game_columns={"text_source": [r["text_source"] for r in descriptions],
                          "text_status": [r["text_status"] for r in descriptions]},
        )
        del matrix
        gc.collect()

        text_documents = [str(row["description_clean"] or "") for row in descriptions]
        for folder_name in ("T1_sublinear_tf", "T2_unigram_bigram"):
            cfg = TEXT_VARIANTS[folder_name]
            ngram_range = tuple(int(x) for x in cfg["ngram_range"])
            vec = make_text_vectorizer(sublinear_tf=bool(cfg["sublinear_tf"]), ngram_range=ngram_range,
                                       max_df=float(cfg["max_df"]))
            with threadpool_limits(limits=2):
                variant_matrix = vec.fit_transform(text_documents).tocsr().astype(np.float32)
            variant_matrix.sort_indices()
            terms = vec.get_feature_names_out().tolist()
            params = {
                "variant": cfg["id"],
                "change": cfg["change"],
                "cleaning": CLEANING_CONFIG,
                "tfidf": {**VECTORIZER_CONFIG, "ngram_range": list(ngram_range), "sublinear_tf": bool(cfg["sublinear_tf"]), "max_df": cfg["max_df"]},
                "fit_scope": "All fixed input catalog cleaned descriptions including zero text rows; no validation/test text or review labels.",
                "experiment_scoring": False,
            }
            _write_model_artifact(
                stage / folder_name, model_id=cfg["id"], app_ids=app_ids, linux=linux,
                matrix=variant_matrix, terms=terms, idf=np.asarray(vec.idf_, dtype=np.float32),
                parameters=params, source_hashes=source_hashes, code_hashes=code_hashes,
                game_columns={"text_source": [r["text_source"] for r in descriptions],
                              "text_status": [r["text_status"] for r in descriptions]},
            )
            del variant_matrix, vec
            gc.collect()

        if t3_gate["t3_choice"] == "build":
            cfg = {"sublinear_tf": False, "ngram_range": (1, 1), "max_df": 0.5}
            vec = make_text_vectorizer(**cfg)
            with threadpool_limits(limits=2):
                matrix_t3 = vec.fit_transform(text_documents).tocsr().astype(np.float32)
            matrix_t3.sort_indices()
            _write_model_artifact(
                stage / "T3_max_df_0_5", model_id="T3_conditional_max_df_0_5", app_ids=app_ids, linux=linux,
                matrix=matrix_t3, terms=vec.get_feature_names_out().tolist(), idf=np.asarray(vec.idf_, dtype=np.float32),
                parameters={"variant": "max_df_0.5_only", "change": "max_df=0.5 only; built because the predeclared T0 gate passed.",
                            "cleaning": CLEANING_CONFIG,
                            "tfidf": {**VECTORIZER_CONFIG, "max_df": 0.5},
                            "fit_scope": "All fixed input catalog cleaned descriptions; no validation/test review labels.",
                            "experiment_scoring": False},
                source_hashes=source_hashes, code_hashes=code_hashes, gate=t3_gate,
            )
            del matrix_t3, vec
            gc.collect()

        # Refine concentration game titles in T0 examples, keeping title lookup
        # restricted to the pinned public metadata source.
        for row in case_rows:
            row["top4_names"] = [c_names.get(int(app)) for app in row["top4_app_ids"]]
            row["target_name"] = c_names.get(int(row["target_app_id"]))
            row["wrong_top_recommended_name"] = c_names.get(int(row["wrong_top_recommended_app_id"])) if row["wrong_top_recommended_app_id"] is not None else None
        # Rewrite case parquet after adding display titles.
        _table_write(stage / "T0_cases.parquet", case_rows)

        support_counts_by_bucket = {}
        for bucket in support_labels:
            mask = support_buckets == bucket
            support_counts_by_bucket[bucket] = {
                "users": int(mask.sum()),
            }
            for label in ("success", "zero_score_failure", "positive_rank_failure"):
                support_counts_by_bucket[bucket][label] = int(np.count_nonzero(mask & (outcomes == label)))

        t0_summary = {
            "scope": "Fixed validation cohort only; outcomes and recommendations are from saved v1 TF-IDF results. No v2 experiment ranking scores were computed.",
            "seed": SAMPLE_SEED,
            "users": n_users,
            "outcome_counts": outcome_counts,
            "outcome_shares": {key: float(value / n_users) for key, value in outcome_counts.items()},
            "target_positive_personalized_score_users": t0_counts["target_positive_score_users"],
            "target_zero_personalized_score_users": t0_counts["target_zero_score_users"],
            "target_train_support_source_and_validation": {
                "source": t0_counts["support_source"],
                "unique_train_supported_apps": len(train_support),
                "minimum_positive_app_support": int(min(train_support.values())) if train_support else None,
                "maximum_app_support": int(max(train_support.values())) if train_support else None,
                "target_support_buckets": support_counts_by_bucket,
                "target_train_support_matches_v1_per_user": True,
            },
            "positive_input_counts": {
                "total_positive_query_rows": int(n_inputs_per_user.sum()),
                "mean_per_user": float(np.mean(n_inputs_per_user)),
                "median_per_user": float(np.median(n_inputs_per_user)),
                "total_review_input_rows_in_query": int(v1_outputs["n_total"].sum()),
                "mean_total_review_inputs_per_user": float(np.mean(v1_outputs["n_total"])),
                "median_total_review_inputs_per_user": float(np.median(v1_outputs["n_total"])),
                "positive_inputs_with_supported_description": int(input_supported.sum()),
                "positive_inputs_without_supported_description": int(n_inputs_per_user.sum() - input_supported.sum()),
                "users_with_zero_supported_positive_inputs": int(np.count_nonzero(input_supported == 0)),
                "mean_supported_input_share": float(np.mean(input_supported / n_inputs_per_user)),
            },
            "outcomes_by_support_and_positive_input_count": support_input_aggregates,
            "target_text_by_outcome": target_text_by_outcome,
            "positive_input_text_by_outcome": input_text_by_outcome,
            "text_catalog": {"all_input_catalog": all_text_summary, "linux_candidates": linux_text_summary},
            "recommendation_concentration": concentration,
            "term_document_frequency": {
                "documents": int(len(app_ids)),
                "vocabulary_size": len(feature_names),
                "over_50_percent_df_terms": int(np.count_nonzero(document_frequency / len(app_ids) > 0.5)),
                "over_50_percent_df_term_list": [
                    {"term": feature_names[i], "df": int(document_frequency[i]), "share": float(document_frequency[i] / len(app_ids)), "idf": float(idf[i])}
                    for i in np.flatnonzero(document_frequency / len(app_ids) > 0.5)
                ],
            },
            "sampling": sampling,
            "selected_examples": {
                "users": len(case_rows),
                "outcome_counts": _aggregate_rows(row["outcome"] for row in case_rows),
                "sampled_case_counts_by_stratum": {key: int(value) for key, value in sorted(sampled_strata.items())},
                "term_contribution_rows": len(term_rows),
                "contribution_definition": "exact profile term weight × candidate TF-IDF term weight; profile is the L2-normalized equal mean of saved positive input game vectors; all shared positive terms are stored and sum to the saved v1 cosine score within 2e-5.",
                "example_rows_include": ["top four saved recommendations and scores", "target and wrong highest recommendation", "input/target/wrong-candidate source, script profile, token count and duplicate group size", "per-term DF and IDF"],
            },
            "T3_conditional_gate": t3_gate,
            "method_definitions": {
                "failure_groups": {"success": "v1 target rank 1-10", "zero_score_failure": "target score <= 0 and target rank 0", "positive_rank_failure": "target score > 0 and target rank 0"},
                "support_log_buckets": {"edges": support_edges, "labels": support_labels, "ties": "integer support counts are assigned to the half-open label ranges encoded by these fixed edges"},
                "positive_input_bins": {label: {"min": low, "max": high} for low, high, label in INPUT_COUNT_BINS},
                "script_profile": "Observed Unicode script blocks only; not an automatic language identification result.",
                "token_percentiles": "numpy.quantile(method='linear'); upper group is strictly greater than the threshold; every game tied at the boundary stays in the lower group and no tie is split by App ID. Inclusive-boundary sensitivity is also reported.",
                "historical_stop_words": "The earlier 21-word list is absent. The report carries the recorded scikit-learn English stop word set as a new separate diagnostic and does not claim the old statistic is reproduced.",
            },
        }

        log("Finalizing metadata source and v2 artifact provenance")
        report_body = {
            "version": VERSION,
            "status": "complete",
            "created_utc": utc_now(),
            "protocol": {
                "version": protocol["version"],
                "status": protocol["status"],
                "split_seed": SAMPLE_SEED,
                "split_manifest_sha256": source_hashes["split_manifest"],
                "improvement_protocol_sha256": source_hashes["improvement_protocol"],
                "baseline_run_lock_sha256": source_hashes["baseline_run_lock"],
            },
            "provenance": pinned["files"],
            "code_sha256": code_hashes,
            "diagnostics": t0_summary,
            "metadata_channel": metadata_summary,
            "models_built": [
                {"id": "v1_control", "directory": "baseline_control", "ranking_metrics_computed": False},
                {"id": "sublinear_tf_only", "directory": "T1_sublinear_tf", "ranking_metrics_computed": False},
                {"id": "word_unigram_bigram_only", "directory": "T2_unigram_bigram", "ranking_metrics_computed": False},
                *([{"id": "T3_conditional_max_df_0_5", "directory": "T3_max_df_0_5", "ranking_metrics_computed": False}] if t3_gate["t3_choice"] == "build" else []),
                {"id": "t4_separate_metadata_channel", "directory": "metadata_channel", "ranking_metrics_computed": False},
            ],
            "resource_limits": {"duckdb_threads": 2, "duckdb_memory_limit": "1GB", "fit_threadpool_limit": 2,
                                "max_rss_policy": "less than 3GB; no dense all-pairs or user-by-candidate score matrix"},
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        # Include supported/unsupported validation target detail in the T0 parquet
        # and report, but leave ranking selection/evaluation to the root evaluator.
        root_manifest = {
            "version": VERSION,
            "status": "complete",
            "created_utc": utc_now(),
            "source_file_hashes": source_hashes,
            "code_hashes": code_hashes,
            "protocol_sha256": source_hashes["improvement_protocol"],
            "split_manifest_sha256": source_hashes["split_manifest"],
            "t3_conditional_gate": t3_gate,
            "models": report_body["models_built"],
            "ranking_metrics_computed": False,
            "validation_users_read": n_users,
        }
        root_manifest["files"] = _file_inventory(stage)
        write_json(stage / "manifest.json", root_manifest)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
        report_body["artifact_directory"] = output.relative_to(EDA).as_posix()
        report_body["artifact_manifest_sha256"] = digest(output / "manifest.json")
        write_json(report, report_body)
        log(f"Wrote TF-IDF v2 artifacts: {output}")
        log(f"Wrote T0 report: {report}")
        return {"status": "complete", "artifact_directory": str(output), "report": str(report),
                "manifest_sha256": digest(output / "manifest.json"), "T0": t0_summary,
                "metadata": metadata_summary, "T3": t3_gate}
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--v1", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-lock", type=Path, default=DEFAULT_RUN_LOCK)
    parser.add_argument("--samples-per-stratum", type=int, default=SAMPLES_PER_STRATUM)
    args = parser.parse_args()
    if args.samples_per_stratum < 1:
        parser.error("--samples-per-stratum must be >= 1")
    result = build_all(args.output, args.report, args.split, args.v1, args.protocol, args.run_lock,
                       args.samples_per_stratum)
    print(json.dumps({"status": result["status"], "artifact_directory": result["artifact_directory"],
                      "report": result["report"], "manifest_sha256": result["manifest_sha256"],
                      "outcome_counts": result["T0"]["outcome_counts"],
                      "selected_examples": result["T0"]["selected_examples"]["users"],
                      "T3_choice": result["T3"]["t3_choice"],
                      "metadata_feature_vocabulary": result["metadata"]["metadata_vocab_size"]}, indent=2))


if __name__ == "__main__":
    main()
