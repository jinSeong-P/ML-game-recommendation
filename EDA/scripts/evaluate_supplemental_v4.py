"""Reporting-only target-seed and hidden-negative checks for v4 controls.

This evaluator is intentionally separate from experiment_v4_common.py. The
primary v4 evaluator stays fingerprinted and unchanged; this script rebuilds
alternate selection queries from each account's visible positive history plus
its original reserved target, then saves predictions and auditable summaries.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import EDA, digest, log
from evaluate_baselines import rank_positive
from experiment_v4_common import (
    ROOT,
    SPLIT,
    PROTOCOL as PRIMARY_PROTOCOL,
    ContentScorer,
    MatrixScorer,
    catalog,
    load_cohort,
    masked_scores,
    rank_values,
)
from build_experiments_v4_split import TARGET_NAMESPACE, uniform_target
from improvement_common import write_json


PROTOCOL = EDA / "data/reference/supplemental_protocol_v4.json"
OUTPUT = ROOT / "supplemental"
TARGET_OUTPUT = OUTPUT / "target_seed_variation"
NEGATIVE_OUTPUT = OUTPUT / "hidden_negative"
SEEDS = (20261108, 20261109, 20261110)
CONTROLS = (
    "binary_item_knn_k100",
    "activity_inverse_degree_cosine_k100",
    "metadata_extended_tfidf",
)
HIDDEN_NEGATIVE_NAMESPACE = "steam-v4-hidden-negative"
HIDDEN_NEGATIVE_SEED = 20261108
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261108
THREAD_LIMIT = 2
BATCH_SIZE = 128


def pair_matrix(rows, cols, shape, label):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    result = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
    ).tocsr()
    if result.nnz != len(rows) or np.any(result.data != 1):
        raise ValueError(f"Duplicate {label} user-game pairs")
    return result


def verify_frozen_inputs(require_selection_lock=True):
    protocol = json.loads(PROTOCOL.read_text("utf-8"))
    if protocol.get("status") != "fixed_before_supplemental_v4_outcomes":
        raise ValueError("Supplemental protocol is not frozen before outcomes")
    if tuple(protocol["seed_variation"]["target_seeds"]) != SEEDS:
        raise ValueError("Target-seed values differ from the frozen protocol")
    if tuple(protocol["controls"]) != CONTROLS:
        raise ValueError("Control list differs from the frozen protocol")
    if protocol["seed_variation"].get("accounts") != "same v4 selection accounts; reporting only, never used for setting selection":
        raise ValueError("Target-seed roster scope differs from the frozen protocol")
    if protocol["seed_variation"].get("target_sampler") != "same uniform positive Linux target sampler and original role namespace as v4 split":
        raise ValueError("Target-seed sampler differs from the frozen protocol")
    if protocol["seed_variation"].get("denominator") != "same accounts in each seed; report primary and short-history strata separately":
        raise ValueError("Target-seed denominators differ from the frozen protocol")
    if protocol["seed_variation"].get("uncertainty") != "average each user's outcomes across seeds before paired user resampling; never treat user x seed rows as independent":
        raise ValueError("Target-seed uncertainty method differs from the frozen protocol")
    if protocol["hidden_negative"].get("accounts") != "primary v4 selection accounts with at least one visible negative Linux game":
        raise ValueError("Hidden-negative account scope differs from the frozen protocol")
    if "seed20261108 namespace steam-v4-hidden-negative" not in protocol["hidden_negative"].get("sample", ""):
        raise ValueError("Hidden-negative sampler differs from the frozen protocol")
    if protocol["hidden_negative"].get("query") != "keep the positive target hidden and additionally remove sampled negative from query and exclusion mask; retain all selected users even if remaining total inputs become4":
        raise ValueError("Hidden-negative query definition differs from the frozen protocol")
    if protocol["hidden_negative"].get("score") != "fraction of selected users with hidden negative in top10; lower is better; one negative per user":
        raise ValueError("Hidden-negative score definition differs from the frozen protocol")
    if TARGET_NAMESPACE != "steam-experiments-v4-hidden-positive-linux-v1":
        raise ValueError("Original v4 target namespace changed")

    split_manifest_path = SPLIT / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text("utf-8"))
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("v4 split is not a verified completed split")
    for relative in (
        "catalog/input_apps.parquet",
        "selection/users.parquet",
        "selection/targets.parquet",
        "selection/query.parquet",
    ):
        expected = split_manifest.get("files", {}).get(relative, {}).get("sha256")
        if not expected or digest(SPLIT / relative) != expected:
            raise ValueError(f"Frozen selection input fingerprint mismatch: {relative}")

    lock_path = ROOT / "selection_frozen.json"
    lock = None
    if require_selection_lock:
        if not lock_path.is_file():
            raise ValueError("Primary v4 selection is not frozen; supplemental outcomes must wait")
        lock = json.loads(lock_path.read_text("utf-8"))
        if (lock.get("status") != "frozen_after_selection_before_confirmation"
                or lock.get("split_manifest_sha256") != digest(split_manifest_path)
                or lock.get("protocol_sha256") != digest(PRIMARY_PROTOCOL)):
            raise ValueError("Primary selection lock does not match the frozen v4 split")
        if lock.get('execution_plan_sha256') != digest(EDA/'data/reference/primary_execution_plan_v4.json'):
            raise ValueError('Execution schedule changed after primary selection')

    return dict(
        protocol=protocol,
        protocol_sha256=digest(PROTOCOL),
        split_manifest=split_manifest,
        split_manifest_sha256=digest(split_manifest_path),
        selection_lock=lock,
        selection_lock_sha256=digest(lock_path) if lock is not None else None,
    )


def load_control_specs():
    """Verify and pin the three frozen control model artifacts."""
    specs = {}
    for name in CONTROLS:
        if name == "metadata_extended_tfidf":
            source = EDA / "data/processed/modeling/improvement_v3/content/metadata_extended_tfidf"
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text("utf-8"))
            for filename in ("tfidf_csr.npz", "features_csr.npz"):
                expected = manifest["files"][filename]["sha256"]
                if digest(source / filename) != expected:
                    raise ValueError(f"Content control artifact fingerprint mismatch: {filename}")
            specs[name] = dict(
                source_manifest_path=manifest_path.relative_to(EDA).as_posix(),
                source_manifest_sha256=digest(manifest_path),
                model_id=manifest["model_id"],
                parameters=manifest["parameters"],
                score_spec=manifest["score_spec"],
                artifacts={
                    filename: dict(sha256=digest(source / filename), bytes=(source / filename).stat().st_size)
                    for filename in ("tfidf_csr.npz", "features_csr.npz")
                },
            )
            continue

        folder = ROOT / "models" / name
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        matrix_path = folder / "similarities_csr.npz"
        matrix_sha = digest(matrix_path)
        if manifest.get("status") != "complete" or manifest.get("matrix_sha256") != matrix_sha:
            raise ValueError(f"Control model artifact fingerprint mismatch: {name}")
        specs[name] = dict(
            source_manifest_path=manifest_path.relative_to(EDA).as_posix(),
            source_manifest_sha256=digest(manifest_path),
            model_settings={k: manifest.get(k) for k in (
                "variant", "rho", "weight", "fixed_neighbors", "training_sha256", "code_sha256"
            )},
            matrix_path=matrix_path.relative_to(EDA).as_posix(),
            matrix_sha256=matrix_sha,
            matrix_bytes=matrix_path.stat().st_size,
        )
    return specs


def load_control_models(specs):
    models = {}
    for name in CONTROLS:
        if name == "metadata_extended_tfidf":
            models[name] = ContentScorer()
        else:
            path = ROOT / "models" / name / "similarities_csr.npz"
            models[name] = MatrixScorer(
                sparse.load_npz(path),
                mode="knn",
                spec={"pinned_control": specs[name]},
            )
    return models


def target_pool_rows(cohort):
    """Rebuild ascending Linux positive-target pools for every selection user."""
    apps, linux = catalog()
    positive = cohort["positive"].tocsr()
    original_target_cols = np.searchsorted(apps, cohort["target_ids"])
    if np.any(original_target_cols >= len(apps)) or np.any(apps[original_target_cols] != cohort["target_ids"]):
        raise ValueError("Original target is outside the frozen full catalog")
    pools = []
    for row in range(len(cohort["user_ids"])):
        start, stop = positive.indptr[row:row + 2]
        visible_cols = positive.indices[start:stop]
        visible_linux = visible_cols[linux[visible_cols]]
        pool_cols = np.unique(np.append(visible_linux, original_target_cols[row]))
        if not len(pool_cols) or np.any(~linux[pool_cols]):
            raise ValueError("Reconstructed positive target pool is empty or non-Linux")
        pool_apps = apps[pool_cols]
        if np.any(np.diff(pool_apps) <= 0):
            raise ValueError("Target pool is not ascending and unique")
        pools.append(pool_apps)
    return pools


def sample_target_ids(cohort, seed):
    """Use the v4 split sampler and its original selection role namespace."""
    pools = target_pool_rows(cohort)
    target_ids = np.fromiter(
        (uniform_target(int(user), pool.tolist(), int(seed), "selection")
         for user, pool in zip(cohort["user_ids"], pools)),
        dtype=np.int64,
        count=len(pools),
    )
    return target_ids, np.fromiter((len(pool) for pool in pools), dtype=np.int32, count=len(pools))


def build_seed_cohort(cohort, seed):
    """Swap the hidden positive target while retaining each account's inputs."""
    apps, linux = catalog()
    candidates = apps[linux]
    target_ids, pool_size = sample_target_ids(cohort, seed)
    original_ids = cohort["target_ids"]
    if seed == SEEDS[0] and not np.array_equal(target_ids, original_ids):
        raise ValueError("Seed 20261108 does not reproduce the original selection targets")
    original_cols = np.searchsorted(apps, original_ids)
    target_cols = np.searchsorted(apps, target_ids)
    if np.any(apps[target_cols] != target_ids) or np.any(~linux[target_cols]):
        raise ValueError("Resampled target is outside the Linux catalog")
    target_candidates = np.searchsorted(candidates, target_ids)
    if np.any(target_candidates >= len(candidates)) or np.any(candidates[target_candidates] != target_ids):
        raise ValueError("Resampled target is outside the Linux candidate catalog")

    rows = np.arange(len(target_ids), dtype=np.int64)
    original_target_pairs = pair_matrix(rows, original_cols, cohort["positive"].shape, "original target")
    full_positive_history = (cohort["positive"] + original_target_pairs).tocsr()
    target_pairs = pair_matrix(rows, target_cols, cohort["positive"].shape, "resampled target")
    positive = (full_positive_history - target_pairs).tocsr()
    positive.eliminate_zeros()

    original_target_candidates = np.searchsorted(candidates, original_ids)
    original_target_candidate_pairs = pair_matrix(
        rows, original_target_candidates, cohort["exclusions"].shape, "original target exclusion"
    )
    full_exclusions = (cohort["exclusions"] + original_target_candidate_pairs).tocsr()
    target_exclusion_pairs = pair_matrix(rows, target_candidates, cohort["exclusions"].shape, "target exclusion")
    exclusions = (full_exclusions - target_exclusion_pairs).tocsr()
    exclusions.eliminate_zeros()
    if np.any(positive.data < 0) or np.any(exclusions.data < 0):
        raise ValueError("Target swap produced negative query/exclusion counts")
    if not np.array_equal(np.diff(positive.indptr), cohort["n_positive"]):
        raise ValueError("Target swap changed positive input counts")
    if not np.array_equal(np.diff(exclusions.indptr), np.diff(cohort["exclusions"].indptr)):
        raise ValueError("Target swap changed the visible Linux exclusion count")
    if np.any(positive[rows, target_cols]) or np.any(exclusions[rows, target_candidates]):
        raise ValueError("Resampled target remains visible or excluded")

    result = dict(cohort)
    result.update(positive=positive, exclusions=exclusions, target_ids=target_ids,
                  target_candidates=target_candidates, target_pool_size=pool_size)
    return result


def hidden_negative_index(user_key, count, seed=HIDDEN_NEGATIVE_SEED):
    """Unbiased counter-mode SHA256 draw in the hidden-negative namespace.

    Payload: namespace:seed:selection:user_key:counter. The role is explicit so
    the draw remains scoped to the v4 selection roster, as in the target sampler.
    """
    import hashlib

    count = int(count)
    if count < 1:
        raise ValueError("Cannot sample from an empty negative pool")
    limit = (1 << 256) - (1 << 256) % count
    counter = 0
    while True:
        payload = f"{HIDDEN_NEGATIVE_NAMESPACE}:{int(seed)}:selection:{int(user_key)}:{counter}"
        value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        if value < limit:
            return value % count
        counter += 1


def build_hidden_negative_cohort(cohort):
    """Select primary users with a visible Linux negative and hide one negative."""
    apps, linux = catalog()
    candidates = apps[linux]
    primary_rows = np.flatnonzero(cohort["n_total"] >= 5)
    negative_linux = cohort["negative"][primary_rows][:, linux].tocsr()
    eligible_local = np.flatnonzero(np.diff(negative_linux.indptr) > 0)
    selected_rows = primary_rows[eligible_local]
    sampled_ids = np.empty(len(selected_rows), dtype=np.int64)
    for local, row in enumerate(selected_rows):
        start, stop = negative_linux.indptr[eligible_local[local]:eligible_local[local] + 2]
        candidate_cols = negative_linux.indices[start:stop]
        if np.any(np.diff(candidates[candidate_cols]) <= 0):
            raise ValueError("Visible negative Linux games are not unique and ascending")
        index = hidden_negative_index(cohort["user_ids"][row], len(candidate_cols))
        sampled_ids[local] = candidates[candidate_cols[index]]
    selected_candidate_cols = np.searchsorted(candidates, sampled_ids)
    if np.any(candidates[selected_candidate_cols] != sampled_ids):
        raise ValueError("Sampled hidden negative is outside the Linux candidate catalog")

    result = {}
    row_fields = (
        "user_ids", "positive", "negative", "exclusions", "target_ids", "target_candidates",
        "n_total", "n_positive",
    )
    for key, value in cohort.items():
        if key in row_fields:
            result[key] = value[selected_rows]
        else:
            result[key] = value
    remove_pairs = pair_matrix(
        np.arange(len(selected_rows), dtype=np.int64), selected_candidate_cols,
        result["exclusions"].shape, "sampled hidden negative",
    )
    if np.any(np.asarray(result["exclusions"][np.arange(len(selected_rows)), selected_candidate_cols]).ravel() != 1):
        raise ValueError("Sampled negative is not in the original visible exclusion mask")
    sampled_full_cols = np.searchsorted(apps, sampled_ids)
    if np.any(apps[sampled_full_cols] != sampled_ids):
        raise ValueError("Sampled negative is outside the full input catalog")
    remove_negative_pairs = pair_matrix(
        np.arange(len(selected_rows), dtype=np.int64), sampled_full_cols,
        result["negative"].shape, "sampled hidden negative query row",
    )
    if np.any(np.asarray(result["negative"][np.arange(len(selected_rows)), sampled_full_cols]).ravel() != 1):
        raise ValueError("Sampled negative is not present in the original visible query")
    result["negative"] = (result["negative"] - remove_negative_pairs).tocsr()
    result["negative"].eliminate_zeros()
    result["exclusions"] = (result["exclusions"] - remove_pairs).tocsr()
    result["exclusions"].eliminate_zeros()
    result["hidden_negative_ids"] = sampled_ids
    result["n_total_after_negative_hide"] = result["n_total"] - 1
    if len(selected_rows) and np.any(result["n_total_after_negative_hide"] < 4):
        raise ValueError("Hidden-negative selection retained a user below the protocol's expected minimum")
    return result


def group_masks(cohort):
    n_total = np.asarray(cohort["n_total"])
    return {
        "primary_5_plus": n_total >= 5,
        "short_history_1_to_4": n_total <= 4,
        "history_1_to_2": n_total <= 2,
        "history_3_to_4": (n_total >= 3) & (n_total <= 4),
    }


def summarize_rank_group(ranks, lengths, target_ids, mask):
    recall, ndcg = rank_values(np.asarray(ranks)[mask])
    selected_targets = np.asarray(target_ids)[mask]
    n = int(mask.sum())
    return dict(
        users=n,
        target_games=int(len(np.unique(selected_targets))),
        recall_at_10=float(recall.mean()) if n else None,
        ndcg_at_10=float(ndcg.mean()) if n else None,
        empty_users=int((np.asarray(lengths)[mask] == 0).sum()),
        average_recommendation_count=float(np.asarray(lengths)[mask].mean()) if n else None,
    )


def _bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap requires a nonempty one-dimensional user population")
    unique, counts = np.unique(values, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(len(values), counts / len(values), size=reps)
    samples = draws @ unique / len(values)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def paired_user_bootstrap(left, right, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Paired user bootstrap for arrays already averaged across target seeds."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError("Paired bootstrap requires aligned, nonempty user arrays")
    diff = right - left
    unique, counts = np.unique(diff, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(len(diff), counts / len(diff), size=reps)
    samples = draws @ unique / len(diff)
    return dict(
        delta=float(diff.mean()),
        ci95=[float(x) for x in np.quantile(samples, [0.025, 0.975])],
    )


def averaged_seed_summary(seed_outcomes, cohort):
    """Average within each user across seeds before user-level uncertainty."""
    masks = group_masks(cohort)
    transformed = {}
    lengths = {}
    for model_name, by_seed in seed_outcomes.items():
        transformed[model_name] = {"recall": [], "ndcg": []}
        lengths[model_name] = []
        for seed in SEEDS:
            ranks, counts = by_seed[seed]
            recall, ndcg = rank_values(ranks)
            transformed[model_name]["recall"].append(recall)
            transformed[model_name]["ndcg"].append(ndcg)
            lengths[model_name].append(counts)
        transformed[model_name] = {
            metric: np.mean(np.stack(values, axis=1), axis=1)
            for metric, values in transformed[model_name].items()
        }
        lengths[model_name] = np.mean(np.stack(lengths[model_name], axis=1), axis=1)

    model_summary = {}
    for model_name in seed_outcomes:
        model_summary[model_name] = {}
        for group, mask in masks.items():
            n = int(mask.sum())
            model_summary[model_name][group] = dict(
                users=n,
                mean_user_recall_at_10=float(transformed[model_name]["recall"][mask].mean()) if n else None,
                mean_user_recall_ci95=_bootstrap_mean_ci(transformed[model_name]["recall"][mask]) if n else None,
                mean_user_ndcg_at_10=float(transformed[model_name]["ndcg"][mask].mean()) if n else None,
                mean_user_ndcg_ci95=_bootstrap_mean_ci(transformed[model_name]["ndcg"][mask], seed=BOOTSTRAP_SEED + 1) if n else None,
                mean_recommendation_count=float(lengths[model_name][mask].mean()) if n else None,
            )

    pairwise = {}
    for left_name, right_name in itertools.combinations(seed_outcomes, 2):
        key = f"{right_name}_minus_{left_name}"
        pairwise[key] = {}
        for group, mask in masks.items():
            pairwise[key][group] = dict(
                users=int(mask.sum()),
                recall_at_10=paired_user_bootstrap(
                    transformed[left_name]["recall"][mask],
                    transformed[right_name]["recall"][mask],
                ),
                ndcg_at_10=paired_user_bootstrap(
                    transformed[left_name]["ndcg"][mask],
                    transformed[right_name]["ndcg"][mask],
                    seed=BOOTSTRAP_SEED + 1,
                ),
            )
    return model_summary, pairwise, transformed


def _per_user_table(cohort, target_ids, ranks, lengths, extra=None):
    columns = dict(
        user_key=pa.array(cohort["user_ids"], type=pa.uint64()),
        target_app_id=pa.array(target_ids, type=pa.int64()),
        target_rank=pa.array(ranks, type=pa.int8()),
        recommendation_count=pa.array(lengths, type=pa.int8()),
        n_total_inputs=pa.array(cohort["n_total"], type=pa.int32()),
        n_positive_inputs=pa.array(cohort["n_positive"], type=pa.int32()),
    )
    for key, values in (extra or {}).items():
        columns[key] = pa.array(values)
    return pa.table(columns)


def _checked_existing(folder, provenance):
    receipt = folder / "manifest.json"
    if not receipt.exists():
        if any(folder.iterdir()):
            raise ValueError(f"Incomplete supplemental output exists without a manifest: {folder}")
        return None
    manifest = json.loads(receipt.read_text("utf-8"))
    if manifest.get("provenance") != provenance:
        raise ValueError(f"Supplemental output provenance changed: {folder}")
    for filename, expected in manifest.get("files", {}).items():
        if digest(folder / filename) != expected["sha256"]:
            raise ValueError(f"Saved supplemental predictions changed: {folder / filename}")
    return manifest


def evaluate_predictions(name, model, cohort, target_ids, folder, provenance, *, hidden_negative=False):
    folder.mkdir(parents=True, exist_ok=True)
    existing = _checked_existing(folder, provenance)
    if existing is not None:
        return existing
    apps, linux = catalog()
    candidate_ids = apps[linux]
    n = len(cohort["user_ids"])
    ranks = np.zeros(n, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int8)
    recommendation_schema = pa.schema([
        ("user_key", pa.uint64()), ("rank", pa.int8()), ("app_id", pa.int64()), ("score", pa.float64())
    ])
    recommendation_path = folder / "recommendations.parquet"
    per_user_path = folder / "per_user.parquet"
    started = time.perf_counter()
    with pq.ParquetWriter(recommendation_path, recommendation_schema, compression="zstd") as writer:
        for begin in range(0, n, BATCH_SIZE):
            end = min(n, begin + BATCH_SIZE)
            scores = masked_scores(model, cohort, begin, end)
            out_users, out_ranks, out_apps, out_scores = [], [], [], []
            for local, row in enumerate(range(begin, end)):
                chosen = rank_positive(scores[local], candidate_ids, 10)
                lengths[row] = len(chosen)
                target_col = int(np.searchsorted(candidate_ids, int(target_ids[row])))
                if target_col >= len(candidate_ids) or candidate_ids[target_col] != target_ids[row]:
                    raise ValueError("Outcome target is outside the candidate catalog")
                hit = np.flatnonzero(chosen == target_col)
                if len(hit):
                    ranks[row] = int(hit[0]) + 1
                out_users.extend([int(cohort["user_ids"][row])] * len(chosen))
                out_ranks.extend(range(1, len(chosen) + 1))
                out_apps.extend(int(candidate_ids[col]) for col in chosen)
                out_scores.extend(float(scores[local, col]) for col in chosen)
            writer.write_table(pa.Table.from_pydict({
                "user_key": pa.array(out_users, type=pa.uint64()),
                "rank": pa.array(out_ranks, type=pa.int8()),
                "app_id": pa.array(out_apps, type=pa.int64()),
                "score": pa.array(out_scores, type=pa.float64()),
            }, schema=recommendation_schema))
            if begin == 0 or end == n:
                log(f"supplemental {name}: {end:,}/{n:,}")

    extra = {}
    if hidden_negative:
        negative_ranks = np.zeros(n, dtype=np.int8)
        # Recommendations are ordered by user and rank. Read per-user top-10
        # once below to derive the hidden-negative diagnostic rank.
        rec = pq.read_table(recommendation_path).to_pydict()
        cursor = 0
        for row in range(n):
            while cursor < len(rec["user_key"]) and rec["user_key"][cursor] == int(cohort["user_ids"][row]):
                if rec["app_id"][cursor] == int(cohort["hidden_negative_ids"][row]):
                    negative_ranks[row] = rec["rank"][cursor]
                cursor += 1
        extra = {
            "hidden_negative_app_id": np.asarray(cohort["hidden_negative_ids"], dtype=np.int64),
            "hidden_negative_rank": negative_ranks,
            "n_total_inputs_after_negative_hide": np.asarray(cohort["n_total_after_negative_hide"], dtype=np.int32),
        }
    elif "target_pool_size" in cohort:
        extra = {"positive_linux_target_pool_size": np.asarray(cohort["target_pool_size"], dtype=np.int32)}
    pq.write_table(_per_user_table(cohort, target_ids, ranks, lengths, extra), per_user_path, compression="zstd")
    if hidden_negative:
        negative_rank = np.asarray(extra["hidden_negative_rank"])
        hit = negative_rank > 0
        negative_metrics = dict(
            users=int(len(negative_rank)),
            hidden_negative_top10_users=int(hit.sum()),
            hidden_negative_top10_rate=float(hit.mean()) if len(hit) else None,
            hidden_negative_top10_rate_ci95=_bootstrap_mean_ci(hit.astype(float)) if len(hit) else None,
            denominator_scope="primary 5+ selection accounts with at least one visible negative Linux game",
            users_with_4_remaining_inputs=int(np.count_nonzero(cohort["n_total_after_negative_hide"] == 4)),
        )
        metrics = {"hidden_negative": negative_metrics}
    else:
        metrics = {
            group: summarize_rank_group(ranks, lengths, target_ids, mask)
            for group, mask in group_masks(cohort).items()
        }
    manifest = dict(
        status="complete",
        variant=name,
        analysis="hidden_negative" if hidden_negative else "target_seed_variation",
        user_count=n,
        provenance=provenance,
        elapsed_seconds=time.perf_counter() - started,
        metrics=metrics,
        files={
            "per_user.parquet": dict(sha256=digest(per_user_path), rows=pq.ParquetFile(per_user_path).metadata.num_rows),
            "recommendations.parquet": dict(sha256=digest(recommendation_path), rows=pq.ParquetFile(recommendation_path).metadata.num_rows),
        },
    )
    write_json(folder / "manifest.json", manifest)
    return manifest


def _load_per_user(path):
    data = pq.read_table(path).to_pydict()
    return data


def verify_original_seed_predictions(name, folder):
    """Seed 20261108 must reproduce the already saved primary selection run."""
    reference = ROOT / "evaluations/selection" / name
    reference_manifest = json.loads((reference / "manifest.json").read_text("utf-8"))
    if reference_manifest.get("status") != "complete" or reference_manifest.get("variant") != name:
        raise ValueError(f"Original selection evaluation is incomplete: {name}")
    fields = ("user_key", "target_app_id", "target_rank", "recommendation_count",
              "n_total_inputs", "n_positive_inputs")
    current = pq.read_table(folder / "per_user.parquet", columns=list(fields)).to_pydict()
    original = pq.read_table(reference / "per_user.parquet", columns=list(fields)).to_pydict()
    if any(not np.array_equal(current[key], original[key]) for key in fields):
        raise ValueError(f"Seed 20261108 per-user predictions differ from primary evaluation: {name}")
    rec_fields = ("user_key", "rank", "app_id")
    current_rec = pq.read_table(folder / "recommendations.parquet", columns=list(rec_fields)).to_pydict()
    original_rec = pq.read_table(reference / "recommendations.parquet", columns=list(rec_fields)).to_pydict()
    if any(not np.array_equal(current_rec[key], original_rec[key]) for key in rec_fields):
        raise ValueError(f"Seed 20261108 recommendation lists differ from primary evaluation: {name}")


def write_supplemental_report(target_summary, negative_summary, frozen):
    report_json = dict(
        status="complete",
        supplemental_protocol_sha256=frozen["protocol_sha256"],
        split_manifest_sha256=frozen["split_manifest_sha256"],
        selection_lock_sha256=frozen["selection_lock_sha256"],
        primary_selection_finalist=frozen["selection_lock"].get("finalist"),
        confirmation_opened=False,
        target_seed_variation=target_summary,
        hidden_negative=negative_summary,
        limitations=[
            "The selection accounts were included in historical v1-v3 training, although reserved accounts were excluded from fresh v4 refitting. This is not a pristine never-seen final test.",
            "The same users appear in all three target-seed runs. Confidence intervals resample users only after averaging each user's outcomes across seeds; they do not include model-fitting or selection uncertainty.",
            "The hidden-negative rate covers one uniformly sampled visible negative Linux game for primary users with at least one such negative. It is not a false-positive rate over all unobserved games.",
            "The diagnostics evaluate warm-catalog retrieval. A new-game performance claim still requires the separate item-removal retraining protocol.",
            "These diagnostics are reporting-only and do not select settings. Confirmation remained unopened.",
        ],
        artifacts=dict(
            target_seed_variation_directory=str(TARGET_OUTPUT.relative_to(EDA).as_posix()),
            hidden_negative_directory=str(NEGATIVE_OUTPUT.relative_to(EDA).as_posix()),
            supplemental_manifest=str((OUTPUT / "manifest.json").relative_to(EDA).as_posix()),
        ),
    )
    report_json_path = EDA / "reports/supplemental_evaluation_v4.json"
    report_md_path = EDA / "reports/supplemental_evaluation_v4.md"
    write_json(report_json_path, report_json)

    def pct(value):
        return "—" if value is None else f"{100 * float(value):.3f}%"

    def ci_pct(values):
        return "—" if values is None else f"[{100 * values[0]:.3f}%, {100 * values[1]:.3f}%]"

    lines = [
        "# v4 정답 시드 반복·숨긴 부정 게임 보조 평가",
        "",
        "**상태:** 주 모델을 고정한 뒤, 확인용 평가 전에 수행한 보고용 진단이다. 이 결과로 모델이나 설정을 다시 고르지 않았다.",
        "",
        f"**이 보조 결과를 보기 전에 고정한 후보:** `{report_json['primary_selection_finalist']}`.",
        "",
        "## 정답을 다시 뽑았을 때의 변동",
        "",
        "사용자에게 보이는 긍정 Linux 게임과 처음 숨긴 정답을 합쳐 정답 후보를 복원했다. 원래의 균등 추출 규칙과 selection 역할을 유지하고 시드 20261108·20261109·20261110을 비교했다. 매번 같은 20,000명이며, 전체 입력 5개 이상인 주평가 10,000명과 1–4개인 10,000명이다. 첫 시드는 원래 정답·추천 목록·적중 순위를 정확히 재현해야 한다.",
        "",
        "먼저 사용자마다 세 시드의 Recall/NDCG를 평균한 뒤 사용자를 재표집해 신뢰구간을 계산했다. 같은 사용자의 세 행을 독립 표본 세 명처럼 세지 않았다.",
        "",
        "| 모델 | 집단 | 사용자 평균 Recall@10 (95% 구간) | 사용자 평균 NDCG@10 (95% 구간) | 사용자 |",
        "|---|---|---:|---:|---:|",
    ]
    averages = target_summary["user_outcomes_averaged_across_seeds"]
    for name in target_summary['controls']:
        for group, label in (("primary_5_plus", "주평가, 입력 5개 이상"), ("short_history_1_to_4", "짧은 이력, 입력 1–4개")):
            m = averages[name][group]
            lines.append(
                f"| `{name}` | {label} | {pct(m['mean_user_recall_at_10'])} ({ci_pct(m['mean_user_recall_ci95'])}) | "
                f"{m['mean_user_ndcg_at_10']:.6f} ({m['mean_user_ndcg_ci95'][0]:.6f}, {m['mean_user_ndcg_ci95'][1]:.6f}) | {m['users']:,} |"
            )
    lines += [
        "",
        "### 같은 사용자에서 모델 간 차이",
        "",
        "차이는 비교 이름에서 앞의 모델을 기준으로 뺀 값이다. 같은 사용자의 시드별 결과를 평균한 뒤 사용자 단위로 짝지어 재표집했다.",
        "",
        "| 비교 | 집단 | Recall@10 차이 (95% 구간) | NDCG@10 차이 (95% 구간) |",
        "|---|---|---:|---:|",
    ]
    for comparison, by_group in target_summary["paired_user_comparisons_on_averaged_seed_outcomes"].items():
        for group, label in (("primary_5_plus", "주평가"), ("short_history_1_to_4", "짧은 이력")):
            result = by_group[group]
            lines.append(
                f"| `{comparison}` | {label} | {result['recall_at_10']['delta']:.6f} "
                f"[{result['recall_at_10']['ci95'][0]:.6f}, {result['recall_at_10']['ci95'][1]:.6f}] | "
                f"{result['ndcg_at_10']['delta']:.6f} [{result['ndcg_at_10']['ci95'][0]:.6f}, "
                f"{result['ndcg_at_10']['ci95'][1]:.6f}] |"
            )
    lines += [
        "",
        "시드별 지표와 사용자별 예측은 `data/processed/modeling/experiments_v4/supplemental/target_seed_variation/`에 저장했다.",
        "",
        "## 숨긴 부정 게임의 추천 포함률",
        "",
        "주평가 사용자 중 보이는 부정 Linux 게임이 있는 사람을 대상으로, 그 게임 하나를 균등 추출해 입력과 제외 목록에서 함께 뺐다. 추가 숨김 뒤 입력이 4개가 되어도 같은 사용자를 분모에 유지했다.",
        "",
        "| 모델 | 숨긴 부정 게임이 Top 10에 든 사용자 | 포함률 (95% 사용자 재표집 구간) | 사용자 |",
        "|---|---:|---:|---:|",
    ]
    for name in negative_summary['controls']:
        m = negative_summary["outcomes"][name]
        lines.append(
            f"| `{name}` | {m['hidden_negative_top10_users']:,} | "
            f"{pct(m['hidden_negative_top10_rate'])} ({ci_pct(m['hidden_negative_top10_rate_ci95'])}) | {m['users']:,} |"
        )
    lines += [
        "",
        f"주평가 {negative_summary['denominator']['eligible_primary_accounts']:,}명 중 "
        f"{negative_summary['denominator']['selected_accounts']:,}명이 대상이다. "
        f"그중 {negative_summary['denominator']['users_with_four_inputs_after_hiding']:,}명은 추가 숨김 뒤 입력이 4개다.",
        "",
        "사용자가 실제 부정 리뷰한 게임 하나의 포함률이며 낮을수록 좋다. 아직 리뷰하지 않은 모든 게임의 거짓 양성률은 아니다. 가격·버그 등 취향 이외의 이유로 남긴 부정 리뷰도 섞여 있다.",
        "",
        "선택한 부정 게임 ID와 추천 목록은 `data/processed/modeling/experiments_v4/supplemental/hidden_negative/`에 저장했다.",
        "",
        "## 해석 한계",
        "",
        "- 예약 계정은 새 v4 학습에서 제외했지만 과거 v1–v3 학습에는 쓰였다. 프로젝트 전체에서 완전히 미사용된 최종 테스트는 아니다.",
        "- 같은 사용자가 세 시드에 반복된다. 구간은 시드 평균 뒤 사용자 변동을 반영하며 재학습·설정 선택의 불확실성은 포함하지 않는다.",
        "- 짧은 이력은 1–2개와 3–4개 집단에서 같은 수를 뽑았다. 합친 평균을 실제 사용자층의 자연스러운 평균으로 해석하지 않는다.",
        "- 학습 상호작용을 제거한 신규 게임 평가는 아니다. 이를 주장하려면 별도 게임 제거 재학습이 필요하다.",
        "- 부정 게임이 보이는 사용자에게서 게임 하나를 숨긴 결과이며 모든 미관측 게임을 다루지 않는다.",
        "- 이 보조 평가는 확인용 계정을 열지 않았다. 별도 확인 결과는 주 실험 보고서에서 확인한다.",
        "",
        f"Supplemental protocol SHA-256: `{frozen['protocol_sha256']}`  ",
        f"v4 split manifest SHA-256: `{frozen['split_manifest_sha256']}`  ",
        f"Selection lock SHA-256: `{frozen['selection_lock_sha256']}`",
    ]
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(
        json_path=report_json_path,
        json_sha256=digest(report_json_path),
        markdown_path=report_md_path,
        markdown_sha256=digest(report_md_path),
    )


def run():
    frozen = verify_frozen_inputs(require_selection_lock=True)
    control_specs = load_control_specs()
    cohort = load_cohort("selection")
    if len(cohort["user_ids"]) != 20_000:
        raise ValueError("Supplemental diagnostics require the fixed 20,000-user selection roster")
    if int(np.count_nonzero(cohort["n_total"] >= 5)) != 10_000:
        raise ValueError("Primary selection denominator changed from 10,000 users")
    provenance_base = dict(
        supplemental_protocol_sha256=frozen["protocol_sha256"],
        split_manifest_sha256=frozen["split_manifest_sha256"],
        selection_lock_sha256=frozen["selection_lock_sha256"],
        selection_input_sha256=cohort["hashes"],
        supplemental_evaluator_sha256=digest(__file__),
        primary_evaluator_common_sha256=digest(EDA / "scripts/experiment_v4_common.py"),
        control_specs=control_specs,
    )
    models = load_control_models(control_specs)
    names = list(CONTROLS)
    finalist = frozen['selection_lock'].get('finalist')
    if finalist:
        from evaluate_confirmation_v4 import restore_selected
        models[finalist] = restore_selected(finalist, frozen['selection_lock'])
        control_specs[finalist] = dict(role='single_frozen_finalist', score_spec=models[finalist].spec)
        names.append(finalist)

    by_model_seed = {name: {} for name in names}
    for seed in SEEDS:
        seed_cohort = build_seed_cohort(cohort, seed)
        for name in names:
            model_provenance = dict(provenance_base, seed=int(seed), model=name)
            folder = TARGET_OUTPUT / name / str(seed)
            result = evaluate_predictions(
                name, models[name], seed_cohort, seed_cohort["target_ids"], folder, model_provenance
            )
            data = _load_per_user(folder / "per_user.parquet")
            ranks = np.asarray(data["target_rank"], dtype=np.int8)
            lengths = np.asarray(data["recommendation_count"], dtype=np.int8)
            if not np.array_equal(data["user_key"], cohort["user_ids"]):
                raise ValueError("Saved per-user predictions do not match the fixed roster")
            if seed == SEEDS[0]:
                verify_original_seed_predictions(name, folder)
            by_model_seed[name][seed] = (ranks, lengths)

    model_average, pairwise, transformed = averaged_seed_summary(by_model_seed, cohort)
    target_summary = dict(
        status="complete",
        protocol_sha256=frozen["protocol_sha256"],
        split_manifest_sha256=frozen["split_manifest_sha256"],
        selection_lock_sha256=frozen["selection_lock_sha256"],
        controls=control_specs,
        target_seeds=list(SEEDS),
        denominator_counts={group: int(mask.sum()) for group, mask in group_masks(cohort).items()},
        per_seed={
            model: {
                str(seed): json.loads((TARGET_OUTPUT / model / str(seed) / "manifest.json").read_text("utf-8"))["metrics"]
                for seed in SEEDS
            }
            for model in names
        },
        user_outcomes_averaged_across_seeds=model_average,
        paired_user_comparisons_on_averaged_seed_outcomes=pairwise,
        bootstrap=dict(unit="user after averaging that user's outcomes across seeds", reps=BOOTSTRAP_REPS,
                       seed=BOOTSTRAP_SEED, interval="percentile 95%"),
    )
    averaged_paths = {}
    for model in names:
        tables = dict(
            user_key=pa.array(cohort["user_ids"], type=pa.uint64()),
            n_total_inputs=pa.array(cohort["n_total"], type=pa.int32()),
            mean_recall_at_10=pa.array(transformed[model]["recall"], type=pa.float64()),
            mean_ndcg_at_10=pa.array(transformed[model]["ndcg"], type=pa.float64()),
            mean_recommendation_count=pa.array(np.mean(np.stack([by_model_seed[model][s][1] for s in SEEDS], axis=1), axis=1), type=pa.float64()),
        )
        for seed in SEEDS:
            data = _load_per_user(TARGET_OUTPUT / model / str(seed) / "per_user.parquet")
            tables[f"target_app_id_seed_{seed}"] = pa.array(data["target_app_id"], type=pa.int64())
            tables[f"target_rank_seed_{seed}"] = pa.array(data["target_rank"], type=pa.int8())
        path = TARGET_OUTPUT / model / "per_user_averaged.parquet"
        pq.write_table(pa.table(tables), path, compression="zstd")
        averaged_paths[model] = dict(sha256=digest(path), rows=pq.ParquetFile(path).metadata.num_rows)
    target_summary["averaged_per_user_predictions"] = {
        model: dict(
            path=f"{model}/per_user_averaged.parquet",
            sha256=details["sha256"],
            rows=details["rows"],
        )
        for model, details in averaged_paths.items()
    }
    write_json(TARGET_OUTPUT / "evaluation_summary.json", target_summary)

    negative_cohort = build_hidden_negative_cohort(cohort)
    negative_model_outcomes = {}
    for name in names:
        model_provenance = dict(
            provenance_base,
            hidden_negative_namespace=HIDDEN_NEGATIVE_NAMESPACE,
            hidden_negative_seed=HIDDEN_NEGATIVE_SEED,
            hidden_negative_payload="namespace:seed:selection:user_key:counter",
            model=name,
        )
        folder = NEGATIVE_OUTPUT / name
        result = evaluate_predictions(
            name, models[name], negative_cohort, negative_cohort["target_ids"], folder,
            model_provenance, hidden_negative=True,
        )
        data = _load_per_user(folder / "per_user.parquet")
        negative_model_outcomes[name] = np.asarray(data["hidden_negative_rank"], dtype=np.float64) > 0

    negative_comparisons = {}
    for left_name, right_name in itertools.combinations(names, 2):
        negative_comparisons[f"{right_name}_minus_{left_name}"] = paired_user_bootstrap(
            negative_model_outcomes[left_name].astype(float),
            negative_model_outcomes[right_name].astype(float),
        )
    negative_summary = dict(
        status="complete",
        protocol_sha256=frozen["protocol_sha256"],
        split_manifest_sha256=frozen["split_manifest_sha256"],
        selection_lock_sha256=frozen["selection_lock_sha256"],
        controls=control_specs,
        sampling=dict(namespace=HIDDEN_NEGATIVE_NAMESPACE, seed=HIDDEN_NEGATIVE_SEED,
                      payload="namespace:seed:selection:user_key:counter", method="SHA256 counter-mode rejection"),
        denominator=dict(
            eligible_primary_accounts=int(np.count_nonzero(cohort["n_total"] >= 5)),
            selected_accounts=int(len(negative_cohort["user_ids"])),
            all_selected_retained=True,
            users_with_four_inputs_after_hiding=int(np.count_nonzero(negative_cohort["n_total_after_negative_hide"] == 4)),
        ),
        outcomes={
            name: json.loads((NEGATIVE_OUTPUT / name / "manifest.json").read_text("utf-8"))["metrics"]["hidden_negative"]
            for name in names
        },
        paired_user_comparisons=negative_comparisons,
        bootstrap=dict(unit="selected user", reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED,
                       interval="percentile 95%"),
        interpretation="Descriptive fraction of selected users whose one hidden visible negative is recommended in top 10; not a false-positive rate over all unobserved games.",
    )
    write_json(NEGATIVE_OUTPUT / "evaluation_summary.json", negative_summary)

    report_artifacts = write_supplemental_report(target_summary, negative_summary, frozen)

    overall = dict(
        status="complete",
        protocol_sha256=frozen["protocol_sha256"],
        split_manifest_sha256=frozen["split_manifest_sha256"],
        selection_lock_sha256=frozen["selection_lock_sha256"],
        confirmation_opened=False,
        target_seed_variation_summary_sha256=digest(TARGET_OUTPUT / "evaluation_summary.json"),
        hidden_negative_summary_sha256=digest(NEGATIVE_OUTPUT / "evaluation_summary.json"),
        evaluator_sha256=digest(__file__),
        supplemental_report_json_sha256=report_artifacts["json_sha256"],
        supplemental_report_markdown_sha256=report_artifacts["markdown_sha256"],
        controls=control_specs,
    )
    write_json(OUTPUT / "manifest.json", overall)
    return target_summary, negative_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run after the primary selection lock is frozen")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run to execute the frozen reporting-only diagnostics")
    with threadpool_limits(limits=THREAD_LIMIT):
        run()
