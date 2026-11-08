"""Audit a reproducible A-only user split without adopting it or exporting IDs.

The proposed 80/10/10 split assigns whole A users first. Eligibility is counted
after dedup/conflict removal; no hidden target is selected or model fitted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_review_candidates import fingerprint


EDA = Path(__file__).resolve().parents[1]
NAMESPACE = "A-v28-user-disjoint-preview-v1"


def assign_partitions(user_ids):
    """Stable SplitMix64 permutation salted with SHA256(namespace), modulo 100.

    The overflow is intentional uint64 arithmetic. No Python randomized hash
    or row order is involved. This is partitioning, not identity anonymization.
    """
    source = np.asarray(user_ids)
    if not np.issubdtype(source.dtype, np.integer) or np.any(source < 0):
        raise ValueError("A user IDs must be nonnegative exact integers")
    salt = int.from_bytes(hashlib.sha256(NAMESPACE.encode("utf-8")).digest()[:8], "little")
    with np.errstate(over="ignore"):
        value = source.astype(np.uint64) ^ np.uint64(salt)
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value = value ^ (value >> np.uint64(31))
    bucket = value % np.uint64(100)
    return np.where(bucket < 80, 0, np.where(bucket < 90, 1, 2)).astype(np.int8)


def main():
    rec_path = EDA / "data/processed/A_recommendations.parquet"
    games_path = EDA / "data/processed/A_games.parquet"
    frame = pd.read_parquet(rec_path, columns=["user_id", "app_id", "is_recommended"])
    raw_rows = len(frame)
    if frame.isna().any().any() or not pd.api.types.is_bool_dtype(frame.is_recommended):
        raise ValueError("Unexpected nulls or labels")
    distinct = frame.drop_duplicates(["user_id", "app_id", "is_recommended"])
    duplicates = raw_rows - len(distinct)
    del frame
    conflicts = distinct.duplicated(["user_id", "app_id"], keep=False)
    conflict_rows = int(conflicts.sum())
    conflict_pairs = int(distinct.loc[conflicts, ["user_id", "app_id"]].drop_duplicates().shape[0])
    clean = distinct.loc[~conflicts].copy()
    del distinct
    assert not clean.duplicated(["user_id", "app_id"]).any()
    print(f"A clean pairs: {len(clean):,}; conflicts: {conflict_pairs}", flush=True)
    users = clean.user_id.to_numpy()
    max_user = int(users.max())
    history = np.bincount(users, minlength=max_user + 1)
    observed = np.flatnonzero(history)
    partition = np.full(max_user + 1, -1, dtype=np.int8)
    partition[observed] = assign_partitions(observed)
    assert np.array_equal(partition[observed[::-1]], assign_partitions(observed[::-1]))
    games = pd.read_parquet(games_path, columns=["app_id", "linux"])
    linux = set(games.loc[games.linux, "app_id"])
    is_target = clean.is_recommended.to_numpy() & clean.app_id.isin(linux).to_numpy()
    target_counts = np.bincount(users[is_target], minlength=max_user + 1)
    row_partition = partition[users]
    assert np.isin(row_partition, [0, 1, 2]).all()
    assert int(sum((partition[observed] == p).sum() for p in range(3))) == len(observed)
    # Reconstruct user sets from the assigned rows, then measure intersections.
    actual_user_sets = [np.unique(users[row_partition == p]) for p in range(3)]
    intersections = {f"{left}_{right}": int(np.intersect1d(actual_user_sets[i], actual_user_sets[j], assume_unique=True).size)
                     for i, j, left, right in [(0, 1, "train", "validation"), (0, 2, "train", "test"), (1, 2, "validation", "test")]}
    shared_users = sum(intersections.values())
    assert shared_users == 0
    assert sum(len(values) for values in actual_user_sets) == len(observed)
    results = {}
    trained_apps = set(clean.app_id.loc[row_partition == 0])
    trained_linux = trained_apps & linux
    unsupported_target = is_target & ~clean.app_id.isin(trained_apps).to_numpy()
    for number, name in enumerate(["train", "validation", "test"]):
        user_mask = (partition == number) & (history > 0)
        row_mask = row_partition == number
        counts = {}
        for minimum in [1, 3, 5, 10]:
            eligible = user_mask & (history >= minimum + 1) & (target_counts > 0)
            targets = is_target & eligible[users]
            unsupported = targets & unsupported_target
            counts[str(minimum)] = {"potential_users": int(eligible.sum()),
                "possible_linux_positive_target_pairs": int(targets.sum()),
                "possible_target_games": int(clean.app_id.loc[targets].nunique()),
                "possible_target_pairs_on_games_without_train_reviews": int(unsupported.sum())}
        results[name] = {"users": int(user_mask.sum()), "review_pairs": int(row_mask.sum()),
                         "potential_eligibility_after_hiding_one_target": counts}
    all_rows = sum(x["review_pairs"] for x in results.values())
    all_users = sum(x["users"] for x in results.values())
    assert all_rows == len(clean) and all_users == len(observed)
    assignment_hash = hashlib.sha256(observed.astype("<i8").tobytes() + partition[observed].tobytes()).hexdigest()
    report = {"checked_date": "2026-11-07", "status": "A_ONLY_GROUP_SPLIT_PREVIEW_VERIFIED",
        "adopted_as_project_protocol": False, "model_fitted": False, "hidden_targets_selected": False,
        "partitioned_datasets_written": False, "namespace": NAMESPACE,
        "algorithm": "SplitMix64(canonical nonnegative A user ID XOR little-endian first 8 bytes SHA256(namespace)) modulo 100; train <80, validation <90, test >=90",
        "inputs": [fingerprint(rec_path), fingerprint(games_path)],
        "raw_rows": raw_rows, "same_label_duplicate_rows_removed": duplicates,
        "conflicting_pairs_excluded": conflict_pairs, "conflicting_label_rows_excluded": conflict_rows,
        "clean_pairs": len(clean), "users_with_clean_reviews": len(observed),
        "linux_candidate_games_with_A_reviews": len(set(clean.app_id) & linux),
        "linux_games_with_train_reviews": len(trained_linux),
        "assignment_aggregate_sha256": assignment_hash,
        "measured_user_set_intersections": intersections,
        "assertions": {"all_clean_pairs_assigned_once": True, "all_observed_users_assigned_once": True,
            "assignment_independent_of_row_order": True,
            "cross_partition_same_A_user_ids": shared_users,
            "cross_partition_same_user_game_pairs": shared_users,
            "zero_shared_user_game_pairs_follows_from_measured_zero_shared_users": True,
            "scope": "Separation of stable A user IDs within A only; no claim of A/VintageDon independence"},
        "partitions": results,
        "required_before_training": ["Hide positive Linux target(s) only inside held-out validation/test users and assert query-target pair disjointness.",
            "Build interaction counts, k-core, similarity and popularity from training users only; do not k-core the entire source before splitting.",
            "Report training-item support and zero-vector/zero-neighbor behavior; preserve a common evaluation cohort.",
            "Choose model settings only on validation; use test labels only for final evaluation.",
            "Fix an A+C metadata policy separately; this is user-generalization, not a historical time-split benchmark.",
            "Do not add VintageDon interactions to this training union while claiming certified cross-source user independence."]}
    output = EDA / "reports/A_strict_user_split_preview.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "clean_pairs": len(clean), "partitions": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
