"""Whole-column Steam-ID audits and measured user-exclusion examples.

Reads local B and VintageDon, plus optional complete frankjc2022 projections.
Exports aggregate counts only. No source replacement, split files or model fit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_review_candidates import fingerprint
from audit_vintage_independence import load_vintage


EDA = Path(__file__).resolve().parents[1]
STEAM_USER_BASE = 76561197960265728


def require_steam_strings(series):
    """Lossless normalization requires string input; never repair floats by str()."""
    values = series.dropna()
    if not values.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Steam IDs must originate as exact strings, not floats/integers")
    if not values.str.fullmatch(r"\d{17}").all():
        raise ValueError("Noncanonical Steam-ID token")
    integer = pd.to_numeric(values, errors="raise")
    if not integer.between(STEAM_USER_BASE, STEAM_USER_BASE + 2**32 - 1).all():
        raise ValueError("Not a public individual SteamID64 account range")


def clean_reviews(frame):
    require_steam_strings(frame.user_id)
    if frame[["user_id", "app_id", "positive"]].isna().any().any():
        raise ValueError("Unresolved identity/item/label must be quarantined before audit")
    if not pd.api.types.is_bool_dtype(frame.positive):
        raise ValueError("Explicit booleans required")
    labels = frame[["user_id", "app_id", "positive"]].drop_duplicates()
    conflict = labels.duplicated(["user_id", "app_id"], keep=False)
    clean = labels.loc[~conflict].copy()
    facts = {"raw_rows": len(frame), "same_label_duplicate_rows_removed": len(frame) - len(labels),
             "conflicting_pairs_excluded": int(labels.loc[conflict, ["user_id", "app_id"]].drop_duplicates().shape[0]),
             "clean_pairs": len(clean), "users": int(clean.user_id.nunique()), "apps": int(clean.app_id.nunique())}
    if "review_id" in frame:
        original = frame[["review_id", "user_id", "app_id"]].drop_duplicates()
        collision = original.review_id.duplicated(keep=False)
        facts["original_review_id_rows_missing"] = int(frame.review_id.isna().sum())
        facts["unique_original_review_ids"] = int(frame.review_id.nunique())
        facts["original_review_ids_linked_to_different_user_or_app"] = int(original.loc[collision, "review_id"].nunique())
        if facts["original_review_ids_linked_to_different_user_or_app"]:
            raise ValueError("Original review ID maps to incompatible owner/item")
    return clean, facts


def map_b_reviews(profiles, reviews):
    mapping = profiles[["user_id", "steam_id"]].drop_duplicates()
    require_steam_strings(mapping.steam_id)
    if mapping.groupby("user_id").steam_id.nunique().gt(1).any():
        raise ValueError("Ambiguous B alias-to-Steam mapping")
    mapped = reviews.merge(mapping, on="user_id", how="left", validate="many_to_one")
    # Numeric-looking aliases can disagree with the explicit profile Steam ID.
    # Unknown aliases are quarantined, never guessed from their string shape.
    resolved = mapped.loc[mapped.steam_id.notna()]
    canonical = resolved[["steam_id", "item_id", "recommend"]].rename(
        columns={"steam_id": "user_id", "item_id": "app_id", "recommend": "positive"})
    return mapping, mapped, resolved, canonical


def eligible_users(frame, supported, linux, minimum=1):
    inputs = frame.loc[frame.app_id.isin(supported)]
    counts = inputs.groupby("user_id").app_id.nunique()
    targets = inputs.loc[inputs.positive & inputs.app_id.isin(linux)]
    return set(counts.index[counts >= minimum + 1]) & set(targets.user_id)


def review_eligibility(frame, supported, linux):
    output = {}
    for k in [1, 3, 5, 10]:
        users = eligible_users(frame, supported, linux, k)
        targets = frame.loc[frame.user_id.isin(users) & frame.positive & frame.app_id.isin(linux)]
        output[str(k)] = {"potential_users": len(users), "possible_target_pairs": len(targets),
                          "possible_target_apps": int(targets.app_id.nunique())}
    return output


def cross_review_ids(left, right):
    if "review_id" not in left or "review_id" not in right:
        return {"status": "not_available_in_both_sources; user+app checks still apply"}
    a = left[["review_id", "user_id", "app_id", "positive"]].drop_duplicates()
    b = right[["review_id", "user_id", "app_id", "positive"]].drop_duplicates()
    joined = a.merge(b, on="review_id", suffixes=("_left", "_right"))
    conflict = joined.user_id_left.ne(joined.user_id_right) | joined.app_id_left.ne(joined.app_id_right)
    return {"common_original_review_ids": int(joined.review_id.nunique()),
            "common_review_ids_with_different_user_or_app": int(joined.loc[conflict, "review_id"].nunique()),
            "same_original_review_ids_with_changed_label": int(joined.loc[~conflict & joined.positive_left.ne(joined.positive_right), "review_id"].nunique())}


def anti_join_example(train, test, supported, linux, raw_train=None, raw_test=None):
    """Reserve every eligible test participant, including their query histories."""
    blocked = eligible_users(test, supported, linux)
    overlap = set(train.user_id) & blocked
    kept = train.loc[~train.user_id.isin(blocked)]
    observed_test = test.loc[test.user_id.isin(blocked)]
    shared_after = len(set(kept.user_id) & set(observed_test.user_id))
    shared_pairs = pd.MultiIndex.from_frame(kept[["user_id", "app_id"]]).intersection(
        pd.MultiIndex.from_frame(observed_test[["user_id", "app_id"]])).size
    assert shared_after == 0 and shared_pairs == 0
    original_intersection = None
    if raw_train is not None and raw_test is not None and "review_id" in raw_train and "review_id" in raw_test:
        remaining_ids = set(raw_train.loc[~raw_train.user_id.isin(blocked), "review_id"].dropna())
        heldout_ids = set(raw_test.loc[raw_test.user_id.isin(blocked), "review_id"].dropna())
        original_intersection = len(remaining_ids & heldout_ids)
        assert original_intersection == 0
    trained_apps = set(kept.app_id)
    targets = observed_test.loc[observed_test.positive & observed_test.app_id.isin(linux)]
    return {"scope": "Measured complete clean review columns; fixed eligible test users; no target chosen or train/test file written",
            "reserved_test_users": len(blocked), "training_users_removed": len(overlap),
            "training_pairs_removed": len(train) - len(kept), "remaining_training_users": int(kept.user_id.nunique()),
            "remaining_training_pairs": len(kept), "measured_user_intersection_after_exclusion": shared_after,
            "measured_user_app_intersection_after_exclusion": int(shared_pairs),
            "measured_original_review_id_intersection_after_exclusion_when_available": original_intersection,
            "possible_target_pairs_without_any_remaining_train_review_on_app": int((~targets.app_id.isin(trained_apps)).sum()),
            "potential_test_users_if_inputs_and_targets_restricted_to_remaining_train_apps": len(eligible_users(test, trained_apps & supported, trained_apps & linux)),
            "all_training_signals_must_follow_same_exclusion": "Ownership, friendship endpoints, query-derived caches and other source unions must exclude validation/test IDs too."}


def ownership_eligibility(reviews, ownership, supported, linux):
    """Ownership/played histories minus all reviewed games; positive Linux targets."""
    targets = reviews.loc[reviews.positive & reviews.app_id.isin(linux)]
    owned = ownership.loc[ownership.app_id.isin(supported)].drop_duplicates(["user_id", "app_id"])
    reviewed_index = pd.MultiIndex.from_frame(reviews[["user_id", "app_id"]])
    owned_index = pd.MultiIndex.from_frame(owned[["user_id", "app_id"]])
    inputs = owned.loc[~owned_index.isin(reviewed_index)]
    output = {}
    for name, query in [("owned", inputs), ("played_positive_minutes", inputs.loc[inputs.playtime_forever > 0])]:
        counts = query.groupby("user_id").app_id.nunique()
        result = {}
        for k in [1, 3, 5, 10]:
            users = set(counts.index[counts >= k]) & set(targets.user_id)
            selected = targets.loc[targets.user_id.isin(users)]
            result[str(k)] = {"users": len(users), "positive_linux_target_pairs": len(selected),
                              "target_apps": int(selected.app_id.nunique())}
        output[name] = result
    return output


def main():
    p = EDA / "data/processed"
    profile_path = p / "B_users_items_profiles.parquet"
    review_path = p / "B_reviews.parquet"
    profiles = pd.read_parquet(profile_path)
    b_raw = pd.read_parquet(review_path, columns=["user_id", "item_id", "recommend"])
    mapping, b_mapped, resolved, b = map_b_reviews(profiles, b_raw)
    zip_path = EDA / "data/raw/candidate_review/VintageDon/steam_dataset_2025_csv_package_v1.zip"
    v_raw = load_vintage(zip_path).rename(columns={"author_steamid": "user_id", "appid": "app_id", "voted_up": "positive", "recommendationid": "review_id"})
    v = v_raw[["user_id", "app_id", "positive", "review_id"]]
    source_frames = {"B_mapped": b, "VintageDon": v}
    input_files = [profile_path, review_path, zip_path]
    frank_dir = p / "candidate_identity/frankjc2022"
    frank_review_path = frank_dir / "reviews_identity_columns.parquet"
    frank_ownership_path = frank_dir / "user_games_identity_columns.parquet"
    ownership = None
    projection_manifest = None
    if frank_review_path.exists():
        projection_manifest = json.loads((frank_dir / "projection_manifest.json").read_text("utf-8"))
        if fingerprint(frank_review_path)["sha256"] != projection_manifest["files"]["reviews"]["local_projection"]["sha256"]:
            raise ValueError("Frank review projection differs from its pinned manifest")
        f = pd.read_parquet(frank_review_path).rename(columns={"author_steamid": "user_id", "appid": "app_id", "voted_up": "positive", "recommendationid": "review_id"})
        if len(f) != projection_manifest["files"]["reviews"]["rows"]:
            raise ValueError("Frank review projection row count mismatch")
        source_frames["frankjc2022"] = f[["user_id", "app_id", "positive", "review_id"]]
        input_files.append(frank_review_path)
    if frank_ownership_path.exists():
        if projection_manifest is None or fingerprint(frank_ownership_path)["sha256"] != projection_manifest["files"]["user_games"]["local_projection"]["sha256"]:
            raise ValueError("Frank ownership projection differs from its pinned manifest")
        ownership = pd.read_parquet(frank_ownership_path).rename(columns={"steamid": "user_id", "appid": "app_id"})
        if len(ownership) != projection_manifest["files"]["user_games"]["rows"]:
            raise ValueError("Frank ownership projection row count mismatch")
        require_steam_strings(ownership.user_id)
        if ownership[["user_id", "app_id"]].isna().any().any():
            raise ValueError("Unresolved ownership identity")
        input_files.append(frank_ownership_path)
    games = pd.read_parquet(p / "A_games.parquet", columns=["app_id", "linux"])
    supported = set(pd.read_parquet(p / "A_recommendations.parquet", columns=["app_id"]).app_id.unique())
    linux = set(games.loc[games.linux, "app_id"]) & supported
    cleaned, sources = {}, {}
    for name, raw in source_frames.items():
        cleaned[name], sources[name] = clean_reviews(raw)
        sources[name]["review_input_eligibility_on_fixed_A_catalog"] = review_eligibility(cleaned[name], supported, linux)
    comparisons = {}
    for index, left in enumerate(source_frames):
        for right in list(source_frames)[index + 1:]:
            left_frame, right_frame = cleaned[left], cleaned[right]
            shared_users = set(left_frame.user_id) & set(right_frame.user_id)
            pairs = left_frame.merge(right_frame, on=["user_id", "app_id"], suffixes=("_left", "_right"))
            original = cross_review_ids(source_frames[left], source_frames[right])
            if original.get("common_review_ids_with_different_user_or_app", 0):
                raise ValueError("Cross-source original review identity mismatch; refuse certification")
            comparisons[f"{left}__{right}"] = {"common_exact_steam_users": len(shared_users),
                "common_user_app_pairs": len(pairs), "common_user_app_pairs_with_different_label": int(pairs.positive_left.ne(pairs.positive_right).sum()),
                "original_review_id_check": original,
                f"train_{left}_test_{right}": anti_join_example(left_frame, right_frame, supported, linux, source_frames[left], source_frames[right]),
                f"train_{right}_test_{left}": anti_join_example(right_frame, left_frame, supported, linux, source_frames[right], source_frames[left])}
    direct = mapping.user_id.str.fullmatch(r"\d{17}")
    report = {"checked_date": "2026-11-07", "status": "STRUCTURAL_RESEARCH_PLUS_MEASURED_LOCAL_STEAM_ID_COMBINATIONS",
        "source_replacement_or_model_fit": False, "partitioned_datasets_written": False,
        "identity_scope": "Same original public Steam account; not a natural person across multiple accounts; publisher identity semantics assumed and checked where original review IDs overlap",
        "inputs": [fingerprint(path) for path in input_files],
        "catalog": {"input_apps": "A reviewed apps, all platforms", "target_apps": "A linux=True AND A reviewed", "linux_app_count": len(linux),
                    "scope_note": "Fixed comparison catalog only. A interaction rows do not enter these training examples. Replacement-source catalog expansion needs separate metadata/support audit."},
        "B_mapping": {"profile_rows": len(profiles), "unique_local_profile_ids": int(profiles.user_id.nunique()),
            "local_profile_ids_with_multiple_steam_ids": 0,
            "steam_ids_with_multiple_local_aliases": int(mapping.groupby("steam_id").user_id.nunique().gt(1).sum()),
            "17digit_numeric_aliases_different_from_explicit_steam_id": int((direct & mapping.user_id.ne(mapping.steam_id)).sum()),
            "original_review_users": int(b_raw.user_id.nunique()), "mapped_review_users": int(resolved.user_id.nunique()),
            "mapped_original_steam_users": int(resolved.steam_id.nunique()),
            "unmapped_review_users_quarantined": int(b_mapped.loc[b_mapped.steam_id.isna(), "user_id"].nunique()),
            "unmapped_review_rows_quarantined": int(b_mapped.steam_id.isna().sum()),
            "policy": "Always use explicit profile steam_id; never infer Steam identity from a numeric-looking user_id alias. Quarantine reviews without profile mapping."},
        "sources": sources, "pairwise_complete_review_column_audits": comparisons,
        "frank_projection_provenance": projection_manifest}
    if ownership is not None and "frankjc2022" in cleaned:
        frank_users = set(cleaned["frankjc2022"].user_id)
        owner_users = set(ownership.user_id)
        report["frank_ownership"] = {"rows": len(ownership), "unique_steam_users": len(owner_users),
            "unique_user_app_pairs": int(ownership.drop_duplicates(["user_id", "app_id"]).shape[0]),
            "duplicate_user_app_rows": int(ownership.duplicated(["user_id", "app_id"]).sum()),
            "unique_apps": int(ownership.app_id.nunique()),
            "users_with_both_reviews_and_ownership": len(owner_users & frank_users),
            "evaluation": ownership_eligibility(cleaned["frankjc2022"], ownership, supported, linux),
            "anti_join_requirement": "If frank ownership participates in training, remove every validation/test Steam ID from ownership too; review-only exclusion is insufficient."}
    output = EDA / "reports/dataset_combination_overlap_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": sources, "pairs": comparisons, "B_mapping": report["B_mapping"],
                      "frank_ownership": report.get("frank_ownership")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
