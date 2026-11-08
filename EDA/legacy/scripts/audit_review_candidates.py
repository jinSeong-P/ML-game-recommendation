"""Aggregate review-history eligibility; never export individual user IDs.

Uses existing local A/B Parquets. Optionally audits the VintageDon CSV archive.
This measures possible input/target splits, not recommendation performance.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


EDA = Path(__file__).resolve().parents[1]


def fingerprint(path):
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    label = path.relative_to(EDA).as_posix() if path.is_relative_to(EDA) else path.name
    return {"path": label, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def summarize(frame, catalog, supported, linux):
    required = ["user_id", "app_id", "positive"]
    missing = frame[required].isna().any(axis=1)
    valid = frame.loc[~missing, required].copy()
    if not pd.api.types.is_bool_dtype(valid.positive) and not valid.positive.map(
        lambda value: isinstance(value, bool)
    ).all():
        raise ValueError("Labels must be explicit booleans")
    if not valid.user_id.map(lambda value: isinstance(value, str)).all():
        raise ValueError("User IDs must be loaded as strings without float conversion")
    distinct = valid.drop_duplicates()
    conflicts = distinct.duplicated(["user_id", "app_id"], keep=False)
    conflicting_pairs = distinct.loc[conflicts, ["user_id", "app_id"]].drop_duplicates()
    # Exclude contradictory pairs instead of choosing a favorable label.
    pairs = distinct.loc[~conflicts].copy()
    history = pairs.groupby("user_id").app_id.nunique()
    targets = pairs.loc[pairs.positive & pairs.app_id.isin(linux)]
    views = {}
    for name, allowed in [("all_games", None), ("A_catalog", catalog),
                          ("A_reviewed", supported)]:
        inputs = pairs if allowed is None else pairs.loc[pairs.app_id.isin(allowed)]
        counts = inputs.groupby("user_id").app_id.nunique()
        # Targets belong to every allowed set. Removing one target leaves >= k
        # other distinct input games iff the user's count is >= k + 1.
        qualified = {}
        for minimum in [1, 3, 5, 10]:
            users = set(counts[counts >= minimum + 1].index) & set(targets.user_id)
            possible = targets.loc[targets.user_id.isin(users)]
            qualified[str(minimum)] = {
                "users": len(users), "possible_target_pairs": len(possible),
                "possible_target_games": int(possible.app_id.nunique()),
            }
        views[name] = qualified
    user_labels = pairs.groupby("user_id").positive.agg(["any", "all"])
    result = {
        "raw_rows": len(frame), "invalid_rows": int(missing.sum()),
        "duplicate_same_label_rows": len(valid) - len(distinct),
        "conflicting_pairs_excluded": len(conflicting_pairs),
        "unique_user_game_pairs": len(pairs), "users": len(history),
        "games": int(pairs.app_id.nunique()),
        "history_quantiles": {str(q): float(history.quantile(q))
                              for q in [0.25, 0.5, 0.75, 0.9]},
        "maximum_games_per_user": int(history.max()),
        "maximum_reviews_per_game": int(pairs.groupby("app_id").size().max()),
        "single_game_users": int((history == 1).sum()),
        "users_with_at_least_games": {str(k): int((history >= k).sum())
                                     for k in [2, 3, 5, 10]},
        "user_label_mix": {
            "positive_only": int(user_labels["all"].sum()),
            "negative_only": int((~user_labels["any"]).sum()),
            "mixed": int((user_labels["any"] & ~user_labels["all"]).sum()),
        },
        "games_in_A_catalog": len(set(pairs.app_id) & catalog),
        "games_in_A_reviewed": len(set(pairs.app_id) & supported),
        "linux_positive_pairs": len(targets),
        "linux_positive_games": int(targets.app_id.nunique()),
        "eligibility_by_minimum_input_games": views,
    }
    assert sum(result["user_label_mix"].values()) == result["users"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintage-zip", type=Path)
    parser.add_argument("--output", type=Path,
                        default=EDA / "reports/test_dataset_review_history_metrics.json")
    args = parser.parse_args()
    games_path = EDA / "data/processed/A_games.parquet"
    recs_path = EDA / "data/processed/A_recommendations.parquet"
    b_path = EDA / "data/processed/B_reviews.parquet"
    games = pd.read_parquet(games_path, columns=["app_id", "linux"])
    catalog = set(games.app_id)
    supported = set(pd.read_parquet(recs_path, columns=["app_id"]).app_id.unique())
    linux = set(games.loc[games.linux, "app_id"]) & supported
    report = {
        "checked_date": "2026-11-07",
        "protocol": "At least k distinct input games after hiding one positive Linux game; no split or model fit performed",
        "linux_rule": "A games.linux=True AND app_id present in A recommendations",
        "deduplication": "Unique (user_id, app_id); exclude conflicting labels",
        "A_user_disjointness": "Unverified: A internal IDs cannot be directly compared to external Steam IDs",
        "A_reviewed_linux_games": len(linux),
        "inputs": [fingerprint(p) for p in [games_path, recs_path, b_path]],
        "datasets": {},
    }
    baseline = pd.read_parquet(b_path, columns=["user_id", "item_id", "recommend"])
    baseline = baseline.rename(columns={"item_id": "app_id", "recommend": "positive"})
    report["datasets"]["B"] = summarize(baseline, catalog, supported, linux)
    if args.vintage_zip:
        report["inputs"].append(fingerprint(args.vintage_zip))
        with zipfile.ZipFile(args.vintage_zip) as archive:
            names = [n for n in archive.namelist() if Path(n).name == "reviews.csv"]
            if len(names) != 1:
                raise ValueError(f"Expected one reviews.csv, found {names}")
            with archive.open(names[0]) as handle:
                reviews = pd.read_csv(
                    handle, dtype={"author_steamid": "string", "recommendationid": "string"},
                    usecols=["author_steamid", "appid", "voted_up", "recommendationid",
                             "timestamp_created", "language"],
                )
        reviews = reviews.rename(columns={"author_steamid": "user_id", "appid": "app_id", "voted_up": "positive"})
        if not pd.api.types.is_bool_dtype(reviews.positive):
            original = reviews.positive
            reviews["positive"] = original.astype("string").str.lower().map(
                {"true": True, "false": False, "t": True, "f": False})
            if (original.notna() & reviews.positive.isna()).any():
                raise ValueError("Unrecognized VintageDon label representation")
        result = summarize(reviews, catalog, supported, linux)
        dates = pd.to_datetime(reviews.timestamp_created, unit="s", utc=True, errors="coerce")
        result["source_checks"] = {
            "duplicate_review_ids": int(reviews.recommendationid.dropna().duplicated().sum()),
            "missing_review_ids": int(reviews.recommendationid.isna().sum()),
            "nonnull_user_ids_not_17_digits": int((~reviews.user_id.dropna().str.fullmatch(r"\d{17}")).sum()),
            "invalid_timestamps": int(dates.isna().sum()),
            "review_created_utc_min": str(dates.min()),
            "review_created_utc_max": str(dates.max()),
            "language_counts": {str(k): int(v) for k, v in reviews.language.value_counts(dropna=False).items()},
        }
        report["datasets"]["VintageDon"] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["datasets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
