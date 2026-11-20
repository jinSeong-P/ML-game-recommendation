"""Audit A/VintageDon independence; export aggregates only, never identity links.

Full A scan, full VintageDon scan. Fingerprints are compatibility screens, not
proof of review/user identity. Shifted dates are sensitivity controls, not an
estimated false-positive rate. This script does not change train/test sources.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_review_candidates import fingerprint


EDA = Path(__file__).resolve().parents[1]
KEY_ORIGIN = np.datetime64("2000-01-01", "D").astype(int)
KEY_DAYS = 20_000


def day_number(values):
    return np.asarray(values).astype("datetime64[D]").astype(np.int64)


def signature(app, day, positive):
    day = np.asarray(day, dtype=np.int64) - KEY_ORIGIN
    if np.any((day < 0) | (day >= KEY_DAYS)):
        raise ValueError("Day outside collision-free signature range")
    return ((np.asarray(app, dtype=np.int64) * KEY_DAYS + day) * 2
            + np.asarray(positive, dtype=np.int64))


def load_vintage(path):
    columns = ["recommendationid", "author_steamid", "appid", "voted_up",
               "timestamp_created", "timestamp_updated", "author_playtime_at_review",
               "author_playtime_forever", "votes_up", "votes_funny"]
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if Path(n).name == "reviews.csv"]
        if len(names) != 1:
            raise ValueError("Expected one reviews.csv")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, usecols=columns,
                                dtype={"recommendationid": "string", "author_steamid": "string"})
    if not pd.api.types.is_bool_dtype(frame.voted_up):
        raise ValueError("Expected explicit boolean labels")
    if frame[columns[:6]].isna().any().any():
        raise ValueError("Missing identifier/label/timestamp")
    if not frame.author_steamid.str.fullmatch(r"\d{17}").all():
        raise ValueError("Steam IDs must remain exact 17-digit strings")
    frame["day"] = frame.timestamp_created // 86400
    frame["rid_numeric"] = pd.to_numeric(frame.recommendationid, errors="raise").astype("int64")
    if frame.rid_numeric.duplicated().any():
        raise ValueError("Duplicate original review IDs require a separate policy")
    return frame


def clean_pairs(vintage):
    frame = vintage.rename(columns={"author_steamid": "user_id", "appid": "app_id", "voted_up": "positive"})
    labels = frame[["user_id", "app_id", "positive"]].drop_duplicates()
    conflict = labels.duplicated(["user_id", "app_id"], keep=False)
    allowed = labels.loc[~conflict, ["user_id", "app_id"]]
    # Earliest creation date is conservative if duplicate pair records differ.
    pairs = (frame.groupby(["user_id", "app_id", "positive"], as_index=False)
             .agg(day=("day", "min")))
    clean = pairs.merge(allowed, on=["user_id", "app_id"], validate="many_to_one")
    assert not clean.duplicated(["user_id", "app_id"]).any()
    return clean


def eligibility(pairs, supported, linux, cutoff, excluded_users=None, recent_inputs=False, recent_targets=False):
    frame = pairs if excluded_users is None else pairs.loc[~pairs.user_id.isin(excluded_users)]
    inputs = frame.loc[frame.app_id.isin(supported)]
    if recent_inputs:
        inputs = inputs.loc[inputs.day > cutoff]
    counts = inputs.groupby("user_id").app_id.nunique()
    targets = inputs.loc[inputs.positive & inputs.app_id.isin(linux)]
    if recent_targets:
        targets = targets.loc[targets.day > cutoff]
    output = {}
    for k in [1, 3, 5, 10]:
        users = counts.index[counts >= k + 1].intersection(targets.user_id.unique())
        chosen = targets.loc[targets.user_id.isin(users)]
        output[str(k)] = {"users": len(users), "possible_target_pairs": len(chosen),
                          "possible_target_games": int(chosen.app_id.nunique())}
    return output


def relation_summary(records, vintage):
    if not records:
        return {"vintage_users_with_multi_game_same_A_user_candidate": {str(k): 0 for k in [2, 3, 5]},
                "candidate_user_links_with_at_least_games": {str(k): 0 for k in [2, 3, 5]}}
    links = pd.concat(records, ignore_index=True).drop_duplicates()
    links["vuser"] = vintage.author_steamid.to_numpy()[links.vrow.to_numpy()]
    counts = links.groupby(["vuser", "auser"]).app_id.nunique()
    return {"vintage_users_with_multi_game_same_A_user_candidate": {
                str(k): int(counts.loc[counts >= k].index.get_level_values("vuser").nunique()) for k in [2, 3, 5]},
            "candidate_user_links_with_at_least_games": {str(k): int((counts >= k).sum()) for k in [2, 3, 5]}}


def audit(a_path, vintage, batch_size):
    parquet = pq.ParquetFile(a_path)
    date_min = min(parquet.metadata.row_group(i).column(parquet.schema_arrow.get_field_index("date")).statistics.min
                   for i in range(parquet.num_row_groups))
    date_max = max(parquet.metadata.row_group(i).column(parquet.schema_arrow.get_field_index("date")).statistics.max
                   for i in range(parquet.num_row_groups))
    lower, cutoff = day_number([date_min, date_max])
    historical = vintage.index[vintage.day.between(lower, cutoff)].to_numpy()
    probes = []
    offsets = {"actual_date": 0, "date_plus_7_days_control": 7, "date_plus_365_days_control": 365}
    for index, (name, offset) in enumerate(offsets.items()):
        shifted = vintage.day.to_numpy()[historical] + offset
        valid = (shifted >= lower) & (shifted <= cutoff)
        rows = historical[valid]
        probes.append(pd.DataFrame({"key": signature(vintage.appid.to_numpy()[rows], shifted[valid],
                                                     vintage.voted_up.to_numpy()[rows]),
                                    "vrow": rows, "control": np.int8(index)}))
    probes = pd.concat(probes, ignore_index=True)
    probe_keys = pd.Index(probes.key.unique())
    criteria = ["app_day_label", "plus_playtime_at_review", "plus_playtime_forever", "plus_either_playtime",
                "plus_either_playtime_and_votes", "plus_either_playtime_and_nonzero_votes"]
    counts = {name: np.zeros((3, len(vintage)), dtype=np.uint32) for name in criteria}
    rich_links = [[] for _ in offsets]
    original_id_index = pd.Index(vintage.rid_numeric)
    original_id_matches = 0
    id_matches_same_basic_fields = 0
    ordinal = True
    scanned = 0
    a_user_min, a_user_max = None, None
    a_review_min, a_review_max = None, None
    non_midnight = 0
    nulls = {c: 0 for c in parquet.schema_arrow.names}
    cols = ["app_id", "date", "is_recommended", "hours", "helpful", "funny", "user_id", "review_id"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=cols):
        a = batch.to_pandas()
        for c in cols:
            nulls[c] += int(a[c].isna().sum())
        if a.isna().any().any():
            raise ValueError("Missing A fields require an explicit matching policy")
        days = day_number(a.date)
        non_midnight += int((a.date.to_numpy() != a.date.to_numpy().astype("datetime64[D]")).sum())
        ordinal &= bool(np.array_equal(a.review_id.to_numpy(), np.arange(scanned, scanned + len(a))))
        for column in ["user_id", "review_id"]:
            lo, hi = int(a[column].min()), int(a[column].max())
            if column == "user_id":
                a_user_min = lo if a_user_min is None else min(a_user_min, lo)
                a_user_max = hi if a_user_max is None else max(a_user_max, hi)
            else:
                a_review_min = lo if a_review_min is None else min(a_review_min, lo)
                a_review_max = hi if a_review_max is None else max(a_review_max, hi)
        # Deliberately demonstrate why equal numeric review IDs do not establish identity.
        loc = original_id_index.get_indexer(a.review_id)
        matched = loc >= 0
        original_id_matches += int(matched.sum())
        vloc = loc[matched]
        id_matches_same_basic_fields += int(((a.app_id.to_numpy()[matched] == vintage.appid.to_numpy()[vloc])
            & (days[matched] == vintage.day.to_numpy()[vloc])
            & (a.is_recommended.to_numpy()[matched] == vintage.voted_up.to_numpy()[vloc])).sum())
        a["key"] = signature(a.app_id, days, a.is_recommended)
        a = a.loc[probe_keys.get_indexer(a.key) >= 0]
        if len(a):
            hit = a.merge(probes, on="key", how="inner")
            vrow = hit.vrow.to_numpy()
            control = hit.control.to_numpy()
            hours10 = np.rint(hit.hours.to_numpy(dtype=np.float64) * 10).astype(np.int64)
            # A's collection/rounding semantics are not assumed to equal either V field.
            # Exclude 1000h saturation; report both minute->0.1h quantizations as sensitivity.
            at = vintage.author_playtime_at_review.to_numpy()[vrow] / 6
            forever = vintage.author_playtime_forever.to_numpy()[vrow] / 6
            at_match = np.isfinite(at) & ((hours10 == np.floor(at)) | (hours10 == np.rint(at))) & (hours10 < 10000)
            forever_match = np.isfinite(forever) & ((hours10 == np.floor(forever)) | (hours10 == np.rint(forever))) & (hours10 < 10000)
            votes = ((hit.helpful.to_numpy() == vintage.votes_up.to_numpy()[vrow])
                     & (hit.funny.to_numpy() == vintage.votes_funny.to_numpy()[vrow]))
            nonzero = (hit.helpful.to_numpy() + hit.funny.to_numpy()) > 0
            masks = [np.ones(len(hit), dtype=bool), at_match, forever_match, at_match | forever_match,
                     (at_match | forever_match) & votes, (at_match | forever_match) & votes & nonzero]
            for name, mask in zip(criteria, masks):
                np.add.at(counts[name], (control[mask], vrow[mask]), 1)
            for c in range(3):
                mask = masks[4] & (control == c)
                if mask.any():
                    rich_links[c].append(hit.loc[mask, ["vrow", "user_id", "app_id"]].rename(columns={"user_id": "auser"}))
        scanned += batch.num_rows
        if scanned % (batch_size * 10) == 0 or scanned == parquet.metadata.num_rows:
            print(f"Scanned A rows: {scanned:,}/{parquet.metadata.num_rows:,}", flush=True)
    result = {"rows_scanned": scanned, "date_min": str(date_min), "date_max": str(date_max),
              "user_id_range": [a_user_min, a_user_max], "review_id_range": [a_review_min, a_review_max],
              "review_id_equals_zero_based_row_number_for_every_row": ordinal,
              "non_midnight_dates": non_midnight, "nulls": nulls,
              "numeric_review_id_collisions": original_id_matches,
              "numeric_review_id_collisions_with_same_app_day_label": id_matches_same_basic_fields}
    screens = {}
    for c, name in enumerate(offsets):
        result_c = {"probe_review_rows": int((probes.control == c).sum()), "criteria": {}}
        for criterion in criteria:
            values = counts[criterion][c]
            rows = np.flatnonzero(values)
            result_c["criteria"][criterion] = {
                "vintage_review_rows_with_candidate": len(rows),
                "vintage_users_with_candidate": int(vintage.author_steamid.iloc[rows].nunique()),
                "candidate_A_V_review_links": int(values.sum()),
                "vintage_review_rows_with_exactly_one_A_candidate": int((values == 1).sum())}
        result_c["multi_game_rich_signature_consistency"] = relation_summary(rich_links[c], vintage)
        screens[name] = result_c
    return result, screens, counts, int(cutoff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintage-zip", type=Path, default=EDA / "data/raw/candidate_review/VintageDon/steam_dataset_2025_csv_package_v1.zip")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=EDA / "reports/vintage_independence_metrics.json")
    args = parser.parse_args()
    a_path = EDA / "data/processed/A_recommendations.parquet"
    games_path = EDA / "data/processed/A_games.parquet"
    users_path = EDA / "data/processed/A_users.parquet"
    vintage = load_vintage(args.vintage_zip)
    print(f"Loaded all {len(vintage):,} VintageDon reviews", flush=True)
    a, screens, counts, cutoff = audit(a_path, vintage, args.batch_size)
    supported = set(pd.read_parquet(a_path, columns=["app_id"]).app_id.unique())
    games = pd.read_parquet(games_path, columns=["app_id", "linux"])
    linux = set(games.loc[games.linux, "app_id"]) & supported
    pairs = clean_pairs(vintage)
    base_users = set(vintage.author_steamid.iloc[np.flatnonzero(counts["app_day_label"][0])])
    rich_users = set(vintage.author_steamid.iloc[np.flatnonzero(counts["plus_either_playtime_and_votes"][0])])
    summaries = {"original": eligibility(pairs, supported, linux, cutoff),
        "targets_created_after_A_max_day": eligibility(pairs, supported, linux, cutoff, recent_targets=True),
        "all_input_and_target_reviews_created_after_A_max_day": eligibility(pairs, supported, linux, cutoff, recent_inputs=True, recent_targets=True),
        "exclude_any_user_with_app_day_label_candidate": eligibility(pairs, supported, linux, cutoff, excluded_users=base_users),
        "exclude_any_user_with_rich_signature_candidate": eligibility(pairs, supported, linux, cutoff, excluded_users=rich_users)}
    timestamp = pd.to_datetime(vintage.timestamp_created, unit="s", utc=True)
    updated = pd.to_datetime(vintage.timestamp_updated, unit="s", utc=True)
    result = {"checked_date": "2026-11-07", "decision": "STRICT_USER_DISJOINTNESS_NOT_CERTIFIABLE_WITH_CURRENT_A",
        "strict_split_created": False,
        "reason": "A user IDs are source-local; A review IDs are row ordinals. No original SteamID mapping is available in local A files. Fingerprint compatibility is not verified identity or nonidentity.",
        "inputs": [fingerprint(p) for p in [a_path, users_path, games_path, args.vintage_zip]],
        "A": a,
        "VintageDon": {"rows_scanned": len(vintage), "original_review_id_range": [int(vintage.rid_numeric.min()), int(vintage.rid_numeric.max())],
            "unique_steam_users": int(vintage.author_steamid.nunique()),
            "review_rows_by_created_year_UTC": {str(k): int(v) for k, v in timestamp.dt.year.value_counts().sort_index().items()},
            "created_on_or_before_A_max_day": int((vintage.day <= cutoff).sum()),
            "created_after_A_max_day": int((vintage.day > cutoff).sum()),
            "old_review_rows_updated_after_A_max_day": int(((vintage.day <= cutoff) & (vintage.timestamp_updated // 86400 > cutoff)).sum()),
            "missing_playtime_at_review": int(vintage.author_playtime_at_review.isna().sum()),
            "missing_playtime_forever": int(vintage.author_playtime_forever.isna().sum()),
            "created_min_UTC": str(timestamp.min()), "created_max_UTC": str(timestamp.max()),
            "updated_min_UTC": str(updated.min()), "updated_max_UTC": str(updated.max())},
        "fingerprint_policy": {"base": "App ID + A source date / Vintage UTC creation day + current explicit label",
            "playtime": "A hours <1000; tenths equal floor(minutes/6) or nearest-even integer(minutes/6), separately for V at-review and forever fields",
            "rich": "Base + either playtime compatibility + equal helpful/funny votes",
            "limitations": ["Creation day is low resolution; different reviews can match.",
                "A source-date timezone and creation-vs-update semantics are not independently established; aligning its day with Vintage UTC creation day is conditional.",
                "Playtime, votes and labels may change between collection dates; absence of a match does not certify independence.",
                "A hours collection semantics and original rounding are not established by this audit.",
                "Shifted-date results are sensitivity controls, not calibrated false-positive rates.",
                "Same App IDs are expected in item-based evaluation and do not alone imply leakage."]},
        "fingerprint_screens": screens, "eligibility_sensitivity_not_certified_splits": summaries,
        "guarantees": {"same_game_catalog_overlap_allowed": True,
            "same_original_user_excluded": "UNVERIFIABLE", "same_original_review_excluded": "UNVERIFIABLE for the original mixed-date source",
            "post_cutoff_creation_dates": "Excludes identical original review records with immutable creation dates only if both sources store compatible original creation days; A timezone/date semantics remain unverified. Does not exclude returning users or establish a full temporal evaluation.",
            "safe_strict_user_split": "Use original Steam IDs in every training source and split users before fitting; alternatively construct all splits within one source retaining exact IDs. A cannot be added to that training union without its original-ID mapping."}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "A": a, "fingerprint_screens": screens,
                      "eligibility": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
