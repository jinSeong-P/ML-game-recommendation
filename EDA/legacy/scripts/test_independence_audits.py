"""Small regressions for identity preservation, conflict policy and split isolation."""

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from audit_a_user_split import assign_partitions
from audit_vintage_independence import audit, clean_pairs, eligibility, load_vintage, signature
from audit_steam_id_combinations import anti_join_example, map_b_reviews, require_steam_strings


class IndependenceAuditTests(unittest.TestCase):
    def test_steam_identity_rejects_float_and_invalid_account_range(self):
        require_steam_strings(pd.Series(["76561198000000001", "76561198000000002"]))
        with self.assertRaises(ValueError):
            require_steam_strings(pd.Series([76561198000000001.0]))
        with self.assertRaises(ValueError):
            require_steam_strings(pd.Series(["12345678901234567"]))

    def test_b_numeric_alias_is_not_used_as_original_steam_id(self):
        profiles = pd.DataFrame({"user_id": ["76561198000000001"], "steam_id": ["76561198000000002"]})
        reviews = pd.DataFrame({"user_id": ["76561198000000001", "76561198000000003"],
                                "item_id": [10, 20], "recommend": [True, False]})
        _, mapped, _, canonical = map_b_reviews(profiles, reviews)
        self.assertEqual(canonical.user_id.tolist(), ["76561198000000002"])
        self.assertEqual(int(mapped.steam_id.isna().sum()), 1)

    def test_user_exclusion_removes_all_training_games_and_keeps_shared_items(self):
        heldout, other = "76561198000000001", "76561198000000002"
        train = pd.DataFrame({"user_id": [heldout, heldout, other], "app_id": [1, 2, 10],
                              "positive": [True, False, True], "review_id": ["101", "102", "103"]})
        test = pd.DataFrame({"user_id": [heldout, heldout], "app_id": [10, 11],
                             "positive": [True, False], "review_id": ["104", "105"]})
        result = anti_join_example(train, test, {1, 2, 10, 11}, {10}, train, test)
        self.assertEqual(result["training_pairs_removed"], 2)
        self.assertEqual(result["remaining_training_pairs"], 1)
        self.assertEqual(result["measured_user_intersection_after_exclusion"], 0)
        self.assertEqual(result["measured_original_review_id_intersection_after_exclusion_when_available"], 0)
        self.assertEqual(result["possible_target_pairs_without_any_remaining_train_review_on_app"], 0)

    def test_conflicts_removed_and_earliest_duplicate_day_retained(self):
        source = pd.DataFrame({"author_steamid": ["x", "x", "y", "y", "z"],
            "appid": [1, 1, 2, 2, 3], "voted_up": [True, True, True, False, False],
            "day": [5, 3, 5, 5, 7]})
        pairs = clean_pairs(source)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(int(pairs.loc[pairs.user_id == "x", "day"].iloc[0]), 3)
        self.assertNotIn("y", set(pairs.user_id))

    def test_eligibility_requires_other_supported_inputs(self):
        pairs = pd.DataFrame({"user_id": ["one", "two", "two", "outside", "outside"],
            "app_id": [1, 1, 2, 1, 99], "positive": [True, True, False, True, False],
            "day": [3, 3, 1, 3, 1]})
        baseline = eligibility(pairs, {1, 2}, {1}, 2)
        self.assertEqual(baseline["1"]["users"], 1)
        self.assertEqual(baseline["3"]["users"], 0)
        self.assertEqual(eligibility(pairs, {1, 2}, {1}, 2, recent_inputs=True)["1"]["users"], 0)
        self.assertEqual(eligibility(pairs, {1, 2}, {1}, 2, excluded_users={"two"})["1"]["users"], 0)

    def test_partition_is_user_based_and_order_independent(self):
        users = np.arange(10_000, dtype=np.int32)
        first = assign_partitions(users)
        self.assertTrue(np.array_equal(first[::-1], assign_partitions(users[::-1])))
        repeated = np.repeat(users, 3)
        self.assertTrue(np.array_equal(assign_partitions(repeated), np.repeat(first, 3)))
        self.assertEqual(set(np.unique(first)), {0, 1, 2})
        with self.assertRaises(ValueError):
            assign_partitions(np.array([1.0]))
        with self.assertRaises(ValueError):
            assign_partitions(np.array([-1]))

    def test_archive_load_preserves_adjacent_large_steam_ids(self):
        source = pd.DataFrame({"recommendationid": [100, 101],
            "author_steamid": ["76561198000000001", "76561198000000002"], "appid": [10, 20],
            "voted_up": [True, False], "timestamp_created": [1_640_995_200, 1_641_081_600],
            "timestamp_updated": [1_640_995_200, 1_641_081_600],
            "author_playtime_at_review": [210, 210], "author_playtime_forever": [210, 210],
            "votes_up": [2, 2], "votes_funny": [1, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("reviews.csv", source.to_csv(index=False))
            loaded = load_vintage(path)
        self.assertEqual(list(loaded.author_steamid), list(source.author_steamid))
        self.assertEqual(loaded.author_steamid.nunique(), 2)

    def test_full_scan_screen_and_ordinal_ids(self):
        source = pd.DataFrame({"app_id": [10, 20, 30],
            "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-12-31"]),
            "is_recommended": [True, False, True], "hours": [3.5, 3.5, 0.1],
            "helpful": [2, 2, 0], "funny": [1, 1, 0], "user_id": [7, 7, 8], "review_id": [0, 1, 2]})
        dates = pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True)
        vintage = pd.DataFrame({"appid": [10, 20], "voted_up": [True, False],
            "day": dates.tz_localize(None).to_numpy().astype("datetime64[D]").astype(np.int64),
            "rid_numeric": [1000, 1001], "author_steamid": ["76561198000000001"] * 2,
            "author_playtime_at_review": [210, 210], "author_playtime_forever": [210, 210],
            "votes_up": [2, 2], "votes_funny": [1, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.parquet"
            source.to_parquet(path, index=False)
            stats, screens, counts, _ = audit(path, vintage, 2)
        self.assertEqual(stats["rows_scanned"], 3)
        self.assertTrue(stats["review_id_equals_zero_based_row_number_for_every_row"])
        self.assertEqual(screens["actual_date"]["criteria"]["plus_either_playtime_and_votes"]["vintage_review_rows_with_candidate"], 2)
        self.assertEqual(screens["actual_date"]["multi_game_rich_signature_consistency"]["vintage_users_with_multi_game_same_A_user_candidate"]["2"], 1)
        self.assertEqual(int(counts["plus_either_playtime_and_votes"][1].sum()), 0)
        self.assertEqual(int(counts["plus_either_playtime_and_votes"][2].sum()), 0)
        self.assertEqual(len(set(signature([1, 1, 2], [19000] * 3, [True, False, True]))), 3)


if __name__ == "__main__":
    unittest.main()
