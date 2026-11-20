"""Korean presentation helpers for EDA tables and report outputs.

This module only changes presentation copies. It never renames columns in the
source CSV/JSON files and never mutates a DataFrame or Series passed by a caller.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd


# Explicit labels for every CSV header in reports/tables and every additional
# heading emitted by the 02 sufficiency report. Keep these mappings semantic:
# do not infer Korean labels by splitting identifiers or abbreviations.
_COLUMN_LABELS = {
    "empty_values": "비어 있는 값",
    "B_profile_duplicates": "B 프로필 중복",
    "B_review_duplicates": "B 리뷰 중복",
    "A_users_consistency": "A 사용자·게임 파일 일치 여부",
    "per_user": "사용자별",
    "per_item": "게임별",
    "hours": "플레이 시간(시간)",
    "linux_positive_per_game": "Linux 게임별 긍정 리뷰 수",
    "B_input_cf_per_user": "B 사용자별 협업 필터링 입력 게임 수",
    "pop_A_all_linux": "인기순: A 전체 기간, 전체 Linux 후보",
    "pop_A_linux_released_le2015": "인기순: 2015년까지 출시된 Linux 후보",
    "pop_A_reviews_le2015": "인기순: 2015년까지의 A 리뷰 사용",
    "random_expected_recall@10": "무작위 추천의 기대 Recall@10",
    "p25": "하위 25% 값",
    "p50": "중앙값",
    "p75": "하위 75% 값",
    "p90": "하위 90% 값",
    "p99": "하위 99% 값",
    "mean": "평균",
    "max": "최댓값",
    "recall@10": "Recall@10",
    "ndcg@10": "NDCG@10",
    "A_meta_tags_empty": "A에서 태그가 빈 게임 수",
    "A_meta_description_empty": "A에서 설명이 빈 게임 수",
    "B_steam_games_id_missing": "B 게임 목록에서 ID가 없는 행 수",
    "B_steam_games_tags_missing": "B 게임 목록에서 태그가 없는 행 수",
    "B_steam_games_genres_missing": "B 게임 목록에서 장르가 없는 행 수",
    "C_tags_empty": "C에서 태그가 빈 게임 수",
    "C_genres_empty": "C에서 장르가 빈 게임 수",
    "C_short_description_empty": "C에서 짧은 설명이 빈 게임 수",
    "profile_lines": "프로필 행 수",
    "unique_users": "서로 다른 사용자 수",
    "users_with_dup_lines": "여러 행에 나타난 사용자 수",
    "dup_users_identical_item_lists": "그중 게임 목록이 모두 같은 사용자 수",
    "users_with_multiple_review_lines": "리뷰 행이 여러 개인 사용자 수",
    "dup_user_item_rows": "중복된 사용자-게임 행 수",
    "user_item_with_conflicting_label": "긍정과 부정이 엇갈린 사용자-게임 수",
    "rec_users_missing_in_users_csv": "리뷰는 있으나 사용자 파일에 없는 사용자 수",
    "rec_items_missing_in_games_csv": "리뷰는 있으나 게임 파일에 없는 게임 수",
    "games_without_any_rec": "리뷰가 하나도 없는 게임 수",
    "games_metadata_ids_equal_games_ids": "게임 파일과 설명 파일의 게임 ID 일치 여부",
    "positive_rate": "긍정 리뷰 비율(0~1)",
    "sparsity": "행렬 희소도(빈 칸 비율, 0~1)",
    "users_with_1_review_pct": "리뷰가 1건인 사용자 비율(%)",
    "users_with_ge5_reviews_pct": "리뷰가 5건 이상인 사용자 비율(%)",
    "users_with_ge10_reviews_pct": "리뷰가 10건 이상인 사용자 비율(%)",
    "users_with_ge5_positive_pct": "긍정 리뷰가 5건 이상인 사용자 비율(%)",
    "users_with_no_positive_pct": "긍정 리뷰가 없는 사용자 비율(%)",
    "items_with_ge10_positive": "긍정 리뷰가 10건 이상인 게임 수",
    "items_with_ge50_positive": "긍정 리뷰가 50건 이상인 게임 수",
    "hours_zero_pct": "플레이 시간이 0인 리뷰 비율(%)",
    "hours_at_cap_1000_pct": "플레이 시간이 상한 1,000시간인 리뷰 비율(%)",
    "hours_median_pos": "추천 리뷰의 플레이 시간 중앙값",
    "hours_median_neg": "비추천 리뷰의 플레이 시간 중앙값",
    "users_joined": "두 파일로 연결한 사용자 수",
    "reviews_col_equals_rec_rows_pct": "사용자 파일의 리뷰 수와 실제 리뷰 행 수가 같은 비율(%)",
    "products_median": "사용자별 보유 제품 수 중앙값",
    "reviews_to_products_median_ratio": "보유 제품 대비 리뷰 수 비율의 중앙값",
    "users_csv_users_with_0_reviews_pct": "사용자 파일에서 리뷰 수가 0인 사용자 비율(%)",
    "games_total": "전체 게임 수",
    "linux_games_pct": "Linux 게임 비율(%)",
    "linux_games_with_any_review": "리뷰가 있는 Linux 게임 수",
    "linux_games_with_ge10_positive": "긍정 리뷰가 10건 이상인 Linux 게임 수",
    "linux_games_with_ge50_positive": "긍정 리뷰가 50건 이상인 Linux 게임 수",
    "linux_games_released_le_2015": "2015년까지 출시된 Linux 게임 수",
    "reviews_on_linux": "Linux 게임의 리뷰 수",
    "reviews_on_linux_pct": "Linux 게임의 리뷰 비율(%)",
    "positive_reviews_on_linux": "Linux 게임의 긍정 리뷰 수",
    "positive_rate_linux": "Linux 게임의 긍정 비율(0~1)",
    "positive_rate_non_linux": "Linux 미지원 게임의 긍정 비율(0~1)",
    "users_with_ge1_linux_positive": "Linux 게임 긍정 리뷰가 1건 이상인 사용자 수",
    "users_with_ge2_linux_positive": "Linux 게임 긍정 리뷰가 2건 이상인 사용자 수",
    "users_with_ge5_linux_positive": "Linux 게임 긍정 리뷰가 5건 이상인 사용자 수",
    "top1pct_linux_games_share_of_linux_positive": "상위 1% Linux 게임이 차지하는 긍정 리뷰 비중(0~1)",
    "top10pct_linux_games_share_of_linux_positive": "상위 10% Linux 게임이 차지하는 긍정 리뷰 비중(0~1)",
    "linux_positive_reviews_le_2015": "2015년까지의 Linux 게임 긍정 리뷰 수",
    "owned_rows_in_A_pct": "B 보유 기록 중 A에 있는 비율(%)",
    "owned_rows_played_in_A_pct": "B 플레이 기록 중 A에 있는 비율(%)",
    "playtime_weighted_in_A_pct": "플레이 시간 기준 A에 있는 비율(%)",
    "review_rows_in_A_pct": "B 리뷰 중 A에 있는 비율(%)",
    "positive_review_rows_in_A_pct": "B 긍정 리뷰 중 A에 있는 비율(%)",
    "positive_review_rows_in_A_linux_pct": "B 긍정 리뷰 중 A의 Linux 게임 비율(%)",
    "A_vs_C_title_match_pct": "A와 C의 제목 일치 비율(%)",
    "A_vs_C_release_date_match_pct": "A와 C의 출시일 일치 비율(%)",
    "A_vs_B_steam_games_common": "A와 B 게임 목록의 공통 게임 수",
    "A_vs_B_steam_games_title_match_pct": "A와 B 게임 목록의 제목 일치 비율(%)",
    "A_tags_pct": "A 태그가 있는 비율(%)",
    "A_desc_pct": "A 설명이 있는 비율(%)",
    "C_tags_pct": "C 태그가 있는 비율(%)",
    "C_genres_pct": "C 장르가 있는 비율(%)",
    "A_or_C_tags_pct": "A 또는 C 태그가 있는 비율(%)",
    "no_text_at_all": "글 정보가 전혀 없는 게임 수",
    "A_tags_per_game_median": "게임당 A 태그 수 중앙값",
    "review_users": "리뷰를 쓴 사용자 수",
    "review_users_with_item_profile": "그중 보유 목록이 있는 사용자 수",
    "reviewed_items_also_owned_pct": "리뷰한 게임이 보유 목록에도 있는 비율(%)",
    "positive_reviewed_items_also_owned_pct": "긍정 리뷰한 게임이 보유 목록에도 있는 비율(%)",
    "B_positive_rate": "B 리뷰의 긍정 비율(0~1)",
    "users_ge2_positive_in_A": "A 게임 긍정 리뷰가 2건 이상인 B 사용자 수",
    "of_which_ge1_linux": "그중 Linux 게임 긍정 리뷰가 있는 사용자 수",
    "linux_items_compared": "비교한 Linux 게임 수",
    "spearman_A_pos_vs_B_pos": "A 긍정 리뷰 수와 B 긍정 리뷰 수의 순위 상관",
    "spearman_A_pos_vs_B_owners": "A 긍정 리뷰 수와 B 보유자 수의 순위 상관",
    "spearman_A_pos_le2015_vs_B_pos": "2015년까지의 A 긍정 리뷰 수와 B 긍정 리뷰 수의 순위 상관",
    "top3_items_share": "상위 3개 게임이 차지하는 정답 비중(0~1)",
    "top10_items_share": "상위 10개 게임이 차지하는 정답 비중(0~1)",
    "targets_in_A_top50_linux_share": "A 인기 상위 50개 Linux 게임이 차지하는 정답 비중(0~1)",
    "longtail_users": "인기 상위 50개를 뺀 평가의 사용자 수",
    "longtail_target_pairs": "인기 상위 50개를 뺀 평가의 정답 쌍 수",
    "longtail_unique_items": "인기 상위 50개를 뺀 평가의 정답 게임 수",
    "pop_recall@10_longtail_targets_full_candidates": "인기순 Recall@10(상위 50개를 정답에서만 뺌)",
    "pop_recall@10_longtail_targets_head_excluded": "인기순 Recall@10(상위 50개를 정답과 후보에서 뺌)",
    "pop_ndcg@10_longtail_head_excluded": "인기순 NDCG@10(상위 50개를 정답과 후보에서 뺌)",
    "A_users_lt5_positive_pct": "긍정 리뷰가 5건 미만인 A 사용자 비율(%)",
    "B_test_users_with_0_cf_input": "협업 필터링 입력이 없는 B 사용자 수",
    "B_test_users_with_ge5_cf_input": "협업 필터링 입력이 5개 이상인 B 사용자 수",
    "B_test_users": "Linux 정답이 있는 B 사용자 수",
    "B_input_items_cf_share_pct": "입력 게임 중 협업 필터링에 쓸 수 있는 비율(%)",
    "B_targets_reachable_by_itemknn_pct": "정답 중 협업 필터링 후보에 있는 비율(%)",
    "games_with_exactly_1_target": "정답이 1쌍뿐인 게임 수",
    "games_with_ge5_targets": "정답이 5쌍 이상인 게임 수",
    "games_with_ge10_targets": "정답이 10쌍 이상인 게임 수",
    "ge5_games_share_candidates_pct": "정답 5쌍 이상 게임이 후보에서 차지하는 비율(%)",
    "targets": "정답: A의 Linux 긍정 게임",
    "targets_cf": "정답: 그중 협업 필터링 기준 충족 게임",
    "A linux (2024)": "A 자료의 Linux 지원 여부(2024년 기준)",
    "A_actual_reviews": "A 자료에 실제로 포함된 리뷰 수(표본 점검)",
    "A_description_length_chars_nonempty": "A 설명이 비어 있지 않은 게임의 설명 글자 수",
    "A_description_missing": "A 설명이 없는 게임 수",
    "A_description_missing_pct": "A 설명이 없는 게임 비율(대상 게임 중, %)",
    "A_linux_pos_rank": "A Linux 게임의 긍정 리뷰 순위",
    "A_or_C_available_pct": "A 또는 C 설명을 쓸 수 있는 비율(대상 게임 중, %)",
    "A_or_C_description_available": "A 또는 C 설명을 쓸 수 있는 게임 수",
    "A_or_C_description_available_pct": "A 또는 C 설명을 쓸 수 있는 비율(대상 게임 중, %)",
    "A_positive_pct": "이웃 부족 게임이 차지하는 A 긍정 리뷰 비율(전체 A 긍정 리뷰 중, %)",
    "A_positive_rate": "A 리뷰의 긍정 비율(A 리뷰 중, 0~1)",
    "A_positive_reviews": "A 긍정 리뷰 수",
    "A_release_date": "A 자료의 출시일",
    "A_reviewed_games": "A 리뷰가 있는 게임 수",
    "A_reviews": "A 리뷰 수",
    "A_tags_empty_games": "A 태그가 비어 있는 게임 수",
    "A_users": "A 리뷰 작성 사용자 수",
    "B_genres": "B 장르 정보",
    "B_input_pair_pct": "B 입력 쌍 비율(B 입력 쌍 중, %)",
    "B_input_pairs": "B 입력 사용자-게임 쌍 수",
    "B_owners": "B 보유자 수",
    "B_reviews": "B 리뷰 수",
    "B_reviews_dated": "날짜 정보가 있는 B 리뷰 수",
    "B_set": "B 항목 집합 이름",
    "B_specs": "B 상세 사양 정보",
    "B_target_games": "B 정답 게임 수",
    "B_target_pair_pct": "B 정답 쌍 비율(B 정답 쌍 중, %)",
    "B_target_pairs": "B 정답 사용자-게임 쌍 수",
    "B_title": "B 게임 제목",
    "C_about_the_game_can_fill_A_missing": "C 상세 설명으로 A 설명 결측을 보완할 수 있는 게임 수",
    "C_both_fields_can_fill_A_missing": "C 짧은 설명과 상세 설명이 모두 있어 A 결측을 보완할 수 있는 게임 수",
    "C_categories": "C 분류 정보",
    "C_either_field_can_fill_A_missing": "C 설명 중 하나 이상으로 A 결측을 보완할 수 있는 게임 수",
    "C_fillable": "C 자료로 보완할 수 있는 게임 수",
    "C_genres": "C 장르 정보",
    "C_linux_items": "C Linux 게임 항목 수",
    "C_short_description_can_fill_A_missing": "C 짧은 설명으로 A 설명 결측을 보완할 수 있는 게임 수",
    "C_side_empty_games": "C 쪽 태그가 비어 있는 게임 수",
    "C_title": "C 게임 제목",
    "False": "아니요",
    "MB": "파일 크기(MB)",
    "True": "예",
    "actual_bytes": "실제 파일 크기(바이트)",
    "actual_sha256": "실제 파일 SHA-256 값",
    "addresses": "보완하려는 데이터 공백",
    "all_distribution": "전체 값 분포",
    "app_id": "Steam 게임 ID(App ID)",
    "asserts_passed": "재현 확인 항목 통과 여부",
    "ascii_ratio_threshold": "영문·ASCII 비율 판정 기준",
    "baseline_ci_halfwidth_pp": "기준선 재현율 신뢰구간 반폭(퍼센트포인트)",
    "baseline_recall": "인기순 기준선 재현율",
    "bundle_id": "묶음 상품 ID(Bundle ID)",
    "candidate": "데이터 후보 이름",
    "candidate_games": "후보 게임 수",
    "candidates": "후보 수",
    "catalog_loss_users": "카탈로그에서 입력이 줄어든 사용자 수",
    "category": "범주",
    "ci_high": "신뢰구간 상한",
    "ci_low": "신뢰구간 하한",
    "ci_width": "신뢰구간 너비",
    "classification": "분류 결과",
    "classification_basis": "분류 근거",
    "cohort": "분석 대상 묶음",
    "cols": "열 수",
    "column": "원본 열 이름",
    "common_games": "두 자료에 공통으로 있는 게임 수",
    "comparison": "비교 대상",
    "cosine_similarity": "코사인 유사도",
    "cumulative_owner_pct": "기존 기타 항목의 전체 보유 기록 중 누적 비율(%)",
    "cumulative_owner_rows": "누적 보유자 행 수",
    "cutoff": "시간 분할 기준일",
    "dataset": "자료 이름",
    "date_max": "가장 늦은 날짜",
    "date_min": "가장 이른 날짜",
    "date_release": "출시일",
    "decision": "판단",
    "density": "행렬 밀도(비어 있지 않은 칸 비율, 0~1)",
    "description_ascii_character_share": "설명에서 ASCII 문자가 차지하는 비율(0~1)",
    "description_length": "설명 글자 수",
    "descriptions_with_any_non_ascii_pct": "ASCII가 아닌 문자가 하나라도 있는 설명 비율(비어 있지 않은 설명 중, %)",
    "dimension": "구분 기준",
    "document_frequency": "태그가 나타난 게임 수",
    "downloaded_at": "다운로드 날짜",
    "dtype": "자료형",
    "dup_key": "중복 확인 기준 열",
    "dup_rows": "중복 행 수",
    "duplicate_games_pct_of_nonempty_descriptions": "중복 설명이 있는 게임 비율(설명이 있는 게임 중, %)",
    "empty_union_games_jaccard_undefined": "두 태그 집합이 모두 비어 Jaccard 값을 계산할 수 없는 게임 수",
    "estimate": "추정값",
    "evaluated": "평가 대상 여부",
    "evaluation_users": "평가 사용자 수",
    "excess_games_from_exact_duplicates": "완전히 같은 설명 때문에 추가로 중복 집계된 게임 수",
    "exact_description_excerpt": "중복 설명 일부",
    "exact_duplicate_description_text_groups": "완전히 같은 설명 묶음 수",
    "excluded_head_games": "제외한 인기 상위 게임 수",
    "expected_bytes": "기대 파일 크기(바이트)",
    "expected_sha256": "기대 SHA-256 값",
    "field": "항목 이름",
    "file": "파일 이름",
    "free_games": "무료 게임 수",
    "games": "게임 수",
    "games_ge20_positive_neighbors": "긍정 사용자 20명 이상을 공유하는 이웃이 있는 게임 수",
    "games_ge20_positive_neighbors_pct": "긍정 사용자 20명 이상을 공유하는 이웃이 있는 게임 비율(게임 중, %)",
    "games_in_duplicate_descriptions": "완전히 같은 설명 묶음에 포함된 게임 수",
    "games_in_exact_duplicate_text_groups": "완전히 같은 설명 묶음에 포함된 게임 수",
    "games_pct": "리뷰가 있는 Linux 후보 게임 중 비율(%)",
    "games_with_tags": "태그가 있는 게임 수",
    "genres": "장르 목록",
    "group": "대상 묶음",
    "has_A_description": "A 설명 보유 여부",
    "has_C_description": "C 설명 보유 여부",
    "in_A_catalog": "A 카탈로그 포함 여부",
    "in_A_games": "A 게임 자료 포함 수",
    "in_A_games_pct": "B 집합 게임 중 A 게임 자료에 포함된 비율(%)",
    "in_A_linux": "A Linux 게임 자료 포함 수",
    "in_A_or_C": "A 또는 C 자료 포함 수",
    "in_A_rec_items": "A 추천 대상 자료 포함 수",
    "in_B_metadata": "B 메타데이터 포함 여부",
    "in_B_steam_games": "B Steam 게임 목록 포함 수",
    "in_C": "C 자료 포함 수",
    "in_C_metadata": "C 메타데이터 포함 여부",
    "in_C_pct": "B 집합 게임 중 C 자료에 포함된 비율(%)",
    "in_neither": "A와 C 어느 쪽에도 없는 수",
    "input": "입력 조건",
    "input_pairs": "입력 사용자-게임 쌍 수",
    "insufficient_A_positive_pct": "이웃이 부족한 게임의 A 긍정 리뷰 비율(전체 A 긍정 리뷰 중, %)",
    "insufficient_games": "필요한 이웃 수에 못 미치는 게임 수",
    "insufficient_ge5_neighbors": "긍정 이웃이 5명 미만인 게임 수",
    "insufficient_median_A_positive": "이웃 부족 게임의 A 긍정 리뷰 수 중앙값",
    "insufficient_pct": "이웃이 부족한 게임 비율(후보 게임 중, %)",
    "insufficient_target_games": "이웃이 부족한 정답 게임 수",
    "insufficient_target_pairs": "이웃이 부족한 정답 사용자-게임 쌍 수",
    "insufficient_target_pairs_pct": "이웃이 부족한 정답 쌍 비율(정답 쌍 중, %)",
    "interactions": "사용자-게임 상호작용 수",
    "interpretation": "질의 결과에 대한 사람의 검토 의견",
    "item_id": "게임 항목 ID",
    "item_name": "게임 이름",
    "items": "게임 항목 수",
    "iterations": "반복 횟수",
    "jaccard_nonempty_union": "합집합이 비어 있지 않은 게임의 Jaccard 유사도(0~1)",
    "kaggle_last_updated": "Kaggle 자료의 마지막 갱신일",
    "kaggle_ref": "Kaggle 자료 주소",
    "kaggle_version": "Kaggle 자료 버전",
    "kcore_games": "k-core에 남은 게임 수",
    "kcore_interactions": "k-core에 남은 사용자-게임 상호작용 수",
    "kcore_linux_games": "k-core에 남은 Linux 게임 수",
    "kcore_users": "k-core에 남은 사용자 수",
    "license": "이용 허가 조건",
    "linux": "Linux 지원 여부",
    "linux_candidate_pct": "Linux 후보 비율(전체 Linux 후보 중, %)",
    "linux_candidates": "Linux 후보 게임 수",
    "linux_games": "Linux 게임 수",
    "linux_input_pairs_pct": "Linux 입력 쌍 비율(입력 쌍 중, %)",
    "linux_interactions": "Linux 사용자-게임 상호작용 수",
    "linux_items": "Linux 게임 항목 수",
    "linux_reviewed_games": "리뷰가 있는 Linux 게임 수",
    "local_path": "로컬 파일 경로",
    "loss_rate_bin": "입력 손실 비율 구간",
    "matrix_density_pct": "행렬 밀도(%, 전체 칸 중 값이 있는 칸)",
    "matrix_nonzero_values": "행렬에서 0이 아닌 값의 수",
    "matrix_sparsity_pct": "행렬 희소도(%, 값이 없는 칸)",
    "mean_jaccard_nonempty_union": "합집합이 비어 있지 않은 게임들의 Jaccard 평균",
    "median_A_positive_reviews": "게임별 A 긍정 리뷰 수 중앙값",
    "median_release_year": "출시 연도 중앙값",
    "metric": "평가 지표",
    "min_input_items": "사용자당 필요한 최소 입력 게임 수",
    "min_item_pos": "게임별 최소 긍정 리뷰 수 기준",
    "min_user_pos": "사용자별 최소 긍정 리뷰 수 기준",
    "minimum_cooccurrence_users": "이웃으로 인정하는 최소 공동 긍정 사용자 수",
    "missing_pct": "결측 비율(대상 행 중, %)",
    "n": "항목 수",
    "n_B_tags": "B 태그 수",
    "n_C_tags": "C 태그 수",
    "n_null": "결측값 수",
    "name": "게임 이름",
    "nearest_similarity_max": "가장 가까운 게임과의 유사도 최댓값",
    "nearest_similarity_mean": "가장 가까운 게임과의 유사도 평균",
    "nearest_similarity_p25": "가장 가까운 게임 유사도의 25백분위수",
    "nearest_similarity_p50": "가장 가까운 게임 유사도의 중앙값",
    "nearest_similarity_p75": "가장 가까운 게임 유사도의 75백분위수",
    "nearest_similarity_p90": "가장 가까운 게임 유사도의 90백분위수",
    "nearest_similarity_p99": "가장 가까운 게임 유사도의 99백분위수",
    "neighbor_app_id": "이웃 게임 ID(App ID)",
    "neighbor_title": "이웃 게임 제목",
    "neighbors_ge1": "긍정 사용자 1명 이상을 공유하는 이웃 수",
    "neighbors_ge10": "긍정 사용자 10명 이상을 공유하는 이웃 수",
    "neighbors_ge10_median": "긍정 사용자 10명 이상을 공유하는 이웃 수 중앙값",
    "neighbors_ge1_median": "긍정 사용자 1명 이상을 공유하는 이웃 수 중앙값",
    "neighbors_ge20": "긍정 사용자 20명 이상을 공유하는 이웃 수",
    "neighbors_ge20_median": "긍정 사용자 20명 이상을 공유하는 이웃 수 중앙값",
    "neighbors_ge5": "긍정 사용자 5명 이상을 공유하는 이웃 수",
    "neighbors_ge5_median": "긍정 사용자 5명 이상을 공유하는 이웃 수 중앙값",
    "neighbors_ge5_p10": "긍정 사용자 5명 이상을 공유하는 이웃 수의 10백분위수",
    "neighbors_ge5_p90": "긍정 사용자 5명 이상을 공유하는 이웃 수의 90백분위수",
    "new_target_game_pct": "신규 정답 게임 비율(정답 게임 중, %)",
    "new_target_games": "신규 정답 게임 수",
    "new_target_pair_pct": "신규 정답 쌍 비율(정답 쌍 중, %)",
    "new_target_pairs": "신규 정답 사용자-게임 쌍 수",
    "no_positive_similarity_neighbor_games": "긍정 유사도 이웃이 없는 게임 수",
    "no_positive_similarity_neighbor_pct": "긍정 유사도 이웃이 없는 게임 비율(게임 중, %)",
    "no_tags_or_genres_games": "태그와 장르가 모두 없는 게임 수",
    "nonempty_descriptions": "설명이 비어 있지 않은 게임 수",
    "nonempty_length_median": "비어 있지 않은 설명 글자 수 중앙값",
    "nonempty_union_games": "두 태그 집합의 합집합이 비어 있지 않은 게임 수",
    "nonzero_distribution": "0이 아닌 값의 분포",
    "nonzero_vector_games_pct": "텍스트 벡터가 비어 있지 않은 게임 비율(전체 게임 중, %)",
    "null_pct": "결측 비율(전체 행 중, %)",
    "observed_to_store_reviews_ratio": "A에서 관측된 리뷰 수 ÷ 원자료 표시 리뷰 수(비율)",
    "original_reason": "원래 기록된 사유",
    "owner_row_pct": "기존 기타 항목의 전체 보유 기록 중 비율(%)",
    "owner_rows": "보유 기록 행 수",
    "owners": "보유자 수",
    "partial_download": "일부만 내려받은 파일 여부",
    "per_description_ascii_character_ratio": "게임별 설명의 ASCII 문자 비율(0~1)",
    "phrase": "반복해서 나타난 다섯 단어 구절",
    "pop_ndcg10": "인기순 기준선 NDCG@10",
    "pop_recall10": "인기순 기준선 Recall@10",
    "popularity_band": "긍정 리뷰 수 기준 인기 구간",
    "popularity_stratum": "긍정 리뷰 인기도 층",
    "population": "평가 대상 범위",
    "positive_neighbors_p25": "긍정 이웃 수의 25백분위수",
    "positive_neighbors_p50": "긍정 이웃 수의 중앙값",
    "positive_neighbors_p75": "긍정 이웃 수의 75백분위수",
    "positive_neighbors_p90": "긍정 이웃 수의 90백분위수",
    "positive_neighbors_p99": "긍정 이웃 수의 99백분위수",
    "positive_pairs": "긍정 사용자-게임 쌍 수",
    "positive_ratio": "긍정 리뷰 비율(전체 리뷰 중, %)",
    "positive_review_threshold": "게임에 필요한 최소 긍정 리뷰 수",
    "positive_reviews": "긍정 리뷰 수",
    "positive_reviews_per_game": "게임당 긍정 리뷰 수",
    "positive_rows": "긍정으로 기록된 행 수",
    "positive_similarity": "긍정 사용자 기반 이웃 여부",
    "positive_users_in_source": "해당 기준에서 게임에 긍정한 사용자 수",
    "price_final": "최종 표시 가격(원자료 표기 단위)",
    "prior_positive_requirement": "기준일 전 필요한 최소 긍정 리뷰 수",
    "problem": "확인한 문제",
    "published_size": "공개 자료에 표시된 크기",
    "query_app_id": "질의 게임 ID(App ID)",
    "query_positive_reviews": "질의 게임의 긍정 리뷰 수",
    "query_positive_similarity_neighbors": "긍정 사용자를 공유하는 질의 이웃 수",
    "query_stratum": "질의 게임의 인기도 층",
    "query_title": "질의 게임 제목",
    "query_zero_vector": "질의 게임의 텍스트 벡터가 비어 있는지 여부",
    "rank": "순위",
    "reachable_pairs_pct": "연결 경로가 있는 정답 쌍 비율(정답 쌍 중, %)",
    "reachable_target_pairs": "연결 경로가 있는 정답 사용자-게임 쌍 수",
    "reachable_target_games": "연결 경로가 있는 정답 게임 수",
    "reason_guess": "이름과 메타데이터로 추정한 분류",
    "remaining_all_items": "기준 적용 후 전체 게임 항목 수",
    "resolution": "처리 방법",
    "result": "처리 결과",
    "review_space_loss_users": "리뷰 자료 범위에서 입력이 줄어든 사용자 수",
    "risk": "확인해야 할 위험과 한계",
    "rows": "행 수",
    "schema_source": "자료 구조 정보의 출처",
    "seed": "무작위 추출 기준값",
    "selection_order": "질의 선택 순서",
    "selection_rule": "질의 선택 기준",
    "sha256_match": "SHA-256 값 일치 여부",
    "share": "전체 정답 쌍 중 비중(0~1)",
    "share_of_all_candidates_pct": "전체 후보 중 비율(%)",
    "share_of_games_pct": "해당 대상 묶음의 게임 중 비율(%)",
    "share_of_other_owner_rows_pct": "기존 기타 항목의 전체 보유 기록 중 비율(%)",
    "share_of_target_games_pct": "정답 게임 수 비율(전체 정답 게임 중, %)",
    "size_match": "파일 크기 일치 여부",
    "source": "비교 기준 또는 출처",
    "sparsity_pct": "행렬 희소도(%, 값이 없는 칸)",
    "steam_deck": "Steam Deck 지원 여부",
    "status": "확인 상태",
    "table": "표 이름",
    "tag": "태그 이름",
    "tag_source": "사용한 태그·장르 정보의 출처",
    "tagged_games_pct": "태그가 있는 게임 비율(해당 대상 묶음의 게임 중, %)",
    "tags_only_top5": "태그만 사용했을 때 가장 가까운 게임 5개",
    "tags_per_game": "게임당 태그 수",
    "tags_per_game_median": "게임당 태그 수 중앙값",
    "tags_plus_description_top5": "태그와 설명을 함께 사용했을 때 가장 가까운 게임 5개",
    "tags_seen_in_fewer_than_5_games": "5개 미만 게임에만 나타난 태그 수",
    "tags_seen_in_fewer_than_5_games_pct": "5개 미만 게임에만 나타난 태그 비율(전체 태그 중, %)",
    "target": "정답 조건",
    "target_games": "정답 게임 수",
    "target_pairs": "정답 사용자-게임 쌍 수",
    "target_pairs_pct": "정답 쌍 비율(비교 대상 전체 정답 쌍 중, %)",
    "title": "게임 제목",
    "title_pattern_games": "제목 패턴에 해당하는 게임 수",
    "title_pattern_pct": "제목 패턴에 해당하는 비율(대상 게임 중, %)",
    "top_owned_games": "보유자 수 상위 게임 수",
    "unique_input_games": "중복을 제외한 입력 게임 수",
    "unique_tags": "서로 다른 태그 수",
    "unique_target_items": "중복을 제외한 정답 게임 수",
    "unlabeled_games": "평가 정답 라벨이 없는 게임 수",
    "unlabeled_pct": "정답 라벨이 없는 게임 비율(후보 게임 중, %)",
    "usable_input_games": "텍스트 벡터를 만들 수 있는 입력 게임 수",
    "usable_input_pairs": "텍스트 벡터를 만들 수 있는 입력 쌍 수",
    "usable_input_pairs_pct": "텍스트 벡터를 만들 수 있는 입력 쌍 비율(전체 입력 쌍 중, %)",
    "user_positive_label": "사용자별 긍정 평가 라벨 제공 여부",
    "user_reviews": "사용자 리뷰 수",
    "users": "사용자 수",
    "users_retaining_any_target": "정답이 하나 이상 남은 사용자 수",
    "users_retaining_both_conditions": "입력 5개 이상과 정답 1개 이상을 모두 충족하는 사용자 수",
    "users_retaining_ge5_inputs": "입력이 5개 이상 남은 사용자 수",
    "users_with_all_reachable_targets": "정답 모두에 연결 경로가 있는 사용자 수",
    "users_with_all_reachable_targets_pct": "정답 모두에 연결 경로가 있는 사용자 비율(사용자 중, %)",
    "users_with_any_reachable_target": "하나 이상 정답에 연결 경로가 있는 사용자 수",
    "users_with_any_reachable_target_pct": "하나 이상 정답에 연결 경로가 있는 사용자 비율(사용자 중, %)",
    "users_with_fewer_than_5_usable_inputs": "사용 가능한 입력이 5개 미만인 사용자 수",
    "users_with_zero_usable_inputs": "사용 가능한 입력이 하나도 없는 사용자 수",
    "variant": "텍스트 입력 방식",
    "version_period": "버전 또는 자료 수집 기간",
    "vocabulary": "텍스트 분석에 사용한 단어 수",
    "vocabulary_size": "텍스트 분석에 사용한 단어 수",
    "with_B_or_C_metadata_games": "B 또는 C 메타데이터가 있는 게임 수",
    "year": "연도",
    "zero_neighbor_games": "긍정 이웃이 없는 게임 수",
    "zero_neighbor_max_A_positive": "긍정 이웃이 없는 게임의 A 긍정 리뷰 최대값",
    "zero_neighbor_pct": "긍정 이웃이 없는 게임 비율(후보 게임 중, %)",
    "zero_pct": "값이 0인 비율(대상 행 중, %)",
    "zero_rows": "값이 0인 행 수",
    "zero_vector_games": "텍스트 벡터가 비어 있는 게임 수",
    "zero_vector_games_pct": "텍스트 벡터가 비어 있는 게임 비율(게임 중, %)",
    "zero_vector_input_games": "텍스트 벡터가 비어 있는 입력 게임 수",
    "zero_vector_input_games_pct": "텍스트 벡터가 비어 있는 입력 게임 비율(입력 게임 중, %)",
    "zero_vectors_all": "전체 후보 중 텍스트 벡터가 비어 있는 게임 수",
    "zero_vectors_reviewed": "리뷰가 있는 후보 중 텍스트 벡터가 비어 있는 게임 수",
    "question": "검토 질문",
    "scope": "판단이 적용되는 범위",
    "released_after_2015_games": "2016년 이후 출시된 게임 수",
    "released_after_2015_pct": "2016년 이후 출시된 게임 비율(대상 게임 중, %)",
    "rare_tags_lt5_games_pct": "5개 미만 게임에만 있는 희귀 태그 비율(전체 태그 중, %)",
    "low_ascii_descriptions": "ASCII 문자 비율이 기준보다 낮은 설명 수",
    "non_english_proxy_pct": "영어가 아닐 가능성이 있는 설명 비율(ASCII 기준 추정, %)",
    "repeated_5word_phrases_in_ge5_games": "5개 이상 게임에 반복된 다섯 단어 구절 수",
}


# Only exact, known category or marker strings are translated. Unrecognized
# strings (including game names, filenames, URLs, and source prose) pass through.
_VALUE_LABELS = {
    "target_games": "정답 기록이 있는 후보 게임",
    "unlabeled_candidates": "정답 기록이 없는 후보 게임",
    "A": "A 자료",
    "B": "B 자료",
    "C": "C 자료",
    "all": "전체 대상",
    "all_candidates": "전체 후보 게임",
    "all_positive": "전체 긍정 리뷰 기준",
    "kcore_5_10": "사용자 5건·게임 10건 이상으로 거른 기준",
    "tags_only": "태그만 사용",
    "tags_plus_description": "태그와 설명을 함께 사용",
    "tags_plus_description_top5": "태그와 설명을 함께 사용한 상위 5개",
    "longtail_full_candidates": "인기 상위 50개를 정답에서 뺀 평가(후보는 전체)",
    "longtail_head50_excluded_candidates": "인기 상위 50개를 정답과 후보에서 뺀 평가",
    "longtail_head50_excluded_targets": "인기 상위 50개를 정답에서 뺀 평가",
    "targets_and_candidates_without_absent_games": "정답과 후보에서 관측되지 않은 게임을 제외한 범위",
    "B_owned_items": "B 보유 게임 항목",
    "B_pos_review_items": "B 긍정 리뷰 게임 항목",
    "B_review_items": "B 리뷰 게임 항목",
    "B_steam_games": "B Steam 게임 항목",
    "targets": "정답 게임",
    "targets_cf": "협업 필터링 기준 정답 게임",
    "input_owned": "보유 기록 기반 입력",
    "input_owned_cf": "보유 기록 기반 협업 필터링 입력",
    "input_played": "플레이 기록 기반 입력",
    "all_games": "전체 게임",
    "linux": "Linux 게임",
    "non_linux": "Linux 미지원 게임",
    "all_positive_reviews": "전체 긍정 리뷰 기준",
    "Linux all": "전체 Linux 게임",
    "Linux with any A review": "A 리뷰가 하나 이상 있는 Linux 게임",
    "A non-Linux": "A 자료에서 Linux를 지원하지 않는 게임",
    "Linux popularity: lowest 20%": "Linux 게임 중 긍정 리뷰 수 하위 20%",
    "Linux popularity: 20–40%": "Linux 게임 중 긍정 리뷰 수 20~40% 구간",
    "Linux popularity: 40–60%": "Linux 게임 중 긍정 리뷰 수 40~60% 구간",
    "Linux popularity: 60–80%": "Linux 게임 중 긍정 리뷰 수 60~80% 구간",
    "Linux popularity: highest 20%": "Linux 게임 중 긍정 리뷰 수 상위 20%",
    "All common A/C App IDs": "A와 C에 공통으로 있는 전체 게임 ID",
    "Linux common A/C App IDs": "A와 C에 공통으로 있는 Linux 게임 ID",
    "A tags vs C tags": "A 태그와 C 태그 비교",
    "A tags vs C tags plus genres": "A 태그와 C 태그·장르 비교",
    "dlc (metadata)": "메타데이터에 DLC로 표시된 항목",
    "dlc/pack (name inference)": "이름으로 DLC·묶음 상품으로 추정한 항목",
    "game-like metadata; missingness unresolved": "게임 메타데이터가 있으나 A 누락 사유는 확인되지 않은 항목",
    "metadata present; type unresolved": "메타데이터는 있으나 유형이 확인되지 않은 항목",
    "server/SDK/tool (name inference)": "이름으로 서버·개발 도구로 추정한 항목",
    "software/tool (metadata inference)": "메타데이터로 소프트웨어·도구로 추정한 항목",
    "soundtrack/video (name inference)": "이름으로 사운드트랙·영상으로 추정한 항목",
    "test/beta/demo (name inference)": "이름으로 테스트·베타·체험판으로 추정한 항목",
    "unresolved_no_game_metadata": "게임 메타데이터가 없어 유형이 확인되지 않은 항목",
    "App ID exists in B/C": "B 또는 C에 게임 ID가 있음",
    "B specs: Downloadable Content": "B 사양에 내려받을 콘텐츠로 표시됨",
    "B/C genre metadata": "B 또는 C 장르 메타데이터",
    "name pattern": "이름 패턴에 따른 추정",
    "no category evidence": "분류를 뒷받침할 정보 없음",
    "software-like genre": "소프트웨어 성격의 장르 정보",
    "reviewed candidates": "리뷰가 있는 후보 게임",
    "Linux items without A reviews": "A 리뷰가 없는 Linux 게임 항목",
    "C tags only": "C 태그만 사용",
    "C genres only": "C 장르만 사용",
    "A tags": "A 태그 사용",
    "no tags/genres": "태그와 장르 없음",
    "match": "일치",
    "funny": "재미있음 표시 수",
    "helpful": "도움이 됨 표시 수",
    "Recall@10": "Recall@10(상위 10개 재현율)",
    "NDCG@10": "NDCG@10(상위 10개 순위 품질)",
    "0": "0건",
    "1-9": "1~9건",
    "10-49": "10~49건",
    "50-199": "50~199건",
    "200-999": "200~999건",
    "1000+": "1,000건 이상",
    "0-10%": "0~10%",
    "10-20%": "10~20%",
    "20-30%": "20~30%",
    "30-40%": "30~40%",
    "40-50%": "40~50%",
    "50-60%": "50~60%",
    "60-70%": "60~70%",
    "70-80%": "70~80%",
    "80-90%": "80~90%",
    "90-100%": "90~100%",
    "dlc/pack": "DLC·묶음 상품",
    "server/SDK/tool": "서버·개발 도구",
    "soundtrack/video": "사운드트랙·영상",
    "test/beta/demo": "테스트·베타·체험판",
    "other": "기타",
    "action": "액션",
    "adventure": "어드벤처",
    "casual": "캐주얼",
    "early access": "앞서 해보기",
    "free to play": "무료 플레이",
    "indie": "인디",
    "racing": "레이싱",
    "rpg": "롤플레잉",
    "simulation": "시뮬레이션",
    "sports": "스포츠",
    "strategy": "전략",
    "unknown": "미분류",
    "genres": "장르",
    "popularity_band": "긍정 리뷰 수 기준 인기 구간",
    "year": "연도",
}

_CATEGORY_COLUMNS = {
    "group",
    "A linux (2024)",
    "B_set",
    "category",
    "classification",
    "classification_basis",
    "cohort",
    "comparison",
    "dataset",
    "dimension",
    "field",
    "input",
    "license",
    "loss_rate_bin",
    "metric",
    "original_reason",
    "popularity_band",
    "popularity_stratum",
    "population",
    "reason_guess",
    "source",
    "status",
    "tag_source",
    "target",
    "variant",
}

_BOOLEAN_COLUMNS = {
    "asserts_passed",
    "evaluated",
    "has_A_description",
    "has_C_description",
    "in_A_catalog",
    "in_B_metadata",
    "in_C_metadata",
    "linux",
    "partial_download",
    "positive_similarity",
    "query_zero_vector",
    "sha256_match",
    "size_match",
    "steam_deck",
    "user_positive_label",
}

_TEXT_COLUMNS = {
    "B_title",
    "C_title",
    "exact_description_excerpt",
    "interpretation",
    "item_name",
    "local_path",
    "name",
    "neighbor_title",
    "phrase",
    "query_title",
    "risk",
    "title",
}


def column_label(name: Any) -> Any:
    """Return the explicit Korean display label for a report column name."""
    if isinstance(name, bool):
        return "예" if name else "아니요"
    if type(name).__name__ == "bool_" and type(name).__module__.startswith("numpy"):
        return "예" if bool(name) else "아니요"
    if not isinstance(name, str):
        return name
    return _COLUMN_LABELS.get(name, name)


def display_value(value: Any, *, column: Any = None) -> Any:
    """Translate known markers and category values while preserving source text.

    Pass ``column`` when the value comes from a table cell. That lets the
    helper translate structured category values while leaving game titles,
    URLs, paths, and free-text fields untouched.
    """
    if isinstance(value, bool):
        return "예" if value else "아니요"
    if type(value).__name__ == "bool_" and type(value).__module__.startswith("numpy"):
        return "예" if bool(value) else "아니요"
    if isinstance(value, str):
        if column in _TEXT_COLUMNS:
            return value
        if column in _BOOLEAN_COLUMNS and value in ("True", "False"):
            return "예" if value == "True" else "아니요"
        if value in {
            "all_positive",
            "all_positive_reviews",
            "kcore_5_10",
            "longtail_full_candidates",
            "longtail_head50_excluded_candidates",
            "longtail_head50_excluded_targets",
            "tags_only",
            "tags_plus_description",
            "targets_and_candidates_without_absent_games",
        }:
            return _VALUE_LABELS.get(value, value)
        if column in _CATEGORY_COLUMNS:
            return _VALUE_LABELS.get(value, value)
    return value


def friendly_frame(frame: Any) -> Any:
    """Return a presentation copy of a pandas DataFrame or Series."""
    if isinstance(frame, pd.DataFrame):
        result = frame.copy(deep=True)
        source_columns = list(result.columns)
        result.columns = [column_label(column) for column in source_columns]
        if result.index.name is not None:
            source_index_name = result.index.name
            result.index.name = column_label(result.index.name)
            result.index = result.index.map(lambda value: display_value(value, column=source_index_name))
        if isinstance(result.index, pd.MultiIndex):
            result.index = result.index.set_names([column_label(name) for name in result.index.names])
        # map returns a new object here; the caller's frame and values are untouched.
        # Positions keep this safe when two source columns share one Korean label.
        for position, source_column in enumerate(source_columns):
            result.isetitem(
                position,
                frame.iloc[:, position].map(
                    lambda value, source_column=source_column: display_value(value, column=source_column)
                ),
            )
        return result
    if isinstance(frame, pd.Series):
        result = frame.copy(deep=True)
        result.name = column_label(result.name)
        if result.index.name is not None:
            source_index_name = result.index.name
            result.index.name = column_label(result.index.name)
            result.index = result.index.map(lambda value: display_value(value, column=source_index_name))
        return result.map(lambda value: display_value(value, column=frame.name))
    return frame


def _format_scalar(value: Any) -> Any:
    """Format one mixed-table value so counts stay whole numbers next to ratios."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.4f}"
    return value


def _mapping_rows(mapping: dict[Any, Any], prefix: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, item in mapping.items():
        key_label = str(column_label(key))
        path = (*prefix, key_label)
        if isinstance(item, dict):
            rows.extend(_mapping_rows(item, path))
        else:
            rows.append({"항목": " / ".join(path), "값": _format_scalar(display_value(item, column=key))})
    return rows


def _friendly_object(obj: Any) -> Any:
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return friendly_frame(obj)
    if isinstance(obj, dict):
        return pd.DataFrame(_mapping_rows(obj))
    if isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
        # DataFrame copies the row dictionaries before any display conversion.
        return friendly_frame(pd.DataFrame(obj))
    return display_value(obj)


def display_readable(*objects: Any, **kwargs: Any) -> None:
    """Display copies of the objects with Korean labels in a notebook.

    Returning nothing keeps a trailing call from echoing the copies a second time.
    """
    from IPython.display import display as ipython_display

    friendly = tuple(_friendly_object(obj) for obj in objects)
    # Long Korean labels must not be cut off; very long free text still is.
    with pd.option_context("display.max_colwidth", 120):
        ipython_display(*friendly, **kwargs)


def setup_korean_font() -> str | None:
    """Choose an installed Korean-capable Matplotlib font when available."""
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
    except ImportError:
        return None

    for family in (
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "NanumGothic",
        "Noto Sans CJK JP",
        "AppleGothic",
    ):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except (ValueError, RuntimeError):
            continue
        mpl.rcParams["font.family"] = family
        mpl.rcParams["axes.unicode_minus"] = False
        return family

    mpl.rcParams["axes.unicode_minus"] = False
    return None


def _table_description(filename: str) -> str:
    """Short Korean explanation for each exported table file."""
    descriptions = {
        "00_sources.csv": "분석에 사용한 원자료의 버전, 이용 조건, 내려받은 파일과 날짜를 정리합니다.",
        "01_schema_nulls.csv": "각 원자료 열의 자료형과 결측값 수·비율을 보여 줍니다.",
        "01_structure.csv": "각 자료의 행·열 수, 사용자·게임 수, 중복과 날짜 범위를 요약합니다.",
        "02_A_positive_kcore.csv": "A 긍정 리뷰에서 사용자·게임 최소 활동 기준을 적용한 결과입니다.",
        "02_reviews_by_year.csv": "연도별 A 리뷰와 긍정 비율, 날짜가 있는 B 리뷰 수입니다.",
        "03_linux_flag_A_vs_C.csv": "A와 C 자료에서 기록된 Linux 지원 여부가 어떻게 겹치는지 보여 줍니다.",
        "04_B_owned_not_in_A_reasons.csv": "B에서 보유 기록은 있지만 A 게임 목록에는 없는 항목을 추정 사유별로 셉니다.",
        "04_B_owned_not_in_A_top30.csv": "A에 없는 B 보유 항목 중 보유자가 많은 상위 30개를 보여 줍니다.",
        "04_id_overlap.csv": "B의 여러 게임 집합이 A·C 자료에 얼마나 포함되는지 비교합니다.",
        "05_external_test_eligibility.csv": "입력과 정답 조건을 바꾸었을 때 외부 평가에 쓸 수 있는 규모를 비교합니다.",
        "05_top_target_items.csv": "정답 사용자-게임 쌍이 많은 상위 게임과 그 비중을 보여 줍니다.",
        "06_cli_file_verification.csv": "공개 명령행 도구의 파일 크기와 SHA-256 값이 기대값과 일치하는지 기록합니다.",
        "07_A_internal_time_split.csv": "기준일 전 입력과 기준일 후 Linux 정답으로 만든 시간 분할 규모입니다.",
        "07_candidate_labels.csv": "후보 게임별 제목, 출시일, 긍정 리뷰, 정답 여부, 인기 구간과 장르입니다.",
        "07_candidate_target_comparison.csv": "장르·인기 구간·연도별 후보 수와 정답·라벨 누락 규모를 비교합니다.",
        "07_longtail_sensitivity.csv": "인기 상위 게임을 제외할 때 평가 사용자와 정답 규모가 어떻게 바뀌는지 보여 줍니다.",
        "07_popularity_bootstrap_ci.csv": "사용자 자료를 반복해서 뽑아 계산한 인기순 기준 지표의 불확실성 범위입니다.",
        "07_pre2016_kcore.csv": "2015년까지의 A 긍정 리뷰에 사용자·게임 최소 활동 기준을 적용한 결과입니다.",
        "07_reproduction.csv": "주요 A 자료, 평가셋과 인기순 기준선 수치의 재현 확인 결과입니다.",
        "08_B_target_cooccurrence_reachability.csv": "입력 게임과 정답 게임 사이에 긍정 공동 출현 경로가 있는지 요약합니다.",
        "08_positive_cooccurrence_by_popularity.csv": "인기 구간별 긍정 공동 출현 이웃 수와 이웃 부족 비율입니다.",
        "08_positive_cooccurrence_neighbors.csv": "각 게임과 긍정 사용자를 공유하는 이웃 수를 기준별로 기록합니다.",
        "09_cf_item_threshold_sensitivity.csv": "게임별 최소 긍정 리뷰 기준을 바꿀 때 후보·입력·정답이 얼마나 남는지 비교합니다.",
        "10_C_A_C_tag_jaccard.csv": "A와 C 자료에 공통인 게임의 태그 집합 유사도를 비교합니다.",
        "10_C_A_description_summary.csv": "설명 결측, 길이, 중복 및 비영어 설명의 대략적 비율을 요약합니다.",
        "10_C_A_exact_duplicate_descriptions.csv": "완전히 같은 설명 문구가 있는 게임 묶음의 예를 보여 줍니다.",
        "10_C_A_tag_summary.csv": "게임별 태그 수, 태그 보유율과 드문 태그 분포를 요약합니다.",
        "10_C_A_tag_top20.csv": "각 대상 묶음에서 자주 나타나는 상위 태그를 보여 줍니다.",
        "10_C_tfidf_queries.csv": "정성 검토를 위해 미리 고른 질의 게임과 선택 기준입니다.",
        "11_C_qualitative_review.csv": "질의 게임마다 태그만 쓴 경우와 설명을 더한 경우의 상위 결과를 검토합니다.",
        "11_C_repeated_phrases.csv": "여러 게임 설명에 반복된 다섯 단어 구절을 요약합니다.",
        "11_C_tfidf_input_coverage.csv": "입력 게임의 텍스트 벡터 생성 가능 여부와 사용자별 입력 수를 비교합니다.",
        "11_C_tfidf_query_top5_neighbors.csv": "단어 중요도로 비교한 각 질의 게임의 상위 5개 텍스트 유사도 이웃입니다.",
        "11_C_tfidf_summary.csv": "태그·설명을 단어 중요도로 바꾼 뒤 게임 간 유사도 분포를 요약합니다.",
        "12_B_input_loss_distribution.csv": "카탈로그나 리뷰 범위 때문에 입력을 잃는 사용자 규모를 손실률별로 나눕니다.",
        "12_B_missing_other_classification_summary.csv": "기존 other 분류 항목을 재검토한 결과를 유형별로 요약합니다.",
        "12_B_missing_other_reclassification.csv": "기존 other 항목마다 B·C 메타데이터와 추정 분류 근거를 기록합니다.",
        "12_B_missing_other_top_coverage.csv": "보유자가 많은 기존 other 항목에서 메타데이터가 있는 정도를 보여 줍니다.",
        "12_C_only_linux_B_positive_games.csv": "C에는 있고 A에는 없으며 B에서 긍정 기록이 있는 Linux 게임을 나열합니다.",
        "13_bundle_linux.csv": "묶음 상품별 전체 게임과 Linux 지원 게임 수입니다.",
        "13_game_metadata_distributions.csv": "전체·Linux·비Linux 게임의 가격, 무료 여부, 긍정 비율과 리뷰 수 분포입니다.",
        "13_helpful_funny.csv": "도움·재미 표시 수의 0값과 분포를 요약합니다.",
        "13_linux_steam_deck_crosstab.csv": "Linux와 Steam Deck 지원 표시의 교차표입니다.",
        "13_review_sampling_proxy.csv": "게임별 표시 리뷰 수와 A에서 관측된 리뷰 수의 비율을 비교합니다.",
        "14_external_dataset_candidates.csv": "추가 외부 평가 자료 후보와 공개된 범위·이용 조건·위험을 정리합니다.",
        "15_B_team_fortress_dota_entries.csv": "B에 포함된 Team Fortress·Dota 항목과 A 카탈로그 포함 여부입니다.",
        "15_baseline_uncertainty.csv": "기준선 Recall@10과 사용자 bootstrap 신뢰구간의 반폭입니다.",
        "15_baseline_without_absent_games.csv": "관측되지 않은 게임을 제외했을 때 인기순 기준 성능의 불확실성 범위입니다.",
        "15_candidates_without_A_tags.csv": "A 태그가 없거나 다른 태그 출처를 쓰는 후보와 정답·입력 규모입니다.",
        "15_cf_insufficient_weighted.csv": "협업 필터링 이웃 부족 게임이 긍정 리뷰·정답에 차지하는 비율입니다.",
        "15_product_type_title_pattern.csv": "상품 유형을 제목 패턴으로 추정한 규모와 태그·장르 결손을 요약합니다.",
        "15_tag_source_weighted_impact.csv": "태그·장르 정보 출처별 후보와 긍정 리뷰·정답·입력 비중입니다.",
        "15_target_games_absent_from_B_ownership.csv": "B 보유 기록에 나타나지 않는 정답 게임을 보여 줍니다.",
        "15_target_games_released_after_2015.csv": "2016년 이후 출시된 정답 게임과 정답 쌍 수입니다.",
    }
    return descriptions.get(filename, "이 표 파일의 열별 설명은 아래 항목을 참고하세요.")


def write_table_guide(eda_dir: str | Path) -> Path:
    """Write a Korean guide for every CSV under ``EDA/reports/tables``."""
    root = Path(eda_dir)
    if root.name == "tables":
        tables_dir = root
    elif (root / "reports" / "tables").is_dir():
        tables_dir = root / "reports" / "tables"
    else:
        tables_dir = root / "tables"
    output = tables_dir / "README.md"

    lines = [
        "# 분석 표 안내",
        "",
        "이 안내서는 분석에서 내보낸 CSV 표의 용도와 열 이름을 설명합니다. 표의 원본 열 이름과 수치는 그대로 두고, 화면에 표시할 때만 쉬운 한국어 이름을 붙입니다.",
        "",
        "## 읽는 법",
        "",
        "- **정답**은 평가에서 맞혀야 하는 사용자별 긍정 게임입니다. 이 분석에서는 한 사용자의 한 게임을 하나의 정답으로 셉니다.",
        "- **쌍**은 사용자와 게임을 묶은 한 건입니다. 같은 게임이라도 사용자가 다르면 서로 다른 쌍입니다.",
        "- **입력**은 추천을 만들 때 사용자에게 이미 알려져 있다고 보는 게임입니다.",
        "- **CF**는 협업 필터링입니다. 비슷한 긍정 이력을 가진 사용자들이 함께 좋아한 게임을 이용하는 방식입니다.",
        "- **공동 출현**은 같은 사용자가 두 게임 모두 긍정적으로 평가한 경우입니다.",
        "- **k-core**는 사용자와 게임에 최소 긍정 리뷰 수를 정해, 기준을 만족하지 않는 항목을 반복해서 걷어낸 자료입니다.",
        "- **텍스트 벡터**는 태그와 설명을 단어별 숫자로 바꿔, 게임 설명의 비슷한 정도를 계산할 수 있게 만든 표현입니다.",
        "- **기준선**은 다른 추천 방식과 비교할 때 참고하는 간단한 방법입니다. 여기서 인기순 기준은 긍정 리뷰가 많은 게임을 먼저 보여 줍니다.",
        "- **bootstrap 신뢰구간**은 사용자를 다시 뽑는 과정을 여러 번 반복해 계산한 성능의 불확실성 범위입니다.",
        "- **Jaccard 유사도**는 두 태그 목록이 얼마나 겹치는지 나타내며, 0은 겹침 없음, 1은 완전히 같음을 뜻합니다.",
        "- **A·B·C**는 보고서에서 쓰는 세 원자료의 짧은 이름입니다. 원자료와 출처는 `00_sources.csv`에서 확인할 수 있습니다.",
        "- `비율(…, %)`은 괄호 안에 적힌 분모를 기준으로 한 백분율입니다. 분모가 표마다 다를 수 있으므로 열 이름을 함께 확인하세요.",
        "- 분류가 이름이나 메타데이터에서 추정된 경우에는 확정된 사실이 아니라는 점을 표 설명과 `분류 근거` 열에서 확인하세요.",
        "",
        "## 표별 안내",
        "",
    ]

    csv_files = sorted(tables_dir.glob("*.csv"))
    for path in csv_files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            headers = next(csv.reader(stream), [])
        lines.extend([f"### [{path.name}]({path.name})", "", _table_description(path.name), "", "열 이름과 뜻:", ""])
        for header in headers:
            lines.append(f"- `{header}` → {column_label(header)}")
        lines.append("")

    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output

