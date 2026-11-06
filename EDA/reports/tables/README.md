# 분석 표 안내

이 안내서는 분석에서 내보낸 CSV 표의 용도와 열 이름을 설명합니다. 표의 원본 열 이름과 수치는 그대로 두고, 화면에 표시할 때만 쉬운 한국어 이름을 붙입니다.

## 읽는 법

- **정답**은 평가에서 맞혀야 하는 사용자별 긍정 게임입니다. 이 분석에서는 한 사용자의 한 게임을 하나의 정답으로 셉니다.
- **쌍**은 사용자와 게임을 묶은 한 건입니다. 같은 게임이라도 사용자가 다르면 서로 다른 쌍입니다.
- **입력**은 추천을 만들 때 사용자에게 이미 알려져 있다고 보는 게임입니다.
- **CF**는 협업 필터링입니다. 비슷한 긍정 이력을 가진 사용자들이 함께 좋아한 게임을 이용하는 방식입니다.
- **공동 출현**은 같은 사용자가 두 게임 모두 긍정적으로 평가한 경우입니다.
- **k-core**는 사용자와 게임에 최소 긍정 리뷰 수를 정해, 기준을 만족하지 않는 항목을 반복해서 걷어낸 자료입니다.
- **텍스트 벡터**는 태그와 설명을 단어별 숫자로 바꿔, 게임 설명의 비슷한 정도를 계산할 수 있게 만든 표현입니다.
- **기준선**은 다른 추천 방식과 비교할 때 참고하는 간단한 방법입니다. 여기서 인기순 기준은 긍정 리뷰가 많은 게임을 먼저 보여 줍니다.
- **bootstrap 신뢰구간**은 사용자를 다시 뽑는 과정을 여러 번 반복해 계산한 성능의 불확실성 범위입니다.
- **Jaccard 유사도**는 두 태그 목록이 얼마나 겹치는지 나타내며, 0은 겹침 없음, 1은 완전히 같음을 뜻합니다.
- **A·B·C**는 보고서에서 쓰는 세 원자료의 짧은 이름입니다. 원자료와 출처는 `00_sources.csv`에서 확인할 수 있습니다.
- `비율(…, %)`은 괄호 안에 적힌 분모를 기준으로 한 백분율입니다. 분모가 표마다 다를 수 있으므로 열 이름을 함께 확인하세요.
- 분류가 이름이나 메타데이터에서 추정된 경우에는 확정된 사실이 아니라는 점을 표 설명과 `분류 근거` 열에서 확인하세요.

## 표별 안내

### [00_sources.csv](00_sources.csv)

분석에 사용한 원자료의 버전, 이용 조건, 내려받은 파일과 날짜를 정리합니다.

열 이름과 뜻:

- `dataset` → 자료 이름
- `kaggle_ref` → Kaggle 자료 주소
- `kaggle_version` → Kaggle 자료 버전
- `kaggle_last_updated` → Kaggle 자료의 마지막 갱신일
- `license` → 이용 허가 조건
- `file` → 파일 이름
- `MB` → 파일 크기(MB)
- `partial_download` → 일부만 내려받은 파일 여부
- `downloaded_at` → 다운로드 날짜

### [01_schema_nulls.csv](01_schema_nulls.csv)

각 원자료 열의 자료형과 결측값 수·비율을 보여 줍니다.

열 이름과 뜻:

- `table` → 표 이름
- `column` → 원본 열 이름
- `dtype` → 자료형
- `n_null` → 결측값 수
- `null_pct` → 결측 비율(전체 행 중, %)

### [01_structure.csv](01_structure.csv)

각 자료의 행·열 수, 사용자·게임 수, 중복과 날짜 범위를 요약합니다.

열 이름과 뜻:

- `table` → 표 이름
- `rows` → 행 수
- `cols` → 열 수
- `users` → 사용자 수
- `items` → 게임 항목 수
- `dup_key` → 중복 확인 기준 열
- `dup_rows` → 중복 행 수
- `date_min` → 가장 이른 날짜
- `date_max` → 가장 늦은 날짜

### [02_A_positive_kcore.csv](02_A_positive_kcore.csv)

A 긍정 리뷰에서 사용자·게임 최소 활동 기준을 적용한 결과입니다.

열 이름과 뜻:

- `min_user_pos` → 사용자별 최소 긍정 리뷰 수 기준
- `min_item_pos` → 게임별 최소 긍정 리뷰 수 기준
- `users` → 사용자 수
- `items` → 게임 항목 수
- `interactions` → 사용자-게임 상호작용 수
- `density` → 행렬 밀도(비어 있지 않은 칸 비율, 0~1)
- `linux_items` → Linux 게임 항목 수
- `linux_interactions` → Linux 사용자-게임 상호작용 수

### [02_reviews_by_year.csv](02_reviews_by_year.csv)

연도별 A 리뷰와 긍정 비율, 날짜가 있는 B 리뷰 수입니다.

열 이름과 뜻:

- `year` → 연도
- `A_reviews` → A 리뷰 수
- `A_positive_rate` → A 리뷰의 긍정 비율(A 리뷰 중, 0~1)
- `B_reviews_dated` → 날짜 정보가 있는 B 리뷰 수

### [03_linux_flag_A_vs_C.csv](03_linux_flag_A_vs_C.csv)

A와 C 자료에서 기록된 Linux 지원 여부가 어떻게 겹치는지 보여 줍니다.

열 이름과 뜻:

- `A linux (2024)` → A 자료의 Linux 지원 여부(2024년 기준)
- `False` → 아니요
- `True` → 예

### [04_B_owned_not_in_A_reasons.csv](04_B_owned_not_in_A_reasons.csv)

B에서 보유 기록은 있지만 A 게임 목록에는 없는 항목을 추정 사유별로 셉니다.

열 이름과 뜻:

- `reason_guess` → 이름과 메타데이터로 추정한 분류
- `items` → 게임 항목 수
- `owner_rows` → 보유 기록 행 수
- `in_C` → C 자료 포함 수
- `in_B_steam_games` → B Steam 게임 목록 포함 수

### [04_B_owned_not_in_A_top30.csv](04_B_owned_not_in_A_top30.csv)

A에 없는 B 보유 항목 중 보유자가 많은 상위 30개를 보여 줍니다.

열 이름과 뜻:

- `item_id` → 게임 항목 ID
- `owners` → 보유자 수
- `name` → 게임 이름
- `reason_guess` → 이름과 메타데이터로 추정한 분류
- `in_C` → C 자료 포함 수
- `in_B_steam_games` → B Steam 게임 목록 포함 수

### [04_id_overlap.csv](04_id_overlap.csv)

B의 여러 게임 집합이 A·C 자료에 얼마나 포함되는지 비교합니다.

열 이름과 뜻:

- `B_set` → B 항목 집합 이름
- `n` → 항목 수
- `in_A_games` → A 게임 자료 포함 수
- `in_A_games_pct` → B 집합 게임 중 A 게임 자료에 포함된 비율(%)
- `in_A_rec_items` → A 추천 대상 자료 포함 수
- `in_A_linux` → A Linux 게임 자료 포함 수
- `in_C` → C 자료 포함 수
- `in_C_pct` → B 집합 게임 중 C 자료에 포함된 비율(%)
- `in_A_or_C` → A 또는 C 자료 포함 수
- `in_neither` → A와 C 어느 쪽에도 없는 수

### [05_external_test_eligibility.csv](05_external_test_eligibility.csv)

입력과 정답 조건을 바꾸었을 때 외부 평가에 쓸 수 있는 규모를 비교합니다.

열 이름과 뜻:

- `input` → 입력 조건
- `target` → 정답 조건
- `min_input_items` → 사용자당 필요한 최소 입력 게임 수
- `users` → 사용자 수
- `target_pairs` → 정답 사용자-게임 쌍 수
- `unique_target_items` → 중복을 제외한 정답 게임 수

### [05_top_target_items.csv](05_top_target_items.csv)

정답 사용자-게임 쌍이 많은 상위 게임과 그 비중을 보여 줍니다.

열 이름과 뜻:

- `app_id` → Steam 게임 ID(App ID)
- `target_pairs` → 정답 사용자-게임 쌍 수
- `share` → 전체 정답 쌍 중 비중(0~1)
- `A_linux_pos_rank` → A Linux 게임의 긍정 리뷰 순위
- `title` → 게임 제목

### [06_cli_file_verification.csv](06_cli_file_verification.csv)

공개 명령행 도구의 파일 크기와 SHA-256 값이 기대값과 일치하는지 기록합니다.

열 이름과 뜻:

- `dataset` → 자료 이름
- `file` → 파일 이름
- `local_path` → 로컬 파일 경로
- `expected_bytes` → 기대 파일 크기(바이트)
- `actual_bytes` → 실제 파일 크기(바이트)
- `expected_sha256` → 기대 SHA-256 값
- `actual_sha256` → 실제 파일 SHA-256 값
- `size_match` → 파일 크기 일치 여부
- `sha256_match` → SHA-256 값 일치 여부
- `status` → 확인 상태

### [07_A_internal_time_split.csv](07_A_internal_time_split.csv)

기준일 전 입력과 기준일 후 Linux 정답으로 만든 시간 분할 규모입니다.

열 이름과 뜻:

- `cutoff` → 시간 분할 기준일
- `users` → 사용자 수
- `target_pairs` → 정답 사용자-게임 쌍 수
- `target_games` → 정답 게임 수
- `new_target_games` → 신규 정답 게임 수
- `new_target_game_pct` → 신규 정답 게임 비율(정답 게임 중, %)
- `new_target_pairs` → 신규 정답 사용자-게임 쌍 수
- `new_target_pair_pct` → 신규 정답 쌍 비율(정답 쌍 중, %)
- `prior_positive_requirement` → 기준일 전 필요한 최소 긍정 리뷰 수

### [07_candidate_labels.csv](07_candidate_labels.csv)

후보 게임별 제목, 출시일, 긍정 리뷰, 정답 여부, 인기 구간과 장르입니다.

열 이름과 뜻:

- `app_id` → Steam 게임 ID(App ID)
- `title` → 게임 제목
- `date_release` → 출시일
- `positive_reviews` → 긍정 리뷰 수
- `target_pairs` → 정답 사용자-게임 쌍 수
- `evaluated` → 평가 대상 여부
- `year` → 연도
- `popularity_band` → 긍정 리뷰 수 기준 인기 구간
- `genres` → 장르 목록

### [07_candidate_target_comparison.csv](07_candidate_target_comparison.csv)

장르·인기 구간·연도별 후보 수와 정답·라벨 누락 규모를 비교합니다.

열 이름과 뜻:

- `dimension` → 구분 기준
- `category` → 범주
- `candidates` → 후보 수
- `target_games` → 정답 게임 수
- `unlabeled_games` → 평가 정답 라벨이 없는 게임 수
- `unlabeled_pct` → 정답 라벨이 없는 게임 비율(후보 게임 중, %)
- `target_pairs` → 정답 사용자-게임 쌍 수
- `share_of_all_candidates_pct` → 전체 후보 중 비율(%)
- `share_of_target_games_pct` → 정답 게임 수 비율(전체 정답 게임 중, %)

### [07_longtail_sensitivity.csv](07_longtail_sensitivity.csv)

인기 상위 게임을 제외할 때 평가 사용자와 정답 규모가 어떻게 바뀌는지 보여 줍니다.

열 이름과 뜻:

- `excluded_head_games` → 제외한 인기 상위 게임 수
- `users` → 사용자 수
- `target_pairs` → 정답 사용자-게임 쌍 수
- `target_games` → 정답 게임 수

### [07_popularity_bootstrap_ci.csv](07_popularity_bootstrap_ci.csv)

사용자 자료를 반복해서 뽑아 계산한 인기순 기준 지표의 불확실성 범위입니다.

열 이름과 뜻:

- `population` → 평가 대상 범위
- `users` → 사용자 수
- `metric` → 평가 지표
- `estimate` → 추정값
- `ci_low` → 신뢰구간 하한
- `ci_high` → 신뢰구간 상한
- `ci_width` → 신뢰구간 너비
- `iterations` → 반복 횟수
- `seed` → 무작위 추출 기준값

### [07_pre2016_kcore.csv](07_pre2016_kcore.csv)

2015년까지의 A 긍정 리뷰에 사용자·게임 최소 활동 기준을 적용한 결과입니다.

열 이름과 뜻:

- `min_user_pos` → 사용자별 최소 긍정 리뷰 수 기준
- `min_item_pos` → 게임별 최소 긍정 리뷰 수 기준
- `users` → 사용자 수
- `games` → 게임 수
- `interactions` → 사용자-게임 상호작용 수
- `linux_games` → Linux 게임 수
- `linux_interactions` → Linux 사용자-게임 상호작용 수

### [07_reproduction.csv](07_reproduction.csv)

주요 A 자료, 평가셋과 인기순 기준선 수치의 재현 확인 결과입니다.

열 이름과 뜻:

- `A_reviews` → A 리뷰 수
- `A_users` → A 리뷰 작성 사용자 수
- `A_reviewed_games` → A 리뷰가 있는 게임 수
- `linux_games` → Linux 게임 수
- `linux_reviewed_games` → 리뷰가 있는 Linux 게임 수
- `kcore_users` → k-core에 남은 사용자 수
- `kcore_games` → k-core에 남은 게임 수
- `kcore_interactions` → k-core에 남은 사용자-게임 상호작용 수
- `kcore_linux_games` → k-core에 남은 Linux 게임 수
- `evaluation_users` → 평가 사용자 수
- `target_pairs` → 정답 사용자-게임 쌍 수
- `target_games` → 정답 게임 수
- `pop_recall10` → 인기순 기준선 Recall@10
- `pop_ndcg10` → 인기순 기준선 NDCG@10
- `asserts_passed` → 재현 확인 항목 통과 여부

### [08_B_target_cooccurrence_reachability.csv](08_B_target_cooccurrence_reachability.csv)

입력 게임과 정답 게임 사이에 긍정 공동 출현 경로가 있는지 요약합니다.

열 이름과 뜻:

- `source` → 비교 기준 또는 출처
- `population` → 평가 대상 범위
- `users` → 사용자 수
- `target_pairs` → 정답 사용자-게임 쌍 수
- `reachable_target_pairs` → 연결 경로가 있는 정답 사용자-게임 쌍 수
- `reachable_pairs_pct` → 연결 경로가 있는 정답 쌍 비율(정답 쌍 중, %)
- `users_with_any_reachable_target` → 하나 이상 정답에 연결 경로가 있는 사용자 수
- `users_with_any_reachable_target_pct` → 하나 이상 정답에 연결 경로가 있는 사용자 비율(사용자 중, %)
- `users_with_all_reachable_targets` → 정답 모두에 연결 경로가 있는 사용자 수
- `users_with_all_reachable_targets_pct` → 정답 모두에 연결 경로가 있는 사용자 비율(사용자 중, %)
- `minimum_cooccurrence_users` → 이웃으로 인정하는 최소 공동 긍정 사용자 수

### [08_positive_cooccurrence_by_popularity.csv](08_positive_cooccurrence_by_popularity.csv)

인기 구간별 긍정 공동 출현 이웃 수와 이웃 부족 비율입니다.

열 이름과 뜻:

- `source` → 비교 기준 또는 출처
- `popularity_band` → 긍정 리뷰 수 기준 인기 구간
- `candidate_games` → 후보 게임 수
- `insufficient_games` → 필요한 이웃 수에 못 미치는 게임 수
- `insufficient_pct` → 이웃이 부족한 게임 비율(후보 게임 중, %)
- `neighbors_ge1_median` → 긍정 사용자 1명 이상을 공유하는 이웃 수 중앙값
- `neighbors_ge5_median` → 긍정 사용자 5명 이상을 공유하는 이웃 수 중앙값
- `neighbors_ge10_median` → 긍정 사용자 10명 이상을 공유하는 이웃 수 중앙값
- `neighbors_ge20_median` → 긍정 사용자 20명 이상을 공유하는 이웃 수 중앙값
- `neighbors_ge5_p10` → 긍정 사용자 5명 이상을 공유하는 이웃 수의 10백분위수
- `neighbors_ge5_p90` → 긍정 사용자 5명 이상을 공유하는 이웃 수의 90백분위수

### [08_positive_cooccurrence_neighbors.csv](08_positive_cooccurrence_neighbors.csv)

각 게임과 긍정 사용자를 공유하는 이웃 수를 기준별로 기록합니다.

열 이름과 뜻:

- `source` → 비교 기준 또는 출처
- `app_id` → Steam 게임 ID(App ID)
- `positive_users_in_source` → 해당 기준에서 게임에 긍정한 사용자 수
- `A_positive_reviews` → A 긍정 리뷰 수
- `neighbors_ge1` → 긍정 사용자 1명 이상을 공유하는 이웃 수
- `neighbors_ge5` → 긍정 사용자 5명 이상을 공유하는 이웃 수
- `neighbors_ge10` → 긍정 사용자 10명 이상을 공유하는 이웃 수
- `neighbors_ge20` → 긍정 사용자 20명 이상을 공유하는 이웃 수
- `insufficient_ge5_neighbors` → 긍정 이웃이 5명 미만인 게임 수
- `popularity_band` → 긍정 리뷰 수 기준 인기 구간

### [09_cf_item_threshold_sensitivity.csv](09_cf_item_threshold_sensitivity.csv)

게임별 최소 긍정 리뷰 기준을 바꿀 때 후보·입력·정답이 얼마나 남는지 비교합니다.

열 이름과 뜻:

- `positive_review_threshold` → 게임에 필요한 최소 긍정 리뷰 수
- `remaining_all_items` → 기준 적용 후 전체 게임 항목 수
- `linux_candidates` → Linux 후보 게임 수
- `linux_candidate_pct` → Linux 후보 비율(전체 Linux 후보 중, %)
- `B_input_pairs` → B 입력 사용자-게임 쌍 수
- `B_input_pair_pct` → B 입력 쌍 비율(B 입력 쌍 중, %)
- `B_target_pairs` → B 정답 사용자-게임 쌍 수
- `B_target_pair_pct` → B 정답 쌍 비율(B 정답 쌍 중, %)
- `B_target_games` → B 정답 게임 수
- `users_retaining_ge5_inputs` → 입력이 5개 이상 남은 사용자 수
- `users_retaining_any_target` → 정답이 하나 이상 남은 사용자 수
- `users_retaining_both_conditions` → 입력 5개 이상과 정답 1개 이상을 모두 충족하는 사용자 수

### [10_C_A_C_tag_jaccard.csv](10_C_A_C_tag_jaccard.csv)

A와 C 자료에 공통인 게임의 태그 집합 유사도를 비교합니다.

열 이름과 뜻:

- `cohort` → 분석 대상 묶음
- `comparison` → 비교 대상
- `common_games` → 두 자료에 공통으로 있는 게임 수
- `A_tags_empty_games` → A 태그가 비어 있는 게임 수
- `C_side_empty_games` → C 쪽 태그가 비어 있는 게임 수
- `empty_union_games_jaccard_undefined` → 두 태그 집합이 모두 비어 Jaccard 값을 계산할 수 없는 게임 수
- `nonempty_union_games` → 두 태그 집합의 합집합이 비어 있지 않은 게임 수
- `jaccard_nonempty_union` → 합집합이 비어 있지 않은 게임의 Jaccard 유사도(0~1)
- `mean_jaccard_nonempty_union` → 합집합이 비어 있지 않은 게임들의 Jaccard 평균

### [10_C_A_description_summary.csv](10_C_A_description_summary.csv)

설명 결측, 길이, 중복 및 비영어 설명의 대략적 비율을 요약합니다.

열 이름과 뜻:

- `cohort` → 분석 대상 묶음
- `games` → 게임 수
- `A_description_missing` → A 설명이 없는 게임 수
- `A_description_missing_pct` → A 설명이 없는 게임 비율(대상 게임 중, %)
- `A_description_length_chars_nonempty` → A 설명이 비어 있지 않은 게임의 설명 글자 수
- `exact_duplicate_description_text_groups` → 완전히 같은 설명 묶음 수
- `games_in_exact_duplicate_text_groups` → 완전히 같은 설명 묶음에 포함된 게임 수
- `excess_games_from_exact_duplicates` → 완전히 같은 설명 때문에 추가로 중복 집계된 게임 수
- `duplicate_games_pct_of_nonempty_descriptions` → 중복 설명이 있는 게임 비율(설명이 있는 게임 중, %)
- `descriptions_with_any_non_ascii_pct` → ASCII가 아닌 문자가 하나라도 있는 설명 비율(비어 있지 않은 설명 중, %)
- `description_ascii_character_share` → 설명에서 ASCII 문자가 차지하는 비율(0~1)
- `per_description_ascii_character_ratio` → 게임별 설명의 ASCII 문자 비율(0~1)
- `C_short_description_can_fill_A_missing` → C 짧은 설명으로 A 설명 결측을 보완할 수 있는 게임 수
- `C_about_the_game_can_fill_A_missing` → C 상세 설명으로 A 설명 결측을 보완할 수 있는 게임 수
- `C_both_fields_can_fill_A_missing` → C 짧은 설명과 상세 설명이 모두 있어 A 결측을 보완할 수 있는 게임 수
- `C_either_field_can_fill_A_missing` → C 설명 중 하나 이상으로 A 결측을 보완할 수 있는 게임 수
- `A_or_C_description_available` → A 또는 C 설명을 쓸 수 있는 게임 수
- `A_or_C_description_available_pct` → A 또는 C 설명을 쓸 수 있는 비율(대상 게임 중, %)

### [10_C_A_exact_duplicate_descriptions.csv](10_C_A_exact_duplicate_descriptions.csv)

완전히 같은 설명 문구가 있는 게임 묶음의 예를 보여 줍니다.

열 이름과 뜻:

- `cohort` → 분석 대상 묶음
- `rank` → 순위
- `games` → 게임 수
- `description_length` → 설명 글자 수
- `exact_description_excerpt` → 중복 설명 일부

### [10_C_A_tag_summary.csv](10_C_A_tag_summary.csv)

게임별 태그 수, 태그 보유율과 드문 태그 분포를 요약합니다.

열 이름과 뜻:

- `cohort` → 분석 대상 묶음
- `games` → 게임 수
- `games_with_tags` → 태그가 있는 게임 수
- `tagged_games_pct` → 태그가 있는 게임 비율(해당 대상 묶음의 게임 중, %)
- `unique_tags` → 서로 다른 태그 수
- `tags_per_game` → 게임당 태그 수
- `document_frequency` → 태그가 나타난 게임 수
- `positive_reviews_per_game` → 게임당 긍정 리뷰 수
- `tags_seen_in_fewer_than_5_games` → 5개 미만 게임에만 나타난 태그 수
- `tags_seen_in_fewer_than_5_games_pct` → 5개 미만 게임에만 나타난 태그 비율(전체 태그 중, %)

### [10_C_A_tag_top20.csv](10_C_A_tag_top20.csv)

각 대상 묶음에서 자주 나타나는 상위 태그를 보여 줍니다.

열 이름과 뜻:

- `cohort` → 분석 대상 묶음
- `rank` → 순위
- `tag` → 태그 이름
- `document_frequency` → 태그가 나타난 게임 수
- `share_of_games_pct` → 해당 대상 묶음의 게임 중 비율(%)

### [10_C_tfidf_queries.csv](10_C_tfidf_queries.csv)

정성 검토를 위해 미리 고른 질의 게임과 선택 기준입니다.

열 이름과 뜻:

- `selection_order` → 질의 선택 순서
- `popularity_stratum` → 긍정 리뷰 인기도 층
- `app_id` → Steam 게임 ID(App ID)
- `title` → 게임 제목
- `positive_reviews` → 긍정 리뷰 수
- `selection_rule` → 질의 선택 기준

### [11_C_qualitative_review.csv](11_C_qualitative_review.csv)

질의 게임마다 태그만 쓴 경우와 설명을 더한 경우의 상위 결과를 검토합니다.

열 이름과 뜻:

- `query_app_id` → 질의 게임 ID(App ID)
- `query_title` → 질의 게임 제목
- `interpretation` → 질의 결과에 대한 사람의 검토 의견
- `tags_only_top5` → 태그만 사용했을 때 가장 가까운 게임 5개
- `tags_plus_description_top5` → 태그와 설명을 함께 사용했을 때 가장 가까운 게임 5개

### [11_C_repeated_phrases.csv](11_C_repeated_phrases.csv)

여러 게임 설명에 반복된 다섯 단어 구절을 요약합니다.

열 이름과 뜻:

- `cohort` → 분석 대상 묶음
- `phrase` → 반복해서 나타난 다섯 단어 구절
- `document_frequency` → 태그가 나타난 게임 수

### [11_C_tfidf_input_coverage.csv](11_C_tfidf_input_coverage.csv)

입력 게임의 텍스트 벡터 생성 가능 여부와 사용자별 입력 수를 비교합니다.

열 이름과 뜻:

- `variant` → 텍스트 입력 방식
- `unique_input_games` → 중복을 제외한 입력 게임 수
- `usable_input_games` → 텍스트 벡터를 만들 수 있는 입력 게임 수
- `zero_vector_input_games` → 텍스트 벡터가 비어 있는 입력 게임 수
- `input_pairs` → 입력 사용자-게임 쌍 수
- `usable_input_pairs` → 텍스트 벡터를 만들 수 있는 입력 쌍 수
- `usable_input_pairs_pct` → 텍스트 벡터를 만들 수 있는 입력 쌍 비율(전체 입력 쌍 중, %)
- `users` → 사용자 수
- `users_with_zero_usable_inputs` → 사용 가능한 입력이 하나도 없는 사용자 수
- `users_with_fewer_than_5_usable_inputs` → 사용 가능한 입력이 5개 미만인 사용자 수

### [11_C_tfidf_query_top5_neighbors.csv](11_C_tfidf_query_top5_neighbors.csv)

단어 중요도로 비교한 각 질의 게임의 상위 5개 텍스트 유사도 이웃입니다.

열 이름과 뜻:

- `variant` → 텍스트 입력 방식
- `query_stratum` → 질의 게임의 인기도 층
- `query_app_id` → 질의 게임 ID(App ID)
- `query_title` → 질의 게임 제목
- `query_positive_reviews` → 질의 게임의 긍정 리뷰 수
- `query_zero_vector` → 질의 게임의 텍스트 벡터가 비어 있는지 여부
- `query_positive_similarity_neighbors` → 긍정 사용자를 공유하는 질의 이웃 수
- `rank` → 순위
- `neighbor_app_id` → 이웃 게임 ID(App ID)
- `neighbor_title` → 이웃 게임 제목
- `cosine_similarity` → 코사인 유사도
- `positive_similarity` → 긍정 사용자 기반 이웃 여부

### [11_C_tfidf_summary.csv](11_C_tfidf_summary.csv)

태그·설명을 단어 중요도로 바꾼 뒤 게임 간 유사도 분포를 요약합니다.

열 이름과 뜻:

- `variant` → 텍스트 입력 방식
- `games` → 게임 수
- `vocabulary_size` → 텍스트 분석에 사용한 단어 수
- `matrix_nonzero_values` → 행렬에서 0이 아닌 값의 수
- `matrix_density_pct` → 행렬 밀도(%, 전체 칸 중 값이 있는 칸)
- `matrix_sparsity_pct` → 행렬 희소도(%, 값이 없는 칸)
- `zero_vector_games` → 텍스트 벡터가 비어 있는 게임 수
- `zero_vector_games_pct` → 텍스트 벡터가 비어 있는 게임 비율(게임 중, %)
- `nonzero_vector_games_pct` → 텍스트 벡터가 비어 있지 않은 게임 비율(전체 게임 중, %)
- `positive_neighbors_p25` → 긍정 이웃 수의 25백분위수
- `positive_neighbors_p50` → 긍정 이웃 수의 중앙값
- `positive_neighbors_p75` → 긍정 이웃 수의 75백분위수
- `positive_neighbors_p90` → 긍정 이웃 수의 90백분위수
- `positive_neighbors_p99` → 긍정 이웃 수의 99백분위수
- `games_ge20_positive_neighbors` → 긍정 사용자 20명 이상을 공유하는 이웃이 있는 게임 수
- `games_ge20_positive_neighbors_pct` → 긍정 사용자 20명 이상을 공유하는 이웃이 있는 게임 비율(게임 중, %)
- `nearest_similarity_p25` → 가장 가까운 게임 유사도의 25백분위수
- `nearest_similarity_p50` → 가장 가까운 게임 유사도의 중앙값
- `nearest_similarity_p75` → 가장 가까운 게임 유사도의 75백분위수
- `nearest_similarity_p90` → 가장 가까운 게임 유사도의 90백분위수
- `nearest_similarity_p99` → 가장 가까운 게임 유사도의 99백분위수
- `nearest_similarity_mean` → 가장 가까운 게임과의 유사도 평균
- `nearest_similarity_max` → 가장 가까운 게임과의 유사도 최댓값
- `no_positive_similarity_neighbor_games` → 긍정 유사도 이웃이 없는 게임 수
- `no_positive_similarity_neighbor_pct` → 긍정 유사도 이웃이 없는 게임 비율(게임 중, %)

### [12_B_input_loss_distribution.csv](12_B_input_loss_distribution.csv)

카탈로그나 리뷰 범위 때문에 입력을 잃는 사용자 규모를 손실률별로 나눕니다.

열 이름과 뜻:

- `loss_rate_bin` → 입력 손실 비율 구간
- `catalog_loss_users` → 카탈로그에서 입력이 줄어든 사용자 수
- `review_space_loss_users` → 리뷰 자료 범위에서 입력이 줄어든 사용자 수

### [12_B_missing_other_classification_summary.csv](12_B_missing_other_classification_summary.csv)

기존 other 분류 항목을 재검토한 결과를 유형별로 요약합니다.

열 이름과 뜻:

- `classification` → 분류 결과
- `games` → 게임 수
- `owner_rows` → 보유 기록 행 수
- `owner_row_pct` → 기존 기타 항목의 전체 보유 기록 중 비율(%)

### [12_B_missing_other_reclassification.csv](12_B_missing_other_reclassification.csv)

기존 other 항목마다 B·C 메타데이터와 추정 분류 근거를 기록합니다.

열 이름과 뜻:

- `app_id` → Steam 게임 ID(App ID)
- `owners` → 보유자 수
- `name` → 게임 이름
- `original_reason` → 원래 기록된 사유
- `B_title` → B 게임 제목
- `C_title` → C 게임 제목
- `in_B_metadata` → B 메타데이터 포함 여부
- `in_C_metadata` → C 메타데이터 포함 여부
- `B_genres` → B 장르 정보
- `C_genres` → C 장르 정보
- `B_specs` → B 상세 사양 정보
- `C_categories` → C 분류 정보
- `classification` → 분류 결과
- `classification_basis` → 분류 근거
- `cumulative_owner_rows` → 누적 보유자 행 수
- `cumulative_owner_pct` → 기존 기타 항목의 전체 보유 기록 중 누적 비율(%)

### [12_B_missing_other_top_coverage.csv](12_B_missing_other_top_coverage.csv)

보유자가 많은 기존 other 항목에서 메타데이터가 있는 정도를 보여 줍니다.

열 이름과 뜻:

- `top_owned_games` → 보유자 수 상위 게임 수
- `owner_rows` → 보유 기록 행 수
- `share_of_other_owner_rows_pct` → 기존 기타 항목의 전체 보유 기록 중 비율(%)
- `with_B_or_C_metadata_games` → B 또는 C 메타데이터가 있는 게임 수

### [12_C_only_linux_B_positive_games.csv](12_C_only_linux_B_positive_games.csv)

C에는 있고 A에는 없으며 B에서 긍정 기록이 있는 Linux 게임을 나열합니다.

열 이름과 뜻:

- `item_id` → 게임 항목 ID
- `positive_pairs` → 긍정 사용자-게임 쌍 수

### [13_bundle_linux.csv](13_bundle_linux.csv)

묶음 상품별 전체 게임과 Linux 지원 게임 수입니다.

열 이름과 뜻:

- `bundle_id` → 묶음 상품 ID(Bundle ID)
- `items` → 게임 항목 수
- `linux_items` → Linux 게임 항목 수
- `C_linux_items` → C Linux 게임 항목 수

### [13_game_metadata_distributions.csv](13_game_metadata_distributions.csv)

전체·Linux·비Linux 게임의 가격, 무료 여부, 긍정 비율과 리뷰 수 분포입니다.

열 이름과 뜻:

- `population` → 평가 대상 범위
- `games` → 게임 수
- `price_final` → 최종 표시 가격(원자료 표기 단위)
- `free_games` → 무료 게임 수
- `positive_ratio` → 긍정 리뷰 비율(전체 리뷰 중, %)
- `user_reviews` → 사용자 리뷰 수

### [13_helpful_funny.csv](13_helpful_funny.csv)

도움·재미 표시 수의 0값과 분포를 요약합니다.

열 이름과 뜻:

- `field` → 항목 이름
- `rows` → 행 수
- `zero_rows` → 값이 0인 행 수
- `zero_pct` → 값이 0인 비율(대상 행 중, %)
- `positive_rows` → 긍정으로 기록된 행 수
- `all_distribution` → 전체 값 분포
- `nonzero_distribution` → 0이 아닌 값의 분포

### [13_linux_steam_deck_crosstab.csv](13_linux_steam_deck_crosstab.csv)

Linux와 Steam Deck 지원 표시의 교차표입니다.

열 이름과 뜻:

- `linux` → Linux 지원 여부
- `steam_deck` → Steam Deck 지원 여부
- `games` → 게임 수

### [13_review_sampling_proxy.csv](13_review_sampling_proxy.csv)

게임별 표시 리뷰 수와 A에서 관측된 리뷰 수의 비율을 비교합니다.

열 이름과 뜻:

- `app_id` → Steam 게임 ID(App ID)
- `title` → 게임 제목
- `linux` → Linux 지원 여부
- `user_reviews` → 사용자 리뷰 수
- `positive_ratio` → 긍정 리뷰 비율(전체 리뷰 중, %)
- `A_actual_reviews` → A 자료에 실제로 포함된 리뷰 수(표본 점검)
- `observed_to_store_reviews_ratio` → A에서 관측된 리뷰 수 ÷ 원자료 표시 리뷰 수(비율)

### [14_external_dataset_candidates.csv](14_external_dataset_candidates.csv)

추가 외부 평가 자료 후보와 공개된 범위·이용 조건·위험을 정리합니다.

열 이름과 뜻:

- `candidate` → 데이터 후보 이름
- `addresses` → 보완하려는 데이터 공백
- `license` → 이용 허가 조건
- `version_period` → 버전 또는 자료 수집 기간
- `app_id` → Steam 게임 ID(App ID)
- `user_positive_label` → 사용자별 긍정 평가 라벨 제공 여부
- `published_size` → 공개 자료에 표시된 크기
- `risk` → 확인해야 할 위험과 한계
- `source` → 비교 기준 또는 출처
- `schema_source` → 자료 구조 정보의 출처

### [15_B_team_fortress_dota_entries.csv](15_B_team_fortress_dota_entries.csv)

B에 포함된 Team Fortress·Dota 항목과 A 카탈로그 포함 여부입니다.

열 이름과 뜻:

- `item_id` → 게임 항목 ID
- `item_name` → 게임 이름
- `B_owners` → B 보유자 수
- `in_A_catalog` → A 카탈로그 포함 여부

### [15_baseline_uncertainty.csv](15_baseline_uncertainty.csv)

기준선 Recall@10과 사용자 bootstrap 신뢰구간의 반폭입니다.

열 이름과 뜻:

- `population` → 평가 대상 범위
- `users` → 사용자 수
- `baseline_recall` → 인기순 기준선 재현율
- `baseline_ci_halfwidth_pp` → 기준선 재현율 신뢰구간 반폭(퍼센트포인트)

### [15_baseline_without_absent_games.csv](15_baseline_without_absent_games.csv)

관측되지 않은 게임을 제외했을 때 인기순 기준 성능의 불확실성 범위입니다.

열 이름과 뜻:

- `population` → 평가 대상 범위
- `users` → 사용자 수
- `metric` → 평가 지표
- `estimate` → 추정값
- `ci_low` → 신뢰구간 하한
- `ci_high` → 신뢰구간 상한
- `ci_width` → 신뢰구간 너비
- `iterations` → 반복 횟수
- `seed` → 무작위 추출 기준값

### [15_candidates_without_A_tags.csv](15_candidates_without_A_tags.csv)

A 태그가 없거나 다른 태그 출처를 쓰는 후보와 정답·입력 규모입니다.

열 이름과 뜻:

- `app_id` → Steam 게임 ID(App ID)
- `title` → 게임 제목
- `tag_source` → 사용한 태그·장르 정보의 출처
- `A_positive_reviews` → A 긍정 리뷰 수
- `target_pairs` → 정답 사용자-게임 쌍 수
- `input_pairs` → 입력 사용자-게임 쌍 수
- `has_A_description` → A 설명 보유 여부
- `has_C_description` → C 설명 보유 여부
- `n_C_tags` → C 태그 수
- `n_B_tags` → B 태그 수

### [15_cf_insufficient_weighted.csv](15_cf_insufficient_weighted.csv)

협업 필터링 이웃 부족 게임이 긍정 리뷰·정답에 차지하는 비율입니다.

열 이름과 뜻:

- `source` → 비교 기준 또는 출처
- `insufficient_games` → 필요한 이웃 수에 못 미치는 게임 수
- `insufficient_pct` → 이웃이 부족한 게임 비율(후보 게임 중, %)
- `insufficient_A_positive_pct` → 이웃이 부족한 게임의 A 긍정 리뷰 비율(전체 A 긍정 리뷰 중, %)
- `insufficient_target_pairs` → 이웃이 부족한 정답 사용자-게임 쌍 수
- `insufficient_target_pairs_pct` → 이웃이 부족한 정답 쌍 비율(정답 쌍 중, %)
- `insufficient_target_games` → 이웃이 부족한 정답 게임 수
- `insufficient_median_A_positive` → 이웃 부족 게임의 A 긍정 리뷰 수 중앙값
- `zero_neighbor_games` → 긍정 이웃이 없는 게임 수
- `zero_neighbor_pct` → 긍정 이웃이 없는 게임 비율(후보 게임 중, %)
- `zero_neighbor_max_A_positive` → 긍정 이웃이 없는 게임의 A 긍정 리뷰 최대값

### [15_product_type_title_pattern.csv](15_product_type_title_pattern.csv)

상품 유형을 제목 패턴으로 추정한 규모와 태그·장르 결손을 요약합니다.

열 이름과 뜻:

- `cohort` → 분석 대상 묶음
- `games` → 게임 수
- `title_pattern_games` → 제목 패턴에 해당하는 게임 수
- `title_pattern_pct` → 제목 패턴에 해당하는 비율(대상 게임 중, %)
- `no_tags_or_genres_games` → 태그와 장르가 모두 없는 게임 수

### [15_tag_source_weighted_impact.csv](15_tag_source_weighted_impact.csv)

태그·장르 정보 출처별 후보와 긍정 리뷰·정답·입력 비중입니다.

열 이름과 뜻:

- `tag_source` → 사용한 태그·장르 정보의 출처
- `games` → 게임 수
- `games_pct` → 리뷰가 있는 Linux 후보 게임 중 비율(%)
- `A_positive_pct` → 이웃 부족 게임이 차지하는 A 긍정 리뷰 비율(전체 A 긍정 리뷰 중, %)
- `target_pairs` → 정답 사용자-게임 쌍 수
- `target_pairs_pct` → 정답 쌍 비율(비교 대상 전체 정답 쌍 중, %)
- `linux_input_pairs_pct` → Linux 입력 쌍 비율(입력 쌍 중, %)

### [15_target_games_absent_from_B_ownership.csv](15_target_games_absent_from_B_ownership.csv)

B 보유 기록에 나타나지 않는 정답 게임을 보여 줍니다.

열 이름과 뜻:

- `app_id` → Steam 게임 ID(App ID)
- `title` → 게임 제목
- `target_pairs` → 정답 사용자-게임 쌍 수
- `B_reviews` → B 리뷰 수

### [15_target_games_released_after_2015.csv](15_target_games_released_after_2015.csv)

2016년 이후 출시된 정답 게임과 정답 쌍 수입니다.

열 이름과 뜻:

- `app_id` → Steam 게임 ID(App ID)
- `title` → 게임 제목
- `A_release_date` → A 자료의 출시일
- `target_pairs` → 정답 사용자-게임 쌍 수
