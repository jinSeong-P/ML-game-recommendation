# Kieran 원본 — 학습·내부 검증

[Kaggle: kieranpoc/steam-reviews](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) **v2, 2023-12-10**을 고정했습니다. `all_reviews.csv` 전체 **113,883,717행**을 46개 Parquet으로 투영했습니다. `weighted_score_above_08.csv`는 도움됨 점수로 선택된 부분집합이므로 주 리뷰에 합치지 않습니다. ZIP 내 두 파일의 CRC를 확인했습니다.

| 원본 열 | 분석 열 | 의미·처리 |
|---|---|---|
| `author_steamid` | `user_id` | 원 Steam 계정. 문자열 → 범위 검증 → 손실 없는 정수 비교 |
| `appid` | `app_id` | Steam 게임 식별자 |
| `recommendationid` | `review_id` | 원 리뷰 식별자 |
| `voted_up` | `positive` | 긍정·부정 리뷰 |
| `timestamp_created` | 동일 | 최초 작성 Unix 시각 |
| `timestamp_updated` | 동일 | 갱신 Unix 시각 |
| `author_playtime_forever` | `playtime_forever` | 리뷰 작성자 스냅샷의 누적 플레이 분. 전체 보유 목록이 아님 |
| `language` | 동일 | 리뷰 언어 |

동일 원 리뷰가 여러 게임에 연결된 332,865개 ID·686,161행과 frank 대조로 추가 발견한 3개 ID를 격리했습니다. 같은 라벨 반복 2,375행과 라벨 충돌 72쌍도 정리했습니다. 정리 후 **113,195,034쌍·37,875,512계정·105,838개 App ID**입니다. 긍정 비율은 85.60%이며 사용자별 리뷰 게임 수 중앙값은 1, p90은 6, p99는 25입니다.

frank 예비 평가 계정 **87,573명 전체**를 예약하고 Kieran에서 일치한 79,879계정·3,360,025쌍을 모든 게임에서 제외했습니다. 최종 평가 후보가 줄어도 예약 계정을 학습으로 되돌리지 않습니다. 나머지 계정 중 `MD5("steam-eda-validation-v1:<정확한 Steam ID>")`의 앞 15자리 16진 값을 정수로 바꾼 뒤 modulo 10이 0이면 검증용입니다. 생성한 전체 학습 분할은 **98,857,701쌍·34,014,115계정**이며 모든 앱의 긍정 84,791,317쌍을 포함합니다. Item-KNN의 확정 학습 범위는 비Linux 게임도 포함하는 고정 입력 카탈로그 **37,610개 앱**의 긍정 리뷰입니다. 이 범위의 긍정 **78,315,268쌍**을 `train/positive_catalog.parquet`에 저장했으며, 긍정 singleton도 앱 지원 수와 코사인 정규화에 유지합니다. 내부 검증 예약은 10,977,308쌍·3,781,518계정입니다. 기본 평가 방식은 적격 긍정 Linux 게임 1개를 사용자별 균등 무작위로 숨기는 것입니다. seed **20261107**과 실제 입력·정답 manifest 및 최종 분할 파일을 생성·검증했습니다. 검증 예약 3,781,518명 중 평가 적격자는 **325,639명**입니다. [분할 파일 안내](../modeling_splits.md)를 참고하세요.

원본 위치는 `EDA/data/raw/kieran_original/`, 분석 투영·작업 DB는 `EDA/data/processed/kieran_original/`입니다. ZIP SHA256는 `ec3234d1ff0831f5a20752059df1c85dfe66ab0efde95b886c7cc3a11d0430b7`입니다. 원본·개별 계정 자료는 Git에서 제외합니다.

[전수 EDA 수치](../../reports/kieran_original_frank_eda_metrics.json) · [게임별 학습 지원 수](../../reports/tables/16_kieran_original_app_support.csv) · [종합 보고서](../../reports/kieran_frank_eda.md)
