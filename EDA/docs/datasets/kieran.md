# Kieran 원본 — 학습·내부 검증

[Kaggle: kieranpoc/steam-reviews](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) **v2, 2023-12-10**을 사용하도록 확정했습니다. `all_reviews.csv`의 **113,883,717행**에서 분석에 필요한 열만 추려 46개 Parquet 파일로 저장했습니다. `weighted_score_above_08.csv`는 도움됨 점수에 따라 선택된 일부 리뷰이므로 전체 리뷰에 합치지 않습니다. ZIP 안의 두 파일에 대해 파일 오류를 검사하는 CRC 확인을 마쳤습니다.

| 원본 열 | 분석 열 | 의미·처리 |
|---|---|---|
| `author_steamid` | `user_id` | 원 Steam 계정 ID. 문자열로 읽고 범위를 확인한 뒤 정밀도 손실 없이 정수로 비교 |
| `appid` | `app_id` | Steam 게임 식별자 |
| `recommendationid` | `review_id` | 원 리뷰 식별자 |
| `voted_up` | `positive` | 긍정·부정 리뷰 |
| `timestamp_created` | 동일 | 최초 작성 Unix 시각 |
| `timestamp_updated` | 동일 | 갱신 Unix 시각 |
| `author_playtime_forever` | `playtime_forever` | 리뷰 작성 당시 작성자에게서 수집한 누적 플레이 분. 사용자가 보유한 전체 게임 목록은 아님 |
| `language` | 동일 | 리뷰 언어 |

같은 원 리뷰가 여러 게임에 연결된 332,865개 ID·686,161행과 frank 자료 대조에서 추가로 찾은 3개 ID는 분석에서 제외했습니다. 같은 라벨이 반복된 2,375행과 라벨이 서로 다른 72쌍도 정리했습니다. 정리 후 데이터는 **113,195,034쌍·37,875,512계정·105,838개 App ID**입니다. 긍정 리뷰 비율은 85.60%입니다. 사용자별 리뷰 게임 수의 중앙값은 1개이며, 90백분위수(p90)는 6개, 99백분위수(p99)는 25개입니다.

frank 예비 평가 사용자 **87,573명 전체**를 테스트용으로 예약했다. Kieran에서 일치한 79,879계정·3,360,025쌍은 모든 게임에서 제외했다. 최종 테스트 사용자 수가 줄어도 예약한 계정을 학습에 다시 넣지 않는다. 나머지 계정은 `MD5("steam-eda-validation-v1:<정확한 Steam ID>")` 앞 15자리의 16진수를 정수로 바꿔 10으로 나눈다. 나머지가 0이면 검증용, 그 외에는 학습용으로 분류한다. 전체 학습 분할은 **98,857,701쌍·34,014,115계정**이며 모든 게임의 긍정 리뷰 84,791,317쌍을 포함한다. Item-KNN은 비Linux 게임도 포함하는 고정 입력 카탈로그 **37,610개 앱**의 긍정 리뷰로 학습한다. 이 범위의 긍정 리뷰 **78,315,268쌍**을 `train/positive_catalog.parquet`에 저장했다. 긍정 리뷰가 1개인 사용자도 게임별 긍정 사용자 수와 코사인 정규화에 포함한다. 내부 검증용 분할에는 10,977,308쌍·3,781,518계정이 있다. 기본 평가에서는 조건을 만족하는 긍정 Linux 게임 1개를 사용자마다 같은 확률로 숨긴다. seed **20261107**, 실제 입력·정답 파일 목록·검증 정보(manifest), 최종 분할 파일을 만들고 검증했다. 검증 분할에 예약된 사용자 3,781,518명 중 **325,639명**이 평가 조건을 만족한다. [분할 파일 안내](../modeling_splits.md)를 참고하세요.

원본 위치는 `EDA/data/raw/kieran_original/`, 분석용 추출 파일·작업 DB는 `EDA/data/processed/kieran_original/`입니다. ZIP SHA256는 `ec3234d1ff0831f5a20752059df1c85dfe66ab0efde95b886c7cc3a11d0430b7`입니다. 원본·개별 계정 자료는 Git에서 제외합니다.

[전체 데이터 EDA 수치](../../reports/kieran_original_frank_eda_metrics.json) · [게임별 긍정 사용자 수](../../reports/tables/16_kieran_original_app_support.csv) · [종합 보고서](../../reports/kieran_frank_eda.md)
