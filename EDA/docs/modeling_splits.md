# 모델링 분할 파일 — split_v1

2026-11-07 확정한 Kieran → frank 규칙에 따라 **데이터 분할 파일을 만들고 저장된 전체 데이터를 검증했다.** 버전은 `kieran-frank-split-v1`, 난수 seed는 **20261107**이다. 계정별 자료는 Git 제외 경로인 `EDA/data/processed/modeling/split_v1/`에 있다. 2026-11-08에는 소개글 원문·토큰·TF-IDF 인덱스·코사인 검색 자료도 준비했다. [콘텐츠 준비 결과](tfidf_content.md)를 참고한다. 2026-11-08 이 분할로 Item-KNN 학습과 두 모델의 공통 검증·frank 테스트도 완료했다. [모델 결과](../reports/baseline_v1.md)에 실제 성능을 기록했다.

## 생성한 파일

아래 경로는 `split_v1/` 기준이다. 앱·계정 수와 행 수를 구분한다.

| 파일 | 실제 저장 규모 | 용도 |
|---|---:|---|
| `train/positive_catalog.parquet` | **78,315,268쌍·29,836,980계정** | Item-KNN 학습 입력. 고정 입력 카탈로그에서 긍정 리뷰만 포함하고 비Linux 게임과 긍정 리뷰가 1개인 사용자도 유지 |
| `train/interactions.parquet` | 98,857,701쌍·34,014,115계정 | 모든 앱·긍정/부정 라벨을 보존한 학습 분할. 모델에 그대로 넣는 파일이 아님 |
| `validation/interactions.parquet` | 10,977,308쌍·3,781,518계정 | 전체 검증 예약 이력. 모델 학습에 사용하지 않음 |
| `validation/users.parquet` | **325,639계정** | 정답을 숨긴 뒤 전체 입력 ≥5·긍정 입력 ≥1인 검증 사용자 |
| `validation/query.parquet` | 3,920,857쌍 | 정답을 뺀 검증 입력 이력. 긍정·부정 라벨 모두 보존 |
| `validation/targets.parquet` | 325,639쌍 | 검증 사용자당 긍정 Linux 정답 1개 |
| `test/users.parquet` | **24,342계정** | 동일 조건의 frank 테스트 사용자 |
| `test/query.parquet` | 461,599쌍 | 정답을 뺀 frank 입력 이력. 긍정·부정 라벨 모두 보존 |
| `test/targets.parquet` | 24,342쌍 | 테스트 사용자당 긍정 Linux 정답 1개 |
| `{validation,test}/exclusions.parquet` | 각 입력 이력과 같은 행 수 | 입력에 남은 긍정·부정 게임을 모두 추천에서 제외. 숨긴 정답은 포함하지 않음 |
| `{train,validation,test}/review_lineage.parquet` | 98,859,771 / 10,977,506 / 485,947행 | 해당 분할의 원 리뷰 ID를 추적하는 자료. 원본에서 같은 라벨로 반복된 행도 보존하므로 상호작용 쌍 수와 다를 수 있음 |
| `reserved_test_accounts.parquet` | **87,573계정** | 학습·검증에서 제외한 frank 예약 사용자 전체. 테스트 평가 조건을 만족하는 사용자보다 많음 |
| `catalog/input_apps.parquet` | 37,610개 앱 | 고정 입력 범위와 Linux 표시 |
| `catalog/linux_candidates.parquet` | 6,630개 앱 | 모델 공통 추천·정답 후보 |
| `manifest.json` | 파일별 행 수·열 구조·SHA256 | seed, 분할 규칙, 자료·코드 해시값, 실행 환경, 전체 데이터 검사 결과 |

Parquet 파일은 총 **17개, 1,894,504,662바이트**다. `user_key`와 `review_key`는 64비트 부호 없는 정수(`uint64`)로 저장해 정밀도 손실이 생기는 실수형으로 바꾸지 않는다. `interactions`는 `(user_key, app_id, positive)` 열을, 입력 이력과 정답 파일은 여기에 `linux` 열을 포함한다. 사용자 명단에는 정답을 숨긴 뒤의 전체·긍정 입력 수와 숨기기 전 적격 정답 수를 기록했다.

## 분할과 정답 고정

Kieran에는 이전과 같이 게임 연결이 모호한 리뷰와 라벨 충돌을 분석에서 제외한 정리본을 사용한다. frank 예약 사용자 **87,573명 전체**를 먼저 제외한다. 나머지 계정은 `MD5("steam-eda-validation-v1:<정확한 SteamID>")`의 앞 15자리를 16진수 정수로 해석해 10으로 나눈 나머지가 0이면 검증, 아니면 학습으로 분리했다. 평가 조건을 만족하지 못한 frank 예약 계정도 학습에 되돌리지 않았다.

검증·테스트 사용자는 고정 입력 카탈로그의 리뷰에서 **긍정 Linux 정답 1개를 숨긴 뒤 전체 입력 ≥5·긍정 입력 ≥1**이어야 한다. 사용자마다 조건을 만족하는 정답 App ID를 오름차순으로 정렬하고, SHA256 counter-mode 거절 샘플링(해시를 반복 계산해 선택 편향을 막는 방식)으로 같은 확률의 의사무작위 인덱스를 골랐다. 해시에 넣은 값은 `steam-hidden-positive-linux-v1:20261107:<validation 또는 test>:<정확한 계정 ID>:<counter>`다. 입력 행 순서와 무관하게 같은 결과가 나오며, 실제 선택 결과도 `targets.parquet`에 저장했다. 시간순 미래 예측을 위한 분할은 아니다.

원본·참조 파일의 해시값을 확인하고 기존 EDA DB에 저장된 **전처리 결과 테이블을 읽기 전용으로 사용**했다. 이전 경로를 가리키는 데이터베이스 뷰(view)인 `raw`·`frank_input`은 사용하지 않았다. 재사용한 DB의 SHA256도 파일 목록·검증 정보(manifest)에 기록했다. 과거 EDA JSON과 그림은 당시 후보 규모와 기대값 기록으로 보존한다.

## 검증 결과와 모델에서 읽는 방법

저장한 파일을 다시 읽어 **83개 오류 검사 모두 통과했고 위반 수가 0**임을 확인했다. 학습·검증 계정, 사용자–게임 조합, 원 리뷰 ID가 겹치지 않는다. 예약 계정이 학습·검증에 들어오지 않고, 입력 이력과 정답, 제외 목록과 정답도 겹치지 않는다. 정답 오류·중복·최소 입력 조건 위반도 없다. 긍정 리뷰가 1개인 사용자 **17,042,189명**도 게임별 긍정 사용자 수와 코사인 정규화에 포함한다. 추가 [소스 대조 감사](../reports/modeling_split_v1_source_audit.json)는 예약 사용자 전체의 원 리뷰 ID 중복, 모호 리뷰 제외 누락, 입력 이력 누락과 seed에 따른 정답 재현을 확인한다. [생성 결과](../reports/modeling_split_v1.json)에 집계와 파일 해시값이 있다.

협업 모델은 `train/positive_catalog.parquet`만 학습에 사용한다. 검증·테스트 사용자의 취향은 각 `query.parquet` 입력 이력에서 **긍정 리뷰만 같은 비중으로 평균**한다. 추천 제외 목록은 `exclusions.parquet` 전체를 읽는다. 두 모델은 같은 사용자·정답·Linux 후보 6,630개를 사용한다. 점수가 양수인 후보를 최대 10개 반환하고 남는 자리는 비워 둔다. 추천이 비어도 조건을 만족한 사용자는 Recall@10·NDCG@10 계산에 포함한다. TF-IDF 모델은 별도로 생성한 `tfidf_content_v1`에서 게임 ID와 희소 벡터를 읽는다. 분할 파일에 소개글 특징을 중복 저장하지 않는다.

## 재현

프로젝트 루트에서 실행한다. 생성기는 기존 출력 폴더를 덮어쓰지 않는다. 검증을 통과한 임시 출력만 최종 경로로 옮긴다.

```powershell
EDA/.venv/Scripts/python EDA/scripts/build_modeling_splits.py --output EDA/data/processed/modeling/split_v1_rebuild --seed 20261107
EDA/.venv/Scripts/python EDA/scripts/verify_modeling_splits.py EDA/data/processed/modeling/split_v1
EDA/.venv/Scripts/python EDA/scripts/audit_modeling_split_sources.py
EDA/.venv/Scripts/python -m unittest discover -s EDA/scripts -p test_modeling_splits.py -v
```

정답 선택과 데이터 분할 규칙이 다시 만들어지는지, Parquet 파일의 바이트 순서까지 같은지를 구분한다. 공개 버전의 파일 SHA256은 저장된 파일이 바뀌었는지 확인하는 데 쓴다. 전체 입력 ≥3, 다른 가중 집계, 인기도순 빈자리 보완, 여러 정답 숨김은 기존 [백로그](modeling_backlog.md)에 남아 있다.
