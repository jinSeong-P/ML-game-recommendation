# 모델링 분할 파일 — split_v1

2026-11-07 확정한 Kieran → frank 규칙으로 **분할 파일 생성과 저장 파일 전수 검증을 완료**했다. 버전은 `kieran-frank-split-v1`, seed는 **20261107**이다. 계정별 자료는 Git 제외 경로인 `EDA/data/processed/modeling/split_v1/`에 있다. 2026-11-08 소개글 원문·토큰·TF-IDF 인덱스·코사인 검색 준비도 완료했다. [콘텐츠 준비 결과](tfidf_content.md)를 참고한다. 2026-11-08 이 분할로 Item-KNN 학습과 두 모델의 공통 검증·frank 테스트도 완료했다. [모델 결과](../reports/baseline_v1.md)를 참고한다.

## 생성한 파일

아래 경로는 `split_v1/` 기준이다. 앱·계정 수와 행 수를 구분한다.

| 파일 | 실제 저장 규모 | 용도 |
|---|---:|---|
| `train/positive_catalog.parquet` | **78,315,268쌍·29,836,980계정** | Item-KNN 학습 입력. 고정 입력 카탈로그의 긍정만 포함, 비Linux·긍정 singleton 유지 |
| `train/interactions.parquet` | 98,857,701쌍·34,014,115계정 | 모든 앱·긍정/부정 라벨을 보존한 학습 분할. 모델에 그대로 넣는 파일이 아님 |
| `validation/interactions.parquet` | 10,977,308쌍·3,781,518계정 | 전체 검증 예약 이력. 모델 학습에 사용하지 않음 |
| `validation/users.parquet` | **325,639계정** | 숨김 후 전체 ≥5·긍정 ≥1을 만족하는 검증 평가 사용자 |
| `validation/query.parquet` | 3,920,857쌍 | 정답을 제거한 검증 입력 이력, 두 라벨 보존 |
| `validation/targets.parquet` | 325,639쌍 | 검증 사용자당 긍정 Linux 정답 1개 |
| `test/users.parquet` | **24,342계정** | 동일 조건의 frank 테스트 사용자 |
| `test/query.parquet` | 461,599쌍 | 정답을 제거한 frank 입력 이력, 두 라벨 보존 |
| `test/targets.parquet` | 24,342쌍 | 테스트 사용자당 긍정 Linux 정답 1개 |
| `{validation,test}/exclusions.parquet` | 각 query와 같은 행 수 | 남은 긍정·부정 게임 모두의 재추천 제외 목록. 숨긴 정답 없음 |
| `{train,validation,test}/review_lineage.parquet` | 98,859,771 / 10,977,506 / 485,947행 | 해당 분할의 원 리뷰 ID 계보. 원본의 동일 라벨 반복 행도 보존하므로 상호작용 쌍 수와 다를 수 있음 |
| `reserved_test_accounts.parquet` | **87,573계정** | 학습·검증에서 제외하는 전체 frank 예약 집단. 테스트 적격자보다 큼 |
| `catalog/input_apps.parquet` | 37,610개 앱 | 고정 입력 범위와 Linux 표시 |
| `catalog/linux_candidates.parquet` | 6,630개 앱 | 모델 공통 추천·정답 후보 |
| `manifest.json` | 파일별 행 수·스키마·SHA256 | seed, 분할 규칙, 소스·코드 지문, 런타임, 전수 검사 결과 |

Parquet은 총 **17개, 1,894,504,662바이트**다. `user_key`와 `review_key`는 손실 없는 `uint64`이며 float로 변환하지 않는다. `interactions`는 `(user_key, app_id, positive)`, query/target은 여기에 `linux`를 포함한다. 사용자 명단에는 숨김 후 전체·긍정 입력 수와 선택 전 적격 정답 수를 기록했다.

## 분할과 정답 고정

Kieran은 기존 모호 리뷰·라벨 충돌 격리를 유지한 정리본을 사용한다. frank 예약 **87,573명 전체**를 먼저 제외하고, 나머지 계정은 `MD5("steam-eda-validation-v1:<정확한 SteamID>")`의 앞 15자리 16진 정수 modulo 10이 0이면 검증, 나머지는 학습으로 분리했다. 평가 조건 때문에 탈락한 테스트 계정을 학습에 되돌리지 않았다.

검증·테스트 모두 고정 입력 카탈로그 안의 이력에서 **긍정 Linux 정답을 하나 숨긴 뒤 전체 입력 ≥5·긍정 입력 ≥1**을 요구한다. 사용자별 적격 정답 App ID를 오름차순 정렬하고, SHA256의 counter-mode 거절 샘플링으로 균등 의사무작위 인덱스를 선택했다. 해시 입력은 `steam-hidden-positive-linux-v1:20261107:<validation 또는 test>:<정확한 계정 ID>:<counter>`다. 입력 행 순서와 무관하며 실제 선택 결과도 `targets.parquet`에 저장했다. 시간순 미래 예측 분할은 아니다.

원본·참조 파일 지문을 검증하고 기존 EDA DB의 **영속 전처리 테이블을 읽기 전용으로 사용**했다. 마이그레이션 전 경로를 가리키는 `raw`·`frank_input` 뷰는 사용하지 않았다. 재사용한 DB의 SHA256도 manifest에 고정했다. 과거 EDA JSON과 그림은 당시의 후보 규모·기대값 기록으로 보존한다.

## 검증 결과와 모델에서 읽는 방법

저장한 파일을 다시 읽어 **83개 위반 검사 모두 0**임을 확인했다. 학습/검증 계정·사용자–게임·원 리뷰 ID 교집합, 전체 예약 계정의 학습·검증 유입, query–정답 및 제외 목록–정답 겹침, 잘못된 정답·중복·최소 입력 위반이 없다. 긍정 singleton **17,042,189계정**도 학습 빈도·코사인 정규화에 남아 있다. 추가 [소스 대조 감사](../reports/modeling_split_v1_source_audit.json)는 예약 집단 전체의 원 리뷰 ID 겹침·격리 누락·입력 이력 누락과 seed 정답 재현을 확인한다. [생성 결과](../reports/modeling_split_v1.json)에 집계와 파일 지문을 기록했다.

모델은 `train/positive_catalog.parquet`만 협업 학습에 사용한다. 검증·테스트 취향은 각 `query.parquet`의 **긍정 행만 동일 가중 평균**하고, 제외 마스크는 `exclusions.parquet` 전체를 읽는다. 두 모델 모두 같은 사용자·정답·6,630개 Linux 후보를 사용한다. 양수 개인화 점수만 최대 10개 반환하고 부족한 자리는 비운다. 빈 추천을 받은 적격 사용자도 Recall@10·NDCG@10 분모에 남긴다. TF-IDF는 별도로 생성한 `tfidf_content_v1`의 게임 ID 매핑과 희소 벡터를 읽는다. 분할 파일에는 소개글 특징을 중복 저장하지 않는다.

## 재현

프로젝트 루트에서 실행한다. 생성기는 기존 출력 폴더를 덮어쓰지 않고, 검증을 통과한 임시 출력만 최종 경로로 이동한다.

```powershell
EDA/.venv/Scripts/python EDA/scripts/build_modeling_splits.py --output EDA/data/processed/modeling/split_v1_rebuild --seed 20261107
EDA/.venv/Scripts/python EDA/scripts/verify_modeling_splits.py EDA/data/processed/modeling/split_v1
EDA/.venv/Scripts/python EDA/scripts/audit_modeling_split_sources.py
EDA/.venv/Scripts/python -m unittest discover -s EDA/scripts -p test_modeling_splits.py -v
```

정답 선택·논리적 분할의 재현과 Parquet의 바이트 순서를 구분한다. 공개한 버전의 파일 SHA256은 저장 파일의 변경 감지에 사용한다. 전체 입력 ≥3, 다른 가중 집계, 인기순 보완, 여러 정답 숨김은 기존 [백로그](modeling_backlog.md)에 남아 있다.
