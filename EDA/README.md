# Kieran → frank 메인 EDA

<!-- v3-current -->
**현재 결과: [백로그 개선 실험 v3](reports/model_improvement_v3.md).** 같은 검증의 설정 83개와 별도 보조 평가·시간 cutoff 재학습을 완료했습니다.

- **협업:** 사용자 활동량의 역수로 학습 기여를 보정한 Item-KNN. 검증 Recall@10 40.08% → **40.93%**, frank 참조 11.19% → **10.86%**.
- **콘텐츠:** 소개글 TF-IDF와 확장 메타데이터 결합. 검증 Recall@10 9.51% → **14.51%**, frank 참조 6.70% → **9.64%**.

콘텐츠는 두 평가에서 개선됐습니다. 협업 모델은 내부 검증에서 개선됐지만 frank에서는 하락했으므로 일반적인 성능 향상으로 단정하지 않습니다. frank 결과로 재선택하지 않았습니다.

아래의 소개글 전용 TF-IDF·단순 코사인 Item-KNN 설명은 고정 **v1 베이스라인 이력**입니다. 현재 선택과 실행은 [모델 안내](docs/baseline_models.md), 실험별 완료·미실행 조건은 [백로그](docs/modeling_backlog.md)를 따릅니다. frank는 재사용한 참조 평가로 구분합니다.
<!-- /v3-current -->

**확정 조합: Kieran 원본(v2)으로 학습·내부 검증하고 frank 리뷰로 외부 테스트합니다.** 2026-11-07 전수 EDA 결과를 이 조합 중심으로 정리했습니다. 최종 분할 파일과 seed **20261107**·정답 manifest를 생성하고 저장 파일 전수 검증을 완료했습니다. [분할 파일 안내](docs/modeling_splits.md)에 실제 규모·경로·검증 결과를 정리했습니다. 2026-11-08 소개글 보충·정리·토큰화·TF-IDF 인덱스와 코사인 검색도 준비했습니다. [콘텐츠 준비 결과](docs/tfidf_content.md)를 참고하세요. 2026-11-08 Item-KNN 학습과 두 모델의 공통 내부 검증·frank 테스트도 완료했습니다. [학습·평가 결과](reports/baseline_v1.md)와 [모델 실행 안내](docs/baseline_models.md)에 수치·설정·검증 근거를 기록했습니다.

먼저 [결과 보고서](reports/kieran_frank_eda.md), [실행 노트북](notebooks/01_kieran_frank_eda.ipynb), [데이터 사전](docs/datasets/README.md)을 읽으면 됩니다. 모델별 문제와 근거는 [Item-KNN·TF-IDF 개선 기획안](docs/model_improvement_plan.md), 사용자가 승인한 네 단계와 시각화는 [v2 실행 절차](docs/model_improvement_execution.md)에 정리했습니다. 실제 측정·보호 지표·선정은 [v2 실행 결과](reports/model_improvement_v2.md)에서 확인합니다.

EDA·모델링 계획의 의심점 10개에 대한 [중간점검 판정](reports/kieran_frank_midpoint_audit.md)을 추가했습니다. 분포 차이와 불완전 이력을 검증하고 인기도·무작위 기준선 기대값을 계산했습니다. 중간점검 당시에는 모델 성능이 없었으며, 이후 완료한 실제 모델 결과는 [베이스라인 보고서](reports/baseline_v1.md)로 구분합니다.

Item-KNN 학습 범위는 **고정 입력 카탈로그 37,610개 앱(비Linux 포함)의 긍정 리뷰**이며 긍정 singleton도 앱 지원 수와 코사인 정규화에 유지합니다. EDA 미리보기는 이 범위에서 78,315,268개 긍정 사용자–게임 쌍입니다. 학습안 전체 앱의 모든 라벨은 98,857,701쌍(긍정 84,791,317쌍)으로 집계되어 있으며, 이는 Item-KNN의 고정 카탈로그 범위와 구분합니다.

v1 TF-IDF는 A `description`을 우선하고 결측·빈 문자열·공백뿐인 경우 C `short_description`으로 보충한 **소개글만** 사용합니다. 태그·장르는 v1에 넣지 않고 별도 v2 채널 결합으로 비교합니다. 두 모델의 사용자 취향·추천 점수는 남은 **긍정 리뷰만** 사용하며, 남은 긍정·부정 리뷰 게임은 재추천에서 제외하고 숨긴 정답은 제외 마스크에서 제거합니다. 기본 평가는 사용자별 적격 긍정 Linux 게임 하나를 균등 무작위로 숨기며, 숨김 후 전체 입력 ≥5·긍정 입력 ≥1을 요구합니다. 실제 생성한 테스트 명단은 **24,342명**이며 내부 검증 평가 명단은 **325,639명**입니다. 두 모델에 같은 입력·정답·후보를 적용하고, 동일 가중 평균으로 집계합니다. TF-IDF는 게임별 L2 정규화 벡터 평균 후 후보와 코사인 유사도를 계산하며, Item-KNN은 후보별 긍정 입력 유사도 합을 전체 긍정 입력 수로 나눕니다. 양수 개인화 점수만 최대 10개 추천하고 부족한 자리는 비웁니다. 적격 사용자는 추천 목록이 비어도 지표 분모에 유지합니다. seed **20261107**과 실제 명단을 저장했습니다. 부정 신호 반영, 여러 정답 동시 숨김, 전체 입력 ≥3 평가, 대안 집계, 모델 개선 후 인기순 보완은 후속 백로그입니다. 세부 사항은 [모델링 결정·백로그](docs/modeling_backlog.md), [정답 숨김 방법 비교](docs/holdout_options.md)에 기록했습니다.

| 단계 | 사용자 | 사용자–게임 쌍 |
|---|---:|---:|
| Kieran 정리 후 | 37,875,512 | 113,195,034 |
| frank 예비 평가 계정 전체 제외 후 | 37,795,633 | 109,835,009 |
| 내부 검증 예약 | 3,781,518 | 10,977,308 |
| 학습안 | 34,014,115 | 98,857,701 |

frank 예비 평가 계정 **87,573명**을 통째로 예약하며 Kieran에서 실제로 제외된 계정은 **79,879명**입니다. 이 예약 집단은 최종 적격 평가자가 줄어도 학습에 되돌리지 않습니다. 알려진 리뷰–게임 연결 모호성을 제거하고 긍정 Linux 정답 하나를 숨긴 뒤, 전체 입력 ≥5인 EDA 후보는 **24,571명**입니다. 이 중 숨김 뒤 긍정 입력이 없는 229명을 공통 제외한 평가 조건의 **잠재 사용자 수는 24,342명**입니다. 동일한 24,342명을 실제 명단에 저장하고 검증했습니다. 가능한 정답 102,602쌍을 동시에 숨기는 실험이 아닙니다.

## 현재 구조

```text
EDA/
├── docs/datasets/             # Kieran·frank 데이터 사전, 분리 원칙
├── data/
│   ├── source_manifest.json   # 메인 데이터셋 버전·위치·역할
│   ├── reference/             # 고정 게임 목록·Linux 플래그·연결 여부·모호 리뷰 ID
│   ├── raw/kieran_original/   # 원본 ZIP, Git 제외
│   └── processed/            # kieran_original/, frank/; Git 제외
├── scripts/                  # 메인 수집·전수 EDA·집계 검증·노트북 생성
├── notebooks/01_kieran_frank_eda.{py,ipynb}
├── reports/
│   ├── kieran_frank_eda.md
│   ├── kieran_frank_eda_metrics.json           # 재검증·frank 재집계
│   ├── kieran_original_frank_eda_metrics.json  # Kieran 전체 원본 EDA 증거
│   └── tables/               # Kieran 지원 수·메인 요약표
├── figures/                  # 메인 EDA 그래프
└── legacy/                   # 이전 데이터셋·코드·노트북·조사 결과
```

`data/reference/`는 이전 A의 리뷰 존재 게임·공식 Linux 표시와 A+C 메타데이터 **존재 여부**를 고정한 참조입니다. 현재 입력 37,610개·Linux 후보 6,630개 기준을 보존합니다. 사용자 이력·리뷰 라벨·설명 본문·태그 원문은 들어 있지 않습니다. 확정된 A/C 소개글 원문·토큰·TF-IDF 벡터는 `data/processed/modeling/tfidf_content_v1/`에 별도로 생성·검증했습니다. 유효 벡터는 전체 37,527개·Linux 6,613개이며 결측·영벡터 게임도 카탈로그에 남겼습니다. 참조 파일의 출처와 지문은 [참조 안내](data/reference/README.md)에 있습니다.

## 실행 방법

저장소 루트 PowerShell 기준입니다. 기존 완료 파일이 있으면 다시 다운로드하거나 수집하지 않습니다.

```powershell
EDA/.venv/Scripts/python -m pip install -r EDA/requirements.txt

# 기존 전수 분석 파일의 지문·행 수 확인 + frank 후보·지원 범위 재집계
EDA/.venv/Scripts/python EDA/scripts/summarize_kieran_frank.py

# 메인 노트북 실행 및 표·그래프 갱신
EDA/.venv/Scripts/python EDA/scripts/build_notebook.py

# 사용자 전체 제외·원 리뷰 충돌 격리 회귀 검사
EDA/.venv/Scripts/python -m unittest discover -s EDA/scripts -p 'test_*.py' -v
```

최초 수집 또는 Kieran 전수 분석을 다시 수행할 때만 아래를 실행합니다. Kieran 압축 원본 약 17.24GB, 압축 내부 주 CSV 약 42.49GB이며, 변환 파일·DuckDB 작업 공간도 필요합니다. 전수 EDA 기본 설정은 메모리 8GB·4스레드입니다.

```powershell
EDA/.venv/Scripts/python EDA/scripts/ingest_kieran_original.py
EDA/.venv/Scripts/python EDA/scripts/fetch_frank_identity_columns.py
EDA/.venv/Scripts/python EDA/scripts/eda_large_review_frank.py
EDA/.venv/Scripts/python EDA/scripts/summarize_kieran_frank.py
EDA/.venv/Scripts/python EDA/scripts/build_notebook.py
```

frank는 고정 commit의 리뷰·보유 이력 Parquet에서 필요한 열의 전체 행을 투영했습니다. 현재 테스트는 **리뷰만 사용**합니다. 원본 전체 파일과 리뷰 본문을 다운로드·검증했다는 뜻은 아닙니다. 기존 수집 manifest의 경로는 당시 위치를 보존했으며 현재 위치는 [메인 출처 manifest](data/source_manifest.json)와 [이동 기록](legacy/migration_manifest.json)에서 확인합니다.

기존 Kieran 전수 JSON·지원 CSV는 바이트를 보존했습니다. 당시 분석 코드도 `legacy/scripts/eda_large_review_frank.py`에 보존되어 JSON의 코드 지문을 추적할 수 있습니다. 활성 코드는 새 경로와 고정 참조를 읽도록 갱신했습니다. 집계 검증 스크립트는 Kieran 46개 파일의 SHA256·행 수·수집 manifest·지원 CSV를 확인하고, frank와 참조 지문을 확인한 뒤 평가 후보를 다시 계산합니다.

## 평가 시 유지할 원칙

- 학습·검증·테스트는 원 Steam 계정 단위로 분리합니다. 같은 게임의 다른 계정 리뷰는 학습할 수 있습니다.
- 테스트 계정 87,573명은 최종 적격 후보보다 넓게 예약합니다. 적격 후보 정리 후에도 학습에 되돌리지 않습니다.
- Item-KNN은 고정 입력 37,610개 게임(비Linux 포함)의 긍정 리뷰를 학습에 사용합니다. 긍정 singleton은 아이템 지원 수와 코사인 정규화에 유지합니다. 모든 리뷰 라벨은 분할 자료에 보존하되 베이스라인 점수에는 긍정만 사용합니다.
- 두 모델 모두 긍정 입력을 동일 가중 평균으로 집계합니다. 후보 점수가 0 이하이거나 지원이 없으면 추천하지 않으며, 개인화 추천을 최대 10개까지만 내고 빈자리는 유지합니다. 적격 사용자의 빈 목록도 Recall@10·NDCG@10 분모에서 제외하지 않습니다.
- 숨김 뒤 전체 입력 ≥5·긍정 입력 ≥1을 만족하는 같은 24,342명 잠재 평가 집단, 정답, 6,630개 Linux 후보를 두 모델에 공통 적용합니다. 남은 긍정·부정 게임은 후보 제외 마스크에 넣고 숨긴 정답은 마스크에서 제거합니다.
- 최종 파일을 만든 뒤 계정·원 리뷰·사용자–게임 분리, 입력–정답 분리와 Linux 긍정 정답 조건을 저장 파일에서 다시 검사합니다.
- frank의 앱당 최대 100리뷰·활동적인 리뷰어 선정 편향을 보고합니다. 현재 EDA는 시간순 미래 예측이나 추천 성능 검증이 아닙니다.

이전 A·B·C 기반 EDA와 후보 비교는 [레거시 안내](legacy/README.md)에서 확인합니다. 레거시의 당시 권장안과 성능·충분성 수치를 현재 조합의 결과로 사용하지 않습니다.
