# 데이터 분석과 모델 평가

Steam 리뷰 데이터의 품질을 확인하고 Linux 게임 추천 모델을 평가하는 폴더입니다. 처음 읽는 경우 [최근 모델 평가](reports/forgemaster_model_evaluation.md), [데이터 분석 결과](reports/forgemaster_eda.md), [용어 설명](docs/terminology.md) 순서로 보면 됩니다.

## 최근 결과

2026-11-08에 Forgemaster 원본 CSV 11개 전체를 다운로드하고 리뷰 15,437,471행을 검사했습니다. 같은 사용자·게임의 중복과 긍정·부정 충돌을 정리한 뒤, 게임당 리뷰 수를 제한하지 않고 395,421명을 평가했습니다.

| 모델 | Recall@10 | NDCG@10 |
|---|---:|---:|
| 기본 Item-KNN | 29.05% | 0.17003 |
| 활동량 보정 Item-KNN | 29.49% | 0.17218 |
| 콘텐츠 TF-IDF | 13.09% | 0.07338 |
| 학습 데이터의 인기순 추천 | 21.24% | 0.12993 |

이번 평가를 위해 Kieran 원본 v2를 다시 받았습니다. Forgemaster 평가 계정과 기존 frank 평가용 계정의 리뷰를 학습에서 모두 제외하고, 긍정 리뷰 72,292,011쌍으로 협업 모델을 다시 학습했습니다. 학습·평가 계정 중복은 0입니다. 콘텐츠 모델은 기존 게임 설명과 메타데이터로 다시 만들었습니다.

**리뷰 수 제한이 없는 평가를 중심으로 모델 성능을 설명합니다.** 100개 제한 실험은 제한 자체의 영향을 따로 확인합니다. 같은 사용자·정답에서 기본 Item-KNN의 Recall은 평균 1.34%p, 활동량 보정 모델은 1.77%p 낮아졌습니다. 콘텐츠 모델의 Recall은 일관되게 낮아지지 않았습니다. 제한 후에는 평가 가능한 사용자도 약 6천 명으로 줄었습니다.

전체 64개 평가 조건에서 저장된 추천과 사용자별 지표를 검산했습니다. 상세 조건·신뢰구간·학습 계정 중복 검사는 [모델 평가 보고서](reports/forgemaster_model_evaluation.md)에 있습니다. 이 평가는 2021년까지의 일부 게임을 다루므로 최신 게임 전체를 대표하지 않습니다.

## 파일 위치

| 경로 | 내용 |
|---|---|
| `data/raw/forgemaster_v1/` | 원본 CSV와 다운로드 기록 |
| `data/raw/kieran_original/` | 복원한 Kieran 원본 ZIP |
| `data/processed/forgemaster_v1/` | 중복·충돌을 정리한 리뷰와 분석용 DB |
| `data/processed/modeling/forgemaster_cap_study_v1/` | 이번 실험의 학습 데이터, 모델, 평가 대상과 추천 결과 |
| `data/reference/` | 실험에서 공통으로 쓰는 게임 목록과 사전에 정한 실험 설정 |
| `scripts/` | 다운로드·분석·학습·평가·보고서 생성 코드 |
| `notebooks/` | 분석 코드와 실행 결과를 함께 보는 노트북 |
| `reports/` | 평가 조건·수치·해석을 정리한 보고서 |
| `reports/tables/`, `figures/` | 집계표와 그래프 |
| `legacy/` | 이전 데이터셋·코드·분석 기록 |

대용량 원본과 가공 데이터, 계정별 파일은 Git에서 제외합니다. 파일 이름과 JSON·CSV의 열 이름은 재현 코드에서 사용하므로 문서의 설명과 구분합니다. 예를 들어 `query`는 사용자의 입력 리뷰 이력, `target`은 평가에서 숨긴 정답 게임, `support`는 해당 게임을 긍정 평가한 학습 사용자 수를 뜻합니다.

## 실행 방법

저장소 루트에서 PowerShell로 실행합니다. 현재 환경은 `.venv/Scripts/python`을 사용합니다. 기존 파일의 크기와 해시값이 맞으면 다운로드와 학습 결과를 재사용합니다.

```powershell
.venv/Scripts/python -m pip install -r EDA/requirements.txt
.venv/Scripts/python EDA/scripts/download_forgemaster.py
.venv/Scripts/python EDA/scripts/fetch_frank_review_comparison.py
.venv/Scripts/python EDA/scripts/verify_forgemaster_csv.py
.venv/Scripts/python EDA/scripts/eda_forgemaster.py
.venv/Scripts/python EDA/scripts/audit_forgemaster_results.py
.venv/Scripts/python EDA/scripts/build_forgemaster_report.py
```

모델 재학습과 평가 명령은 [모델 평가 보고서의 재현 방법](reports/forgemaster_model_evaluation.md#검증과-재현-방법)을 따릅니다. Kieran 원본 ZIP은 약 17.24GB이며 변환 파일과 작업 공간도 필요합니다. 과거 `split_v1`과 새 실험의 분할은 별도 경로에 저장합니다.

## 분석 노트북

- [Forgemaster 데이터 분석](notebooks/02_forgemaster_eda.ipynb): 품질, 분포, 평가 가능 사용자와 100개 제한 영향
- [기존 Kieran·frank 데이터 분석](notebooks/01_kieran_frank_eda.ipynb): 당시 원본 크기와 계정 분리 검사

노트북의 수치는 각 분석 당시의 결과입니다. 모델 성능과 최근 평가 분할은 해당 보고서에서 확인합니다.

## 이전 실험과 설계 문서

<!-- v3-current -->
이전 v3 실험은 내부 검증에서 83개 설정을 비교해 활동량 보정 Item-KNN과 소개글·확장 메타데이터 TF-IDF를 선택했습니다. 당시 분할의 내부 검증 Recall@10은 각각 40.93%, 14.51%이며 frank 결과는 10.86%, 9.64%입니다. 이번 Forgemaster 실험은 평가 계정을 추가로 제외해 다시 학습했으므로 이 수치와 직접 비교하지 않습니다.

[v3 실험 결과](reports/model_improvement_v3.md) · [v2 실험 결과](reports/model_improvement_v2.md) · [초기 모델 결과](reports/baseline_v1.md)
<!-- /v3-current -->

- [모델 수식과 실행 방법](docs/baseline_models.md)
- [데이터 분할 방법과 저장 파일](docs/modeling_splits.md)
- [데이터셋별 설명](docs/datasets/README.md)
- [콘텐츠 모델 입력과 TF-IDF](docs/tfidf_content.md)
- [정답 게임 선택 방법](docs/holdout_options.md)
- [모델 개선 계획](docs/model_improvement_plan.md)
- [후속 작업 목록](docs/modeling_backlog.md)
- [이전 Kieran·frank 분석 보고서](reports/kieran_frank_eda.md)
- [이전 평가 방법 점검](reports/kieran_frank_midpoint_audit.md)

과거 파일의 날짜·수치·실험 조건을 확인한 뒤 해석합니다. 읽기 쉬운 표현과 용어의 의미는 [문장 작성 기준](docs/terminology.md)에 정리했습니다.
