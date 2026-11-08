# Linux 게임 추천

Steam 리뷰 이력을 바탕으로 Linux 지원 게임을 추천합니다. 사용자들이 함께 긍정 평가한 게임을 찾는 **Item-KNN**과, 게임 소개글·태그 등의 텍스트를 비교하는 **TF-IDF 콘텐츠 추천**을 사용합니다. Linux 지원 여부는 프로젝트에 저장된 게임 목록의 표시를 따릅니다.

## 최근 모델 평가

2026-11-08에 Forgemaster 데이터 전체를 내려받아 품질을 확인하고 모델을 평가했습니다. **게임당 리뷰 수를 100개로 제한하지 않은 결과를 중심으로 해석합니다.** 제한을 적용한 결과는 사용자 이력이 줄어들 때 성능이 어떻게 달라지는지 확인하는 데 사용합니다.

다음은 평가 조건을 만족한 **395,421명**의 결과입니다. 평가 계정의 모든 Kieran 리뷰를 학습에서 제외하고 모델을 다시 학습했습니다. 학습 사용자와 평가 사용자는 겹치지 않습니다.

| 모델 | Recall@10 | NDCG@10 |
|---|---:|---:|
| 기본 Item-KNN | 29.05% | 0.17003 |
| 활동량 보정 Item-KNN | 29.49% | 0.17218 |
| 콘텐츠 TF-IDF | 13.09% | 0.07338 |
| 학습 데이터의 인기순 추천 | 21.24% | 0.12993 |

같은 사용자와 정답을 유지한 5개 표본 비교에서는 100개 제한 후 협업 모델의 Recall이 평균 1.34%p와 1.77%p 낮아졌습니다. 콘텐츠 모델의 Recall은 일관되게 낮아지지 않았습니다. 제한 후에는 평가 대상 사용자도 약 6천 명으로 달라지므로, 전체 평가의 점수 차이를 모두 입력 손실로 해석하면 안 됩니다. 이 데이터는 2021년 4월까지의 일부 게임을 다루며 최신 게임 전체의 성능을 나타내지는 않습니다.

[모델 평가 결과와 비교 조건](EDA/reports/forgemaster_model_evaluation.md) · [데이터 품질과 분포 분석](EDA/reports/forgemaster_eda.md)

## 데이터의 역할

| 데이터 | 용도 | 해석할 때 확인할 점 |
|---|---|---|
| [Kieran Steam Reviews v2](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) | 모델 학습과 내부 검증 | Steam 계정별로 나눠 같은 사람이 학습과 평가에 함께 들어가지 않게 합니다. |
| [Forgemaster Steam Reviews v1](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset) | 리뷰 수 제한이 없는 추가 평가와 제한 영향 분석 | 고정 Linux 후보 6,630개 중 2,210개가 관측됩니다. |
| [frank Steam Dataset](https://huggingface.co/datasets/frankjc2022/steam-dataset/tree/75694cc07a6cdc69f48f9c316ca1e3e38a3f0f89) | 기존 실험과 비교하는 외부 데이터 평가 | 게임당 최대 100개 리뷰가 수집된 자료입니다. 이미 여러 번 확인한 결과이므로 새 독립 테스트로 부르지 않습니다. |

## 추천과 평가 방법

모델이 입력으로 받는 게임은 37,610개이며, 추천할 수 있는 Linux 게임은 그중 6,630개입니다. 사용자가 긍정 평가한 게임으로 추천 점수를 계산하고, 긍정·부정 여부와 관계없이 이미 리뷰한 게임은 추천에서 제외합니다.

평가에서는 사용자별 긍정 Linux 게임 하나를 무작위로 골라 숨기고 정답으로 사용합니다. 남은 리뷰 게임이 5개 이상이고 그중 긍정 게임이 1개 이상인 사용자를 평가합니다. 추천 점수가 양수인 게임을 최대 10개 제시하며, 빈 추천 목록도 평가에 포함합니다.

**Recall@10**은 정답 게임을 추천 10개 안에서 찾은 사용자의 비율입니다. **NDCG@10**은 정답을 더 높은 순위로 추천할수록 높은 점수를 줍니다. 게임당 리뷰 100개 제한과 Item-KNN의 이웃 수 **K=100**은 서로 다른 설정입니다.

## 실행과 문서

최근 실험의 모델·학습 데이터·사용자별 평가 결과는 `EDA/data/processed/modeling/forgemaster_cap_study_v1/`에 저장되어 있습니다. 대용량 데이터와 계정별 파일은 Git에서 제외합니다. 과거 모델 파일을 그대로 복구한 것이 아니라, 기존 설정으로 별도 학습한 결과입니다.

- [EDA 실행과 파일 안내](EDA/README.md)
- [최근 모델 평가의 재현 명령](EDA/reports/forgemaster_model_evaluation.md#검증과-재현-방법)
- [모델 수식과 과거 실험 실행 방법](EDA/docs/baseline_models.md)
- [학습·검증·테스트 데이터 구성](EDA/docs/modeling_splits.md)
- [데이터별 설명](EDA/docs/datasets/README.md)
- [용어와 문장 작성 기준](EDA/docs/terminology.md)

## 이전 모델 개선 실험

<!-- v3-current -->
2026-11-08에 완료한 v3 실험은 같은 내부 검증 데이터에서 설정 83개를 비교했습니다. 아래 수치는 당시 학습·평가 분할의 결과이며, 위 Forgemaster 재학습 결과와 직접 비교하지 않습니다.

| 선택 모델 | 내부 검증 Recall@10 | frank Recall@10 |
|---|---:|---:|
| 활동량 보정 Item-KNN | 40.93% | 10.86% |
| 소개글·확장 메타데이터 TF-IDF | 14.51% | 9.64% |

콘텐츠 모델은 두 평가에서 개선됐습니다. 협업 모델은 내부 검증에서 개선됐지만 frank에서는 낮아졌습니다. 모델 설정은 내부 검증으로 선택했으며 frank 결과를 보고 다시 고르지 않았습니다.

[v3 실험 결과](EDA/reports/model_improvement_v3.md) · [v2 실험 결과](EDA/reports/model_improvement_v2.md) · [초기 모델 결과](EDA/reports/baseline_v1.md) · [후속 작업 목록](EDA/docs/modeling_backlog.md)
<!-- /v3-current -->

이전 데이터셋과 분석 코드는 [보관 자료](EDA/legacy/README.md)에 있습니다. 게임 목록과 게임 설명의 출처는 [참조 파일 안내](EDA/data/reference/README.md)에서 확인할 수 있습니다.
