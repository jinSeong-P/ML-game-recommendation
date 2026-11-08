# ML Game Recommendation

<!-- v3-current -->
2026-11-08 백로그 개선 실험 v3를 완료했습니다. 같은 내부 검증에서 사전 지정 설정 **83개**를 비교하고 전체 성능·게임별 평균·희귀 게임·빈 목록 보호 기준으로 선택했습니다.

- **협업:** 사용자 활동량의 역수로 학습 기여를 보정한 Item-KNN. 검증 Recall@10 40.08% → **40.93%**, frank 참조 11.19% → **10.86%**.
- **콘텐츠:** 소개글 TF-IDF와 확장 메타데이터 결합. 검증 Recall@10 9.51% → **14.51%**, frank 참조 6.70% → **9.64%**.

콘텐츠는 두 평가에서 개선됐지만, 협업 모델은 내부 검증에서 개선되고 frank에서는 하락했습니다. 협업 모델의 일반적인 성능 향상으로 단정하지 않으며, frank 결과로 모델을 다시 선택하지 않았습니다.

Recall@10은 숨긴 긍정 게임을 추천 10개 안에서 찾은 사용자 비율입니다. frank는 이미 관측한 **참조 평가**이고 모델 선택에는 사용하지 않았습니다. [v3 결과·전체 실험표](EDA/reports/model_improvement_v3.md) · [현재 모델 실행](EDA/docs/baseline_models.md) · [백로그 상태](EDA/docs/modeling_backlog.md). 아래 v1/v2 수치는 초기 실험 이력입니다.
<!-- /v3-current -->

Steam 사용자의 리뷰 이력으로 **공식 Linux 지원 게임**을 추천하는 프로젝트입니다. 게임 소개글을 이용하는 TF-IDF 추천과 공동 긍정 리뷰를 이용하는 Item-KNN을 비교합니다. TF-IDF는 A `description`을 우선 사용하고 값이 결측·빈 문자열·공백뿐이면 C `short_description`으로 보충합니다. v1 베이스라인은 소개글만 쓰며, 별도 태그·장르 채널 결합은 [v2 개선 실험](EDA/reports/model_improvement_v2.md)에서 비교합니다.

두 모델의 베이스라인은 **긍정 리뷰만으로 선호·순위 점수를 계산**합니다. 부정 리뷰한 게임 자체는 재추천에서 제외하고, 부정 게임 유사도 감점·유사 후보 제외·긍정/부정 관계 학습은 후속 실험으로 둡니다.

2026-11-08 v2 진단·개선 실험을 완료했습니다. 희소 게임 보호 기준에 따라 Item-KNN은 K=100을 유지하고, 콘텐츠는 **bigram 소개글 50%+태그·장르 50%**로 선정했습니다. 콘텐츠 Recall@10은 내부 검증 **1.93%→9.51%**, 이미 관측한 frank 참조 평가 **3.58%→6.70%**입니다. [v2 전체 결과](EDA/reports/model_improvement_v2.md)에 후보별 성능·보호 지표·선정 근거와 검증을 기록했습니다.

기본 평가는 사용자별 적격 긍정 Linux 게임 하나를 균등 무작위로 숨기고, 두 모델에 같은 입력·정답·후보를 적용합니다. 숨김 뒤 남은 리뷰 게임이 전체 5개 이상이고 긍정 게임이 1개 이상인 사용자가 적격입니다. **24,342명**의 실제 테스트 사용자·입력·정답 파일을 생성하고 seed **20261107**로 고정했습니다. **여러 정답 동시 숨김**은 후속 실험입니다.

메인 데이터셋은 **Kieran 원본 → frank 리뷰**로 확정했습니다. 데이터셋 선택과 전수 EDA는 완료했으며, 최종 분할 파일과 TF-IDF 소개글·토큰·벡터·코사인 검색 준비를 완료했습니다. 2026-11-08 두 베이스라인의 구현·Item-KNN 학습·공통 내부 검증·frank 테스트를 완료했습니다. [모델 안내](EDA/docs/baseline_models.md)와 [성능 보고서](EDA/reports/baseline_v1.md)를 참고하세요. [분할 파일 안내](EDA/docs/modeling_splits.md)를 참고하세요.

| 역할 | 데이터셋 | 사용 원칙 |
|---|---|---|
| 학습·내부 검증 | [Kieran Steam Reviews v2](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) | 정확한 원 Steam 계정으로 분리. 테스트 계정의 모든 리뷰를 학습에서 제외 |
| 외부 테스트 | [frank Steam Dataset 고정 버전](https://huggingface.co/datasets/frankjc2022/steam-dataset/tree/75694cc07a6cdc69f48f9c316ca1e3e38a3f0f89) | 리뷰 일부를 입력으로 사용하고 긍정 Linux 게임 하나를 숨겨 평가 |

Kieran은 정리 후 113,195,034쌍입니다. 테스트 계정 제외와 검증 계정 예약 후 **모든 게임·긍정/부정을 합친 학습 EDA 미리보기**는 98,857,701쌍·34,014,115계정이며, 그중 모든 앱의 긍정은 84,791,317쌍입니다. Item-KNN의 확정 학습 범위는 고정 입력 카탈로그 37,610개 앱(비Linux 포함) 안의 긍정 리뷰이며, 긍정 singleton도 앱 지원 수와 코사인 정규화에 유지합니다. 이 범위의 긍정 **78,315,268쌍**을 실제 모델 학습 입력으로 저장했습니다. frank에서 전체 입력 5개 이상인 후보 24,571명 중 숨김 뒤 긍정 입력이 없는 229명을 공통 제외한 **24,342명**이 생성된 테스트 명단에 포함됩니다. 내부 검증 평가 명단은 **325,639명**입니다. 저장된 분할의 수치는 EDA 예상과 일치합니다. 같은 24,342명 테스트에서 Recall@10은 TF-IDF **3.5782%**, Item-KNN(K=100) **11.1905%**입니다. 초기 설정을 고정한 뒤 평가했으며 튜닝·검증 재학습은 하지 않았습니다.

리뷰 학습에는 비Linux 게임도 포함합니다. 현재 Item-KNN 입력 범위는 고정 참조 카탈로그의 리뷰 존재 게임 37,610개이고, 추천 후보·정답은 그중 공식 Linux 지원 6,630개입니다. 입력 집계는 긍정 게임들의 동일 가중 평균입니다. TF-IDF는 게임별 L2 정규화 소개글 벡터를 평균한 뒤 후보와 코사인 유사도를 계산하고, Item-KNN은 각 후보의 긍정 입력 유사도 합을 전체 긍정 입력 수로 나눕니다. 양수 개인화 점수만 최대 10개 추천하며 부족한 자리는 비워 둡니다. 나머지 긍정·부정 리뷰 게임은 후보에서 제외하고, 숨긴 정답은 제외 마스크에서 빼 추천 가능하게 둡니다. 적격 사용자의 빈 추천 목록도 지표 분모에 남깁니다. 게임은 Steam App ID로, 사용자 분리는 손실 없는 원 Steam 계정 ID로 확인합니다. 다른 출처라는 이유만으로 사용자 독립성을 가정하지 않습니다.

- [두 베이스라인 학습·평가 결과](EDA/reports/baseline_v1.md)
- [모델 수식·실행 안내](EDA/docs/baseline_models.md)
- [모델 개선 네 단계·시각화](EDA/docs/model_improvement_execution.md)
- [모델 개선 v2 측정 결과](EDA/reports/model_improvement_v2.md)
- [메인 EDA 안내](EDA/README.md)
- [TF-IDF 소개글·검색 인덱스](EDA/docs/tfidf_content.md)
- [분할 파일·검증 결과](EDA/docs/modeling_splits.md)
- [모델링 결정·백로그](EDA/docs/modeling_backlog.md)
- [정답 숨김 방법 비교](EDA/docs/holdout_options.md)
- [Kieran→frank EDA 결과](EDA/reports/kieran_frank_eda.md)
- [EDA·모델링 중간점검 판정](EDA/reports/kieran_frank_midpoint_audit.md)
- [데이터 사전과 분리 원칙](EDA/docs/datasets/README.md)
- [실행 노트북](EDA/notebooks/01_kieran_frank_eda.ipynb)
- [레거시 자료·후보 검토 이력](EDA/legacy/README.md)

이전 A·B·C, VintageDon, artermiloff 및 기타 후보의 자료·코드·결과는 `EDA/legacy/`에 보존했습니다. 기존 Linux 카탈로그와 메타데이터 연결 여부만 출처가 기록된 `EDA/data/reference/`의 고정 참조로 유지합니다. 이전 데이터셋의 사용자 이력을 새 학습에 섞지 않습니다.
