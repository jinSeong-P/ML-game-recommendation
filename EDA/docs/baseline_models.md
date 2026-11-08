# TF-IDF · Item-KNN 베이스라인 v1

<!-- v3-current -->
**현재 선택 모델은 [v3 보고서](../reports/model_improvement_v3.md)와 [선택 잠금](../data/reference/improvement_selection_v3.json)을 따릅니다.** 이 문서 아래의 v1 설정·명령은 원래 베이스라인을 재현하기 위해 보존합니다.

- **협업:** 사용자 활동량의 역수로 학습 기여를 보정한 Item-KNN. 검증 Recall@10 40.08% → **40.93%**, frank 참조 11.19% → **10.86%**.
- **콘텐츠:** 소개글 TF-IDF와 확장 메타데이터 결합. 검증 Recall@10 9.51% → **14.51%**, frank 참조 6.70% → **9.64%**.

콘텐츠는 두 평가에서 개선됐지만 협업 모델은 frank에서 하락했습니다. 협업 모델의 일반적인 성능 향상을 입증한 결과로 해석하지 않습니다. frank 결과로 재선택하지 않았습니다.

현재 모델은 검증에서 선택한 뒤 고정했고, 같은 모델을 frank 참조 평가와 보조 평가에 재사용했습니다. 이미 리뷰한 긍정·부정 게임을 제외하며 양수 점수만 최대 10개 추천하고 빈자리는 유지합니다. 인기도 보완은 별도 정책 실험으로만 보고했습니다.

```powershell
EDA/.venv/Scripts/python EDA/scripts/recommend_selected_v3.py --family collaborative --positive-apps 10 --exclude-apps 10 240
EDA/.venv/Scripts/python EDA/scripts/recommend_selected_v3.py --family content --positive-apps 10 --exclude-apps 10 240
```

활동량 가중 Item-KNN은 같은 사용자가 남기는 게임 쌍의 기여를 활동량으로 보정하고 가중 코사인을 계산합니다. 콘텐츠는 선택 manifest의 각 채널을 독립 정규화한 뒤 결합합니다. 정확한 설정명·식·출처는 모델 manifest와 v3 보고서에 기록합니다.
<!-- /v3-current -->

2026-11-08 고정한 초기 설정이다. 성능을 보고 설정을 고르지 않고, 두 모델을 그대로 내부 검증한 뒤 같은 모델로 frank 테스트를 평가한다. 모델별 사용자·숨긴 정답·입력·제외 목록·Linux 후보는 동일하다. 상세 결과는 [평가 보고서](../reports/baseline_v1.md)에 기록한다. 후속 개선은 사용자가 승인한 [v2 실행 절차](model_improvement_execution.md)에 따라 별도 코드·프로토콜·산출물로 진행하며, [v2 결과](../reports/model_improvement_v2.md)에 측정·선정을 구분한다. 이 문서의 고정 v1 설정은 유지한다.

v2 비교는 완료했다. 내부 검증 보호 기준에 따라 Item-KNN은 이 문서의 K=100을 유지하고, 콘텐츠는 unigram+bigram 소개글과 별도 태그·장르 코사인을 50:50으로 결합한 모델을 선정했다. [선정 스냅샷](../data/reference/improvement_selection_v2.json)과 v2 보고서가 후속 선택의 기준이며 아래 명령은 계속 원래 v1을 재현한다.

## 두 모델의 계산

**TF-IDF**는 기존 `tfidf_content_v1`을 사용한다. A 소개글 우선·C 짧은 소개글 결측 보충, 단어 unigram·smoothed IDF·게임별 L2 정규화 설정을 유지한다. 남은 긍정 게임 벡터를 같은 비중으로 평균한 뒤, Linux 후보와 코사인 유사도를 계산한다. 설명·어휘·IDF는 고정 카탈로그의 공개 메타데이터에서만 얻으며 리뷰 정답 라벨을 학습하지 않는다. 별도 벡터DB는 사용하지 않는다.

**Item-KNN**은 `split_v1/train/positive_catalog.parquet`의 긍정 사용자–게임 78,315,268쌍으로 이진 사용자×게임 행렬을 만든다. 고정 입력 37,610개 게임과 긍정 singleton을 유지하며 검증·테스트 리뷰를 공동 선호 학습에 넣지 않는다.

두 게임의 유사도는 `공동 긍정 사용자 수 / sqrt(게임 i 긍정 사용자 수 × 게임 j 긍정 사용자 수)`다. 각 Linux 후보는 전체 입력 게임 중 코사인 유사도가 가장 높은 **100개 이웃**을 유지한다. 자기 자신은 제외하며, 공동 사용자 1명 이상·shrinkage 0을 초기 기준으로 쓴다. 동점은 App ID 오름차순이다. 이웃 100개는 검증으로 최적화한 값이 아니다.

추천 후보별 점수는 `남은 긍정 입력과의 저장된 유사도 합 / 전체 긍정 입력 수`다. 이웃에 없거나 학습 지원이 없는 입력도 분모에 포함하고 기여는 0이다. 원래 코사인은 대칭이지만 후보마다 이웃을 선택한 저장 그래프는 대칭일 필요가 없다. 후보별 이웃 100개와 사용자 입력 중 상위 100개 선택은 다른 방식이다.

## 공통 추천·평가 규칙

- 남은 긍정 리뷰만 프로필·점수에 사용한다. 남은 긍정·부정 리뷰 게임은 모두 재추천에서 제외한다.
- 숨긴 긍정 Linux 정답 1개는 입력·프로필·제외 목록에 없다. 사용자는 숨김 뒤 전체 입력 ≥5·긍정 입력 ≥1을 만족한다.
- Linux 후보 6,630개 중 양수 개인화 점수만 최대 10개 추천한다. 동점은 App ID 오름차순이며 부족한 자리는 비운다.
- 빈 목록과 미지원 정답을 포함한 고정 사용자 전체가 지표 분모다. 단일 정답이므로 Recall@10은 적중 사용자 비율이고 NDCG@10은 적중 순위 `r`에서 `1/log2(r+1)`, 미적중이면 0이다.
- 내부 검증 비교 기준은 전체 사용자 NDCG@10 우선·동점이면 Recall@10이다. 이번에는 탐색·튜닝·검증 이력 재학습 없이 두 초기 모델을 모두 보고한다.
- train 인기도와 균등 무작위 기대값은 별도 진단이다. 개인화 추천의 빈자리를 채우는 데 사용하지 않는다.

인기도 구간은 train의 양수 지원 Linux 후보에 대한 25·50·75% 분위수로만 정한다. 지원 0은 별도 구간이며 경계 동점은 상위 구간에 넣는다. 빈 추천·짧은 목록·카탈로그 커버리지·미지원 입력/정답도 함께 기록한다.

## 재현과 저장

프로젝트 루트의 PowerShell에서 실행한다. 학습·평가는 기존 완료 디렉터리를 덮어쓰지 않는다. [고정 프로토콜](../data/reference/baseline_protocol_v1.json)과 `baseline_run_v1.json`에는 성능 계산 전 고정한 설정·모델·평가 코드·분할 지문을 남긴다. 테스트 실행에는 같은 지문의 완료된 검증 결과가 필요하다.

```powershell
EDA/.venv/Scripts/python EDA/scripts/build_item_knn.py --neighbors 100 --block-size 64
EDA/.venv/Scripts/python EDA/scripts/verify_item_knn.py --report EDA/reports/item_knn_v1_audit.json
EDA/.venv/Scripts/python EDA/scripts/freeze_baseline_run.py
EDA/.venv/Scripts/python EDA/scripts/evaluate_baselines.py --cohort validation --output EDA/data/processed/modeling/baseline_v1/validation
EDA/.venv/Scripts/python EDA/scripts/evaluate_baselines.py --cohort test --validation-result EDA/data/processed/modeling/baseline_v1/validation --output EDA/data/processed/modeling/baseline_v1/test
EDA/.venv/Scripts/python EDA/scripts/tfidf_content.py --positive-apps 10 --exclude-apps 10 240 --top-k 10
EDA/.venv/Scripts/python EDA/scripts/recommend_baseline.py --model item_knn --positive-apps 10 --exclude-apps 10 240 --top-k 10
```

게임별 학습 지원 수·이웃 유사도·ID 매핑·파라미터·출처는 `data/processed/modeling/item_knn_v1/`에 저장한다. 평가 경로의 모델별 `recommendations.parquet`에는 사용자·추천 게임·순위·점수, `per_user.parquet`에는 숨긴 정답·적중 순위·지표·지원 상태를 저장한다. 사용자별 파일은 Git 제외 경로에만 보관한다. 요약 보고서에는 집계와 지문만 기록한다.

위 명령은 최초 생성 순서다. 현재 완료 산출물을 다시 평가할 때는 새 출력 경로를 사용한다. 모델·프로토콜·평가 코드가 바뀌면 기존 v1 실행 잠금을 덮어쓰지 않고 새 버전과 지문으로 별도 실험을 만든다.

행렬은 희소 형식으로 저장하고, 학습 중 공동 선호는 후보 블록별로 계산한다. 전체 사용자×게임 밀집 행렬이나 전체 게임 쌍의 밀집 유사도 행렬을 저장하지 않는다. 평가 시에도 사용자 배치 단위로만 후보 점수를 계산한다.

## 해석 범위

시간순 미래 예측이 아닌 리뷰 이력의 일부를 숨긴 평가다. frank의 게임당 리뷰 상한·리뷰어 선정 편향과 고정된 과거 Linux 카탈로그의 한계를 유지한다. TF-IDF의 Linux 영벡터 17개와 CJK 분절 한계도 그대로다. 이 결과를 전체 Steam 사용자·최신 출시 게임·현재 Linux 지원 여부에 일반화하지 않는다.

[분할](modeling_splits.md) · [TF-IDF 준비](tfidf_content.md) · [확정 규칙·후속 백로그](modeling_backlog.md)
