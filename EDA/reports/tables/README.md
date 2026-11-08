# 메인 EDA 결과표

[model_improvement_v4.csv](model_improvement_v4.csv)는 새 v4 분할의 설정별 성능, 대조군과의 차이·신뢰구간, 보호 기준 통과 여부를 담습니다. 실행 완료 수와 범위는 [v4 보고서](../model_improvement_v4.md)를 기준으로 확인합니다. v3·Forgemaster와 평가 대상이 달라 점수의 직접 비교에는 사용하지 않습니다.

[multi_target_hit_rates_v3.csv](multi_target_hit_rates_v3.csv)는 기존 v3 정답 2개 평가의 모델·평가 집단별 1개 이상·2개 모두 적중률, 평균 적중 수, Precision@10을 담습니다. 지표의 정의와 범위는 [정답 2개 적중률 보고서](../multi_target_hit_rates_v3.md)를 참고하세요.

<!-- v3-current -->
[model_improvement_v3.csv](model_improvement_v3.csv)는 v3의 검증·기존 frank 평가·보조 평가 결과와 채택 기준을 통과하지 못한 이유, 실행 시간을 담습니다. `variant`는 재현에 쓰는 설정 ID입니다. 방법명과 지표의 정의·해석은 [v3 보고서](../model_improvement_v3.md)를 참고하세요. 사용자별 자료는 Git에서 제외한 경로에 보관합니다. 아래 01~17은 이전 EDA 기록입니다.
<!-- /v3-current -->

| 파일 | 단위·내용 |
|---|---|
| [01_main_split_preview.csv](01_main_split_preview.csv) | Kieran 정리 후·테스트 계정 제외 후·검증 예약·학습 데이터의 계정/리뷰/긍정 쌍 수 |
| [02_frank_eligibility.csv](02_frank_eligibility.csv) | 긍정 정답 1개를 숨긴 뒤 남는 최소 입력 수별 잠재 평가 후보·가능한 정답 쌍 |
| [03_kieran_quality.csv](03_kieran_quality.csv) | 원본 식별자·게임 연결·라벨·플레이 시간 품질 검사 |
| [16_kieran_original_app_support.csv](16_kieran_original_app_support.csv) | Kieran 원본 전체 행을 분석해 집계한 App ID별 학습 리뷰·긍정 리뷰 수와 고정 카탈로그 포함 여부 |
| [17_quarantined_app_groups.csv](17_quarantined_app_groups.csv) | 앱 연결이 서로 달랐던 원 리뷰 ID의 앱 조합·행 수·카탈로그 교차. 올바른 앱 연결이나 다시 연결해도 안전한 규칙을 확정하지 않음 |

01~03은 [메인 노트북](../../notebooks/01_kieran_frank_eda.ipynb)에서 생성합니다. `16_` 파일은 이전 전체 원본 집계 산출물의 파일명과 내용을 유지해 해시값을 검증하는 데 사용합니다.

`16_`의 `all_reviews/all_positive`는 정리 후 전체, `post_test_reviews/post_test_positive`는 평가 계정을 모두 제외한 뒤, `final_train_reviews/final_train_positive`는 검증 계정까지 학습에서 제외한 뒤의 학습 데이터입니다. 사용자–게임 중복을 제거했으므로 긍정 리뷰 수는 해당 게임에 긍정 리뷰를 남긴 서로 다른 계정 수입니다. `in_fixed_A_input_catalog`, `linux_in_A_catalog`는 이전 A에서 고정한 게임 목록에 들어 있는지 나타내는 값이며, A의 사용자 이력을 썼다는 뜻은 아닙니다.

현재 표는 EDA·분할 미리보기입니다. 추천 정확도·공동 긍정 이웃의 존재나 최종 저장 파일의 비중복을 인증하지 않습니다. 이전 데이터셋의 표는 [레거시 결과표](../../legacy/reports/tables/README.md)에 있습니다.
