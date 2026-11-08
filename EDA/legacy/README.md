# 레거시 데이터셋·EDA 보관

2026-11-07 **Kieran 원본 → frank 리뷰**를 확정하면서 이전 자료를 이 폴더로 이동했습니다. 현재 기준은 [메인 EDA](../README.md)와 [확정 조합 보고서](../reports/kieran_frank_eda.md)입니다. 아래 문서는 당시의 입력 데이터·판정·수치를 보존한 이력이며 현재 프로젝트의 기본 학습·평가 구성을 뜻하지 않습니다.

| 보관 자료 | 위치·이유 |
|---|---|
| 이전 A: Game Recommendations on Steam | `data/raw/A_game_recs/`, `data/processed/A_*`; 익명 사용자 이력은 새 학습에 혼합하지 않음 |
| 이전 B: Australian users/items/reviews | `data/raw/B_ucsd_steam/`, `data/processed/B_*`; 리뷰 기반 평가 규모 부족 |
| 이전 C: Steam Games 2025 | `data/raw/C_steam_games_2025/`, `data/processed/C_*`; 전체 자료 보관, 게임 단위 연결 여부만 고정 참조로 유지 |
| artermiloff 2024 | `data/raw/candidate_review/artermiloff_2024/`, `data/processed/candidate_identity/artermiloff_2024/`; 원본 계정 ID 반올림 손상으로 사용 제외 |
| VintageDon 및 다른 후보 조사 | `data/raw/candidate_review/`, `data/processed/combination_research/`, `reports/`; 채택 전 비교·독립성 감사 이력 |
| 이전 노트북·그래프·표·코드 | `notebooks/`, `figures/`, `reports/`, `scripts/`; 실행 당시 결과·근거 보존 |

로컬에서 확보한 원본과 중간 파일도 **실제로 이동**했으며 대용량 자료와 계정별 파일은 계속 Git에서 제외합니다. 다운로드하지 않은 후보는 조사 문서만 있습니다. Kieran과 frank의 원본·분석 투영은 이 폴더에 중복 복사하지 않고 메인 `../data/`에 둡니다.

이전 비교 보고서의 링크를 보존하기 위해 Kieran 전수 집계 JSON·게임별 지원 CSV와 당시 공통 분석 코드는 여기에 사본을 남겼습니다. 레거시 문서·manifest 속 경로와 재현 명령은 **이동 전 경로**를 포함할 수 있습니다. 이 폴더의 스크립트는 당시 근거를 보존한 것이며 새 메인 실행 진입점은 `../scripts/`입니다. 현재 경로 대응은 [이동 기록](migration_manifest.json)에서 확인합니다.

- [A·B·C 데이터 사전](docs/datasets/README.md)
- [기본 EDA](reports/eda_summary.md), [기존 데이터 충분성 EDA](reports/eda_sufficiency_summary.md)
- [후보 조사](reports/test_dataset_candidate_review.md), [전체 조합 조사](reports/dataset_train_test_combinations.md)
- [A–VintageDon 독립성 감사](reports/vintage_independence_audit.md)
- [artermiloff·Kieran 비교 전수 EDA](reports/large_review_frank_eda.md)
- [이전 EDA 안내 원문](ORIGINAL_EDA_README.md), [이전 프로젝트 안내 원문](ORIGINAL_PROJECT_README.md)

고정 Linux 게임 카탈로그·메타데이터 연결 여부·알려진 모호 리뷰 목록은 [메인 참조](../data/reference/README.md)로 추출해 출처와 지문을 기록했습니다. 이 예외는 이전 사용자 이력을 메인에 남기거나 학습에 넣는다는 뜻은 아닙니다.
