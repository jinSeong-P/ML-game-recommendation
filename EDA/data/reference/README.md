# 게임 목록과 실험 설정 기록

<!-- v3-current -->
[improvement_protocol_v3.json](improvement_protocol_v3.json)은 v3 실험 전에 정한 모델 설정·평가 방법·선택 기준을 담습니다. [improvement_selection_v3.json](improvement_selection_v3.json)은 내부 검증으로 고른 모델과 코드·결과의 해시값을 frank 재평가 전에 기록한 파일입니다. 기존 v1/v2 설정과 데이터 분할·게임 목록은 보존했습니다. [v3 결과](../../reports/model_improvement_v3.md)를 참고하세요.
<!-- /v3-current -->

Kieran 학습·frank 평가에 사용할 게임 목록과 제외할 충돌 리뷰를 기록한 참조 파일입니다. 분석은 아래 파일을 읽습니다. 이전 A·B·C 또는 artermiloff 원본을 다시 읽을 필요는 없습니다. [용어 설명](../../docs/terminology.md)에서 파일별 기록의 의미를 확인할 수 있습니다.

| 파일 | 내용 | 행 수 |
|---|---|---:|
| `catalog.parquet` | `app_id`, `linux`, `reviewed`. 이전 A v28의 게임 목록·Linux 표시·리뷰 존재 여부 | 50,872 |
| `content_catalog.parquet` | `app_id`, `in_A_metadata`, `in_C_metadata`, `has_description`, `has_tags`. 이전 A+C의 메타데이터 연결·존재 여부 | 104,614 |
| `review_app_quarantine.parquet` | `review_key`. 전체 리뷰 연결 검사에서 frank와의 게임 연결이 모호하다고 확인한 원 리뷰 ID | 296 |
| [manifest.json](manifest.json) | 출처 파일·검사 결과·참조 파일 해시값, frank 데이터의 지정 Git 커밋 | — |
| [baseline_protocol_v1.json](baseline_protocol_v1.json) | 성능 계산 전에 정한 두 모델·추천·평가·비교 규칙 | — |
| [baseline_run_v1.json](baseline_run_v1.json) | 검증 실행 전에 기록한 모델·분할·평가 코드 해시값 | — |
| [improvement_protocol_v2.json](improvement_protocol_v2.json) | v2 실험 전에 정한 후보 설정·성능 하락 허용 기준·frank 재평가 방법 | — |
| [improvement_selection_v2.json](improvement_selection_v2.json) | 내부 검증으로 고른 최종 모델·코드·검증 결과의 해시값. frank 재평가 전에 기록 | — |

입력은 `reviewed=True`인 37,610개 게임, 정답 후보는 그중 `linux=True`인 6,630개 게임입니다. 기존 Kieran 최종 학습 데이터에 리뷰가 남는 Linux 게임은 6,603개입니다. 학습 리뷰가 없는 후보도 모든 모델의 평가 목록에 유지합니다.

2026-11-07 확정한 기준 Item-KNN은 이 37,610개 게임의 긍정 리뷰를 사용합니다. 비Linux 게임과 긍정 리뷰가 한 개뿐인 사용자도 빈도·정규화 계산에 포함합니다. TF-IDF 원문은 A의 `description`을 우선 사용하고, 없으면 C의 `short_description`으로 보충했습니다. 토큰·TF-IDF 벡터·코사인 검색을 준비했으며, 원문은 파일 해시값을 확인한 이전 메타데이터에서 연결했습니다. [콘텐츠 준비 결과](../../docs/tfidf_content.md)에 저장 위치가 있습니다. 확정한 규칙은 [데이터 출처·처리 기록](../source_manifest.json)과 [모델링 결정과 남은 작업](../../docs/modeling_backlog.md)에 기록했습니다. 기존 참조 파일과 해시값은 보존했습니다.

이 참조에는 계정 ID·사용자 이력·리뷰 라벨과 실제 설명·태그 원문이 없습니다. `content_catalog`는 텍스트 **존재 여부**만 나타내므로 TF-IDF 모델 입력으로 바로 쓸 수 없습니다. A/C가 들어간 열 이름은 기존 데이터 출처와 연결하기 위해 유지했습니다.

충돌 리뷰 목록은 이전 두 원본에서 **같은 리뷰가 여러 게임에 연결되는 경우**를 합친 것입니다. artermiloff의 손상된 계정 ID와 frank의 정확한 계정 ID가 다르다는 이유만으로 frank 리뷰를 제거하지 않았습니다. 기존 입력 범위에서 해당 목록은 77쌍·77계정과 가능한 Linux 긍정 정답 38쌍에 영향을 줍니다.

이 참조 파일을 다시 만들 때만 `EDA/scripts/freeze_reference_catalog.py`를 사용합니다. 재구축에는 `EDA/legacy/data/`의 기존 자료와 두 검사 DB가 필요합니다. 일반 EDA·노트북 실행에는 이 단계가 필요하지 않습니다. 참조를 바꾸면 평가 대상도 달라지므로 버전·해시값·학습에서 제외한 계정을 다시 확인해야 합니다.

v2 실험은 기존 게임 목록·분할·v1 실행 파일 해시값을 그대로 사용합니다. `run_improvement_v2.py select`는 미리 정한 후보 설정만 허용하고, 선택 기록을 처리 경로와 위 참조 경로에 동일하게 저장합니다. `reference`는 두 기록과 현재 코드·모델·검증 파일의 해시값이 일치하는지 확인한 뒤 선택한 모델만 실행합니다. frank는 개발 중 이미 확인한 자료이므로 재평가를 새 독립 테스트로 해석하지 않습니다. [실행 결과](../../reports/model_improvement_v2.md)를 참고하세요.
