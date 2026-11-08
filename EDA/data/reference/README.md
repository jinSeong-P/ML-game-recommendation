# 고정 게임·평가 참조

<!-- v3-current -->
[improvement_protocol_v3.json](improvement_protocol_v3.json)은 v3 사전 실험값·평가·선택 조건을 담는다. [improvement_selection_v3.json](improvement_selection_v3.json)은 검증 선택과 코드·결과 지문을 새 frank 참조 전에 고정한 공개 기록이다. 기존 v1/v2 참조와 분할·카탈로그 데이터는 보존했다. [v3 결과](../../reports/model_improvement_v3.md) 참조.
<!-- /v3-current -->

Kieran→frank EDA의 게임 범위와 알려진 모호 리뷰를 고정한 작은 참조 파일입니다. 메인 분석은 아래 파일을 읽으며 레거시 A·B·C 또는 artermiloff 원본을 다시 읽지 않습니다.

| 파일 | 내용 | 행 수 |
|---|---|---:|
| `catalog.parquet` | `app_id`, `linux`, `reviewed`. 이전 A v28의 게임 목록·Linux 표시·리뷰 존재 여부 | 50,872 |
| `content_catalog.parquet` | `app_id`, `in_A_metadata`, `in_C_metadata`, `has_description`, `has_tags`. 이전 A+C의 메타데이터 연결·존재 여부 | 104,614 |
| `review_app_quarantine.parquet` | `review_key`. 전수 후보 감사에서 frank와 연결이 모호하다고 확인된 원 리뷰 ID | 296 |
| [manifest.json](manifest.json) | 출처 파일·감사 결과·참조 파일 지문, frank 고정 commit | — |
| [baseline_protocol_v1.json](baseline_protocol_v1.json) | 성능 계산 전 고정한 두 모델·추천·평가·비교 규칙 | — |
| [baseline_run_v1.json](baseline_run_v1.json) | 검증 실행 전 고정한 모델·분할·평가 코드 지문 | — |
| [improvement_protocol_v2.json](improvement_protocol_v2.json) | v2 성능 조회 전 고정한 후보값·보호 기준·frank 참조 정책 | — |
| [improvement_selection_v2.json](improvement_selection_v2.json) | 내부 검증만으로 고른 최종 모델·코드·검증 결과 지문. frank 재평가 전 동결하는 공개 스냅샷 | — |

입력은 `reviewed=True`인 37,610개 게임, 정답 후보는 그중 `linux=True`인 6,630개 게임입니다. Kieran 최종 학습안에 리뷰가 남는 Linux 게임은 6,603개입니다. 나머지 게임을 평가에서 모델별로 임의 제거하지 않습니다.

2026-11-07 확정한 Item-KNN 베이스라인은 이 고정 입력 37,610개 게임의 긍정 리뷰를 사용하며, 비Linux 게임과 긍정 singleton의 빈도·정규화를 유지합니다. TF-IDF 원문은 A `description` 우선·결측 시 C `short_description` 보충으로 별도 구성했고, 토큰·TF-IDF 벡터·코사인 검색까지 준비했습니다. 실제 텍스트는 고정 지문을 확인한 legacy 메타데이터에서 연결했으며 [콘텐츠 준비 결과](../../docs/tfidf_content.md)에 저장 위치를 기록했습니다. 확정 규칙은 [소스 manifest](../source_manifest.json)와 [모델링 결정·백로그](../../docs/modeling_backlog.md)에 기록했습니다. 이 결정으로 기존 참조 파일이나 지문을 변경하지는 않았습니다.

이 참조에는 계정 ID·사용자 이력·리뷰 라벨과 실제 설명·태그 원문이 없습니다. `content_catalog`는 텍스트 **존재 여부**만 나타내므로 TF-IDF 모델 입력으로 바로 쓸 수 없습니다. 과거 A/C 지표 명칭은 계보를 보존한 컬럼 이름입니다.

모호 리뷰 목록은 이전 두 원본의 **리뷰–게임 연결 모호성** 합집합입니다. artermiloff의 손상된 계정 ID와 frank의 정확한 계정 ID가 다르다는 이유만으로 frank 리뷰를 제거하지 않았습니다. 기존 입력 범위에서 해당 목록은 77쌍·77계정과 잠재 Linux 긍정 정답 38쌍에 영향을 줍니다.

완료된 감사의 참조를 다시 만들 때만 `EDA/scripts/freeze_reference_catalog.py`를 사용합니다. 이 재구축 도구는 `EDA/legacy/data/`의 기존 자료와 두 감사 DB가 필요합니다. 일반 메인 EDA·노트북 실행에는 이 단계가 필요하지 않습니다. 참조 변경은 평가 모집단 변경이므로 버전·지문·학습 제외 집단을 다시 검증해야 합니다.

v2 실험은 기존 카탈로그·분할·v1 실행 지문을 그대로 사용합니다. `run_improvement_v2.py select`는 사전 지정 후보만 허용하고 선택 기록을 처리 경로와 위 참조 경로에 동일하게 저장합니다. `reference`는 두 기록의 일치, 현재 코드·모델·검증 파일 지문을 다시 확인한 뒤 선택 모델만 실행합니다. frank는 이미 관측한 자료이므로 재평가도 독립 최종 확인이 아닌 참조 결과입니다. [실행 결과](../../reports/model_improvement_v2.md)를 참고하세요.
