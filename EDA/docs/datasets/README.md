# A·B·C 원본 파일별 데이터 사전

이 문서는 **Linux 지원 Steam 게임 개인화 추천 프로젝트**의 원본 데이터 파일별 특징을 설명하는 목차다. 팀이 베이스라인 학습셋·외부 평가셋을 확정하기 전에 확인할 수 있도록 정리했다.

## 먼저 구분할 것
| 출처 | 역할 | 사용자별 선호 신호 | 게임 정보 | 기본 판단 |
|---|---|---|---|---|
| **A** — Game Recommendations on Steam (v28) | 모델 학습 | **있음:** 긍정·부정 리뷰 | 게임·Linux 플래그·태그·설명 | 사용자 취향 학습의 중심 |
| **B** — Steam Video Game and Bundle Data (v1) | 독립 평가 | **있음:** 보유·플레이 이력 + 긍정·부정 리뷰 | 일부 게임 메타데이터 | **입력 보유 목록**과 **정답 리뷰**를 분리 |
| **C** — Steam Games Dataset (v2) | 콘텐츠 보강 | **없음:** 사용자별 기록 없음 | 게임·태그·장르·설명 등 47열 | A의 텍스트 빈 값 보강 |

**핵심 원칙**
- **게임끼리는 Steam App ID로 조인**: A `app_id` ↔ B `item_id`/`id` ↔ C `appid`.
- **사용자 ID는 데이터셋마다 독립적**. A 사용자 번호와 B 사용자 번호를 같은 사람으로 취급하지 않는다.
- **공식 Linux 지원 판정은 A `games.csv`의 `linux`**. Proton·Steam Deck 호환성을 섞지 않는다.
- 모델의 학습·튜닝은 **A 내부에서만**. B의 리뷰 정답은 마지막 외부 평가에 사용한다.
- **빈 값과 미관측을 부정 선호로 간주하지 않는다.** '게임을 보유함'도 '좋아함'과 동의어가 아니다.

## 파일별 상세 설명

### A · 사용자 추천 리뷰 및 게임 카탈로그

- [**A 데이터셋 전체 구조 및 ER 관계도**](A.md) — 네 파일의 역할, 연결 키, PK·FK에 해당하는 관계를 한눈에 확인

1. [`recommendations.csv`](A_recommendations.csv.md) — 사용자별 긍정/부정 리뷰 4,115만 건, 활동량 희소성·k-core·라벨 설계.
2. [`games.csv`](A_games.csv.md) — 게임 50,872개, 공식 Linux 9,041개, 출시일·가격·지원 조건.
3. [`users.csv`](A_users.csv.md) — 사용자 1,430만 명의 보유 수·리뷰 수 **집계**, 보유 게임 ID 목록은 없음.
4. [`games_metadata.json`](A_games_metadata.json.md) — 설명·태그, 빈 태그의 평가 가중 영향과 C 보강 필요성.

### B · 새로운 사용자 기반 외부 테스트
5. [`australian_users_items.json`](B_australian_users_items.json.md) — 사용자 보유·플레이 입력, 중복, A와의 게임 ID 교집합.
6. [`australian_user_reviews.json`](B_australian_user_reviews.json.md) — 긍정 리뷰 정답, 연도 없는 날짜, 정답 누출 방지.
7. [`steam_games.json`](B_steam_games.json.md) — B 자체 게임 정보와 결측, A와 불일치하는 게임 조사.
8. [`bundle_data.json`](B_bundle_data.json.md) — 게임 번들 관계, **사용자 선호 신호가 아님**.
9. [`steam_reviews.json/steam_new.json`](B_steam_new_partial.md) — 4.3GB 파일의 일부만 검사한 흔적. **분석·학습·평가 미사용**.

### C · 텍스트·게임 속성 보강
10. [`games_march2025_cleaned.csv`](C_games_march2025_cleaned.csv.md) — 게임 89,618개×47열, 태그/설명/장르/플랫폼 필드와 누출 위험.

## 학습셋·평가셋을 정할 때

| 구분 | 학습/평가에 실제 사용할 내용 | 제외·추가 검증할 내용 |
|---|---|---|
| 학습 A | `recommendations.csv`에서 사용자–게임 **긍정 리뷰**(초기 Item-KNN), `games.csv`의 공식 Linux 플래그 | 사용자·게임 k-core 임계값은 **선택되지 않음**. A 비Linux 게임 입력까지 모두 버리지 않기 |
| 콘텐츠 A+C | `games_metadata.json` 태그·설명; C로 부족분 보강 | C의 인기/평판 집계는 시점·라벨 누출 우려가 있어 첫 학습에서 분리 |
| 외부 평가 B | `australian_users_items.json`의 보유 게임을 입력, `australian_user_reviews.json`의 긍정 리뷰를 정답 | 긍정·부정 리뷰를 모두 입력에서 제거. 정답을 보고 모델 튜닝하지 않기 |
| 부가 B | `steam_games.json`·`bundle_data.json` | 첫 베이스라인 필수 아님. B 메타데이터 사용은 사전 규칙 필요 |

**평가 범위의 제약:** B에서 입력 게임 5개 이상·Linux 정답 1개 이상 조건을 만족한 사용자 **15,294명**, 정답 **24,704쌍**, 정답 게임 **829개**. 인기순 기준선 Recall@10은 **0.5386**으로 높다. 모든 Linux 게임을 동일한 신뢰도로 평가하는 것은 아니며 롱테일과 데이터 커버리지를 함께 보고해야 한다.

**2026-11-06 주요 정정:** A 태그 없는 게임 수는 적어도, 리뷰가 있는 Linux 후보 중 태그 없는 **117개**가 B 정답 **58.76%**를 차지한다. 따라서 C는 단순한 소폭 태그 개수 보완 이상으로 **평가 대상 인기 게임의 텍스트 확보에 중요**하다.

## 근거·재현 위치
- [기초 EDA 요약](../../reports/eda_summary.md) · [데이터 충분성 분석](../../reports/eda_sufficiency_summary.md)
- [원본 파일 버전·라이선스·SHA256](../../data/raw/manifest.json)
- [원본에서 파생된 표의 전체 컬럼·자료형·결측](../../reports/tables/01_schema_nulls.csv) · [행 수·ID·중복](../../reports/tables/01_structure.csv)
- [전처리 코드](../../scripts/preprocess.py) · [EDA 재현 안내](../../README.md)

## 작성 범위와 검증 수준
- **실측값**: 저장된 `01_eda`·`02_eda_data_sufficiency` 산출물에서 확인한 값. 2026-11-06 재현 분석은 원본 파일의 크기와 SHA256을 검증했다.
- **컬럼 구조**: Parquet·CSV 컬럼 사전과 `preprocess.py`의 실제 추출 키를 기준으로 작성. 특히 B의 원본 Python 리터럴에는 분석에 사용하지 않은 다른 키가 있을 수 있다.
- **전처리 제안**: 아직 구현하거나 모델 성능을 확인한 결과가 아닌 **향후 실험 아이디어**. 문서에 '미구현'이라고 표시.
- GitHub에는 원본 대용량 데이터가 없으며, B의 `steam_new.json` 전체도 검사하지 않았다. 이 문서의 수치는 원본 전체를 새로 읽어 계산한 것이 아니라 **이미 수행된 EDA 결과**의 기록이다.
