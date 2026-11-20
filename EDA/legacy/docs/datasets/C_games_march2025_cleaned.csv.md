# C · games_march2025_cleaned.csv — 2025년 게임 속성 카탈로그

## 1. 역할과 원본 구조
- **원본**: [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) v2, `games_march2025_cleaned.csv` (2025-03 수집, 약 469 MB).
- **한 행**: Steam App ID 하나에 대한 **게임 메타데이터**. 사용자별 긍정/부정 리뷰 행이 아니다.
- **핵심 용도**: **Linux 여부와 관계없이 A 게임 전체**의 태그·장르·설명 빈 값을 보충하고 콘텐츠 기반 TF-IDF용 게임 정보를 만든다. **C 자체에는 사용자별 선호/리뷰 행이 없다.**
- **연결**: `appid` → A의 `app_id`, B의 `item_id`. 다른 데이터셋 사용자 ID와의 연결은 없음.

## 2. 전체 규모와 중복
| 항목 | EDA 값 |
|---|---:|
| 게임 행 / 고유 `appid` | **89,618 / 89,618** |
| 컬럼 수 | **47** |
| A `games.csv`와 공통 App ID | **35,876** |
| A의 공식 Linux 게임과 공통 App ID | **6,440** |
| A/C의 Linux 표시 일치율(공통 전체) | **99.38%** |
| A에서 Linux True, C에서 False | **58개** |
| A에서 Linux False, C에서 True | **166개** |
| A/C 공통 Linux 게임의 태그 Jaccard 평균 | **약 0.873** (양쪽 태그 합집합이 비지 않은 게임) |

지원 운영체제는 수집 시점 차이에 따라 바뀔 수 있으므로, 이번 실험의 공식 Linux 정답 정의는 **A의 `games.csv` `linux`**를 기준으로 고정한다.

## 3. 컬럼 사전 — 총 47개
아래는 **Parquet로 변환된 C의 실제 컬럼 47개**를 기능별로 묶은 것이다. `scripts/preprocess.py`에서 C는 Windows/macOS/Linux 플래그를 bool로 매핑하고 CSV를 Parquet으로 저장한다.

| 영역 | 컬럼(원문명) | 베이스라인에서의 의미 |
|---|---|---|
| 식별·이름·출시 | `appid`, `name`, `release_date` | 조인 키, 표시 이름, 시점 검증 |
| 가격·판매 | `required_age`, `price`, `dlc_count`, `discount` | 연령·가격·DLC·할인 정보. 사용 시점 점검 필요 |
| 설명 텍스트 | `detailed_description`, `about_the_game`, `short_description` | TF-IDF 입력 후보. HTML/반복 문구·결측 점검 |
| 후기 텍스트 | `reviews`, `notes` | 설명 또는 상점 후기 텍스트; 누출 및 결측 주의 |
| 이미지·지원 주소 | `header_image`, `website`, `support_url`, `support_email`, `metacritic_url` | 추천 핵심 피처보다는 서비스/확인용 |
| 플랫폼 | `windows`, `mac`, `linux` | 플랫폼 지원 표시; C 시점의 스냅샷 |
| 평가·인기도 | `metacritic_score`, `user_score`, `score_rank`, `positive`, `negative`, `recommendations` | 집계 평판·추천 수; 학습 시점 누출 방지 |
| 소유·활동량 | `estimated_owners`, `average_playtime_forever`, `average_playtime_2weeks`, `median_playtime_forever`, `median_playtime_2weeks`, `peak_ccu` | 사용자별 데이터가 아니라 **게임별 집계** |
| 콘텐츠 분류 | `categories`, `genres`, `tags`, `packages` | 게임 분류·태그·묶음 정보 |
| 제작·언어·미디어 | `developers`, `publishers`, `supported_languages`, `full_audio_languages`, `screenshots`, `movies`, `achievements` | 부가 콘텐츠·개발사·언어·미디어 특성 |
| 비율·리뷰 수 | `pct_pos_total`, `num_reviews_total`, `pct_pos_recent`, `num_reviews_recent` | 상점 전체/최근 리뷰의 집계 수치 |

컬럼의 배열·딕셔너리성 값 중 상당수는 **CSV 문자열**로 저장되어 있다. 예를 들어 `tags`와 `genres`는 형식상 `str`이며 추가 분석에서 `ast.literal_eval`로 읽었다. 실질적인 빈 값은 문자열 `{}` 또는 `[]` 등으로도 나타난다.

## 4. 결측과 문자열 구조 — 실측
| 항목 | 빈 값·결측 |
|---|---:|
| `tags`: 빈 구조값 | **16,872 / 89,618** |
| `genres`: 빈 구조값 | **212 / 89,618** |
| `short_description`: null | **120 (0.13%)** |
| `detailed_description`: null | **197 (0.22%)** |
| `about_the_game`: null | **220 (0.25%)** |
| `reviews`: null | **79,217 (88.39%)** |
| `score_rank`: null | **89,579 (99.96%)** |
| `website`: null | **48,504 (54.12%)** |
| `metacritic_url`: null | **86,071 (96.04%)** |

형식상 `tags`, `genres`는 `isna()` 결측 0이지만, 파싱 후 **실질적인 빈 태그/장르가 많다**. 따라서 단순 `dropna`만으로 전처리하면 잘못된 태그 문자열이 TF-IDF에 들어갈 수 있다.

## 5. 다른 데이터와 결합했을 때의 '실효성'
- A 전체 Linux 게임 9,041개 가운데 A 태그가 있는 게임은 8,857개, C 태그만으로 보충되는 게임 126개, C 장르만으로 보충되는 게임 2개, 태그·장르 미확보 56개다.
- **리뷰가 있는 Linux 평가 후보 6,630개**에서는 A 태그 없음이 **117개**, C 태그로 채워지는 게임이 **107개**다.
- 이 117개가 B 평가 정답의 **58.76%**를 차지한다. 태그 보유율 상승이 약 1.4%p밖에 안 된다는 이유로 C를 생략하면 **인기 게임과 정답의 상당 부분**에 영향을 준다.
- A/C 공통 Linux 게임은 태그가 겹치는 편(Jaccard 평균 약 0.873)이지만, **결측이 어떤 게임에 집중되어 있는지**가 더 중요하다.
- A와 C를 합쳐도 설명·태그·장르가 없는 A 리뷰 존재 Linux 후보는 10개이며 B 정답 1,392쌍(5.63%)에 해당한다.
- C의 `positive`, `negative`, `pct_pos_recent` 등은 시점에 따라 정답과 겹치는 집계 신호가 될 수 있으므로, 사용자별 추천 모델의 첫 학습 피처로는 제외(제안).

## 6. 베이스라인 전처리 제안 (미구현)
1. `appid` 정규화와 유일성 확인. **비Linux 게임을 포함하는 A 원본의 `app_id` 전체**를 기준으로 왼쪽 조인하여 **A 정보 우선, C는 결측 보강**. C에만 있는 게임을 추가하려면 추천 후보·Linux 판정·학습 리뷰 유무를 별도로 결정.
2. `tags`·`genres`가 문자열인 경우 안전하게 파싱. 빈 구조·손상값·지원하지 않는 자료형을 구분해 기록.
3. 텍스트는 `A.tags → C.tags → C.genres`, 설명은 `A.description → C.short_description/about_the_game`의 명시적 우선순위로 합성(제안).
4. 사용자의 과거 입력이 비Linux 게임일 수 있으므로 **학습·입력 프로필용 전체 게임 텍스트**와 **Linux 추천 후보용 텍스트**의 범위를 별도로 기록. 기존 EDA의 Linux 후보 중심 TF-IDF 진단은 **전체 게임 학습 완료 결과가 아님**.
5. 태그만 쓴 TF-IDF와 태그+설명 TF-IDF를 각각 검토하고 영벡터 게임 수를 보고. 현재 EDA 진단에서는 전체 Linux 9,041개 중 **56개**가 영벡터였으나, 추천 모델 성능 검증 결과는 아니다.
6. 가격·평점·언어 등 새 피처는 데이터 수집 시점·형식·누출 검증 후 후속 실험으로 분리.

## 7. 근거
[전체 컬럼 스키마](../../reports/tables/01_schema_nulls.csv) · [충분성 분석 3·7절](../../reports/eda_sufficiency_summary.md) · [A/C 태그 유사도](../../reports/tables/10_C_A_C_tag_jaccard.csv) · [태그 보강 가중 영향](../../reports/tables/15_tag_source_weighted_impact.csv) · [TF-IDF 진단 결과](../../reports/tables/11_C_tfidf_summary.csv)
