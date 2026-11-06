# B · australian_users_items.json — 사용자별 보유·플레이 게임

## 1. 역할과 원본 구조
- **원본**: [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) v1, B/`australian_users_items.json` (약 553 MB).
- **한 줄**: 한 사용자 프로필과 그 사용자의 **게임 보유 목록**. 한 줄이 곧 게임 1개는 아니다.
- **원본 형식**: 한 줄에 Python 딕셔너리 리터럴 형태. 엄격한 JSON 문법이 아니므로 `json.loads`가 아닌 `ast.literal_eval`로 읽는다.
- **주 역할**: 새로운 B 사용자의 **추천 입력 프로필**. 이 목록 자체가 긍정적인 선호 라벨은 아니다.

## 2. 중첩 데이터 구조
| 계층 | 필드 | 해석 |
|---|---|---|
| 사용자 | `user_id` | B 내부 문자열 사용자 ID |
| 사용자 | `steam_id` | Steam 식별 관련 문자열; A 사용자 ID와 연결 금지 |
| 사용자 | `items_count` | 프로필이 표기한 보유 개수 |
| 사용자 | `items` | 보유 게임에 대한 중첩 리스트 |
| 게임 목록 원소 | `item_id` | 문자열로 수록된 Steam 게임 ID |
| 게임 목록 원소 | `item_name` | 해당 게임의 이름 |
| 게임 목록 원소 | `playtime_forever` | 누적 플레이 시간 값(통상 분 단위, 원본 정의 확인 필요) |
| 게임 목록 원소 | `playtime_2weeks` | 최근 2주 플레이 시간 값(통상 분 단위, 관측 기준일 정보 없음) |

`scripts/preprocess.py`는 이를 아래 두 개 Parquet으로 펼친다. **`line_no`, `item_id_raw`, `item_id`, `n_items_listed`는 전처리 파생 필드**이며 원본 키가 아니다.

- `B_users_items_profiles.parquet`: `line_no, user_id, steam_id, items_count, n_items_listed` (5열).
- `B_users_items.parquet`: `line_no, user_id, item_id_raw, item_name, playtime_forever, playtime_2weeks, item_id` (7열, `item_id` 정수화).

## 3. EDA 실측
| 항목 | 값 |
|---|---:|
| 사용자 프로필 원본 줄 수 | **88,310** |
| 고유 사용자 ID | **87,626** |
| 중복 프로필이 존재하는 사용자 | **673명** (이들의 보유 목록은 동일) |
| 펼친 보유 게임 행 | **5,153,209** |
| 보유 게임 행에 등장한 사용자 | **70,912명** |
| 고유 게임 `item_id` | **10,978개** |
| 펼친 표의 (`user_id`, `item_id`) 중복 행 | **59,127** |
| 보유 게임 종류 중 A 카탈로그에 있는 비율 | **72.78%** |
| 전체 보유 **행** 중 A 카탈로그에 있는 비율 | **79.29%** |

고유 사용자 수(프로필 기반)와 보유 기록이 1개 이상인 사용자 수(펼친 행 기반)의 분모가 다르다. 중복 사용자 673명과 중복 행 59,127건도 혼동하지 않는다.

## 4. 반드시 확인할 문제
1. **정답 누출**: B 사용자가 리뷰한 게임의 **75.62%**가 이 파일의 보유 목록에도 포함된다. 평가용 입력을 만들기 전에 동일 B 사용자가 **긍정/부정 상관없이 리뷰한 모든 게임**을 제외해야 한다.
2. **학습 공간과 불일치**: A에 없는 보유 게임은 내용 기반/협업 기반 모델이 처리하지 못할 수 있다. 평가 사용자 15,294명 기준 카탈로그 매칭 제외 비율 중앙값은 **20.43%**, 절반 이상 잃는 사용자는 **178명(1.16%)**이다.
3. **보유 ≠ 좋아함**: `playtime_forever=0`인 보유 게임도 존재하므로, 보유 전체 입력과 실제 플레이(0 초과) 입력을 별도 평가한다.
4. **구매·보유 날짜 없음**: 과거 어느 시점에 소유했는지 모른다. B를 완전한 시간순 미래 예측 테스트라고 할 수 없다.
5. **입력에서 플랫폼 제한 금지(기본 계획)**: 입력 중 A 게임과 매칭되는 비Linux 게임도 사용자의 취향 표현에 유용할 수 있다. 2026-11-06 분석에서 유효 입력 쌍의 **59.04%**가 비Linux 게임이었다.

## 5. 베이스라인 전처리 제안 (미구현)
1. 사용자별 동일한 중복 보유 목록을 제거하고 `(user_id,item_id)` 중복을 정리. 중복 상충 여부를 추가 검증.
2. `item_id_raw`는 원문 보존, `item_id`는 숫자 변환을 별도 수행하고 변환 실패를 기록.
3. 같은 사용자에 대한 B 리뷰 게임을 **먼저 제외**, 그다음 A 학습 게임 공간과 조인.
4. A/Linux 필터는 **추천 결과**에만 적용. 사용자 입력은 비Linux 게임 포함한 A 매칭 게임 전체 허용.
5. `playtime_forever>0`인 게임만 쓰는 입력 변형을 나란히 평가; 결과 해석 시 신호 정의를 구분.

## 6. 근거
[기초 EDA 요약](../../reports/eda_summary.md) · [추가 충분성 EDA 4절](../../reports/eda_sufficiency_summary.md) · [게임 ID 겹침](../../reports/tables/04_id_overlap.csv) · [외부 평가 규모](../../reports/tables/05_external_test_eligibility.csv) · [preprocess.py](../../scripts/preprocess.py)
