# 데이터셋 A — 파일 구성 및 데이터 간 관계

**Game Recommendations on Steam (v28)** · [Kaggle 원본](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

> **데이터 출처:** 본 문서의 데이터셋 A는 프로젝트 [`main` README의 ‘A · 학습’](https://github.com/jinSeong-P/ML-game-recommendation/blob/main/README.md)에 명시된 **동일 데이터셋**이다. EDA에서는 해당 데이터셋의 v28을 사용하였다.

## 1. 파일 구성 및 주요 정보

| 파일 | 한 행의 의미 | 핵심 내용 | EDA에서 확인한 행 수 |
|---|---|---|---:|
| [`recommendations.csv`](A_recommendations.csv.md) | **사용자의 게임 리뷰 1건** | 사용자 ID, 게임 ID, 긍정·부정 추천 여부, 플레이 시간, 리뷰 날짜 | 41,154,794 |
| [`games.csv`](A_games.csv.md) | **게임 1개** | 게임 제목, 출시일, 가격, Windows/macOS/Linux 지원 여부, 평가 정보 | 50,872 |
| [`users.csv`](A_users.csv.md) | **사용자 1명** | 사용자 ID, 보유 **제품 수**, 작성한 리뷰 수 | 14,306,064 |
| [`games_metadata.json`](A_games_metadata.json.md) | **게임 1개** | 게임 ID, 설명, 태그 | 50,872 |

## 2. 파일 간 연결 관계 (ER 다이어그램)

```mermaid
erDiagram
    USERS ||--o{ RECOMMENDATIONS : "user_id"
    GAMES ||--o{ RECOMMENDATIONS : "app_id"
    GAMES ||--|| GAMES_METADATA : "app_id"

    USERS {
        int user_id PK
        int products
        int reviews
    }
    RECOMMENDATIONS {
        int review_id
        int user_id FK
        int app_id FK
        boolean is_recommended
        float hours
        datetime date
    }
    GAMES {
        int app_id PK
        string title
        boolean linux
        datetime date_release
    }
    GAMES_METADATA {
        int app_id PK,FK
        string description
        array tags
    }
```

*PK는 기본 키, FK는 외래 키에 해당하는 **논리적 연결 관계**를 나타낸다. 원본은 CSV·JSON 파일로 구성되어 있으므로 실제 데이터베이스의 키 제약조건이 적용된 것은 아니다. 다이어그램에는 주요 필드만 표시하였다.*

| 관계 | 연결 컬럼 | 의미 |
|---|---|---|
| `users` → `recommendations` | `user_id` | **1:N**. 사용자 1명에 0건 이상의 리뷰가 대응 |
| `games` → `recommendations` | `app_id` | **1:N**. 게임 1개에 0건 이상의 리뷰가 대응 |
| `games` ↔ `games_metadata` | `app_id` | **1:1**. 동일 App ID를 기준으로 기본 정보와 상세 정보가 일대일 대응 |

**사용자와 게임은 `recommendations.csv`의 리뷰 기록을 통해 연결된다.** `users.csv`의 `products`는 보유 제품의 **목록이 아닌 개수**를 나타낸다.

## 3. 연결 키의 정합성 검증 결과

- `recommendations.csv`에 등장하는 **모든 `user_id`가 `users.csv`에 존재함**을 확인하였다.
- `recommendations.csv`에 등장하는 **모든 `app_id`가 `games.csv`에 존재함**을 확인하였다.
- `games.csv`와 `games_metadata.json`의 **App ID 집합이 일치한다**. 두 파일은 각각 50,872행으로 구성되며, 파일 내부의 App ID 중복은 확인되지 않았다.
- `recommendations.csv`에서는 **동일한 (`user_id`, `app_id`) 조합의 중복 행 21건**이 확인되었다. 따라서 두 컬럼의 조합을 고유 기본 키로 간주할 수 없으며, `review_id`의 고유성 역시 현재 EDA 결과만으로는 확정할 수 없다.

**근거:** [EDA 구조·중복 집계](../../reports/tables/01_structure.csv), [원본·키 정합성 결과](../../reports/eda_metrics.json), [원본 파일·버전 기록](../../data/raw/manifest.json)

> **문서 범위:** 기존 EDA에서 확인된 데이터 구조 및 연결 관계를 요약하였다. 추가 분석, 전처리 정책 결정 및 모델 학습셋 확정은 이 문서의 범위에 포함되지 않는다.
