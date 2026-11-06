# 외부 테스트 데이터셋 후보 및 추가 EDA 계획

> **상태: 후보 조사 단계 / 채택 미확정 (2026-11-07)**  
> 기존 B([Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data))의 사용자별 리뷰 이력이 짧다는 문제를 보완하기 위한 후보 목록이다. 아래의 공개 규모·스키마는 **배포 페이지에서 확인한 값 또는 문서 기재값**이며, **원본을 새로 다운로드해 EDA한 결과가 아니다**. 기존 A+C 학습 범위 및 완료된 B 기반 EDA 결과는 변경하지 않는다.

## 1. 선정 기준

**목표:** A의 전체 Steam 게임 리뷰와 A+C 메타데이터로 학습한 추천기가, **학습에 없는 사용자**의 일부 긍정·부정 리뷰(또는 게임 이력)만으로 **숨긴 공식 Linux 게임의 긍정 리뷰**를 얼마나 회수하는지 평가한다.

단순 리뷰 수보다 아래 조건이 중요하다.

1. 동일한 사용자 식별자를 여러 게임 리뷰에 걸쳐 **일관되게 추적**할 수 있는가?
2. `appid`와 긍정/부정 라벨(`voted_up` 등)이 존재하는가?
3. **서로 다른 게임에 리뷰가 2건 이상 있는 사용자**, 특히 **입력용 리뷰 ≥1 + 숨길 Linux 긍정 정답 ≥1**인 사용자가 얼마나 되는가?
4. A 게임 카탈로그와 ID가 연결되고, 대상 Linux 게임에 충분한 정답이 있는가?
5. 원본 보존·라이선스·다운로드·사용자 ID 정밀도 문제가 없는가?

## 2. 후보 비교

| 후보 | 배포자가 밝힌 규모 | 사용자/게임/라벨 | 장점 | 주의 및 검증 상태 |
|---|---|---|---|---|
| [Mendeley — Online games marketplace Steam app/player data](https://data.mendeley.com/datasets/89yxn7hmpt/1) | 리뷰 JSON **약 2GB** (리뷰 행 수 미확인) | 익명 `steamid`, `appid`, `voted_up`, 플레이 시간, 리뷰 시각 | 사용자-리뷰-보유 게임 정보가 함께 있음. **우선 EDA 후보** | 익명 사용자 ID가 전체 게임에 일관되는지, 실사용자별 리뷰 분포·A와의 교집합 미확인. **CC BY 4.0** |
| [Kaggle — forgemaster Steam Reviews Dataset](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset) | 공식 카드 **약 1,500만 리뷰·600만+ 사용자·8,183게임**; [제3자 데이터 사용 문서](https://github.com/thebobmay/applied-ml-player-feedback/blob/main/dataset_access_instructions.md)는 **15,437,471 리뷰 / 6,976,390 사용자 / 약 5.32GB** | `steamid`, `appid`, `voted_up`, 플레이 시간·리뷰 시각 | 리뷰 라벨과 사용자 ID가 있는 중대형 원본. **우선 EDA 후보** | 해당 제3자 수치의 재현 필요. 평균 약 2.21리뷰/사용자이므로 **총량 증가가 다중 리뷰 사용자를 보장하지 않음**. 라이선스 원본에서 재확인 필요 |
| [Kaggle — 100 Million+ Steam Reviews](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) / [HF 미러](https://huggingface.co/datasets/adde023/steam-reviews) | HF 미러 **113,883,717건 / 23.5GB Parquet** | `author_steamid`, `appid`, `voted_up` 등 | 가장 큰 리뷰 후보. 사용자별 다중 상호작용 확보 가능성 | HF 미러는 **Steam ID가 float64로 노출**되어 64비트 ID 정밀도 손실 위험. 이미 손실된 값은 문자 변환으로 복원 불가. 미러 카드 **CC BY-NC 4.0**; 원본과 라이선스 대조 필요 |
| [Steam Dataset 2025 — VintageDon](https://github.com/vintagedon/steam-dataset-2025) | 공개 스키마에 **1,048,148개 리뷰**, 239,664개 앱 | `author_steamid`(text), `appid`, `voted_up`, 앱 유형·플랫폼 메타데이터 | 2025년 게임 메타데이터·라벨을 함께 다룰 수 있음 | 리뷰 총량은 A보다 작음. 분포·공개 파일 다운로드 범위·Linux 교집합 확인 필요. 데이터 **CC BY 4.0**, 코드 MIT |
| [UCSD Steam Video Game and Bundle Data — v2](https://cseweb.ucsd.edu/~jmcauley/datasets.html) | 사이트 통계 **7,793,069리뷰 / 2,567,538사용자 / 15,474아이템**, 리뷰 압축 파일 약 1.3GB | 리뷰 및 사용자·게임 정보. **명시적 추천 라벨 전체 확인 필요** | 기존 B보다 큰 같은 계열의 리뷰 자료 | 기존 B와 **출처 계열이 겹치며 독립 테스트 자료인지 미확정**. 기존 `steam_new.json` 부분 표본에서 `recommend`를 확인하지 못했음. 무조건 긍정/부정 정답으로 간주하지 않음 |

**우선 확인 순서(제안):** Mendeley → forgemaster → 규모가 더 필요할 때 100 Million+ → VintageDon/UCSD 보조 검토. **후보 순위는 실제 추천 성능이나 사용자 적격 수를 측정해 결정한 결과가 아니다.**

### 출처와 구조 참고

- [Mendeley 공식 데이터 설명](https://data.mendeley.com/datasets/89yxn7hmpt/1): `steam_reviews.json`의 `recommendations` 안에 익명 `steamid`·`voted_up`·`timestamp_created`, `player_details.json`의 보유/플레이 목록이 있다고 명시.
- [forgemaster Kaggle 데이터 카드](https://www.kaggle.com/forgemaster/steam-reviews-dataset/code) 및 [사용 예시의 구조/규모](https://github.com/thebobmay/applied-ml-player-feedback/blob/main/dataset_access_instructions.md): 수치는 수집 판본에 따라 재확인.
- [HF 미러 데이터 카드](https://huggingface.co/datasets/adde023/steam-reviews): 원본 Kaggle 링크, 113,883,717개 행, `float64` 스키마, 미러 라이선스 표기.
- [VintageDon 데이터 사전](https://github.com/vintagedon/steam-dataset-2025/blob/main/steam-dataset-2025-v1/DATA_DICTIONARY.md) 및 [저장소 라이선스](https://github.com/vintagedon/steam-dataset-2025).
- [UCSD 공식 데이터셋 페이지](https://cseweb.ucsd.edu/~jmcauley/datasets.html); [기존 B 부분 파일 확인 기록](datasets/B_steam_new_partial.md).

## 3. 추가 EDA 시 공통 확인 항목

### A. 원본·스키마 검증
- 배포 URL, 버전/업데이트일, 실제 다운로드 파일명·크기·SHA256, 라이선스 및 재배포 조건 기록.
- 필요한 최소 컬럼: `user_id`, `app_id`, `review_id`(있는 경우), `label`(긍정/부정), `timestamp`(있는 경우), `playtime`(선택). **해당 데이터셋의 원본 컬럼명을 유지한 매핑표** 작성.
- Steam 사용자 ID는 처음부터 **문자열 또는 정확한 64비트 정수**로 로딩한다. 특히 float 변환과 16자리 이상 숫자의 Excel 자동 변환을 금지한다.
- `(user_id, app_id)` 중복, `review_id` 중복, 라벨 충돌·누락, 비게임 앱/DLC 여부 점검.

### B. 신규 사용자 평가에 필요한 핵심 분포

| 측정 지표 | 이유 |
|---|---|
| 전체 리뷰 수 / 고유 사용자 수 / 고유 App ID | 기본 규모 비교 |
| 사용자당 **고유 리뷰 게임 수의 p25·중앙값·p75·p90** | 대다수 사용자의 이력 길이 확인 |
| 게임 리뷰 ≥2 / ≥3 / ≥5 / ≥10개인 사용자 수 | 입력과 정답을 분리할 수 있는지 확인 |
| 사용자별 긍정/부정 조합(긍정만·부정만·혼합) | **부정 리뷰를 입력 피처로 사용할 수 있는 범위** 판단 |
| A 카탈로그 교집합, 공식 Linux 긍정 리뷰 게임·사용자 수 | 실제 추천·정답 커버리지 확인 |
| **입력 리뷰 ≥1 + 숨길 Linux 긍정 리뷰 ≥1**인 고유 사용자 수 | **최우선 채택 기준** |
| 게임/정답별 인기 집중도 및 미관측 비율 | 일부 인기 게임에만 평가가 치우치는지 확인 |
| A와의 원천·동일 리뷰 중복 가능성 | 외부 테스트의 독립성 판단 |

### C. 평가 프로토콜 비교(아직 미확정)
1. **리뷰 분할안:** B 또는 새 외부 자료의 **동일 사용자** 리뷰를 고정 난수 시드로 입력/숨김 집합에 나눈다. 입력에는 긍정과 부정을 구분하여 전달할 수 있으나 **부정 피처의 모델별 점수 반영 규칙은 별도 실험 결정**이다. 정답은 **입력에서 제외된 Linux 긍정 리뷰**로 구성한다.
2. **보유 이력안:** 기존 B 방식처럼 보유·플레이 게임을 입력에 쓰되, 리뷰 대상 게임을 입력에서 제외하고 긍정 리뷰를 정답으로 삼는다. 이는 2026-11-06 EDA의 **15,294명 / 24,704 정답 쌍**과 대응하는 **기존 평가 프로토콜**이다. *리뷰 분할안에 이 수치를 전용하지 않는다.*
3. **공통 보호 규칙:** 입력과 숨김 정답의 `(user_id, app_id)` 교집합은 공집합이어야 한다. 추천 결과에서 입력에 사용된 게임을 제외하며, 정답은 A `games.csv`의 `linux=True`로 판정한다. 부정 리뷰는 긍정 정답으로 계산하지 않는다.
4. **학습/테스트 분리:** A에서 모델 학습·검증을 수행하고 외부 자료의 숨김 라벨로 하이퍼파라미터를 결정하지 않는다. 공개 Steam 리뷰 수집본끼리 동일 리뷰가 겹칠 수 있으므로, **'새 사용자'가 A와 완전히 독립인지 자동 보장되지 않음**을 명시한다.
5. **시간 해석:** 시간순 분할은 신규 사용자 취향 일반화 실험에 필수는 아니지만, 수집 시점 불일치를 무시하고 미래 예측 성능이라고 주장해서는 안 된다.

## 4. 채택 판정 및 기록 양식

각 후보에 대해 아래 한 줄 표를 추가 EDA로 채운 후 비교한다. **공개 데이터셋의 리뷰 전체 규모만으로 B 교체를 결정하지 않는다.**

| 후보 | 총 리뷰 | 고유 사용자 | 다중 리뷰 사용자(≥2) | 입력≥1 + 숨김 Linux 양성≥1 사용자 | A App ID 교집합 | ID/중복·라이선스 확인 | 최종 판단 |
|---|---:|---:|---:|---:|---:|---|---|
| 현재 B (기준선) | 59,305 | 25,458 | **새 리뷰 분할 기준 별도 EDA 필요** | 미확정 | 추가 EDA 표준으로 재계산 | 기존 v1 확인 | 유지/교체 미정 |
| Mendeley | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 후보 |
| forgemaster | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 후보 |
| 100 Million+ | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | float64 원본 검증 필요 | 후보 |
| VintageDon | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 파일 접근/필드 검증 | 후보 |
| UCSD v2 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 추천 라벨·독립성 검증 | 보류 |

> **현재 상태:** 다운로드, 재전처리, 신규 후보 EDA, 모델 재학습, B 교체는 수행하지 않았다. 수치가 확인되면 해당 원본의 버전과 재현 코드·결과표를 함께 남긴다.
