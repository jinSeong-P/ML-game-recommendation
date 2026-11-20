# 외부 테스트 데이터셋 후보 및 추가 EDA 계획

> **상태: 후보 조사 및 대형 원본 2개 전수 EDA 완료 / artermiloff 원본 ID 손상으로 제외 / Kieran→frank 권장 / 최종 채택 미확정 (2026-11-07)**
>
> 기존 B([Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data))의 사용자별 리뷰 이력 부족을 보완하기 위한 후보 목록이다. **최신 검토 결과와 A 대체 학습 자료 비교는 [후보 전체 검토 보고서](../reports/test_dataset_candidate_review.md)**를 참고한다. 아래 후보 총량은 별도 표시가 없으면 공개 기재값이며, 전체 실측값과 구분한다. 기존 A+C 학습 범위 및 완료된 B 보유 이력 기반 EDA 결과는 유지한다.

> **독립성 후속 판정:** A·VintageDon 리뷰 전체 대조에서 중복 의심 징후를 확인했다. A 원 사용자/리뷰 ID 매핑이 없어 **현재 A 학습 + VintageDon 테스트의 엄격한 사용자 비중복은 인증 불가**다. VintageDon은 보조 외부 평가 후보로 제한하고, 엄격한 주평가는 A 내부 사용자 분할을 권한다. [검증 결과와 분할 규모](../reports/vintage_independence_audit.md).

> **대형 원본 전수 EDA 최종 판정:** **Kieran 원본 학습 → frank 리뷰 테스트를 권장**한다. Kieran 전체 113,883,717행을 검사했고 정리·계정 제외·검증 예약 후 학습 98,857,701쌍이 남았다. **artermiloff 원본 자체도 Steam ID 정밀도가 손상되어 제외**한다. 기존 구조적 추천을 수정했다. 알려진 App ID 모호성을 제거한 frank 평가 후보는 입력≥5 **24,571명**이다. [전수 EDA·누수 근거](../reports/large_review_frank_eda.md).

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
| [Kaggle — forgemaster Steam Reviews Dataset](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset) | 공식 카드 **15,437,471 리뷰 / 6,976,390 사용자 / 8,183게임**; API 파일 총량 약 5.32GB | `steamid`, `appid`, `voted_up`, 플레이 시간·리뷰 시각 | 원본 100행 표본에서 ID·라벨 확인. **B 보완 EDA 후보** | v1 **CC0** 확인. 카드상 평균 약 2.21리뷰/사용자이며 전체 이력 분포는 미측정. A보다 총량·게임 범위가 작음 |
| [Kaggle — 100 Million+ Steam Reviews](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) / [HF 미러](https://huggingface.co/datasets/adde023/steam-reviews) | **113,883,717건** 기재 / 원본 주 CSV 약 42.49GB / HF 약 23.46GB Parquet | `author_steamid`, `appid`, `voted_up` 등; 원본 전체 113,883,717행 검사 완료 | **A 대체 우선 검증 원본 후보**. A 리뷰 수의 약 2.77배 | 원본–미러 대응 100행 중 **93행 Steam ID 불일치** 확인. float64 미러는 보류. 원본 v2 **MIT**, 미러 **CC BY-NC 4.0**으로 표기가 다름 |
| [Steam Dataset 2025 — VintageDon](https://zenodo.org/records/17266923) | **1,048,148개 리뷰 전체 집계**, 배포 메타데이터 239,664개 앱 | `author_steamid`(문자열), `appid`, `voted_up`, 앱 유형·플랫폼 메타데이터 | 입력≥1 + 숨길 Linux 양성≥1 **16,615명 / 잠재 정답 4,323게임** 확인 | **현재 A와 엄격한 사용자 비중복 인증 불가·중복 의심 징후 확인**. 보조 외부 평가 후보. 앱당 최대 100개 표집·84.12%가 리뷰 게임 1개. A 주 학습 대체는 부적합. 데이터 **CC BY 4.0**, 코드 MIT |
| [UCSD Steam Video Game and Bundle Data — v2](https://cseweb.ucsd.edu/~jmcauley/datasets.html) | 사이트 통계 **7,793,069리뷰 / 2,567,538사용자 / 15,474아이템**, 리뷰 압축 파일 약 1.3GB | 4,703행 원본 표본에서 **추천 라벨 없음**, 사용자 ID 42.06%만 존재 | 암묵적 상호작용 연구의 별도 검토 대상 | B `steam_new.json`의 재포장 가능성이 높음(전체 해시 미검증). 원본 페이지 라이선스 미명시. **Linux 긍정 정답용 보류** |

**현재 우선순위:** 현재 A를 유지하면 엄격한 주평가는 **A 내부 사용자 분할**이다. 학습 자료를 교체한다면 **Kieran 원본 + frank 리뷰 테스트**를 우선 검토한다. frank는 원 ID로 학습에서 테스트 사용자를 제외할 수 있으며 같은 기준의 잠재 평가 규모가 VintageDon보다 크다. **VintageDon**도 원 ID 학습 자료와는 외부 테스트 조합을 구성할 수 있으나, 익명 A와의 사용자 비중복은 인증할 수 없다. 다른 후보와 보유 자료의 판정은 [전체 조합 보고서](../reports/dataset_train_test_combinations.md)를 따른다. 추천 성능 우위를 측정한 순위는 아니다.

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

2026-11-07 실측 행의 입력은 **A에 리뷰가 있는 게임**으로 제한하고, 숨길 정답은 A `linux=True`인 긍정 게임으로 정의했다. 적격 수는 **정답 하나를 숨겨도 입력 하나 이상을 남길 수 있는 잠재 사용자 수**다. 총 리뷰는 중복 제거 전, 다중 리뷰 사용자는 중복·충돌 정리 후다.

| 후보 | 총 리뷰 | 고유 사용자 | 다중 리뷰 사용자(≥2) | 입력≥1 + 숨김 Linux 양성≥1 사용자 | A App ID 교집합 | ID/중복·라이선스 확인 | 최종 판단 |
|---|---:|---:|---:|---:|---:|---|---|
| 현재 B (기준선) | 59,305 | 25,458 | **11,929** | **9,287** | **2,997** | 전체 리뷰 집계·중복 처리 완료 | 보유 이력 기준선과 별도 |
| Mendeley | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | CC BY·교차 앱 ID 재사용 표본 확인; 전체 중복 미측정 | 후보 |
| forgemaster | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | CC0·원본 ID/라벨 표본 확인; 전체 중복 미측정 | 후보 |
| 100 Million+ 원본 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | MIT·원본 ID 표본 확인; float64 미러는 보류 | 후보 |
| VintageDon | **1,048,148** | **701,883** | **111,470** | **16,615** | **42,576** | 전체 리뷰 집계·중복/충돌·CC BY 확인. A 원 ID 매핑 없음 | 보조 외부 후보·엄격한 사용자 비중복 인증 불가 |
| UCSD v2 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 원본 표본에서 추천 라벨 없음·B와 계열 중복 | 보류 |

> **현재 상태:** VintageDon CSV 원본 다운로드·전체 리뷰 집계와 B 리뷰 분할 적격 수 재계산을 수행했다. 다른 후보는 전체 EDA 전이다. 학습 자료/B 교체와 모델 재학습은 수행하지 않았다. [검토 보고서](../reports/test_dataset_candidate_review.md) · [재현 코드](../scripts/audit_review_candidates.py) · [집계 JSON](../reports/test_dataset_review_history_metrics.json)
