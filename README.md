# ML Game Recommendation

Steam 사용자의 취향을 바탕으로 **공식 Linux 지원 게임**을 추천하는 머신러닝 팀 프로젝트입니다. 추천 목록에는 공식 Linux 지원 게임만 포함합니다.

## 프로젝트 목표

- **게임 정보로 추천:** 장르·태그·설명을 단어 빈도-역문서 빈도(TF-IDF)라는 방법으로 단어별 중요도를 계산해 수치로 바꿉니다. 코사인 유사도로 두 게임 설명이 비슷할수록 높은 점수를 주어 추천합니다.
- **리뷰 이력으로 추천:** 사용자들이 긍정적으로 평가한 게임을 바탕으로 비슷한 게임을 찾습니다. 아이템 기반 k-최근접 이웃(Item-based KNN)은 함께 긍정 평가된 게임을 서로 가까운 이웃으로 보고 추천하는 방법입니다.
- 두 방법에서 상위 10개까지의 추천 결과를 비교합니다. **학습에는 Linux 지원 여부와 관계없이 A의 게임 이력을 활용**하며, 최종 **추천 출력과 평가 정답은 A의 공식 Linux 지원 게임으로 제한**합니다.

## 사용 데이터

자료를 구분하기 위해 아래에서 A는 **학습 자료**, B는 **외부 평가 자료**, C는 **게임 설명 보완 자료**를 뜻합니다.

| 구분 | 데이터셋 | 주요 활용 |
| --- | --- | --- |
| **A · 학습 자료** | [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) | 추천 리뷰, 플레이 시간, 게임 정보로 추천 방법을 만들고 조정 |
| **B · 외부 평가 자료** | [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) | 사용자의 보유·플레이 이력과 추천 리뷰로 결과 평가 |
| **C · 게임 설명 보완 자료** | [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) | 게임의 장르·태그·설명 및 운영체제 지원 정보 보완 |

## 학습 데이터 구성 (논의된 방향)

- **A — 선호 학습:** `recommendations.csv`의 전체 게임 리뷰를 사용한다. Windows 전용 등 **비Linux 게임의 긍정·부정 리뷰도 버리지 않는다**. Item-KNN의 첫 기준선은 공동 **긍정 리뷰**이며, 부정 리뷰의 가중·감점 활용은 별도 검토할 실험 옵션이다.
- **A+C — 콘텐츠 학습:** A `games_metadata.json`의 설명·태그를 기본으로 하고, 같은 Steam App ID의 C 태그·장르·설명으로 **빈 정보만 우선 보완**한다. TF-IDF에는 비Linux 입력 게임도 표현할 수 있도록 전체 게임 정보의 활용을 검토한다.
- **최종 추천 후보:** A `games.csv`의 `linux=True`로 제한한다. 전체 Linux 카탈로그 9,041개와 리뷰 이력이 있는 6,630개 중 사용할 후보 범위는 실험 전에 확정해야 한다.
- **검증:** A 내부에서 학습·검증을 분리하고, 외부 사용자용 테스트 자료는 별도로 평가한다. 신규 사용자 취향 일반화가 주목적이므로 시간순 분할을 필수로 강제하지 않지만, 수집 시점의 한계는 보고한다.

**EDA 결과와 설계 결정의 구분:** 위 항목은 학습/평가 **방향**이며 모델 학습이나 최종 성능 평가를 완료했다는 뜻은 아니다. [A+C 상세 원칙](EDA/docs/datasets/README.md) · [외부 테스트 자료 후보](EDA/docs/test_dataset_candidates.md)

데이터 사이의 게임은 Steam App ID(각 게임을 가리키는 Steam 식별 번호)로 연결합니다. 사용자 ID는 자료마다 따로 관리하므로 서로 다른 자료의 ID를 같은 사람으로 취급하지 않습니다.
