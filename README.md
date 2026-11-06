# ML Game Recommendation

Steam 사용자의 취향을 바탕으로 **공식 Linux 지원 게임**을 추천하는 머신러닝 팀 프로젝트입니다. 추천 목록에는 공식 Linux 지원 게임만 포함합니다.

## 프로젝트 목표

- **게임 정보로 추천:** 장르·태그·설명을 단어 빈도-역문서 빈도(TF-IDF)라는 방법으로 단어별 중요도를 계산해 수치로 바꿉니다. 코사인 유사도로 두 게임 설명이 비슷할수록 높은 점수를 주어 추천합니다.
- **리뷰 이력으로 추천:** 사용자들이 긍정적으로 평가한 게임을 바탕으로 비슷한 게임을 찾습니다. 아이템 기반 k-최근접 이웃(Item-based KNN)은 함께 긍정 평가된 게임을 서로 가까운 이웃으로 보고 추천하는 방법입니다.
- 두 방법에서 상위 10개까지의 추천 결과를 비교합니다. 평가할 때는 두 방법 모두 공식 Linux 지원 조건을 적용합니다.

## 사용 데이터

자료를 구분하기 위해 아래에서 A는 **학습 자료**, B는 **외부 평가 자료**, C는 **게임 설명 보완 자료**를 뜻합니다.

| 구분 | 데이터셋 | 주요 활용 |
| --- | --- | --- |
| **A · 학습 자료** | [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) | 추천 리뷰, 플레이 시간, 게임 정보로 추천 방법을 만들고 조정 |
| **B · 외부 평가 자료** | [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) | 사용자의 보유·플레이 이력과 추천 리뷰로 결과 평가 |
| **C · 게임 설명 보완 자료** | [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) | 게임의 장르·태그·설명 및 운영체제 지원 정보 보완 |

데이터 사이의 게임은 Steam App ID(각 게임을 가리키는 Steam 식별 번호)로 연결합니다. 사용자 ID는 자료마다 따로 관리하므로 서로 다른 자료의 ID를 같은 사람으로 취급하지 않습니다.
