# ML Game Recommendation

Steam 사용자의 취향을 바탕으로 **공식 Linux 지원 게임**을 추천하는 머신러닝 팀 프로젝트야.

## 프로젝트 목표

- **콘텐츠 기반 필터링:** 게임의 장르·태그·설명을 활용해 유사한 게임 추천 (TF-IDF + 코사인 유사도)
- **협업 필터링:** 사용자들의 긍정 리뷰 이력을 활용해 게임 추천 (Item-based KNN)
- 두 추천 방식의 **Top 10 결과**에 공식 Linux 지원 조건을 공통 적용하고, 추천 성능을 비교

## 사용 데이터셋

| 구분 | 데이터셋 | 주요 활용 |
| --- | --- | --- |
| **A · 학습** | [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) | 사용자-게임 추천 리뷰, 플레이 시간, 게임 정보 |
| **B · 외부 테스트** | [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) | 사용자별 게임 보유·플레이 이력과 추천 리뷰를 이용한 평가 |
| **C · 메타데이터 보강** | [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) | 게임 장르·태그·설명 및 운영체제 지원 정보 |

데이터셋 간 게임 정보는 **Steam App ID**를 기준으로 연결하며, 서로 다른 데이터셋의 사용자 ID는 동일 사용자로 간주하지 않아.
