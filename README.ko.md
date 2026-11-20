# Linux 게임 추천

[English](README.md) · **한국어**

머신러닝 6팀 — 박상민, 박진성, 이은주, 임재윤

사용자의 Steam 리뷰 기록을 보고, 아직 리뷰하지 않은 Linux 공식 지원 게임 10개를 추천합니다. 두 가지 방법을 비교했습니다.

- **콘텐츠 기반 추천:** 게임 소개글·태그·장르·기능 분류·개발사·유통사를 TF-IDF로 바꾸고 코사인 유사도로 순위를 매깁니다.
- **협업 필터링:** 같은 사용자들이 함께 좋아한 게임을 추천하는 Item-KNN입니다.

비교 기준으로 모든 사람에게 같은 순서를 주는 인기순 추천을 씁니다. 평가할 때는 사용자가 좋아한 Linux 게임을 숨기고, 추천 10개 안에 그 게임이 들어오는지 확인합니다(Recall@10, NDCG@10).

- 발표 자료: [한국어](presentation/ML-game-recommendation-final.pptx) · [영어](presentation/ML-game-recommendation-final-en.pptx)
- 보고서: [Source Codes & Descriptions (PDF)](report/Team06_Source_Codes_and_Descriptions.pdf)

## 최종 결과

최종 모델은 **활동량 보정 Item-KNN**입니다. 계정 0.64%가 함께 리뷰된 게임 조합의 62.6%를 만들어서, 리뷰를 많이 쓴 사용자의 영향을 줄였습니다.

대규모 외부 평가셋(Forgemaster, 학습에 없던 사용자 395,421명) 결과:

| 모델 | Recall@10 | NDCG@10 |
|---|---:|---:|
| Item-KNN (K=100) | 29.05% | 0.17003 |
| **활동량 보정 Item-KNN (최종)** | **29.49%** | **0.17218** |
| 콘텐츠 TF-IDF (소개글 + 메타데이터) | 13.09% | 0.07338 |
| 인기순 (학습 데이터) | 21.24% | 0.12993 |

최종 모델은 인기순보다 8.25%p 높습니다. 기본 Item-KNN과의 차이는 작고, 게임당 리뷰 100개 제한 평가셋에서는 조금 낮습니다.

## 두 가지 평가 설계

| | 시간 분할 (먼저 시도) | 계정 분할 (최종) |
|---|---|---|
| 확인하려는 것 | 특정 날짜 이후에 좋아할 게임을 맞히는가 | 학습에 없던 새 사용자에게도 잘 추천하는가 |
| 방법 | ① 익명 리뷰를 2022년 1월 1일로 나눔<br>② 2014년까지 학습 → 2015년 호주 리뷰로 평가 | Steam 계정 단위로 평가 계정을 빼 두고, 대규모 리뷰로 학습해 외부 평가셋 2개로 평가 |
| 결과 | 평가 대상이 7명, 1,536명뿐이고 ②에서는 인기순이 19.44%p 높음. 참고용으로만 사용 | 추천 후보 Linux 게임 6,630개, 평가 대상 395,421명. 최종 모델 선택에 사용 |

## 폴더 구성

폴더는 발표 자료의 분석 단계 순서를 따릅니다.

| 폴더 | 단계 | 담당 |
|---|---|---|
| [`data_exploration/`](data_exploration/) | 데이터 탐색·전처리 노트북 | 이은주 |
| [`time_split/attempt1_2022/`](time_split/attempt1_2022/) | 시간 분할 ①: 익명 리뷰를 2022년 1월 1일로 나눠 TF-IDF와 Item-KNN 비교 | 박상민 |
| [`time_split/attempt2_2014_2015/`](time_split/attempt2_2014_2015/) | 시간 분할 ②: 2014년까지 학습, 2015년 호주 리뷰로 평가 | 임재윤 |
| [`EDA/`](EDA/) | 계정 분할: 데이터 수집·정리, 분할, 학습, 평가, 개선 실험 | 박진성 |
| [`presentation/`](presentation/) | 최종 발표 자료(한국어·영어), 개선 실험 요약 | 팀 |
| [`report/`](report/) | 최종 보고서(PDF) | 팀 |

`EDA/` 내부:

| 경로 | 내용 |
|---|---|
| `EDA/scripts/` | Python 스크립트: `download_*` · `verify_*`(수집), `eda_*`(탐색), `build_*split*`(분할), `build_item_knn` · `train_*` · `experiment_*`(학습), `evaluate_*` · `summarize_*`(평가·정리) |
| `EDA/data/reference/` | 고정한 게임 목록과 실험 설정. 원본·가공 데이터는 Git에 없고 스크립트로 내려받습니다. |
| `EDA/notebooks/` | 데이터 분석 노트북 |
| `EDA/reports/`, `EDA/figures/` | 실험 결과(JSON·Markdown), 표, 그래프 |
| `EDA/docs/` | 모델 수식, 분할 설계, 데이터셋 설명 |
| `EDA/legacy/` | 이전 데이터셋·코드·분석 기록 |

## 실행 방법

**시간 분할 노트북** (`data_exploration/`, `time_split/`): Jupyter나 Google Colab에서 열고 위에서부터 실행합니다. 아래 데이터셋을 먼저 내려받아야 합니다.

**계정 분할** (`EDA/`): 저장소 맨 위 폴더에서 실행합니다(Windows PowerShell 기준).

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r EDA/requirements.txt
.venv/Scripts/python EDA/scripts/download_forgemaster.py
.venv/Scripts/python EDA/scripts/verify_forgemaster_csv.py
.venv/Scripts/python EDA/scripts/eda_forgemaster.py
```

학습과 평가 명령은 [`EDA/reports/forgemaster_model_evaluation.md`](EDA/reports/forgemaster_model_evaluation.md)에 있습니다. 학습용 원본(Kieran)은 ZIP으로 약 17GB라 디스크 공간이 더 필요합니다. 주요 라이브러리는 pandas, DuckDB, PyArrow, scikit-learn, SciPy, NumPy이고, 추가 실험에는 PyTorch, implicit, LightGBM을 씁니다(`EDA/requirements-experiments-v3.txt`).

## 데이터셋

| 역할 | 데이터셋 |
|---|---|
| 시간 분할: 학습 | [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (Kaggle) |
| 게임 설명·Linux 지원 여부 (공통) | [Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) (Kaggle) |
| 시간 분할 ②: 외부 평가 | [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) (Kaggle) |
| 계정 분할: 학습 | [Steam Reviews (Kieran)](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) (Kaggle) |
| 계정 분할: 대규모 평가 | [Steam Reviews Dataset (Forgemaster)](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset) (Kaggle) |
| 계정 분할: 100개 제한 평가 | [steam-dataset (frank)](https://huggingface.co/datasets/frankjc2022/steam-dataset) (Hugging Face) |

## 팀과 기여도

| 이름 | 브랜치 | 역할 | 기여도 |
|---|---|---|---:|
| 이은주 | `EJ` | 데이터 탐색·전처리 | 25% |
| 박상민 | `SM` | 시간 분할 시도 ① | 25% |
| 임재윤 | `JY` | 시간 분할 시도 ② | 25% |
| 박진성 | `JS` | 계정 분할 모델링·평가, 개선 실험, 발표 자료 | 25% |

모든 작업은 `main`에 합쳐져 있고, 팀원 각자의 원래 커밋은 각 브랜치에도 남아 있습니다.
