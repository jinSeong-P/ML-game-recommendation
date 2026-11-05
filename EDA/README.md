# EDA — Linux 지원 Steam 게임 개인화 추천

2026-11-03까지 수행한 기초 탐색적 데이터 분석(EDA)의 코드와 결과를 보관한 폴더입니다. 아직 콘텐츠 기반·협업 기반 추천 모델의 학습·비교 실험은 수행하지 않았습니다. 아래의 인기순 기준선은 평가 프로토콜 점검용입니다.

## 구성

```text
EDA/
├── data/raw/manifest.json        # 원본 데이터 출처·버전·크기·SHA256
├── scripts/
│   ├── download_data.py         # Kaggle 데이터셋 다운로드
│   ├── preprocess.py            # raw → parquet 형식 변환
│   └── build_notebook.py        # EDA 노트북 생성·실행
├── notebooks/
│   ├── 01_eda.py                # EDA 원본 코드 (diff 친화적)
│   └── 01_eda.ipynb             # 실행 결과가 포함된 노트북
├── figures/                      # 그래프 6개
├── reports/
│   ├── eda_summary.md           # 분석 요약·해석
│   ├── eda_metrics.json         # 실제 계산 지표
│   └── tables/                  # CSV 결과표 11개
└── requirements.txt
```

## 사용한 데이터셋

- **A (학습):** [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) v28
- **B (외부 테스트):** [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) v1
- **C (메타데이터):** [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) v2

게임은 Steam App ID를 기준으로 연결하며, 서로 다른 데이터셋의 사용자 ID는 동일 사용자로 가정하지 않습니다. 공식 Linux 지원 여부는 A의 `games.csv` Linux 플래그를 기준으로 분석했습니다.

## 재현 방법

저장소 루트에서 시작하여 EDA 폴더에서 실행합니다.

```bash
cd EDA
python -m venv .venv
# 가상환경을 활성화한 뒤:
python -m pip install -r requirements.txt
python -I scripts/download_data.py
python -I scripts/preprocess.py
python scripts/build_notebook.py
```

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`
- 원본 다운로드 용량 약 3.1GB. `data/raw/`의 원본 및 `data/processed/`의 중간 파일은 Git에 넣지 않으며 스크립트로 생성합니다.
- 추천 메모리: RAM 16GB 이상.
- 실행 시 `notebooks/01_eda.ipynb`, `figures/`, `reports/`의 결과가 갱신될 수 있습니다.

## 현재 완료 범위

데이터 구조·결측·중복 확인, 긍정/부정 리뷰 및 사용자·게임별 활동 분포, 공식 Linux 게임 분석, A·B·C App ID 결합 확인, 외부 평가 사용자 구성, 인기순 기준선 및 롱테일·시점 편향 점검.

**미구현:** TF-IDF·Item-KNN·ALS 추천 모델, 개인화 모델 성능 평가, 추가 데이터셋 D 조사·수집.

자세한 해석은 [EDA 요약 보고서](reports/eda_summary.md)를 참고하세요.

원본 드라이브 폴더: https://drive.google.com/drive/folders/1wBXXRG6XNeJU-mmuef129qnMbneW-Ko0
