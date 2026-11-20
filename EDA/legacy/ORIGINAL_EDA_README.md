# EDA — Linux 지원 Steam 게임 개인화 추천

이 폴더에는 탐색적 데이터 분석(EDA: 자료의 구성과 품질을 살펴보는 분석)의 코드와 결과가 있습니다. 2026-11-03에 기본 분석을, 2026-11-06에 데이터가 추천 평가에 충분한지 확인하는 추가 분석을 수행했습니다. 추천 모델을 만들거나 두 모델의 성능을 비교하는 단계는 아직 진행하지 않았습니다. 보고서의 인기순 추천은 평가 방식을 점검하기 위한 간단한 비교 기준입니다.

## 처음 읽을 때

- **A**는 학습 자료, **B**는 외부 평가 자료, **C**는 게임 설명 보완 자료입니다. 각 자료의 정식 이름과 출처는 아래에 있습니다.
- **입력**은 추천 대상 사용자의 게임 이력입니다. 기존 B 기반 EDA에서는 **보유·플레이 목록**을 입력으로 사용했고, **긍정·부정 리뷰 일부를 입력으로 사용하고 나머지 긍정 리뷰를 숨기는 방식은 검토 중**입니다. **정답**은 입력에서 숨겨 둔 긍정 리뷰 게임 중 A의 공식 Linux 지원 게임입니다. **후보**는 추천 목록에 올릴 수 있는 게임입니다. **한 쌍**은 사용자 한 명과 게임 한 개의 관계를 뜻합니다.
- 추천 결과의 **상위 10개**는 점수가 높은 순서로 고른 10개 게임입니다. **Recall@10**은 사용자마다 상위 10개에 들어간 정답 수를 전체 정답 수로 나눈 뒤 평균한 값이고, **NDCG@10**은 정답 게임을 더 앞 순위에 둘수록 높아지는 점수입니다.
- **롱테일 평가**는 여기서 A의 Linux 지원 게임 중 인기 상위 50개를 제외하고 봅니다. 표나 결과에서 다른 범위를 따로 지정하면 그 범위를 사용합니다.
- 표의 CSV 파일명과 각 열이 뜻하는 내용은 [결과표 안내](reports/tables/README.md)에 정리했습니다.

## 구성

```text
EDA/
├── data/raw/manifest.json        # 원본 데이터 출처·버전·크기·SHA256
├── scripts/
│   ├── download_data.py         # 초기 원본 수집 스크립트 (현재는 아래의 공식 CLI 방법 사용)
│   ├── preprocess.py            # 원본 자료를 분석용 parquet 파일로 변환
│   └── build_notebook.py        # 분석 노트북 생성·실행
├── notebooks/
│   ├── 01_eda.py                # 기본 분석 코드
│   ├── 01_eda.ipynb             # 기본 분석 결과를 담은 노트북
│   ├── 02_eda_data_sufficiency.py    # 데이터 충분성 분석 코드
│   └── 02_eda_data_sufficiency.ipynb # 데이터 충분성 분석 결과를 담은 노트북
├── figures/                      # 분석 그래프 (fig01~06 기본, fig07~12 충분성)
├── reports/
│   ├── eda_summary.md           # 기본 분석 요약·해석
│   ├── eda_metrics.json         # 기본 분석에서 계산한 값
│   ├── eda_sufficiency_summary.md    # 데이터 충분성 판정·근거·권장 사항
│   ├── eda_sufficiency_metrics.json  # 데이터 충분성 계산 지표
│   └── tables/                  # CSV 결과표 (00~05 기본, 06~15 충분성)
└── requirements.txt
```

## 파일별 원본 데이터 사전

- [**A 데이터셋 네 파일의 관계도(ERD)**](docs/datasets/A.md) — A의 파일별 데이터와 서로 연결되는 키를 간단히 시각화했습니다.
- [**B 데이터셋 파일 구성 및 관계도(ERD)**](docs/datasets/B.md) — B의 보유 목록·리뷰·게임·번들 정보의 연결 관계를 시각화했습니다.
- [**신규 외부 테스트 데이터셋 후보와 추가 EDA 체크리스트**](docs/test_dataset_candidates.md) — 기존 B의 사용자별 리뷰 부족을 보완할 후보와 채택 전 확인할 지표입니다.
- [**후보 전체 검토·A 대체 학습 자료 비교 (2026-11-07)**](reports/test_dataset_candidate_review.md) — 후보 5개 원본 표본 확인, B·VintageDon 전체 리뷰 집계, 대형 리뷰 및 보유·플레이 자료 비교 결과입니다.
- [**A–VintageDon 독립성 검증과 엄격한 사용자 분할 대안**](reports/vintage_independence_audit.md) — 전체 리뷰 대조에서 중복 의심 징후를 확인했습니다. 현재 A+VintageDon의 사용자 독립성은 인증할 수 없으며, A 내부 사용자 분할안의 규모·무결성을 별도로 검증했습니다.
- [**중복을 막을 수 있는 학습·테스트 조합 전체 조사**](reports/dataset_train_test_combinations.md) — 원 ID 원본으로 학습 자료를 교체하는 조합, frank 리뷰 테스트 후보 규모, 기존 B의 Steam ID 연결, 세 자료 쌍의 실제 사용자 제외 결과를 정리했습니다. 선행 조사의 구조적 판정과 후속 대형 원본 EDA를 연결합니다.
- [**artermiloff·Kieran 전체 원본 → frank 리뷰 EDA**](reports/large_review_frank_eda.md) — **artermiloff 원본도 ID 정밀도가 손상돼 제외하고 Kieran→frank를 권장**합니다. Kieran 학습 98,857,701쌍, 모호 리뷰 제거 후 입력≥5 평가 후보 24,571명을 확인했습니다. [실행 노트북](notebooks/03_large_review_train_test_eda.ipynb)과 원본 정밀도·실제 누수 증거를 함께 제공합니다.

- [A·B·C **원본 파일별 구조·컬럼·특징·전처리 주의점**](docs/datasets/README.md) — 파일마다 별도 Markdown 문서로 정리했으며, 관측된 EDA 값과 향후 제안을 구분합니다.

## 학습 데이터 방향 — A 전체 + C 메타데이터 보강

| 데이터 | 활용 범위 | 주의 |
|---|---|---|
| **A `recommendations.csv`** | Linux/Windows 구분 없이 **전체 사용자-게임 리뷰**에서 선호·공동 긍정 패턴 학습 | 비Linux 게임을 학습 입력에서 제거하지 않음. 부정 리뷰의 모델별 활용 규칙은 미확정 |
| **A `games.csv` + `games_metadata.json`** | App ID·공식 Linux 플래그·태그·설명 | `linux`는 **최종 추천 후보/정답** 제한에 사용 |
| **C `games_march2025_cleaned.csv`** | A의 빈 태그·장르·설명을 동일 App ID로 보강 | C에는 사용자별 선호 라벨이 없음. 2025년 게임별 평판·인기도 집계는 첫 기준선 입력에서 분리 |
| **평가 데이터 (기존 B 또는 후보)** | 학습에 없던 사용자의 일부 이력으로 다른 긍정 Linux 게임을 찾는 성능 확인 | 외부 평가 자료 교체 및 리뷰 분할 규칙은 **미확정** |

**기존 EDA 수치와 새 프로토콜 구분:** 기존 B의 **15,294명·24,704정답 쌍**은 리뷰를 입력에서 제거한 **보유 이력 기반 평가** 결과다. 일부 리뷰를 입력으로 주는 새 방식의 적격 사용자 수나 성능으로 해석하지 않는다.

- A에서 리뷰가 존재하는 Linux 후보 **6,630개** 가운데 A 태그가 없는 **117개**가 기존 B 정답의 **58.76%**를 차지했다. 해당 117개 중 **107개는 C의 태그로 보완 가능**하므로 C는 콘텐츠 기반 실험에서 중요하다.
- **Item-KNN:** A 전체 사용자의 공동 긍정 기록 기반 이웃 학습을 우선 기준선으로 검토한다. **TF-IDF:** A+C의 게임 설명/태그를 기반으로 비Linux 게임도 사용자 입력에 활용하는 방향이다. **완료된 모델 학습은 없다.**
- A 내 검증에서 입력과 숨긴 정답의 게임을 분리한다. 신규 사용자 일반화를 주목적으로 하므로 시간순 분할은 필수가 아니며, 과거 시점 예측을 재현했다고 주장하지 않는다.

[파일별 A·C 문서](docs/datasets/README.md) · [외부 평가 데이터 추가 후보](docs/test_dataset_candidates.md)

## 사용한 자료

아래에서 A는 **학습 자료**, B는 **외부 평가 자료**, C는 **게임 설명 보완 자료**를 뜻합니다.

- **A · 학습 자료:** [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) v28
- **B · 외부 평가 자료:** [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) v1
- **C · 게임 설명 보완 자료:** [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) v2

게임은 Steam App ID(게임별 Steam 식별 번호)로 연결합니다. 서로 다른 자료의 사용자 ID는 같은 사람으로 취급하지 않습니다. 공식 Linux 지원 여부는 학습 자료 A의 `games.csv`에 있는 Linux 지원 표시를 기준으로 분석했습니다.

## 재현 방법

아래 명령은 Kaggle 공식 CLI(명령줄에서 Kaggle 자료를 받는 도구)로 원본을 받고 분석을 재현하는 방법입니다. 저장소 루트에서 Git Bash 기준으로 실행합니다. 초기 다운로드 스크립트 대신 이 절차를 사용하세요.

**1. Kaggle 토큰 준비**

https://www.kaggle.com/settings/api 에서 **Generate New Token**으로 토큰을 만들어 `~/.kaggle/access_token` 파일에 저장합니다. 토큰은 저장소에 넣거나 다른 사람에게 공유하지 않습니다.

**2. 분석 환경과 CLI 환경 준비**

Kaggle CLI는 분석에 쓰는 고정 패키지와 섞이지 않도록 저장소 밖의 별도 가상환경에 설치합니다. `six` 라이브러리는 kaggle 2.2.5를 실행하는 데 필요하며 자동으로 설치되지 않아 함께 설치합니다.

```bash
cd EDA
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
python -m venv "$HOME/.venvs/kaggle-cli"
"$HOME/.venvs/kaggle-cli/Scripts/python" -m pip install kaggle==2.2.5 six
```

Linux/macOS에서는 경로의 `Scripts/`를 `bin/`으로 바꿉니다.

**3. 원본 다운로드**

```bash
KG="$HOME/.venvs/kaggle-cli/Scripts/kaggle"
A=antonkozyriev/game-recommendations-on-steam
B=pypiahmad/steam-video-game-and-bundle-data
C=artermiloff/steam-games-dataset
mkdir -p data/raw/A_game_recs data/raw/B_ucsd_steam data/raw/C_steam_games_2025 data/raw/_kaggle_tmp
for f in games.csv games_metadata.json users.csv recommendations.csv; do "$KG" datasets download "$A" -f "$f" -p data/raw/A_game_recs --unzip -q; done
for f in australian_user_reviews.json australian_users_items.json steam_games.json; do "$KG" datasets download "$B" -f "$f" -p data/raw/B_ucsd_steam --unzip -q; done
"$KG" datasets download "$C" -f games_march2025_cleaned.csv -p data/raw/C_steam_games_2025 --unzip -q
"$KG" datasets download "$B" -p data/raw/_kaggle_tmp -q
.venv/Scripts/python -I -c "import zipfile; zipfile.ZipFile('data/raw/_kaggle_tmp/steam-video-game-and-bundle-data.zip').extract('bundle_data.json', 'data/raw/B_ucsd_steam')"
rm -rf data/raw/_kaggle_tmp
```

- `bundle_data.json`의 내용은 JSON 문법이 아니라 Python 리터럴 형식입니다. 파일만 바로 풀면 읽기 오류가 나므로 전체 ZIP(약 1.46GB)을 받고 이 파일만 꺼냅니다.
- ZIP을 받을 때 `--unzip`을 쓰지 않습니다. 쓰면 분석에 쓰지 않는 4.3GB `steam_new.json`까지 풀립니다.
- `manifest.json`에 적힌 `steam_new.json`은 전처리와 분석에 쓰지 않으므로 받지 않습니다.

**4. 전처리와 노트북 실행**

```bash
.venv/Scripts/python -I scripts/preprocess.py
.venv/Scripts/python scripts/build_notebook.py 02_eda_data_sufficiency
```

- 02 노트북은 시작할 때 원본 9개 파일의 크기와 SHA256(파일 내용이 바뀌었는지 확인하는 지문)을 `manifest.json`에 기록된 값과 비교하고 기존 수치가 재현되는지 확인합니다. 하나라도 다르면 분석을 멈춥니다.
- 인자 없이 `build_notebook.py`를 실행하면 01 노트북을 실행합니다.
- 원본은 압축 해제 후 약 3.3GB입니다. `data/raw/`의 원본 파일과 `data/processed/`의 중간 파일은 Git에 넣지 않습니다.
- 권장 메모리: 16GB 이상.
- 실행하면 해당 노트북과 `figures/`, `reports/`의 분석 결과가 갱신될 수 있습니다.
- 공개 저장소이므로 B 사용자 ID가 담긴 사용자별 표는 결과물에 저장하지 않습니다.

## 지금까지 살펴본 내용

- **기본 분석:** 자료의 형식, 빈 값과 중복, 긍정·부정 리뷰 수, 사용자와 게임별 활동량, 공식 Linux 지원 게임, 세 자료 사이의 게임 연결 상태를 살펴봤습니다. B에서 평가할 사용자를 정하고, 게임 인기도를 기준으로 한 간단한 추천과 오래된 게임·덜 인기 있는 게임에서의 편향도 점검했습니다.
- **데이터 충분성 분석:** 평가에 쓸 정답의 수, A에서 긍정 평가를 함께 받은 게임의 연결 관계, 게임 설명의 품질, TF-IDF(단어의 중요도를 계산해 게임 설명을 수치로 나타내는 방법) 입력으로 만들 수 있는 게임의 범위를 확인했습니다. 입력 가능 범위는 Linux 미지원 게임도 포함해 계산했습니다. A와 연결되지 않는 B 게임, 사용하지 않은 열, 정답과 입력의 영향, 추가 자료 D 후보도 조사했습니다.

**아직 만들지 않은 것:** 게임 설명 기반 추천, Item-KNN(함께 긍정 평가된 게임을 바탕으로 비슷한 게임을 찾는 방식), ALS(긍정적인 이용 이력에서 게임 취향을 추정하는 방식) 모델, 개인화 추천의 성능 평가, 추가 자료 D의 원본 검증과 수집.

자세한 해석은 [기본 분석 요약 보고서](reports/eda_summary.md)와 [데이터 충분성 보고서](reports/eda_sufficiency_summary.md)를 참고하세요.

원본 드라이브 폴더: https://drive.google.com/drive/folders/1wBXXRG6XNeJU-mmuef129qnMbneW-Ko0
