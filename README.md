# Linux Game Recommendation

**English** · [한국어](README.ko.md)

Machine Learning Team 6 — Sangmin Park, Jinseong Park, Eunju Lee, Jaeyoon Lim

The system recommends 10 Steam games that officially support Linux and that a user has not reviewed yet, based on the user's Steam review history. Two methods are compared:

- **Content-based filtering:** TF-IDF vectors of game text (description, tags, genres, categories, developers, publishers), ranked by cosine similarity.
- **Collaborative filtering:** item-based KNN that recommends games liked by the same users.

A popularity ranking is used as a non-personalized reference. For each test user, liked Linux games are hidden and we check whether they appear in the top 10 (Recall@10, NDCG@10).

- Slides: [Korean](presentation/ML-game-recommendation-final.pptx) · [English](presentation/ML-game-recommendation-final-en.pptx)
- Report: [Source Codes & Descriptions (PDF)](report/Team06_Source_Codes_and_Descriptions.pdf)

## Final result

The final model is **activity-weighted Item-KNN**. It gives less weight to users who reviewed many games, because 0.64% of accounts produced 62.6% of all co-reviewed game pairs.

Large external test set (Forgemaster, 395,421 users never seen in training):

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Item-KNN (K=100) | 29.05% | 0.17003 |
| **Activity-weighted Item-KNN (final)** | **29.49%** | **0.17218** |
| Content TF-IDF (description + metadata) | 13.09% | 0.07338 |
| Popularity (training data) | 21.24% | 0.12993 |

The final model is 8.25 percentage points above popularity. Its margin over plain Item-KNN is small, and it is slightly lower on the 100-reviews-per-game test set.

## Two evaluation designs

| | Time split (tried first) | Account split (final) |
|---|---|---|
| Question | Can the model predict games a user will like after a date? | Can the model recommend well to users never seen in training? |
| Setup | ① anonymous reviews split at 2022-01-01<br>② train up to 2014 → test on 2015 Australian reviews | Whole Steam accounts held out; trained on large review data, tested on 2 external datasets |
| Outcome | Only 7 and 1,536 test users; in ② popularity beat the personalized models by 19.44 pp. Kept for reference only. | 6,630 Linux candidate games, 395,421 test users. Used to choose the final model. |

## Repository map

The folders follow the analysis steps in the slides.

| Folder | Step | Owner |
|---|---|---|
| [`data_exploration/`](data_exploration/) | Data exploration and preprocessing notebook | Eunju Lee |
| [`time_split/attempt1_2022/`](time_split/attempt1_2022/) | Time split ①: anonymous reviews split at 2022-01-01, TF-IDF vs Item-KNN | Sangmin Park |
| [`time_split/attempt2_2014_2015/`](time_split/attempt2_2014_2015/) | Time split ②: train up to 2014, test on 2015 Australian reviews | Jaeyoon Lim |
| [`EDA/`](EDA/) | Account split: download, cleaning, splits, training, evaluation, improvement experiments | Jinseong Park |
| [`presentation/`](presentation/) | Final slides (Korean, English) and improvement-experiment summary | Team |
| [`report/`](report/) | Final report (PDF) | Team |

Inside `EDA/` (documents there are written in Korean):

| Path | Content |
|---|---|
| `EDA/scripts/` | Python scripts: `download_*` · `verify_*` (collect), `eda_*` (explore), `build_*split*` (split), `build_item_knn` · `train_*` · `experiment_*` (train), `evaluate_*` · `summarize_*` (evaluate, report) |
| `EDA/data/reference/` | Fixed game list and experiment settings. Raw and processed data are not in Git; the scripts download them. |
| `EDA/notebooks/` | Data analysis notebooks |
| `EDA/reports/`, `EDA/figures/` | Results (JSON, Markdown), tables and charts |
| `EDA/docs/` | Model formulas, split design, dataset notes |
| `EDA/legacy/` | Earlier datasets, code and analyses |

## How to run

**Time-split notebooks** (`data_exploration/`, `time_split/`): open in Jupyter or Google Colab and run from the top. Download the datasets listed below first.

**Account split** (`EDA/`): run from the repository root (Windows PowerShell shown).

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r EDA/requirements.txt
.venv/Scripts/python EDA/scripts/download_forgemaster.py
.venv/Scripts/python EDA/scripts/verify_forgemaster_csv.py
.venv/Scripts/python EDA/scripts/eda_forgemaster.py
```

Training and evaluation commands are in [`EDA/reports/forgemaster_model_evaluation.md`](EDA/reports/forgemaster_model_evaluation.md). The raw training data (Kieran) is about 17 GB as a ZIP and needs extra disk space. Main libraries: pandas, DuckDB, PyArrow, scikit-learn, SciPy, NumPy (PyTorch, implicit and LightGBM for the extra experiments in `EDA/requirements-experiments-v3.txt`).

## Datasets

| Role | Dataset |
|---|---|
| Time split: training | [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (Kaggle) |
| Game text and Linux flag (both) | [Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) (Kaggle) |
| Time split ②: external test | [Steam Video Game and Bundle Data](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) (Kaggle) |
| Account split: training | [Steam Reviews (Kieran)](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) (Kaggle) |
| Account split: large test | [Steam Reviews Dataset (Forgemaster)](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset) (Kaggle) |
| Account split: 100-cap test | [steam-dataset (frank)](https://huggingface.co/datasets/frankjc2022/steam-dataset) (Hugging Face) |

## Team and contribution

| Member | Branch | Role | Contribution |
|---|---|---|---:|
| Eunju Lee | `EJ` | Data exploration and preprocessing | 25% |
| Sangmin Park | `SM` | Time split ① | 25% |
| Jaeyoon Lim | `JY` | Time split ② | 25% |
| Jinseong Park | `JS` | Account split modeling and evaluation, improvement experiments, slides | 25% |

All work is merged into `main`. Each member's original commits are also kept on their own branch.
