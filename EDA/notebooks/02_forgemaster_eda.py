# %% [markdown]
# # Forgemaster Steam Reviews v1 데이터 분석
#
# 2026-11-08. 원본 11개 파일 전체를 분석한 결과. 실제 학습/정답 분할 및 추천 지표 평가는 수행하지 않았습니다.
#
# 전체 설명: [보고서](../reports/forgemaster_eda.md). 재수집·전수 집계 명령은 보고서 마지막을 참조하세요.
#
# **후속 모델 평가 완료:** Kieran 원본을 복원하고 평가 계정을 학습에서 제외해 다시 학습했다. 결과는 [모델 평가 보고서](../reports/forgemaster_model_evaluation.md)에서 확인할 수 있다. 이 문서의 모델 파일 부재와 학습·평가 계정 중복 미검사는 그 이전 EDA 단계의 상태다.

# %%
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image
EDA = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd() / 'EDA'
m = json.loads((EDA / 'reports/forgemaster_eda_metrics.json').read_text('utf-8'))
T = EDA / 'reports/tables/forgemaster'
display(pd.DataFrame([m['raw'],m['clean']],index=['raw','clean']))

# %% [markdown]
# ## 열별 품질 검사
#
# SteamID는 문자열로 보존합니다. 라벨 충돌 사용자–게임은 전체 제외하고 동일 라벨 반복은 한 쌍으로 합쳤습니다. 두 파서의 파일별 전체 행 수를 대조했습니다.

# %%
display(pd.read_csv(T / '02_schema_quality.csv'))
display(pd.Series(m['duplicates'],name='사용자–게임 중복 검사'))
display(pd.Series(m['quality'],name='품질 검사 결과'))
display(pd.Series(m['additional_saved_output_audit']['timestamps'],name='날짜 검사 결과'))
display(pd.Series(m['reconciliation'],name='집계 일치 검사'))

# %% [markdown]
# ## 게임·사용자·시간 분포

# %%
display(pd.DataFrame([m['reviews_per_game'],m['reviews_per_user']],index=['per game','per user']))
display(pd.read_csv(T / '03_game_support.csv').head(20))
display(pd.read_csv(T / '06_review_years.csv'))
display(Image(filename=str(EDA / 'figures/forgemaster/01_overview.png')))

# %% [markdown]
# ## 입력·추천 게임 목록과 평가 가능 사용자
#
# 정답 한 개를 숨긴 뒤 최소 전체 입력 5개·긍정 입력 1개가 기본 조건입니다. 아래는 EDA 단계에서 계산한 평가 가능 사용자 수입니다.

# %%
display(pd.Series(m['catalog_coverage'],name='게임 목록의 데이터 포함 범위'))
display(pd.read_csv(T / '07_evaluation_eligibility.csv'))
display(pd.Series(m['full_eligible_history'],name='평가 대상자의 리뷰 이력'))

# %% [markdown]
# ## 게임당 최대 100개 제한: 난수 시드 5개
#
# 균등 무작위 제한은 frank의 실제 API 정렬·선별을 복원하지 않습니다. 게임별 긍정률 MAE는 추천 성능 지표가 아닙니다.

# %%
display(pd.read_csv(T / '09_cap_sensitivity.csv'))
display(Image(filename=str(EDA / 'figures/forgemaster/02_cap_distribution.png')))

# %% [markdown]
# ## frank 중복과 학습·평가 계정 중복 검사
#
# 정확한 SteamID와 App ID를 대조했습니다. 이 EDA 당시에는 Kieran 계정별 분할 파일이 없어 학습·평가 계정 중복을 확인하지 못했습니다. 후속 모델 평가에서는 계정을 분리했습니다.

# %%
display(pd.Series(m['frank_comparison']['overlap'],name='계정·게임 ID 중복'))
display(pd.Series(m['frank_comparison']['source_eligible_overlap'],name='평가 가능 사용자 중복'))
display(pd.Series(m['training_identity_audit']))

# %% [markdown]
# ## 결론
#
# 100개 제한의 분포·이력 영향을 살피는 **리뷰 수 제한의 영향을 확인하는 추가 평가 데이터**입니다. 모델 평가에는 Kieran 학습 계정과 중복되지 않게 분리한 데이터를 사용합니다. 2021년 자료·일부 게임만 포함하므로 최신 게임 및 전체 후보의 성능을 대표하지 않습니다.
