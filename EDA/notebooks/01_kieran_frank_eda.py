# %% [markdown]
# # Kieran → frank: 메인 데이터셋 EDA
#
# Kieran v2 원본을 학습·내부 검증, frank 고정 리뷰를 외부 테스트로 채택했다.
# 원본 전수 집계와 재검증 결과를 읽어 규모·품질·계정 분리·평가 가능 범위를 정리한다.
# 최종 split 파일과 추천 모델 성능은 아직 만들지 않았다.
#
# 입력은 고정 카탈로그의 리뷰 게임이며, 정답은 그중 공식 Linux 긍정 게임이다.
# 정답 **하나만** 숨기고 나머지 긍정·부정 리뷰는 입력으로 남길 수 있는 조건이다.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

EDA = next(p for p in [Path.cwd()/'EDA', Path.cwd(), *Path.cwd().parents] if (p/'scripts/summarize_kieran_frank.py').exists())
sys.path.insert(0,str(EDA/'scripts'))
from eda_large_review_frank import digest
from readable_labels import setup_korean_font

assert setup_korean_font(), '한글 글꼴이 필요합니다.'
plt.rcParams.update({'figure.facecolor':'white','axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.2,
                     'axes.axisbelow':True,'font.size':10})
m=json.loads((EDA/'reports/kieran_original_frank_eda_metrics.json').read_text('utf-8'))
e=json.loads((EDA/'reports/kieran_frank_eda_metrics.json').read_text('utf-8'))
assert e['source_metric_sha256']==digest(EDA/'reports/kieran_original_frank_eda_metrics.json'), '집계 검증 스크립트를 다시 실행하세요.'
assert e['reference_manifest_sha256']==digest(EDA/'data/reference/manifest.json')
assert m['whole_account_and_cross_review_identity_checks_passed']
assert all(v==0 for v in m['post_exclusion_checks'].values())
assert e['status']['dataset_choice_final'] and not e['status']['final_train_validation_test_files_written']
tables=EDA/'reports/tables'
figures=EDA/'figures'
figures.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. 두 메인 원천의 크기
#
# Kieran은 원본 주 CSV 전체를, frank는 고정 버전의 선택한 리뷰 열 전체 행을 읽었다.
# frank 전체 리뷰 정리는 알려진 296개 모호 리뷰 ID를 평가 범위에서 제외하기 전 값이다.

# %%
sources=pd.DataFrame([
    {'자료':'Kieran 원본','원본 리뷰 행':m['raw_quality']['raw_rows'],
     '정리 후 사용자–게임 쌍':m['clean_source']['pairs'],'계정':m['clean_source']['users'],
     '게임':m['clean_source']['apps'],'역할':'학습·내부 검증'},
    {'자료':'frank 리뷰','원본 리뷰 행':e['frank_provenance']['raw_rows'],
     '정리 후 사용자–게임 쌍':e['frank_full_review_summary']['clean_pairs'],
     '계정':e['frank_full_review_summary']['users'],'게임':e['frank_full_review_summary']['apps'],
     '역할':'외부 테스트 원천'}])
display(sources)
print('Kieran 긍정 비율:',f"{m['clean_source']['positives']/m['clean_source']['pairs']:.2%}")
print('Kieran 사용자별 리뷰 게임 수 중앙값/p90/p99:',*[m['clean_source']['history_quantiles'][i] for i in [1,3,4]])

# %% [markdown]
# ## 2. 품질 검사와 격리
#
# 원 계정 ID는 float로 읽지 않는다. 같은 원 리뷰가 다른 게임에 연결된 경우는
# 어느 게임이 맞는지 추측하지 않고 해당 원 리뷰 전체를 격리했다.

# %%
q=m['raw_quality']
quality=pd.DataFrame([
    ('형식·결측 무효 행',q['quarantined_rows'],'사용 제외'),
    ('원 리뷰 ID의 계정 충돌',q['review_ids_with_conflicting_owner'],'계정 정밀도 검사'),
    ('원 리뷰 ID의 게임 충돌',q['review_ids_with_conflicting_app'],'해당 ID 전체 격리'),
    ('게임 충돌 ID에 속한 행',q['rows_with_conflicting_review_identity'],'686,161행 격리'),
    ('frank 대조 추가 격리 ID',q['additional_cross_source_review_ids_quarantined'],'추가 3개 ID 격리'),
    ('같은 라벨 반복 행',q['same_label_duplicate_rows'],'중복 제거'),
    ('긍정·부정 충돌 쌍',q['conflicting_label_pairs'],'쌍 전체 격리'),
    ('플레이 시간 결측 행',q['raw_rows']-m['raw_playtime']['rows_with_playtime'],'0분과 구분'),
    ('갱신 시각 역전 행',q['updated_before_created'],'시간 정보 한계 기록')],
    columns=['검사','수','처리·해석'])
display(quality)
quality.to_csv(tables/'03_kieran_quality.csv',index=False,encoding='utf-8-sig')

# %% [markdown]
# ## 3. 계정 전체 제외와 검증 예약
#
# frank 예비 평가 계정 87,573명을 전체 예약한다. 그중 Kieran에서 일치하는
# 79,879계정의 모든 게임 3,360,025쌍을 제외했다. 검증 계정은 나머지 원 계정의
# 고정 MD5 규칙으로 약 10% 예약했다. 아래는 최종 파일을 만들기 전의 규모다.

# %%
splits=pd.DataFrame([{'단계':label,'계정':m[key]['users'],'리뷰 쌍':m[key]['pairs'],
                      '긍정 쌍':m[key]['positives']} for key,label in [
    ('clean_source','정리 후 전체'),('after_test_exclusion','테스트 계정 제외 후'),
    ('validation_preview','내부 검증 예약'),('after_validation_reservation','학습안')]])
display(splits)
splits.to_csv(tables/'01_main_split_preview.csv',index=False,encoding='utf-8-sig')
display(pd.DataFrame([{'검사':k,'교집합':v} for k,v in m['post_exclusion_checks'].items()]))
assert splits.iloc[1]['리뷰 쌍']==splits.iloc[2]['리뷰 쌍']+splits.iloc[3]['리뷰 쌍']
print('원 리뷰 공유:',f"{m['cross_source_overlap']['common_original_review_ids']:,}",
      '/ 작성자 불일치:',m['cross_source_overlap']['common_review_ids_with_different_owner'])

# %% [markdown]
# ## 4. frank 평가 후보
#
# 알려진 리뷰–게임 연결 모호성 296개 ID의 영향을 받는 사용자–게임 쌍을 제거했다.
# 원 계정 제외 상위 집합은 유지한다. 입력은 긍정·부정 리뷰를 함께 세며,
# 숨긴 정답 외 다른 긍정 입력의 존재도 따로 보고한다.

# %%
cohort=pd.DataFrame(e['cohorts']['safe_mapping_fixed_catalog']).rename(columns={
    'minimum_inputs':'최소 다른 입력','potential_users':'평가 후보',
    'potential_users_with_positive_query':'긍정 입력도 남는 후보',
    'potential_users_with_only_negative_query':'부정 입력만 남는 후보',
    'possible_target_pairs':'선택 전 가능한 정답 쌍'})
display(cohort)
cohort.to_csv(tables/'02_frank_eligibility.csv',index=False,encoding='utf-8-sig')
support=[]
for key,label in [('safe_mapping_fixed_catalog','고정 평가 범위'),
                  ('safe_mapping_kieran_review_support','모든 입력·정답에 학습 리뷰 있음'),
                  ('safe_mapping_kieran_positive_support_5','모든 입력·정답에 긍정 학습 계정 ≥5')]:
    row=next(r for r in e['cohorts'][key] if r['minimum_inputs']==5)
    support.append({'조건':label,'입력 ≥5 후보':row['potential_users'],
                    '긍정 입력도 남는 후보':row['potential_users_with_positive_query']})
display(pd.DataFrame(support))
print('지원 조건은 진단입니다. 모델별로 어려운 사용자를 평가 명단에서 제거하지 않습니다.')

# %%
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
counts=[m[k]['pairs']/1_000_000 for k in ['clean_source','after_test_exclusion','after_validation_reservation']]
axes[0].bar(['정리 후 전체','테스트 계정 제외','검증 예약 후 학습'],counts,color=['#abc7e5','#699ace','#2463a6'])
axes[0].set_ylabel('사용자–게임 쌍 (백만)')
axes[0].set_title('Kieran 학습 규모 미리보기')
axes[0].set_ylim(0,max(counts)*1.16)
for i,y in enumerate(counts): axes[0].text(i,y+1,f'{y:.2f}',ha='center')
x=range(len(cohort))
axes[1].bar(x,cohort['긍정 입력도 남는 후보'],label='긍정 입력도 있음',color='#2463a6')
axes[1].bar(x,cohort['부정 입력만 남는 후보'],bottom=cohort['긍정 입력도 남는 후보'],label='부정 입력만 있음',color='#da7732')
axes[1].set_xticks(list(x),[str(v) for v in cohort['최소 다른 입력']])
axes[1].set_xlabel('숨긴 정답 외 최소 입력 게임 수')
axes[1].set_ylabel('평가 후보 (명)')
axes[1].set_title('frank 정리 후 잠재 평가 규모')
axes[1].legend()
fig.suptitle('Kieran → frank | 최종 분할 파일·모델 성능이 아닌 EDA')
fig.tight_layout()
fig.savefig(figures/'fig01_kieran_frank_overview.png',dpi=160,bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. 리뷰 시기와 카탈로그 연결
#
# Kieran 원본은 2010–2023년 다국어 리뷰다. 시간순 분할은 수행하지 않았다.
# 게임 카탈로그와 메타데이터 존재 여부는 별도 고정 참조다.
# 설명·태그 존재는 실제 텍스트 품질과 모델 성능을 뜻하지 않는다.

# %%
years=pd.DataFrame(m['raw_created_years'])
fig,ax=plt.subplots(figsize=(10,3.8))
ax.bar(years['review_created_year'],years['raw_rows']/1_000_000,color='#2463a6')
ax.set(xlabel='최초 작성 연도 (UTC)',ylabel='원본 리뷰 행 (백만)',title='Kieran 원본 리뷰 시기 — 시간순 평가 아님')
fig.tight_layout()
fig.savefig(figures/'fig02_kieran_review_years.png',dpi=160,bbox_inches='tight')
plt.show()
display(pd.DataFrame([{'고정 입력 게임':e['catalog']['input_apps'],
    '고정 Linux 후보':e['catalog']['linux_target_apps'],
    '학습 리뷰가 남는 Linux 게임':m['after_validation_reservation']['linux_apps_in_fixed_catalog'],
    '원본 게임 중 메타데이터 미연결':m['metadata_coverage']['source_apps_without_A_or_C_metadata']}]))

# %% [markdown]
# ## 6. 다음 단계
#
# - 최소 입력 수·긍정 입력 조건·정답 선택 규칙을 고정하고 train/validation/test 파일을 만든다.
# - 저장 파일을 다시 읽어 계정·원 리뷰·사용자–게임 및 입력–정답 비중복을 검사한다.
# - 인기도·공동 출현·이웃·특징은 학습에서만 계산하고 모델 선택은 검증셋에서 한다.
# - frank의 앱당 최대 100리뷰와 활동적인 리뷰어 편향을 결과에 함께 기록한다.
# - Kieran 플레이 시간은 리뷰 작성자 스냅샷이며 전체 보유 목록이 아니다.
#
# 이전 A/B/C·후보 비교·artermiloff 제외 증거는 `legacy/`에 보존한다.
# 메인 요약은 `reports/kieran_frank_eda.md`, 출처·컬럼은 `docs/datasets/`에 있다.
