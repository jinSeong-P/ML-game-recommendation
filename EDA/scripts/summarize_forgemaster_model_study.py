"""Summarize model evaluation with paired uncertainty and cohort decomposition."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forgemaster_model_study import EDA,ROOT,PROTOCOL,SEEDS,MODEL_NAMES,metric_summary,save_json,verify_stage

NAMES={'binary_item_knn_k100':'Item-KNN K=100','activity_inverse_degree_cosine_k100':'활동량 보정 Item-KNN',
    'metadata_extended_tfidf':'확장 메타데이터 TF-IDF','train_popularity':'학습 데이터의 인기순 추천'}


def read(case,model):
    return pd.read_parquet(ROOT / 'evaluations' / case / model / 'per_user.parquet')


def bootstrap_delta(difference,seed=20261108):
    difference=np.asarray(difference,dtype=np.float64)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(40):
        indices=rng.integers(0,len(difference),size=(50,len(difference)),dtype=np.int32)
        draws.extend(difference[indices].mean(axis=1).tolist())
    return {'difference':float(difference.mean()),'bootstrap95':np.quantile(draws,[.025,.975]).tolist(),
            'users':len(difference),'replicates':2000}


def markdown_table(headers,rows):
    lines=['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']
    lines+=['| '+' | '.join(str(x) for x in row)+' |' for row in rows]
    return '\n'.join(lines)


def main():
    result=json.loads((ROOT / 'evaluation_summary.json').read_text('utf-8'))
    if result['status']!='complete':raise ValueError('Evaluation is incomplete')
    prepared=verify_stage(ROOT / 'prepared.json');training=verify_stage(ROOT / 'training_source.json')
    verify_stage(ROOT / 'model_manifest.json');content=verify_stage(ROOT / 'content_manifest.json')
    for case in result['results']:
        verify_stage(ROOT / 'evaluations' / case['case'] / case['model'] / 'manifest.json')
    by_case={(row['case'],row['model']):row for row in result['results']}
    paired=[];decomposition=[]
    for model in MODEL_NAMES:
        full=read('full',model)
        for seed in SEEDS:
            frames={label:read(f'cap{seed}_{label}',model) for label in ['full_history','capped_full_mask','capped']}
            base=frames['full_history']
            for other in frames.values():
                same_users=np.array_equal(base.user_key.to_numpy(dtype=np.uint64),other.user_key.to_numpy(dtype=np.uint64))
                same_targets=np.array_equal(base.target_app_id.to_numpy(dtype=np.int64),other.target_app_id.to_numpy(dtype=np.int64))
                if not same_users or not same_targets:
                    raise AssertionError('Paired users or hidden targets differ')
            if np.any(frames['capped'].ndcg.to_numpy()>frames['capped_full_mask'].ndcg.to_numpy()+1e-12):
                raise AssertionError('Removing exclusions improved the unchanged target ranking, which is impossible')
            if model=='train_popularity' and not np.array_equal(base.ndcg.to_numpy(),frames['capped_full_mask'].ndcg.to_numpy()):
                raise AssertionError('Popularity score changed when only the query profile changed')
            selection=full[full.user_key.isin(base.user_key)]
            if len(selection)!=len(base):raise AssertionError('Cap users were not a subset of full cohort')
            # Each effect follows an explicit observation/target/mask transition.
            for label,left,right in [('profile_only','full_history','capped_full_mask'),
                ('mask_only','capped_full_mask','capped'),('total_paired','full_history','capped')]:
                row={'model':model,'seed':seed,'comparison':label,'users':len(base)}
                for metric,column in [('Recall@10','recall'),('NDCG@10','ndcg')]:
                    row[metric]=bootstrap_delta(frames[right][column].to_numpy()-frames[left][column].to_numpy())
                paired.append(row)
            for metric,column in [('Recall@10','recall'),('NDCG@10','ndcg')]:
                a=float(full[column].mean());b=float(selection[column].mean())
                c=float(base[column].mean());d=float(frames['capped_full_mask'][column].mean());e=float(frames['capped'][column].mean())
                row={'model':model,'seed':seed,'metric':metric,'full_population':a,
                    'selected_users_full_targets':b,'selected_users_cap_targets_full_history':c,
                    'cap_profile_full_mask':d,'cap_profile_cap_mask':e,
                    'cohort_selection_difference':b-a,'target_selection_difference':c-b,
                    'profile_difference':d-c,'exclusion_mask_difference':e-d,'overall_difference':e-a}
                if abs(sum(row[k] for k in ['cohort_selection_difference','target_selection_difference','profile_difference','exclusion_mask_difference'])-row['overall_difference'])>1e-12:
                    raise AssertionError('Metric decomposition did not reconcile')
                decomposition.append(row)
    tables=EDA / 'reports/tables/forgemaster_models';tables.mkdir(parents=True,exist_ok=True)
    flat=[]
    for row in result['results']:
        flat.append({key:value for key,value in row.items() if not isinstance(value,(list,dict))})
    pd.DataFrame(flat).to_csv(tables / '01_model_metrics.csv',index=False,encoding='utf-8-sig')
    pair_flat=[]
    for row in paired:
        item={key:row[key] for key in ['model','seed','comparison','users']}
        for metric in ['Recall@10','NDCG@10']:
            item[metric+'_difference']=row[metric]['difference']
            item[metric+'_ci_low'],item[metric+'_ci_high']=row[metric]['bootstrap95']
        pair_flat.append(item)
    pd.DataFrame(pair_flat).to_csv(tables / '02_paired_differences.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(decomposition).to_csv(tables / '03_population_target_profile_mask_decomposition.csv',index=False,encoding='utf-8-sig')
    public={'protocol':json.loads(PROTOCOL.read_text('utf-8')),'preparation':{k:v for k,v in prepared.items() if k!='files'},
        'training':{k:v for k,v in training.items() if k!='files'},
        'content_summary':{k:v for k,v in content.items() if k!='files'},
        'results':result['results'],'paired_differences':paired,'decomposition':decomposition,
        'interpretation':'Conditional on this corpus and fixed algorithms. Different cap draws overlap; 5 draws are not independent populations. Paired bootstrap is per user within one draw.'}
    save_json(EDA / 'reports/forgemaster_model_evaluation.json',public)
    full_table=markdown_table(['모델','평가 사용자','Recall@10','NDCG@10','추천 없음 비율','게임별 평균 Recall'],[
        [NAMES[name],f"{by_case[('full',name)]['users']:,}",f"{by_case[('full',name)]['Recall@10']:.2%}",
         f"{by_case[('full',name)]['NDCG@10']:.5f}",f"{by_case[('full',name)]['empty_recommendation_rate']:.2%}",
         f"{by_case[('full',name)]['game_macro_Recall@10']:.2%}"] for name in MODEL_NAMES])
    paired_table=markdown_table(['모델','난수 시드','평가 사용자','원래 입력 Recall','제한 입력·전체 제외 목록 Recall','제한 입력·제한 제외 목록 Recall','차이 (%p, 95% 신뢰구간)'],[
        [NAMES[model],seed,f"{by_case[(f'cap{seed}_capped',model)]['users']:,}",
         f"{by_case[(f'cap{seed}_full_history',model)]['Recall@10']:.2%}",
         f"{by_case[(f'cap{seed}_capped_full_mask',model)]['Recall@10']:.2%}",
         f"{by_case[(f'cap{seed}_capped',model)]['Recall@10']:.2%}",
         next(f"{r['Recall@10']['difference']*100:+.2f} [{r['Recall@10']['bootstrap95'][0]*100:+.2f}, {r['Recall@10']['bootstrap95'][1]*100:+.2f}]" for r in paired if r['model']==model and r['seed']==seed and r['comparison']=='total_paired')]
         for model in MODEL_NAMES for seed in SEEDS])
    paired_mean_table=markdown_table(['모델','원래 입력 Recall','제한 입력 Recall','차이 (%p)',
        '원래 입력 NDCG','제한 입력 NDCG','NDCG 차이'],[
        [NAMES[model],
         f"{np.mean([by_case[(f'cap{s}_full_history',model)]['Recall@10'] for s in SEEDS]):.2%}",
         f"{np.mean([by_case[(f'cap{s}_capped',model)]['Recall@10'] for s in SEEDS]):.2%}",
         f"{np.mean([r['Recall@10']['difference'] for r in paired if r['model']==model and r['comparison']=='total_paired'])*100:+.2f}",
         f"{np.mean([by_case[(f'cap{s}_full_history',model)]['NDCG@10'] for s in SEEDS]):.5f}",
         f"{np.mean([by_case[(f'cap{s}_capped',model)]['NDCG@10'] for s in SEEDS]):.5f}",
         f"{np.mean([r['NDCG@10']['difference'] for r in paired if r['model']==model and r['comparison']=='total_paired']):+.5f}"]
        for model in MODEL_NAMES])
    decomposition_table=markdown_table(['모델','평가자 선택 (%p)','정답 변경 (%p)','입력 축소 (%p)','제외 목록 축소 (%p)'],[
        [NAMES[model],*[f"{np.mean([r[key] for r in decomposition if r['model']==model and r['metric']=='Recall@10'])*100:+.2f}"
        for key in ['cohort_selection_difference','target_selection_difference','profile_difference','exclusion_mask_difference']]]
        for model in MODEL_NAMES])
    conclusion=[]
    for name in MODEL_NAMES:
        rows=[r for r in paired if r['model']==name and r['comparison']=='total_paired']
        differences=[r['Recall@10']['difference']*100 for r in rows]
        below=sum(r['Recall@10']['bootstrap95'][1]<0 for r in rows)
        conclusion.append(f"- **{NAMES[name]}:** 같은 사용자와 정답에서 제한 후 Recall 차이는 **{min(differences):+.2f}–{max(differences):+.2f}%p**다. 5개 표본 중 **{below}개**는 차이의 95% 신뢰구간 전체가 0보다 낮았다.")
    figures=EDA / 'figures/forgemaster_models';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    labels=['Binary KNN','Activity-weighted KNN','Extended TF-IDF','Train popularity'];x=np.arange(4);width=.19
    for j,(case,label,color) in enumerate([('full','Full population','#3764a3'),('full_history','Cap users: full histories','#8494a8'),
        ('capped_full_mask','Cap inputs, full exclusions','#d69c45'),('capped','Cap inputs and exclusions','#bd4d3c')]):
        means=[];low=[];high=[]
        for model in MODEL_NAMES:
            vals=([by_case[('full',model)]['Recall@10']*100] if case=='full' else [by_case[(f'cap{s}_{case}',model)]['Recall@10']*100 for s in SEEDS])
            means.append(np.mean(vals));low.append(np.mean(vals)-min(vals));high.append(max(vals)-np.mean(vals))
        axes[0].bar(x+(j-1.5)*width,means,width,label=label,color=color,yerr=[low,high],capsize=2)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Recall@10 (%)',title='Population and history comparisons')
    axes[0].tick_params(axis='x',rotation=18);axes[0].legend(fontsize=8)
    colors=['#3764a3','#377f72','#9a68a8','#d69c45']
    for i,model in enumerate(MODEL_NAMES):
        rows=[r for r in paired if r['model']==model and r['comparison']=='total_paired']
        y=np.array([r['Recall@10']['difference']*100 for r in rows]);ci=np.array([r['Recall@10']['bootstrap95'] for r in rows])*100
        axes[1].errorbar(np.arange(5)+(i-1.5)*.08,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt='o-',color=colors[i],label=labels[i],capsize=2,ms=4)
    axes[1].axhline(0,color='black',lw=1)
    axes[1].set(xticks=np.arange(5),xticklabels=[str(s) for s in SEEDS],xlabel='Per-game cap sampling seed',ylabel='Capped minus full Recall (percentage points)',title='Same users and targets: paired effect')
    axes[1].tick_params(axis='x',rotation=25);axes[1].legend(fontsize=8)
    fig.savefig(figures / '01_model_cap_effect.png',bbox_inches='tight');plt.close(fig)
    audit=training['identity_audit'];counts=training['training']
    report=f'''# Forgemaster 모델 평가와 리뷰 수 제한의 영향

2026-11-08 (KST). [실험 전에 정한 평가 방법](../data/reference/forgemaster_cap_protocol_v1.json) · [데이터 품질과 분포 분석](forgemaster_eda.md) · [용어 설명](../docs/terminology.md)

Kieran 리뷰로 학습한 추천 모델을 Forgemaster 리뷰로 평가했다. **게임당 리뷰 수를 제한하지 않은 전체 평가를 중심으로 성능을 설명한다.** 최대 100개 제한을 적용한 평가는 사용자 이력과 평가 대상이 달라질 때 결과가 어떻게 변하는지 확인한다.

## 100개 제한 없는 전체 평가

평가 조건을 만족한 **{prepared['full_users']:,}명** 모두를 평가했다. 각 사용자가 긍정 평가한 Linux 게임 하나를 무작위로 숨기고, 나머지 관측 이력으로 그 게임을 추천하는지 확인했다. 여기서 전체 이력은 이 데이터셋에 수집된 리뷰 전체를 뜻한다.

{full_table}

Recall@10은 정답을 추천 10개 안에서 찾은 사용자의 비율이다. NDCG@10은 정답을 더 높은 순위로 추천할수록 높다. 게임별 평균 Recall은 게임마다 Recall을 구한 뒤 같은 가중치로 평균한 값이다. 추천이 없는 사용자도 전체 지표 계산에 포함했다.

## 100개 제한의 영향: 같은 사용자와 정답 비교

Forgemaster의 정리된 리뷰에서 게임별로 최대 100개를 무작위로 남겼다. 난수 시드 5개로 표본을 만들었고, 각 표본의 평가 가능 사용자는 5,969–6,130명이었다. 이 사용자의 제한된 이력에서 정답 하나를 선택한 뒤 다음 세 조건에서 같은 정답을 사용했다.

1. **원래 입력과 전체 제외 목록:** 리뷰를 제한하기 전의 긍정 입력을 사용한다. 이미 리뷰한 모든 게임은 추천에서 제외한다.
2. **제한 입력과 전체 제외 목록:** 점수 계산에 쓰는 긍정 입력만 줄인다. 추천 제외 목록은 유지한다. 1번과 비교하면 입력 손실의 영향을 알 수 있다.
3. **제한 입력과 제한 제외 목록:** 입력과 제외 목록을 모두 수집된 제한 이력에 맞춰 줄인다. 2번과 비교하면 수집에서 빠진 기평가 게임이 다시 추천 후보에 들어오는 영향을 알 수 있다.

다음은 5개 표본의 단순 평균이다. 원래 입력 열도 제한 후 평가 가능한 약 6천 명을 대상으로 한다. 따라서 위의 전체 395,421명 결과와 평가 대상과 정답 구성이 다르다.

{paired_mean_table}

차이는 3번에서 1번을 뺀 값이다. 겹치는 사용자가 있는 5개 표본을 독립 표본으로 합산하지 않았다.

{chr(10).join(conclusion)}

각 표본에서 사용자를 복원 추출해 두 조건의 점수 차이를 2,000회 계산하는 짝지은 부트스트랩(paired bootstrap)으로 95% 신뢰구간을 구했다. 부트스트랩 난수 시드는 20261108로 고정했다. 아래는 표본별 결과다.

{paired_table}

![모델별 리뷰 수 제한의 영향](../figures/forgemaster_models/01_model_cap_effect.png)

왼쪽의 전체 집단과 제한 후 집단은 사용자가 다르다. 오른쪽은 같은 사용자와 정답을 비교한다. 왼쪽 오차 막대는 5개 표본의 최소–최대, 오른쪽은 표본별 95% 신뢰구간이다.

## 점수 변화의 요인

전체 평가에서 제한 후에도 평가 가능한 사용자만 추리고, 이후 정답 선택·입력·제외 목록을 차례로 바꿨다. 각 단계의 Recall 차이를 5개 표본에 걸쳐 평균한 결과다. 네 단계 차이의 합이 전체 점수 차이와 일치하는지 확인했다.

{decomposition_table}

앞의 두 열은 평가 사용자와 정답 구성이 달라지는 영향이다. 뒤의 두 열은 같은 사용자와 정답에서 입력과 추천 제외 목록이 달라지는 영향이다. 정답 변경에는 무작위 추출에 따른 변동도 포함된다. [단계별 Recall·NDCG 계산표](tables/forgemaster_models/03_population_target_profile_mask_decomposition.csv)에 전체 수치를 저장했다.

## 학습 데이터 분리와 모델 재학습

기존 저장소에는 과거 모델의 학습 파일과 `split_v1` 데이터 분할이 없었다. Kieran 원본 버전 2 ZIP **17,241,297,932바이트**를 다시 받고 식별 정보가 있는 열을 변환했다. 파일 SHA256은 기존 기록 `ec3234d1ff0831f5a20752059df1c85dfe66ab0efde95b886c7cc3a11d0430b7`와 일치했다. 압축 파일의 모든 멤버 CRC와 원본 113,883,717행을 검사했다. 리뷰 ID 충돌과 라벨 충돌을 기존 규칙으로 제외한 113,195,034쌍에서 새 학습 데이터를 만들었다.

- Forgemaster 평가 가능 **{prepared['full_users']:,}계정**과 기존 frank 평가용 **{prepared['frank_reserved']:,}계정**을 합친 **{prepared['combined_reserved']:,}계정**의 리뷰를 학습에서 미리 제외했다.
- Forgemaster 평가 계정 중 Kieran에도 있는 계정은 **{audit['forge_test_accounts_in_kieran']:,}명**이었다. 이들의 Kieran 리뷰 **{audit['forge_test_pairs_removed_from_kieran']:,}쌍**을 모두 제외했다.
- 과거 분할 방법으로 복원한 긍정 학습 데이터는 **{audit['reconstructed_original_positive_train_pairs']:,}쌍**으로 기존 기록과 같았다. 이 분할에는 Forgemaster 평가자 **{audit['forge_accounts_in_reconstructed_original_positive_train']:,}명**이 포함될 수 있었다. 이번 학습에서는 이 계정을 제외했다.
- 기존 검증 계정 분리 방법(MD5 계정 해시 기준)도 유지했다. 새 학습 데이터는 지정한 게임 목록 안의 긍정 **{counts['pairs']:,}쌍·{counts['users']:,}계정**이다. 긍정 리뷰가 한 개뿐인 사용자도 게임별 학습 사용자 수와 코사인 정규화 계산에 포함했다.
- 학습 데이터와 사전에 분리한 평가 계정의 교집합은 **{training['leakage_checks']['train_reserved_pair_overlap']}**, 검증 계정 교집합은 **{training['leakage_checks']['train_validation_pair_overlap']}**, 중복 학습 쌍은 **{training['leakage_checks']['duplicate_train_pairs']}**이다.

새 평가자를 학습에서 제외했으므로 과거 모델과 가중치가 완전히 같지는 않다. 기존에 선택한 알고리즘, 이웃 수 K=100, 콘텐츠 결합 비중 50:50을 유지해 별도 학습했다. 과거 실험의 분할·보고서·모델 선택 기록은 보존했다. Forgemaster 점수를 보고 설정을 다시 고르지 않았다.

## 공통 평가 방법

Steam 계정 ID는 정밀도 손실 없는 정수 또는 문자열로 읽었다. 입력 게임 37,610개와 Linux 추천 후보 6,630개를 공통으로 사용했다. 학습 리뷰가 없는 후보도 목록에 남겼다.

사용자별 긍정 Linux 게임 하나를 동일한 확률로 골라 정답으로 숨겼다. 숨긴 뒤 남은 리뷰 게임이 5개 이상이고 그중 긍정 게임이 1개 이상인 사용자를 평가했다. 점수에는 남은 긍정 리뷰만 사용하고, 남은 긍정·부정 리뷰 게임은 모두 추천에서 제외했다. 숨긴 정답은 추천 제외 목록에서 뺐다.

양수 점수를 받은 게임을 최대 10개 추천하고 점수가 같으면 App ID 오름차순으로 정렬했다. 추천이 부족해도 다른 방법으로 채우지 않았다. 인기순 추천은 별도의 비교 모델이다. 정답이 한 개이므로 적중 시 NDCG는 `1/log2(정답 순위+1)`이며 계산에는 float64를 사용했다.

기본 Item-KNN은 두 게임을 함께 긍정 평가한 사용자 수로 코사인 유사도를 구하고, 각 추천 후보에 유사한 입력 게임 100개를 저장한다. 활동량 보정 모델은 사용자의 긍정 게임 수 역수를 공동 기여도와 게임별 가중 사용자 수에 적용한다. **이웃 수 K=100은 게임당 리뷰 100개 제한과 다른 설정이다.**

콘텐츠 모델은 `Game Recommendations on Steam`의 소개글을 우선 쓰고, 없으면 `Steam Games 2025`의 짧은 소개글로 보충한다. 설명의 단어·연속 두 단어 TF-IDF 유사도 50%와 태그·장르·범주·개발사·배급사 TF-IDF 유사도 50%를 합친다. 두 채널의 사용자 벡터를 각각 정규화하며 공개 게임 정보만으로 학습한다.

## 해석 범위와 사용 방법

**리뷰 수 제한이 없는 결과를 중심으로 보고하고, 100개 제한 결과는 제한의 영향을 설명하는 데 사용한다.** 협업 모델의 하락은 확인됐지만 기존 Kieran·frank 점수 차이 전체가 100개 제한 때문이라고 단정할 수는 없다. 이번 무작위 표본 추출은 실제 frank API의 정렬과 리뷰어 선택을 재현하지 않는다.

데이터는 2021년 4월까지의 이력이며 Linux 후보 중 2,210개만 관측한다. Kieran 학습 자료와 고정 게임 정보는 더 뒤 시점의 자료도 포함하므로 미래 예측 성능을 측정한 실험이 아니다. 수집 시점과 게임 선택의 편향도 남아 있다.

Forgemaster에는 원 리뷰 ID인 `recommendationid`가 없다. 과거 실험의 리뷰 ID 충돌 제외 방법을 똑같이 적용할 수 없어 사용자–게임 라벨 충돌을 제외하고 평가 계정 전체를 학습에서 차단했다. 이번 모델의 학습·평가 사용자 중복은 해소했다. 다만 모델 설정은 과거 내부 검증에서 고른 값을 사용했으며 그때의 선택 과정을 새 계정 분리 기준으로 다시 수행한 것은 아니다. 따라서 모든 과거 개발 과정에서 독립된 최종 테스트로 해석하지 않는다.

## 검증과 재현 방법

작은 수치 예제로 자기 이웃 제외, 동점 처리, 학습 자료가 없는 긍정 입력의 분모 유지, 두 콘텐츠 채널의 개별 정규화, 빈 추천의 분모 유지, 정확한 Steam ID와 적중 순위·NDCG를 검사했다. 실제 저장 파일에서는 정답이 입력에 없는지, 추천에 제외 대상·중복·잘못된 순위와 점수가 없는지 확인했다. 사용자별 지표와 추천을 저장하고 파일 해시값을 기록했다. 독립 검산은 저장된 적중 순위에서 Recall·NDCG를 다시 계산하고 추천 순서도 확인한다.

- [평가 수치·신뢰구간·학습 정보 JSON](forgemaster_model_evaluation.json)
- [모든 조건의 평가 지표 CSV](tables/forgemaster_models/01_model_metrics.csv)
- [같은 사용자·정답의 점수 차이와 신뢰구간 CSV](tables/forgemaster_models/02_paired_differences.csv)
- [저장 결과 검산 로그](forgemaster_model_evaluation_audit.txt)
- 모델·학습 데이터·사용자별 결과: `data/processed/modeling/forgemaster_cap_study_v1/` (Git 제외)

저장소 루트의 PowerShell에서 실행한다.

```powershell
.venv/Scripts/python EDA/scripts/ingest_kieran_original.py
.venv/Scripts/python EDA/scripts/forgemaster_model_study.py prepare
.venv/Scripts/python EDA/scripts/forgemaster_model_study.py content
.venv/Scripts/python EDA/scripts/forgemaster_model_study.py source
.venv/Scripts/python EDA/scripts/forgemaster_model_study.py fit
.venv/Scripts/python -m unittest discover -s EDA/scripts -p test_forgemaster_model_study.py -v
.venv/Scripts/python EDA/scripts/forgemaster_model_study.py evaluate
.venv/Scripts/python EDA/scripts/summarize_forgemaster_model_study.py
.venv/Scripts/python EDA/scripts/audit_forgemaster_model_evaluation.py
```
'''
    (EDA / 'reports/forgemaster_model_evaluation.md').write_text(report,encoding='utf-8')
    print('Wrote model evaluation report, exact metrics, paired intervals, decomposition, and figure')


if __name__=='__main__':main()
