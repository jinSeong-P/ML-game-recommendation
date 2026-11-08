"""Summarize persisted v4 outcomes; selection freeze is an explicit last step."""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from eda_large_review_frank import EDA,digest
from improvement_common import write_json
from experiment_v4_common import ROOT,PROTOCOL,SPLIT,load_cohort,paired_comparison,catalog,support


def check_guards(m,base,paired):
    a,b=m['metrics'],base['metrics']
    return dict(ndcg_ci_lower_positive=paired['NDCG@10']['ci95'][0]>0,
        overall_recall_noninferior=a['recall']>=b['recall']-1e-12,
        game_macro_ndcg_noninferior=a['game_macro_ndcg']>=b['game_macro_ndcg']-1e-12,
        tail_ndcg_noninferior=a['tail']['ndcg'] is not None and a['tail']['ndcg']>=b['tail']['ndcg']-1e-12,
        empty_users_not_increased=a['empty_users']<=b['empty_users'])


def family(name):
    if name.startswith(('rrf_refit_','lambdamart_refit_')):return '재학습한 결합 대조군'
    if name.startswith('residual_'):return '조건부 결합'
    if name.startswith('knn_teacher_'):return 'KNN 관계를 학습한 콘텐츠'
    if name.startswith('qwen3_'):return '고정 Qwen 콘텐츠'
    if name.startswith('rlae_'):return 'RLAE'
    if name.startswith('rdlae_'):return 'RDLAE'
    if name.startswith('dan_'):return 'DAN'
    if name.startswith('ease_'):return 'EASE'
    if name.startswith(('account_pair_','pair_contribution_')):return '계정 기여량 조절'
    if name.startswith('fslim_'):return 'fSLIM'
    return '대조군'


def run(freeze=False):
    c=load_cohort('selection');folder=ROOT/'evaluations/selection'
    manifests=[json.loads(p.read_text('utf-8')) for p in sorted(folder.glob('*/manifest.json'))]
    execution_plan=EDA/'data/reference/primary_execution_plan_v4.json'
    plan=json.loads(execution_plan.read_text('utf-8'))
    frozen_path=ROOT/'selection_frozen.json'
    if freeze:
        available={m['variant'] for m in manifests if m.get('status')=='complete'}
        missing=sorted(set(plan['required_evaluations'])-available)
        if missing:raise ValueError(f'Primary experiments are not complete: {missing}')
        for m in manifests:
            path=folder/m['variant']
            for name,sha in m['files'].items():
                if digest(path/name)!=sha:raise ValueError('Evaluation files changed before selection freeze')
            report=path/'reporting_metrics.json'
            if not report.exists() or json.loads(report.read_text('utf-8')).get('status')!='passed_actual_recommendation_replay':
                raise ValueError(f'Actual recommendations must be audited before freezing: {m["variant"]}')
        if (ROOT/'selection_frozen.json').exists():
            raise ValueError('Selection is already frozen; never overwrite it after opening reporting or confirmation outcomes')
        if (EDA/'data/reference/improvement_selection_v4.json').exists():
            raise ValueError('The versioned v4 selection reference already exists; preserve it')
    base=next(m for m in manifests if m['variant']=='activity_inverse_degree_cosine_k100')
    main=c['n_total']>=5
    target_support=support()[np.searchsorted(catalog()[0],c['target_ids'])]
    sample_games={}
    for label,mask in [('primary',main),('tail',main&(target_support>0)&(target_support<1000))]:
        _,counts=np.unique(c['target_ids'][mask],return_counts=True)
        sample_games[label]=dict(users=int(mask.sum()),games=len(counts),singleton_games=int((counts==1).sum()))
    br=pq.read_table(folder/base['variant']/'per_user.parquet',columns=['target_rank'])['target_rank'].to_numpy()
    rows=[];outcomes=[]
    for m in manifests:
        name=m['variant'];path=folder/name
        pairedpath=path/'paired_comparison.json'
        if pairedpath.exists():paired=json.loads(pairedpath.read_text('utf-8'))
        else:
            rank=pq.read_table(path/'per_user.parquet',columns=['target_rank'])['target_rank'].to_numpy()
            paired=paired_comparison(br,rank,c);write_json(pairedpath,paired)
        guards=check_guards(m,base,paired)
        row=dict(variant=name,family=family(name),users=m['metrics']['users'],
            recall=m['metrics']['recall'],ndcg=m['metrics']['ndcg'],
            game_macro_ndcg=m['metrics']['game_macro_ndcg'],tail_users=m['metrics']['tail']['users'],
            tail_ndcg=m['metrics']['tail']['ndcg'],empty_users=m['metrics']['empty_users'],
            paired_ndcg_delta=paired['NDCG@10']['delta'],
            paired_ndcg_ci_low=paired['NDCG@10']['ci95'][0],paired_ndcg_ci_high=paired['NDCG@10']['ci95'][1],
            eligible=all(guards.values()),**guards)
        rows.append(row);outcomes.append(dict(variant=name,metrics=m['metrics'],guards=guards,paired=paired))
    rows.sort(key=lambda r:(-r['ndcg'],-r['recall'],r['variant']))
    table=EDA/'reports/tables/model_improvement_v4.csv';table.parent.mkdir(parents=True,exist_ok=True)
    with table.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    eligible=[r for r in rows if r['eligible']]
    selected=freeze or frozen_path.exists()
    confirmation_path=EDA/'reports/confirmation_v4.json'
    confirmation=json.loads(confirmation_path.read_text('utf-8')) if confirmation_path.exists() else None
    result=dict(status='selection_complete' if selected else 'running_partial_results',
        protocol_sha256=digest(PROTOCOL),split_manifest_sha256=digest(SPLIT/'manifest.json'),
        evaluated_conditions=len(rows),primary_selection_users=int((c['n_total']>=5).sum()),
        all_selection_users=len(c['user_ids']),control=base['variant'],
        eligible_candidates=[r['variant'] for r in eligible],outcomes=outcomes,
        scheduled_conditions=plan['required_count'],
        sample_game_counts=sample_games,
        unfinished_scheduled_conditions=sorted(set(plan['required_evaluations'])-{m['variant'] for m in manifests}),
        confirmation_opened=bool(confirmation and confirmation['opened']),
        adopted_variant=confirmation['adopted'] if confirmation else base['variant'],
        historical_exposure='Newly reserved from historical v1 train; not a pristine final test')
    if freeze:
        finalist=eligible[0]['variant'] if eligible else None
        lock=dict(status='frozen_after_selection_before_confirmation',protocol_sha256=digest(PROTOCOL),
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            split_manifest_sha256=digest(SPLIT/'manifest.json'),finalist=finalist,
            execution_plan_sha256=digest(execution_plan),
            confirmation_models=([base['variant'],'binary_item_knn_k100','metadata_extended_tfidf',finalist] if finalist else []),
            evaluation_manifests={m['variant']:digest(folder/m['variant']/'manifest.json') for m in manifests},
            policy='No qualifying challenger: keep current control and preserve confirmation unopened' if finalist is None else
                'Only the single frozen challenger and controls may open confirmation')
        write_json(ROOT/'selection_frozen.json',lock)
        write_json(EDA/'data/reference/improvement_selection_v4.json',lock)
        result['selection']=lock
    elif frozen_path.exists():
        result['selection']=json.loads(frozen_path.read_text('utf-8'))
    if confirmation:result['confirmation']=confirmation
    write_json(EDA/'reports/model_improvement_v4.json',result)
    compact=[]
    for group in ['대조군','재학습한 결합 대조군','조건부 결합','KNN 관계를 학습한 콘텐츠','고정 Qwen 콘텐츠','EASE','RLAE','RDLAE','DAN','계정 기여량 조절','fSLIM']:
        members=[r for r in rows if r['family']==group]
        compact.extend(members if group=='대조군' else members[:1])
    lines=['# v4 성능 개선 실험','',
        '**상태:** '+('첫 단계의 설정 선택 완료.' if selected else '실행 중인 중간 결과다. 미완료 계열의 결과는 포함하지 않는다.'),'',
        '새로 예약한 Kieran 계정에서 현재 모델을 재학습해 비교했다. 각 역할 20,000명 중 전체 입력 ≥5인 10,000명을 주평가로 삼고, 짧은 이력 10,000명은 별도 집단으로 보고한다. 세 역할은 결합 학습·설정 선택·확인용이며, 총 60,000명을 새 기초 학습에서 제외했다. 학습 자료는 긍정 77,881,718쌍이다. 기존 frank·Forgemaster 수치와 개선량을 직접 비교하지 않는다.','',
        '**독립성 한계:** 예약 계정은 과거 v1–v3의 학습에는 쓰였다. 현재 재학습의 학습·평가 계정 중복은 0이지만, 프로젝트 전체 개발 과정에서 완전히 미사용된 최종 테스트는 아니다. Forgemaster 계정과의 대조는 로컬 원자료 부재로 수행하지 않았다.','',
        f'첫 단계 예정 {plan["required_count"]}개 중 저장한 평가 조건은 {len(rows)}개다. 모든 백로그의 완료를 뜻하지 않는다. 길이 512/1,024의 콘텐츠 벡터는 같은 원문에서 동일하게 생성됐으므로 독립적인 개선 증거로 세지 않는다. 아래는 계열별 전체 NDCG 최고 설정이며, 최고 점수와 교체 조건 통과는 구분한다.','',
        '| 계열 | 설정 | Recall@10 | NDCG@10 | 게임별 평균 NDCG | 희귀 게임 NDCG | 조건 통과 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in compact:
        tail='없음' if r['tail_ndcg'] is None else f'{r["tail_ndcg"]:.6f}'
        verdict='대조군' if r['family']=='대조군' else ('통과' if r['eligible'] else '미통과')
        lines.append(f'| {r["family"]} | `{r["variant"]}` | {r["recall"]*100:.3f}% | {r["ndcg"]:.6f} | {r["game_macro_ndcg"]:.6f} | {tail} | {verdict} |')
    if eligible:
        best=eligible[0]
        lines += ['',f'**선택용 보호 조건을 모두 통과한 후보 중 최고 설정:** `{best["variant"]}`. '
            f'Recall@10 {best["recall"]*100:.2f}%, NDCG@10 {best["ndcg"]:.6f}, '
            f'게임별 평균 NDCG {best["game_macro_ndcg"]:.6f}, 희귀 게임 NDCG {best["tail_ndcg"]:.6f}다. '
            '위 표의 계열별 최고 전체 점수와 다를 수 있다. 선택용 통과만으로 교체를 확정하지 않으며, 모든 예정 조건을 마친 뒤 한 후보만 확인한다.']
    if confirmation:
        if confirmation['opened']:
            chosen=confirmation['adopted']
            lines += ['',f'**확인 평가 후 결정:** `{chosen}`. '
                +('고정한 도전 모델이 같은 보호 조건을 모두 통과했다.' if chosen==confirmation['finalist'] else
                  '고정한 도전 모델이 확인용 보호 조건을 모두 통과하지 못해 활동량 보정 KNN을 유지한다. 다른 도전 모델로 확인 집단을 다시 시험하지 않는다.'),
                '', '| 확인용 모델 | Recall@10 | NDCG@10 | 게임별 평균 NDCG | 희귀 게임 NDCG |',
                '|---|---:|---:|---:|---:|']
            for name in [base['variant'],confirmation['finalist']]:
                v=confirmation['outcomes'][name]['metrics']
                tail='없음' if v['tail']['ndcg'] is None else f'{v["tail"]["ndcg"]:.6f}'
                lines.append(f'| `{name}` | {v["recall"]*100:.3f}% | {v["ndcg"]:.6f} | {v["game_macro_ndcg"]:.6f} | {tail} |')
            lines += ['', '[고정한 한 후보의 확인 결과](confirmation_v4.json)']
        else:
            lines += ['', '**확인용 집단 보존:** 선택용 보호 조건을 통과한 도전 모델이 없어 확인 평가를 열지 않았다. 활동량 보정 KNN을 유지한다.']
    lines += ['',
        '희귀 게임은 새 학습 자료에서 긍정 사용자가 1–999명인 정답이다. 전체 NDCG 차이의 짝지은 95% 신뢰구간 하한 >0, 전체 Recall·게임별 평균 NDCG·희귀 게임 NDCG 비감소, 빈 추천 비증가를 모두 요구한다. 기존 0 감소폭 기준을 결과를 본 뒤 완화하지 않았다.','',
        '주평가 10,000명은 정답 게임 1,133개를 포함하고, 그중 희귀 게임 정답은 342명·310개다. 게임 집합을 고정한 신뢰구간은 새로운 게임 모집단·재학습·설정 선택의 불확실성을 포함하지 않는다. 짧은 이력 집단도 정답을 숨긴 뒤 긍정 입력이 남는 사용자만 포함하므로 모든 신규 사용자를 대표하지 않는다.','',
        f'게임 안 재표집에서는 사용자 한 명뿐인 게임의 변동을 추정할 수 없다. 주평가의 이런 게임은 {sample_games["primary"]["singleton_games"]}개, 희귀 구간에서는 {sample_games["tail"]["games"]}개 중 {sample_games["tail"]["singleton_games"]}개다. 게임별 신뢰구간은 이 한계를 가진 조건부 진단이며 모집단의 게임별 정확도를 보장하지 않는다.','',
        '## 후보 생성 진단','',
        '정답을 후보에 추가하지 않았다. 전체 Linux 후보 6,630개를 기준으로 평가하며, 후보 밖 정답과 빈 추천 사용자를 분모에 남겼다. 후보 포함률은 재정렬이 달성할 수 있는 상한이다.','',
        '## 검증과 해석','',
        '- 저장된 추천 목록에서 정답 순위를 재구성하고 후보·관측 게임 제외·점수·정렬·목록 길이를 검사했다. 완료한 감사는 각 평가 폴더의 `reporting_metrics.json`에 있다.',
        '- 전체 차이의 신뢰구간은 같은 사용자를 짝지어 계산했다. 게임별 평균은 게임 집합을 고정하고 게임 안 사용자를 재표집했다. 이 구간은 재학습·여러 설정의 선택 불확실성을 포함하지 않는다.',
        '- NDCG의 순위 로그와 평균은 float64로 계산한다. 초기 v4의 낮은 정밀도 집계는 예측 파일을 바꾸지 않고 수정했으며, 수정 전 영수증과 계산 코드를 보존했다. Recall과 적중 순위는 바뀌지 않았다.',
        '- Recall@10은 숨긴 긍정 게임 하나를 추천 10개 안에서 회수한 비율이다. 사용자 만족도·전환율·추천 전체의 정밀도가 아니다.',
        '- 고정 Qwen의 512/1,024 벡터는 동일하다. 원문 최대 길이가 소개글 220토큰·메타데이터 446토큰이어서 길이를 늘려도 추가 정보가 들어가지 않았다.',
        '- 시간 노출 진단에서 기존 무작위 검증 입력의 57.60%가 정답보다 나중에 작성됐다. 이것은 노출 규모이며 성능 상승의 인과 효과가 아니다. 현재 결과를 미래 추천 성능으로 해석하지 않는다.','',
        '[전체 지표 CSV](tables/model_improvement_v4.csv) · [전체 JSON](model_improvement_v4.json) · [진행 절차](../docs/backlog_experiments_v4.md) · [시간 노출 진단](temporal_exposure_v4.json)']
    diagnostic=EDA/'reports/candidate_diagnostics_v4.json'
    if diagnostic.exists():
        d=json.loads(diagnostic.read_text('utf-8'))
        lines += ['', '## 후보 단계에서 확인한 것', '',
            '| 후보 구성 | 정답 포함률 | 게임별 평균 포함률 | 희귀 게임 포함률 | 평균 후보 수 |',
            '|---|---:|---:|---:|---:|']
        for name,values in d['ceilings'].items():
            p=values['primary'];tail=values['primary_support1-999']
            count=p.get('actual_candidates',{}).get('mean')
            lines.append(f'| `{name}` | {p["recall"]*100:.2f}% | {p["game_macro_recall"]*100:.2f}% | {tail["recall"]*100:.2f}% | {count:.2f} |'
                if count is not None else
                f'| `{name}` | {p["recall"]*100:.2f}% | {p["game_macro_recall"]*100:.2f}% | {tail["recall"]*100:.2f}% | 상위 100개 이하 |')
        o=d['hit_overlap']['primary']
        lines += ['',f'Top 10 적중은 두 모델 모두 {o["both"]:,}명, KNN만 {o["cf_only"]:,}명, 콘텐츠만 {o["content_only"]:,}명, 모두 실패 {o["neither"]:,}명이다. '
            '이 교집합은 진단에만 썼다. 정답을 추론 특징으로 넣지 않았다.', '',
            f'두 Top 10 목록을 모두 살펴 정답이 하나라도 있는 비율은 {(o["both"]+o["cf_only"]+o["content_only"])/o["users"]*100:.2f}%다. '
            '두 목록의 합집합을 확인한 진단값이며, 실제 추천 10개의 성능을 뜻하지 않는다.', '',
            '잔차 비중 0인 세 후보 정책은 모든 선택 사용자에서 저장된 사용자·추천 게임·순위가 KNN 대조군과 완전히 같았다. '
            '구간별 최대 200개 후보는 회수 범위를 넓혔지만, 그 사실만으로 최종 Top 10 개선이 성립하지 않는다.']
    lines += ['', '## 짧은 이력 사용자', '',
        '| 모델 | 남은 전체 입력 | 사용자 | Recall@10 | NDCG@10 | 빈 추천 |',
        '|---|---|---:|---:|---:|---:|']
    control_names=['activity_inverse_degree_cosine_k100','binary_item_knn_k100','metadata_extended_tfidf']
    for name in control_names:
        m=next(x for x in manifests if x['variant']==name)
        per=pq.read_table(folder/name/'per_user.parquet',columns=['n_total_inputs','recommendation_count']).to_pydict()
        total=np.asarray(per['n_total_inputs']);length=np.asarray(per['recommendation_count'])
        for label in ['1-2','3-4','5+']:
            v=m['metrics']['by_total_history'][label]
            mask=total<=2 if label=='1-2' else ((total>=3)&(total<=4) if label=='3-4' else total>=5)
            lines.append(f'| `{name}` | {label} | {v["users"]:,} | {v["recall"]*100:.2f}% | {v["ndcg"]:.6f} | {int((length[mask]==0).sum()):,} |')
    lines += ['', '## 정확도 외 추가 지표', '',
        '| 모델 | 추천 게임 수/6,630 | 노출 지니 | 새로움 | 태그 다양성 | 태그 평가 사용자 |',
        '|---|---:|---:|---:|---:|---:|']
    displayed=control_names + ([eligible[0]['variant']] if eligible else [])
    for name in displayed:
        path=folder/name/'reporting_metrics.json'
        if not path.exists():continue
        audit=json.loads(path.read_text('utf-8'));e=audit['exposure'];div=audit['diversity']
        diversity='없음' if div['tag_jaccard_distance'] is None else f'{div["tag_jaccard_distance"]:.4f}'
        lines.append(f'| `{name}` | {e["recommended_games"]:,} | {e["gini"]:.4f} | {e["novelty"]:.4f} | {diversity} | {div["users_all_recommended_tags_available"]:,} |')
    lines += ['', '노출 지니는 추천되지 않은 후보도 포함한다. 새로움은 smoothing한 학습 사용자 비중의 음의 로그이며, 태그 다양성은 목록 내 게임 쌍의 Jaccard 거리다. '
        '태그가 빠진 목록은 태그 다양성에서 제외하고 별도 분모를 기록했다. 모델마다 평가 가능한 목록이 다르므로 이 값만으로 다양성의 우열을 단정하지 않는다. 인기도 역가중 결과와 유효 표본 수는 각 평가의 `reporting_metrics.json`에 보존했으며, 관측 확률을 보정한 IPS로 해석하지 않는다.', '',
        '[후보 진단 상세](candidate_diagnostics_v4.json)']
    diagnostic=EDA/'reports/alignment_diagnostic_v4.json'
    if diagnostic.exists():
        d=json.loads(diagnostic.read_text('utf-8'))
        lines += ['', '## 콘텐츠 관계 학습의 실패 진단', '',
            f'콘텐츠 정렬 계열에서 NDCG가 가장 높은 설정의 학습 그래프 {d["sampled_edges"]:,}쌍을 점검했다. '
            f'교사 유사도의 중앙값은 {d["teacher_similarity"]["quantiles"]["median"]:.6f}이며, '
            f'{d["teacher_positive_below_weak_upper_margin"]*100:.2f}%가 미관측 쌍에 적용한 상한 {d["weak_upper_margin"]}보다 작다. '
            '따라서 이 상한만으로는 알려진 관계가 미관측 쌍보다 높은 점수를 갖게 만들지 못한다.', '',
            f'학습에 사용한 관계와 미관측 쌍을 점수로 구별하는 AUC도 {d["edge_vs_unlabelled_auc"]:.4f}로 거의 구분되지 않았다. '
            '이 값은 학습 그래프 진단이며 예약 사용자의 추천 성능이나 독립 검증 점수가 아니다. '
            '작은 회귀 오차만으로 정렬 학습이 성공했다고 볼 수 없으며, 콘텐츠 정렬 전체의 가능성을 기각한 결과로도 해석하지 않는다. '
            '관계 사이의 상대 순위를 학습하는 손실과 학습 그래프 안의 별도 검증을 후속 실험으로 남긴다.', '',
            '[진단 수치·출처](alignment_diagnostic_v4.json)']
    temporal=EDA/'reports/temporal_controls_v4.json'
    if temporal.exists():
        d=json.loads(temporal.read_text('utf-8'))
        lines += ['', '## 별도 시간순 대조 평가', '',
            f'주평가 {d["primary_users"]:,}명에서 2022년 말 이전에 작성·수정된 리뷰로 다시 학습한 대조군을 평가했다. '
            '현재 메타데이터를 사용한 상호작용 시간 통제 평가이며 무작위 숨김과 개선량을 직접 비교하지 않는다. '
            '확인용 시간 구간은 열지 않았다. [결과와 추천 목록 검증](temporal_controls_v4.md)']
    fslim=list((ROOT/'models').glob('fslim_m*/manifest.json'))
    if fslim:
        fit=[json.loads(p.read_text('utf-8'))['fit_summary'] for p in fslim]
        lines += ['', '## fSLIM 수치 검증', '',
            '현재 학습의 공동 리뷰 행렬을 충분통계로 재사용했다. 같은 이웃 200/500개의 코사인 대조군과 비음수 L1+L2 회귀를 비교했다. '
            '컴파일된 좌표 하강법의 작은 행렬 해를 독립 최적화와 대조했으며, 생산 학습의 정규화 KKT 잔차도 모두 사전 허용치 0.001보다 작았다. '
            f'최대 잔차는 {max(x["max_normalized_kkt"] for x in fit):.6f}였다. '
            'L1=0인 네 설정에서는 일부 게임 열이 1,000회 반복 상한에 도달해 dual-gap 정지 조건은 충족하지 못했다. '
            '근사 최적성 검사를 통과한 계수의 결과이며, 모든 열이 정지 조건을 충족했다고 보고하지 않는다. '
            '게임별 0 계수율·미수렴 열 수는 각 모델 manifest에 기록했다.']
    supplemental=EDA/'reports/supplemental_evaluation_v4.json'
    if supplemental.exists():
        d=json.loads(supplemental.read_text('utf-8'));name=d['primary_selection_finalist']
        lines += ['', '## 정답 시드 반복과 숨긴 부정 게임', '',
            '주 모델을 고정한 뒤 확인 평가 전에 수행한 보고용 결과다. 첫 시드는 기존 20,000명의 정답·적중 순위·추천 목록을 정확히 재현했다. '
            '세 시드의 결과를 사용자마다 평균한 뒤 신뢰구간을 계산했으며, 같은 사용자를 세 명으로 세지 않았다.']
        if name:
            avg=d['target_seed_variation']['user_outcomes_averaged_across_seeds']
            negative=d['hidden_negative']['outcomes'];base_name=base['variant']
            lines += ['',f'주평가의 시드 평균 Recall은 활동량 보정 KNN {avg[base_name]["primary_5_plus"]["mean_user_recall_at_10"]*100:.2f}%, '
                f'고정 DAN {avg[name]["primary_5_plus"]["mean_user_recall_at_10"]*100:.2f}%다. '
                f'보이는 부정 Linux 게임이 있는 2,395명에게 게임 하나를 추가로 숨긴 포함률은 각각 '
                f'{negative[base_name]["hidden_negative_top10_rate"]*100:.2f}%, {negative[name]["hidden_negative_top10_rate"]*100:.2f}%다. '
                '이 보조 과제에서는 DAN이 싫어할 게임을 피하는 능력을 개선하지 못했다. 실제 추천에서는 이미 보이는 부정 게임을 계속 제외한다. '
                '아직 리뷰하지 않은 모든 게임의 거짓 양성률로 해석하지 않는다.']
        lines += ['', '[보조 평가 전체 결과](supplemental_evaluation_v4.md)']
    if confirmation and confirmation['opened']:
        a=confirmation['outcomes'][base['variant']]['metrics']
        b=confirmation['outcomes'][confirmation['finalist']]['metrics']
        tail_delta=(b['tail']['ndcg']/a['tail']['ndcg']-1)*100
        decision='고정 DAN을 채택한다.' if confirmation['adopted']==confirmation['finalist'] else '기존 활동량 보정 KNN을 유지한다.'
        lines[2]=f'**상태:** 첫 단계 {plan["required_count"]}개 조건의 비교와 고정한 한 후보의 확인 평가를 완료했다.'
        lines[4:4]=['**이번 결정:** '+decision+' 확인용 주평가 10,000명에서 DAN의 Recall@10은 '
            f'{b["recall"]*100:.2f}%, 대조군은 {a["recall"]*100:.2f}%였다. '
            f'전체 NDCG와 게임별 평균은 높아졌지만, 희귀 게임 NDCG는 이 표본에서 {abs(tail_delta):.1f}% '+
            ('낮았다.' if tail_delta<0 else '높았다.'), '',
            f'확인용 희귀 구간은 {b["tail"]["users"]:,}명·{b["tail"]["target_games"]:,}개 게임이다. '
            f'적중 사용자 수는 대조군 {round(a["tail"]["recall"]*a["tail"]["users"])}명, '
            f'DAN {round(b["tail"]["recall"]*b["tail"]["users"])}명이다. '
            '사전에 정한 점추정 비감소 규칙으로 교체 여부를 판정했다. 이 작은 표본의 차이를 실제 서비스 차이로 확정하지 않는다. '
            '다른 후보로 확인 집단을 다시 시험하지 않았다.', '']
    (EDA/'reports/model_improvement_v4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args();run(args.freeze)
