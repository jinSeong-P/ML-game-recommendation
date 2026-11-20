"""Replay temporal recommendation lists and publish a separate Korean report."""
from __future__ import annotations
import json
import numpy as np
import pyarrow.parquet as pq

from eda_large_review_frank import EDA,digest
from improvement_common import write_json
from experiment_v4_common import rank_values
import run_temporal_models_v4 as temporal


def audit(folder,cohort,catalog,support):
    manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
    for name,sha in manifest['files'].items():
        if digest(folder/name)!=sha:raise ValueError('Temporal evaluation file changed')
    p=pq.read_table(folder/'per_user.parquet').to_pydict()
    r=pq.read_table(folder/'recommendations.parquet').to_pydict()
    user=np.asarray(p['user_key'],np.uint64);apps=np.asarray(p['target_app_id'])
    if not np.array_equal(user,cohort['user_ids']) or not np.array_equal(apps,cohort['target_ids']):
        raise ValueError('Temporal target/user order changed')
    if not np.array_equal(p['n_total_inputs'],cohort['n_total']) or not np.array_equal(p['n_positive_inputs'],cohort['n_positive']):
        raise ValueError('Temporal per-user history counts changed')
    ids=catalog['candidate_ids'];ru=np.asarray(r['user_key'],np.uint64)
    ri=np.searchsorted(user,ru);ci=np.searchsorted(ids,r['app_id'])
    if np.any(ri>=len(user)) or not np.array_equal(user[ri],ru):raise ValueError('Unknown recommendation user')
    if np.any(ci>=len(ids)) or not np.array_equal(ids[ci],r['app_id']):raise ValueError('Unknown candidate')
    scores=np.asarray(r['score']);ranks=np.asarray(r['rank'])
    if np.any(~np.isfinite(scores)) or np.any(scores<=0):raise ValueError('Nonpositive/nonfinite temporal score')
    if np.any(~catalog['available'][ci]):raise ValueError('Unavailable-at-T0 candidate was recommended')
    if len(ri) and np.asarray(cohort['exclusions'][ri,ci]).any():raise ValueError('Visible game recommended')
    counts=np.bincount(ri,minlength=len(user));target_rank=np.zeros(len(user),np.int8)
    for row in range(len(user)):
        ix=np.flatnonzero(ri==row);a=ids[ci[ix]];s=scores[ix];rank=ranks[ix]
        if len(a)>10 or len(a)!=len(np.unique(a)) or not np.array_equal(rank,np.arange(1,len(a)+1)):
            raise ValueError('Temporal list duplicates, rank gaps, or excessive length')
        if not np.array_equal(np.lexsort((a,-s)),np.arange(len(a))):raise ValueError('Temporal score/tie order changed')
        found=np.flatnonzero(a==apps[row])
        if len(found):target_rank[row]=rank[found[0]]
    if not np.array_equal(counts,p['recommendation_count']) or not np.array_equal(target_rank,p['target_rank']):
        raise ValueError('Temporal saved target ranks/list counts cannot be replayed')
    target_support=support[np.searchsorted(catalog['apps'],apps)]
    if not np.array_equal(target_support,p['target_train_support']):raise ValueError('Temporal support mapping changed')
    main=temporal.primary_mask(cohort);hit,ndcg=rank_values(target_rank)
    metric=manifest['metrics']['primary']
    if not np.isclose(hit[main].mean(),metric['recall'],atol=1e-15,rtol=0) or not np.isclose(ndcg[main].mean(),metric['ndcg'],atol=1e-15,rtol=0):
        raise ValueError('Temporal primary metrics cannot be reproduced')
    exposure=np.bincount(ci[main[ri]],minlength=len(ids))
    total=exposure.sum();ordered=np.sort(exposure)
    gini=float((2*np.arange(1,len(ids)+1)-len(ids)-1)@ordered/(len(ids)*total)) if total else 0.
    result=dict(status='passed_actual_recommendation_replay',users=len(user),recommendation_rows=len(ri),
        primary_users=int(main.sum()),primary_recommended_games=int((exposure>0).sum()),
        primary_exposure_gini=gini,primary_empty_users=int((counts[main]==0).sum()),
        zero_positive_users=int((cohort['n_positive']==0).sum()),
        zero_positive_nonempty_users=int(((cohort['n_positive']==0)&(counts>0)).sum()),
        evaluation_manifest_sha256=digest(folder/'manifest.json'),audit_code_sha256=digest(__file__))
    if result['zero_positive_nonempty_users']:raise ValueError('Nonpersonalized scores for zero-positive history')
    write_json(folder/'recommendation_audit.json',result)
    return manifest,result


def run():
    cat=temporal.load_temporal_catalog();cohort=temporal.load_selection_cohort(catalog=cat)
    sup=np.load(temporal.TRAINING/'support.npy');outcomes={}
    for name in temporal.VARIANTS:
        m,a=audit(temporal.EVALUATIONS/'selection'/name,cohort,cat,sup)
        outcomes[name]=dict(metrics=m['metrics'],audit=a)
    paired_path=temporal.EVALUATIONS/'selection/paired_comparisons.json'
    paired=json.loads(paired_path.read_text('utf-8'))
    for name,sha in paired['evaluation_receipts'].items():
        if digest(temporal.EVALUATIONS/'selection'/name/'manifest.json')!=sha:
            raise ValueError('Temporal paired comparison sources changed')
    split=json.loads((temporal.TEMPORAL/'manifest.json').read_text('utf-8'))
    result=dict(status='completed_temporal_selection_controls',confirmation_opened=False,
        primary_users=int(temporal.primary_mask(cohort).sum()),all_target_eligible_users=len(cohort['user_ids']),
        split_manifest_sha256=digest(temporal.TEMPORAL/'manifest.json'),protocol_sha256=digest(temporal.REFERENCE_PROTOCOL),
        training=split['training'],availability=split['availability'],outcomes=outcomes,
        paired=paired,interpretation='Interaction-time-controlled supplementary evaluation using current metadata; not a complete historical deployment simulation or pristine final test.')
    write_json(EDA/'reports/temporal_controls_v4.json',result)
    lines=['# v4 상호작용 시간 통제 보조 평가','',
        '학습과 입력에 2022-12-31 이전에 작성·수정된 리뷰만 사용했다. 선택용 정답은 2023년 상반기 자료에서 골랐고, 확인용 기간과 계정은 열지 않았다. 새 학습은 긍정 66,795,430쌍·26,284,538명이며 예약 계정의 학습 중복은 0이다.','',
        '**주평가 820명**, 전체 정답 적격 2,023명이다. 입력 없는 342명과 긍정 입력 없는 354명도 별도 집단으로 보존했다. 무작위 숨김과 사용자·정답·입력 길이가 달라 점수 차이를 시간 통제의 인과 효과나 모델 개선량으로 해석하지 않는다.','',
        '| 모델 | Recall@10 | NDCG@10 | 게임별 평균 NDCG | 빈 추천 | 추천 게임 수/6,630 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,o in outcomes.items():
        m=o['metrics'];a=o['audit'];v=m['primary']
        lines.append(f'| `{name}` | {v["recall"]*100:.3f}% | {v["ndcg"]:.6f} | {m["primary_game_macro_ndcg"]:.6f} | {m["primary_empty_users"]} | {a["primary_recommended_games"]:,} |')
    compare=paired['comparisons']['binary_item_knn_k100_vs_activity_inverse_degree_cosine_k100']['NDCG@10']
    lines += ['',f'활동량 보정 KNN과 기본 KNN의 같은 사용자 NDCG 차이는 {compare["delta"]:.6f}, '
        f'짝지은 95% 신뢰구간은 [{compare["ci95"][0]:.6f}, {compare["ci95"][1]:.6f}]다. 0을 포함하므로 이 표본에서 확실한 우위라고 단정하지 않는다.','',
        '## 입력 길이별 결과','',
        '| 모델 | 남은 전체 입력 | 사용자 | Recall@10 | NDCG@10 |','|---|---|---:|---:|---:|']
    for name,o in outcomes.items():
        for label,v in o['metrics']['by_total_history'].items():
            lines.append(f'| `{name}` | {label} | {v["users"]:,} | {v["recall"]*100:.2f}% | {v["ndcg"]:.6f} |')
    lines += ['', '## 검증과 한계','',
        '- 저장된 추천을 읽어 정답 순위·목록 길이·점수·정렬·중복·관측 게임 제외·출시 가능 여부를 다시 검사했다. 주평가 지표를 float64로 재계산해 일치를 확인했다.',
        '- 후보 배열은 6,630개를 유지했다. T0 이전 출시가 확인된 6,007개만 점수를 허용했으며, 출시일을 모르는 534개와 나중 출시된 89개는 추천에서 제외했다. 정답도 이 출시 조건을 만족하는 자료로 제한되어 결과의 적용 범위가 좁다.',
        '- 현재 메타데이터를 사용했다. 2022년 당시 소개글·태그·Linux 지원 여부를 복원하지 못했으므로 완전한 과거 시점 평가라고 부르지 않는다. 리뷰 작성 순서는 구매·플레이 순서와 다르다.',
        '- 계정은 v4 재학습에서 분리했지만 과거 버전 학습에는 쓰였다. 프로젝트 전체에서 미사용된 최종 테스트가 아니다. 짝지은 구간은 이 고정 예측에 대한 것이며 재학습·설정 선택의 변동을 포함하지 않는다.','',
        '[상세 지표와 검증](temporal_controls_v4.json) · [분할 규칙](../data/reference/temporal_protocol_v4.json) · [주 실험](model_improvement_v4.md)']
    (EDA/'reports/temporal_controls_v4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':run()
