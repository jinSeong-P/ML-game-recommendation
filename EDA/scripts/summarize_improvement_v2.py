"""Publish measured v2 aggregates, guardrail decisions and reproducible commands."""
import json,re
from datetime import datetime
from pathlib import Path
from eda_large_review_frank import EDA,digest
from improvement_common import ROOT,PROTOCOL,write_json
from run_improvement_v2 import load_verified_selection

def publish():
    phases={}
    for key,name in [('reproduction','baseline_review_reproduction_v2.json'),('K0','item_knn_failure_analysis_v2.json'),('T0','tfidf_ranking_analysis_v2.json')]:
        p=EDA/'reports'/name
        phases[key]={'file':'reports/'+name,'sha256':digest(p),'result':json.loads(p.read_text('utf-8'))} if p.exists() else {'status':'missing'}
    evaluations={p.parent.name:json.loads(p.read_text('utf-8')) for p in (ROOT/'evaluations/validation').glob('*/manifest.json')}
    reference={p.parent.name:json.loads(p.read_text('utf-8')) for p in (ROOT/'evaluations/test').glob('*/manifest.json')}
    select=ROOT/'selection_frozen.json'
    selection=load_verified_selection() if select.exists() else None
    for name,m in reference.items():
        item=selection['models'][m['kind']]
        if (name!=item['variant'] or m['model']!=item['model']
            or m['protocol_sha256']!=digest(PROTOCOL)
            or m['evaluator_v2_sha256']!=digest(EDA/'scripts/improvement_common.py')):
            raise ValueError('Reference result differs from frozen selection')
        for relative,info in m['files'].items():
            if digest(ROOT/'evaluations/test'/name/relative)!=info['sha256']:
                raise ValueError('Reference output changed')
    testlog=ROOT/'tests_final.log'
    test_result={'status':'not_run'}
    if testlog.exists():
        txt=testlog.read_text('utf-8-sig',errors='replace')
        match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',txt)
        test_result={'log_sha256':digest(testlog),'passed':bool(match and re.search(r'^OK\s*$',txt,re.M)),
                     'tests':int(match[1]) if match else None,'seconds':float(match[2]) if match else None}
    artifact_audit=EDA/'reports/improvement_v2_artifact_audit.json'
    artifact_result=json.loads(artifact_audit.read_text('utf-8')) if artifact_audit.exists() else None
    reference_complete=bool(selection and set(reference)=={m['variant'] for m in selection['models'].values()})
    costs={}
    for name,m in evaluations.items():
        spec=m['model']; sources=[spec['text'],spec['metadata']] if 'text' in spec else [spec]
        entries=[]
        for source in sources:
            manifest=json.loads((EDA/source['directory']/'manifest.json').read_text('utf-8'))
            entries.append({'directory':source['directory'],'index':manifest.get('index'),
                'listed_file_bytes':sum(info['bytes'] for info in manifest['files'].values()),
                'matrix_bytes':sum(info['bytes'] for file,info in manifest['files'].items() if file.endswith('.npz'))})
        costs[name]={'evaluation_seconds':m['elapsed_seconds'],'channels':entries}
    build=json.loads((ROOT/'knn/manifest.json').read_text('utf-8'))
    build_cost={'knn_models_seconds':(datetime.fromisoformat(build['completed_utc'])-datetime.fromisoformat(build['started_utc'])).total_seconds(),
                'tfidf_build_and_T0_seconds':phases['T0']['result']['elapsed_seconds']}
    report={'version':'model-improvement-v2','status':'complete' if selection and len(evaluations)==18 and reference_complete and all('result' in x for x in phases.values()) and test_result.get('passed') and artifact_result and artifact_result.get('passed') else 'in_progress',
      'protocol_sha256':digest(PROTOCOL),'protocol':json.loads(PROTOCOL.read_text('utf-8')),
      'phases':phases,'validation':evaluations,'selection':selection,'reference_test':reference,'automated_tests':test_result,'artifact_audit':artifact_result,
      'costs':costs,'build_cost':build_cost,
      'limitations':['Same validation used to select among predeclared variants; bootstrap is not independent post-selection confirmation.',
        'frank was previously observed; any reuse is reference only, never a new independent final test.',
        'No unseen-train target in validation; cold-start conclusions are limited.',
        'Game macro averages only target games observed in validation; low-support strata have small samples.']}
    write_json(EDA/'reports/model_improvement_v2.json',report)
    def pct(v):return '—' if v is None else f'{v*100:.4f}%'
    def num(v):return '—' if v is None else f'{v:.6f}'
    guard_labels={'ndcg_ci_lower_positive':'NDCG 개선의 95% 신뢰구간 하한이 0 이하',
       'whole_recall_noninferior':'전체 Recall@10 하락',
       'game_macro_ndcg_noninferior':'정답 게임별 평균 NDCG 하락',
       'tail_ndcg_noninferior':'학습 긍정 사용자 1–999명 게임의 NDCG 하락',
       'empty_users_not_increased':'빈 추천 사용자 증가'}
    lines=['# 모델 개선 실행 결과 — v2','',f"2026-11-08 · 상태: **{report['status']}**. 사용자 승인에 따라 수치 재현 → 평가 기준 보완 → K0/T0 진단 → 사전 지정 개선 실험을 수행했다. v1 결과와 산출물은 보존한다.",'',
      '[실행 절차·시각화](../docs/model_improvement_execution.md) · [고정 프로토콜](../data/reference/improvement_protocol_v2.json) · [전체 측정 JSON](model_improvement_v2.json)','',
      ]
    if selection:
        knn=selection['models']['item_knn']; text=selection['models']['tfidf']
        content_spec=text['model']
        content_setting=(f"bigram 소개글 {content_spec['alpha']*100:g}% + 별도 태그·장르 {(1-content_spec['alpha'])*100:g}%" if 'text' in content_spec
                         else text['variant'])
        lines+=['## 결정 요약','',
          f"**Item-KNN은 v1 설정을 유지**한다. 전체 NDCG가 가장 높았던 `{knn['best_primary_variant']}`도 게임별 평균 NDCG와 학습 긍정 사용자 수가 적은 게임의 NDCG 기준을 통과하지 못했다.",
          f"**콘텐츠 모델은 {content_setting}** (`{text['variant']}`)를 선정했다. 먼저 소개글 단독 모델을 고정하고 태그·장르의 비중을 비교했다. frank 결과를 확인하기 전에 최종 선정과 근거를 기록했다.",'',
          '| 모델 | v1 검증 Recall@10 | 선정 모델 검증 Recall@10 | v1 검증 NDCG@10 | 선정 모델 검증 NDCG@10 |',
          '|---|---:|---:|---:|---:|']
        for kind,control in [('item_knn','knn_v1_control'),('tfidf','text_v1_control')]:
            b=evaluations[control]['protection']; c=evaluations[selection['models'][kind]['variant']]['protection']
            lines.append(f"| {kind} | {pct(b['Recall@10'])} | {pct(c['Recall@10'])} | {num(b['NDCG@10'])} | {num(c['NDCG@10'])} |")
        lines+=['','Recall@10은 숨긴 긍정 게임을 추천 10개 안에서 찾은 사용자 비율이다. 검증 전체 평균의 향상이 frank 향상을 보장하지는 않는다.','']
    lines+=['## 내부 검증 결과','',
      '동일한 325,639명·입력·숨긴 정답·마스크·Linux 후보에서 비교했다. 양수 점수 최대 10개, 빈자리 유지·전체 사용자 분모를 적용했다.','',
      '| 모델 | Recall@10 | NDCG@10 | 게임별 평균 NDCG | 지원 1–999 NDCG | 빈 목록 | 채택 조건 |',
      '|---|---:|---:|---:|---:|---:|---|']
    for name,m in sorted(evaluations.items()):
        p=m['protection'];ok=p.get('eligible_for_automatic_adoption',False)
        lines.append(f"| {name} | {pct(p['Recall@10'])} | {num(p['NDCG@10'])} | {num(p['game_macro_NDCG@10'])} | {num(p['tail']['NDCG@10'])} | {p['empty_users']:,} | {'통과' if ok else '기준점' if name.endswith('v1_control') else '미통과'} |")
    lines+=['','## 사전 지정 채택 기준과 후보 비교','']
    if selection:
        for kind,m in selection['models'].items():
            lines.append(f"- **{kind}: {m['variant']}**. {'v1 유지' if m['retained_v1'] else '개선 후보 선정'}. 전체 NDCG 최고 후보는 `{m['best_primary_variant']}`다.")
        lines+=['','선택에는 다음 조건을 모두 요구했다. NDCG 차이의 bootstrap 95% 신뢰구간 하한이 0보다 커야 하고, 전체 Recall@10·정답 게임별 평균 NDCG·학습 긍정 사용자 수가 1–999명인 정답 게임의 NDCG가 기준 모델보다 낮아지면 안 된다. 빈 추천 사용자 수도 늘어나면 안 된다. 아래는 v1 대비 변화와 통과하지 못한 기준이다.','',
          '| 후보 | NDCG 변화 | 95% 구간 | 통과하지 못한 조건 | 평가 초 |','|---|---:|---|---|---:|']
        for name,m in sorted(evaluations.items()):
            if name.endswith('v1_control'):continue
            p=m['protection'];ci=p['paired_bootstrap']['NDCG@10'];failed=[guard_labels[k] for k,v in p['guard_checks'].items() if not v]
            lines.append(f"| {name} | {ci['delta']:+.6f} | [{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}] | {', '.join(failed) or '없음'} | {m['elapsed_seconds']:.1f} |")
        selected_text=json.loads((ROOT/'text_selection_before_metadata.json').read_text('utf-8'))
        lines+=['','### 소개글 고정 후 태그·장르 비교','',
           f"소개글 단독 `{selected_text['selected_text_variant']}`을 먼저 고정했다. 점수는 `α × 소개글 코사인 + (1−α) × 태그·장르 코사인`이다. 채널별 긍정 입력 프로필을 각각 정규화하며 결측 채널은 0으로 둔다. α=1 대조군의 사용자별 결과는 소개글 단독과 완전히 일치한다.",'',
           '| 소개글 비중 α | Recall@10 | NDCG@10 | 소개글 단독 대비 NDCG 변화 |',
           '|---:|---:|---:|---:|']
        text_ndcg=evaluations[selected_text['selected_text_variant']]['protection']['NDCG@10']
        for name,m in sorted(evaluations.items()):
            if name.startswith('metadata_'):
                p=m['protection']
                lines.append(f"| {m['model']['alpha']:.2f} | {pct(p['Recall@10'])} | {num(p['NDCG@10'])} | {p['NDCG@10']-text_ndcg:+.6f} |")
        lines+=['','### 선정 모델의 게임별 학습 긍정 사용자 수에 따른 성능','',
           '게임별 학습 긍정 사용자 수는 train에서 해당 게임에 긍정 리뷰를 남긴 서로 다른 사람 수다. 검증 정답 게임 3,805개 가운데 긍정 학습 사용자가 1–999명인 게임은 사용자 10,466명·게임 2,715개에 해당한다. 1–9명 구간은 49명뿐이므로 작은 표본의 차이를 일반화하지 않는다.','',
           '| 게임별 학습 긍정 사용자 수 | 사용자 | 정답 게임 | 인기도 Recall | Item-KNN 선정 Recall | 콘텐츠 선정 Recall |',
           '|---|---:|---:|---:|---:|---:|']
        popularity=phases['reproduction']['result']['cohorts']['validation']['popularity']['by_support_log']
        kp=evaluations[knn['variant']]['protection']['by_support_log']
        tp=evaluations[text['variant']]['protection']['by_support_log']
        for label,p in kp.items():
            lines.append(f"| {label} | {p['users']:,} | {p['target_games']:,} | {pct(popularity[label]['Recall@10'])} | {pct(p['Recall@10'])} | {pct(tp[label]['Recall@10'])} |")
    lines+=['','## 진단 근거','',
       '- [검토 수치 재현](baseline_review_reproduction_v2.json): v1 해시값·저장 성능·인기도 기준선·연결 공동 리뷰 수·텍스트 결측·추천 쏠림 정의.',
       '- [K0 원인 분해](item_knn_failure_analysis_v2.json): 원래 train 연결과 v1 이웃 상위 100개 제한·순위 실패를 구분해, 전체 평가 사용자를 분류한 결과.',
       '- [T0 단어 기여](tfidf_ranking_analysis_v2.json): 사례 추출 규칙·정답/오추천 특징 기여·조건부 T3 판단.',
       '- [독립 SQL 모델 감사](improvement_v2_artifact_audit.json): 원시 연결에서 SQL로 감쇠·순위를 다시 계산하고 모델과 대조.', '']
    if 'result' in phases['K0']:
        k0=phases['K0']['result']
        labels={'target_support0':'정답 게임의 긍정 학습 사용자 0명','all_positive_inputs_support0':'학습 긍정 근거가 있는 입력 없음','raw_connection0':'원시 공동 긍정 연결 없음','pruned':'저장 K=100 이웃에서 제외','rank_failure':'양수 점수지만 10위 밖','hit':'적중'}
        lines+=['### K0 — 추천 실패 원인의 전체 사용자 분류','','| 범주 | 검증 사용자 | 기존 frank 평가 사용자 |','|---|---:|---:|']
        for key,label in labels.items():lines.append(f"| {label} | {k0['cohorts']['validation']['categories'][key]['users']:,} | {k0['cohorts']['test']['categories'][key]['users']:,} |")
        lines+=['','입력 게임에 학습 긍정 사용자가 있는 경우와 저장된 K=100 이웃이 그 입력을 포함하는 경우는 다르다. 검증에서 정답 점수가 0인 사용자 가운데 이웃 수 제한으로 연결이 끊긴 경우는 55,487/55,671명이고, frank는 13,448/17,819명이다. 원래 학습 연결이 가능하다는 사실만으로 Top10에 정답을 맞힌 것은 아니다.','']
    if 'result' in phases['T0']:
        t0=phases['T0']['result']['diagnostics'];gate=t0['T3_conditional_gate']
        lines+=['### T0 — 특징 기여 진단','',
           f"고정 seed·층화 규칙으로 {t0['selected_examples']['users']:,}개 사용자 사례와 {t0['selected_examples']['term_contribution_rows']:,}개 단어 기여를 저장했다. 적중·정답 점수 0·양수 점수지만 순위권 밖인 사례와 학습 긍정 사용자 수·입력 수 구간을 포함한다.",
           f"오추천 Top1 점수 중 문서 빈도 50% 초과 단어의 표본 합산 기여는 {gate['sampled_wrong_top1_high_df_contribution_share']*100:.2f}%로, 사전 운영 기준 5%를 넘어 T3 max_df=0.5를 비교했다. 이 비율은 원인 확정이나 전체 사용자를 대표하는 추정치가 아니다. 정답 점수에도 같은 단어가 기여하는지 함께 확인한다.",
           '메타데이터는 공개 고정 카탈로그의 별도 태그·장르 채널이다. 출처 파서·어휘 수·결측·소개글 모델 고정 후 α별 비교 근거는 전체 JSON과 개별 manifest에 남겼다.','']
    lines+=['## 비용·추천 범위','',
       f"KNN 모델 구축 묶음: {build_cost['knn_models_seconds']:.1f}초. TF-IDF 모델·T0 구축 묶음: {build_cost['tfidf_build_and_T0_seconds']:.1f}초. 각 단계의 실제 실행 환경 기준이며 단계별 작업 범위가 달라 직접 속도 비교용은 아니다.",'',
       '| 설정 | 평가 초 | 행렬 파일 MiB | 특징·연결 수(nnz) | 추천 게임 수/6,630 | 10개 미만 사용자 | 추천 게임의 긍정 학습 사용자 수 중앙값 |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in sorted(evaluations.items()):
        c=costs[name];p=m['protection'];channels=c['channels']
        matrix_mb=sum(x['matrix_bytes'] for x in channels)/1024**2
        nnz=' + '.join(str(x['index'].get('nnz','—')) for x in channels)
        lines.append(f"| {name} | {m['elapsed_seconds']:.1f} | {matrix_mb:.2f} | {nnz} | {p['recommended_games']:,} | {p['short_list_users']:,} | {p['recommendation_support_median']:,.0f} |")
    lines+=['','행렬 파일은 압축 NPZ 크기다. 매핑·어휘·원문·진단·추천 저장량과 학습 중 최대 메모리는 포함하지 않는다. 두 채널 설정의 크기는 두 행렬을 합산했다. α=1 대조군도 구현상 두 행렬을 모두 읽는다. 전체 파일 크기·특징 차원과 학습 긍정 사용자 수 구간별 인원·성능은 JSON에 기록했다.','']
    lines+=['## 재현 명령','', 'PowerShell, 저장소 루트에서 실행한다. 기존 산출물은 덮어쓰지 않는다.','', '```powershell',
       '& EDA/.venv/Scripts/python EDA/scripts/reproduce_improvement_review.py',
       '& EDA/.venv/Scripts/python EDA/scripts/improvement_knn.py --build-models --diagnose',
       '& EDA/.venv/Scripts/python EDA/scripts/improvement_tfidf.py',
       '& EDA/.venv/Scripts/python EDA/scripts/run_improvement_v2.py controls',
       '& EDA/.venv/Scripts/python EDA/scripts/run_improvement_v2.py knn',
       '& EDA/.venv/Scripts/python EDA/scripts/run_improvement_v2.py text',
       '& EDA/.venv/Scripts/python EDA/scripts/run_improvement_v2.py metadata',
       '& EDA/.venv/Scripts/python EDA/scripts/run_improvement_v2.py select',
       '& EDA/.venv/Scripts/python EDA/scripts/run_improvement_v2.py reference',
       '& EDA/.venv/Scripts/python EDA/scripts/verify_improvement_v2.py',
       "& EDA/.venv/Scripts/python -m unittest discover -s EDA/scripts -p 'test_*.py' -v *> EDA/data/processed/modeling/improvement_v2/tests_final.log",
       '& EDA/.venv/Scripts/python EDA/scripts/summarize_improvement_v2.py','```','',
       '이미 생성한 모델은 다시 덮어쓰지 않는다. 새 재현은 별도 출력 경로나 별도 저장소 사본에서 수행한다. 집계·평가 재사용은 코드·설정·파일 해시값이 같을 때만 허용한다.','',
       '## 검증과 한계','',
       f"자동 테스트: {test_result.get('tests','미실행')}개, 통과 여부 {test_result.get('passed',False)}. 통과 수는 실행 로그에서 읽었다.",
       '모델별 저장 추천·순위·점수·관측 게임 제외·지표 검사와 단일/배치 검색 대조 결과는 각 평가 manifest에 있다.',
       '생성 코드 해시값 갱신 중 실행된 임시 평가 6개는 Git 제외 superseded_evaluations/manifest_refresh_20261108 경로에 보존하고 최종 선택에서 제외했다. 최종 모델 기록을 고정한 뒤 T1/T2·가중치 4개를 재실행했으며, 추천·사용자별 결과 파일은 앞선 실행과 동일하다. 이 보고서는 최종 해시값과 일치한 평가만 읽는다.',
       'frank 결과는 이미 과거에 확인한 평가 결과라서 새 독립 테스트가 아니다. 이 결과로 후보를 고르지 않았다. 검증 후보 선택 편향, 희소 구간의 작은 표본, 학습에서 보지 못한 정답 게임의 한계는 남는다.','']
    if reference:
        lines+=['## 고정 모델의 기존 frank 평가 결과','','| 모델 | v1 Recall@10 | 선정 모델 Recall@10 | v1 NDCG@10 | 선정 모델 NDCG@10 |','|---|---:|---:|---:|---:|']
        for name,m in sorted(reference.items()):
            baseline=phases['reproduction']['result']['cohorts']['test'][m['kind']]
            lines.append(f"| {name} | {pct(baseline['Recall@10'])} | {pct(m['summary']['Recall@10'])} | {num(baseline['NDCG@10'])} | {num(m['summary']['NDCG@10'])} |")
        lines+=['','모델 설정과 파일 해시값을 고정한 뒤 계산한 결과지만, frank 데이터는 이미 관측한 적이 있으므로 새 독립 최종 확인으로 볼 수 없다.','']
    (EDA/'reports/model_improvement_v2.md').write_text('\n'.join(lines),encoding='utf-8')
    return report

if __name__=='__main__':print(publish()['status'])
