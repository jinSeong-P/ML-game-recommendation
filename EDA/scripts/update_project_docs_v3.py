"""Refresh active project documentation from the completed, frozen v3 report."""
import json
import re
from eda_large_review_frank import EDA
from experiment_v3_common import ROOT
from summarize_experiments_v3 import label


def insert_after_title(path,text):
    original=path.read_text('utf-8')
    marker='<!-- v3-current -->'
    if marker in original:raise ValueError('Review existing v3 documentation before regenerating')
    title,body=original.split('\n',1)
    path.write_text(title+'\n\n'+marker+'\n'+text+'\n<!-- /v3-current -->\n'+body,encoding='utf-8')


def run():
    report=json.loads((EDA/'reports/model_improvement_v3.json').read_text('utf-8'))
    selected=report['selection'];rows={(m['cohort'],m['variant']):m for m in report['evaluations']}
    descriptions=[]
    for family,control in [('collaborative','item_knn_control'),('content','content_control')]:
        chosen=selected[family]['variant'];a=rows['validation',control]['protection'];b=rows['validation',chosen]['protection'];t=rows['test',chosen]['protection']
        descriptions.append(f"- **{'협업' if family=='collaborative' else '콘텐츠'}:** {label(chosen)}. 검증 Recall@10 {a['Recall@10']:.2%} → **{b['Recall@10']:.2%}**, frank 참조 **{t['Recall@10']:.2%}**.")
    result='\n'.join(descriptions)
    insert_after_title(EDA.parent/'README.md',
        f"2026-11-08 백로그 개선 실험 v3를 완료했습니다. 같은 내부 검증에서 사전 지정 설정 **{report['primary_evaluations']}개**를 비교하고 전체 성능·게임별 평균·희귀 게임·빈 목록 보호 기준으로 선택했습니다.\n\n"+result+
        '\n\nRecall@10은 숨긴 긍정 게임을 추천 10개 안에서 찾은 사용자 비율입니다. frank는 이미 관측한 **참조 평가**이고 모델 선택에는 사용하지 않았습니다. [v3 결과·전체 실험표](EDA/reports/model_improvement_v3.md) · [현재 모델 실행](EDA/docs/baseline_models.md) · [백로그 상태](EDA/docs/modeling_backlog.md). 아래 v1/v2 수치는 초기 실험 이력입니다.')
    insert_after_title(EDA/'README.md',
        f"**현재 결과: [백로그 개선 실험 v3](reports/model_improvement_v3.md).** 같은 검증의 설정 {report['primary_evaluations']}개와 별도 보조 평가·시간 cutoff 재학습을 완료했습니다.\n\n"+result+
        '\n\n아래의 소개글 전용 TF-IDF·단순 코사인 Item-KNN 설명은 고정 **v1 베이스라인 이력**입니다. 현재 선택과 실행은 [모델 안내](docs/baseline_models.md), 실험별 완료·미실행 조건은 [백로그](docs/modeling_backlog.md)를 따릅니다. frank는 재사용한 참조 평가로 구분합니다.')
    insert_after_title(EDA/'docs/baseline_models.md',
        '**현재 선택 모델은 [v3 보고서](../reports/model_improvement_v3.md)와 [선택 잠금](../data/reference/improvement_selection_v3.json)을 따릅니다.** 이 문서 아래의 v1 설정·명령은 원래 베이스라인을 재현하기 위해 보존합니다.\n\n'+result+
        '\n\n현재 모델은 검증에서 선택한 뒤 고정했고, 같은 모델을 frank 참조 평가와 보조 평가에 재사용했습니다. 이미 리뷰한 긍정·부정 게임을 제외하며 양수 점수만 최대 10개 추천하고 빈자리는 유지합니다. 인기도 보완은 별도 정책 실험으로만 보고했습니다.\n\n```powershell\nEDA/.venv/Scripts/python EDA/scripts/recommend_selected_v3.py --family collaborative --positive-apps 10 --exclude-apps 10 240\nEDA/.venv/Scripts/python EDA/scripts/recommend_selected_v3.py --family content --positive-apps 10 --exclude-apps 10 240\n```\n\n활동량 가중 Item-KNN은 같은 사용자가 남기는 게임 쌍의 기여를 활동량으로 보정하고 가중 코사인을 계산합니다. 콘텐츠는 선택 manifest의 각 채널을 독립 정규화한 뒤 결합합니다. 정확한 설정명·식·출처는 모델 manifest와 v3 보고서에 기록합니다.')
    insert_after_title(EDA/'docs/backlog_experiments_v3.md',
        f"**실행 완료:** 기본 설정 77개와 새 계열 모델 RRF 6개, 합계 {report['primary_evaluations']}개를 내부 검증했다. 선택·frank 참조·조건별 보조 평가·시간 cutoff 재학습과 저장 추천 감사를 마쳤다. [실제 결과·한계](../reports/model_improvement_v3.md)와 [백로그 상태](modeling_backlog.md)를 함께 읽는다. 아래는 성능 계산 전에 정한 절차·용어다.")
    insert_after_title(EDA/'data/reference/README.md',
        '[improvement_protocol_v3.json](improvement_protocol_v3.json)은 v3 사전 실험값·평가·선택 조건을 담는다. [improvement_selection_v3.json](improvement_selection_v3.json)은 검증 선택과 코드·결과 지문을 새 frank 참조 전에 고정한 공개 기록이다. 기존 v1/v2 참조와 분할·카탈로그 데이터는 보존했다. [v3 결과](../../reports/model_improvement_v3.md) 참조.')
    table=EDA/'reports/tables/README.md'
    insert_after_title(table,'[model_improvement_v3.csv](model_improvement_v3.csv)는 v3의 전체 검증·참조·보조 평가 지표와 채택 조건 탈락 사유·실행 시간을 담습니다. `variant`는 재현용 설정 ID이고, 표준 방법명·정의·해석은 [v3 보고서](../model_improvement_v3.md)를 따릅니다. 사용자별 자료는 Git 제외 경로에 보관합니다. 아래 01~17은 EDA 이력입니다.')
    backlog=EDA/'docs/modeling_backlog.md'
    insert_after_title(backlog,
        f"**v3 실행 상태:** 사전 지정 기본 설정 {report['primary_evaluations']}개와 보조 평가, 2022년 말 cutoff 재학습을 완료했습니다. [결과](../reports/model_improvement_v3.md) · [전체 지표 CSV](../reports/tables/model_improvement_v3.csv) · [절차·용어](backlog_experiments_v3.md). 체크는 해당 실험·진단의 완료를 뜻하며 자동 채택이나 인과 원인 규명을 뜻하지 않습니다.\n\n"
        '| 구분 | 남은 범위 |\n|---|---|\n'
        '| 조건부 모델 | ELSA는 전체 EASE 실패 시 대안이므로 실행하지 않았다. SASRec는 시간 평가와 학습 규모의 가능성 진단을 완료했으나 이벤트 시퀀스·동점 규칙·별도 사전 학습 규칙이 필요한 후속 모델이며 성능은 미측정이다. `improvement_v3/sequence_feasibility_audit.json`에 수치·조건을 남겼다. |\n'
        '| 별도 토큰화 | 문자 n-gram과 다국어 문장 임베딩은 실험했다. 언어별 사전/형태소 분절은 실행하지 않았다. |\n'
        '| 미채택 특징 | 구매 여부·무료 수령 특징을 학습에 넣는 실험은 이번 고정 프로토콜에 포함하지 않았다. |\n'
        '| 데이터 정책·관리 | 모호 리뷰 격리 유지, 카탈로그 갱신은 별도 버전 조건이다. 커밋·푸시는 이번 실험 완료와 별개이며 수행하지 않았다. |\n'
        '| 해석 한계 | 같은 지원 구간의 출처 차이와 시리즈·에디션의 동일 상품 여부는 진단했지만 인과 원인/상품 동일성은 확정하지 않았다. |')
    replacements={
        '후속 — 최소 전체 입력 ≥3 평가':'전체 ≥3·긍정 ≥1 조건의 별도 집단을 고정해 두 대조군과 선택 모델을 평가했다. 검증 535,368명·frank 38,549명이며 기본 ≥5 평가와 직접 개선량을 비교하지 않는다.',
        '후속 — 입력 집계 방식 변경':'동일 입력·정답에서 플레이타임·최신성 가중과 최대 유사도를 실제 비교했다. frank 추가 열은 고정 커밋에서 가져와 기존 리뷰 ID·계정·게임·라벨 다중집합 차이 0건을 확인했다.',
        '후속 — 모델 개선 후 인기순 빈자리 보완':'대조군 두 모델의 빈자리를 train 인기도로만 채우는 별도 정책을 평가하고 보완 사용자·슬롯·적중을 저장했다. 개인화만 추천하는 기본 정책은 유지한다.',
        '후속 — 부정 게임 유사도 감점':'부정 평균 유사도 계수 0.1/0.3을 두 대조군에서 평가했다. 평가용 미관측 후보를 실제 부정 리뷰로 해석하지 않는다.',
        '후속 — 부정 게임의 유사 후보 제외':'부정 게임과의 코사인 0.5/0.8 이상 후보 제외를 비교했다. 입력된 게임 자체의 제외와 구분한다.',
        '후속 — 긍정·부정 관계 학습':'긍정–긍정·긍정–부정·부정–긍정 관계를 train에서 학습하고 공동 부정 항은 제외했다. 부정 입력 계수 0.1/0.3을 실제 평가했다.',
        '후속 — 여러 정답 동시 숨김 평가':'정답 2개를 동시에 숨기고 나머지 전체 ≥5·긍정 ≥1을 요구했다. 검증 217,720명·frank 14,819명이며 Recall=찾은 정답 수/2와 NDCG의 정답 2개 정규화를 적용했다.',
        '예약 계정 Kieran 보조 평가':'예약 계정 64,110명의 Kieran 보조 명단을 고정하고 대조군·선택 모델을 평가했다. 학습으로 되돌리지 않았고 frank와 사용자 중복이 있어 독립 표본으로 합산하지 않는다.',
        '아이디어 — 검증 샘플링 보완':'아래 게임별 리뷰 수 제한 보조 검증을 수행했다. 주 검증·채택 기준을 바꾸거나 frank 추출 분포를 재현했다고 주장하지 않는다.',
        '같은 인기도에서 frank가 더 높은 원인 확인':'지원 100–999명 집단을 전수 재집계했다. 입력 수·최대 콘텐츠 유사도 등 연관성을 저장했지만 차이의 인과 원인은 미확인이다. `backlog_diagnostics_v3.json` 참조.',
        '시리즈작·중복 소개글 영향 분리':'동일 정규화 제목·동일 소개글·높은 유사도를 분리해 집계하고 실제 최대 유사도 입력·제목 사례를 저장했다. 공식 분모는 유지한다. 시리즈/에디션이 동일 상품인지는 별도 App ID 근거가 필요하다.',
        '후속 — Item-KNN·콘텐츠 모델 결합':'실제 RRF 순위 융합과 후보 학습 지원 수에 따른 콘텐츠 비중을 비교했다. 정답을 아는 두 top10 합집합 적중은 실제 추천 성능으로 사용하지 않았다.',
        '아이디어 — 게임별 리뷰 수 제한 보조 검증':'Kieran 검증의 게임별 최대 100리뷰 표본을 고정했다. 입력과 정답 모두 제한한 뒤 남은 14,463명을 대조군·선택 모델로 평가했다. 보고용이며 독립 표본 또는 frank 수집 절차의 재현이 아니다.',
        'frank 입력 보강 평가 (사용자 결정 필요)':'사용자의 전체 백로그 실험 요청 범위에서 전체 Kieran 이력/정답 이전 추가 이력을 각각 추론 입력에만 더해 평가했다. 정답은 제거하고 예약 계정은 학습에서 계속 제외했다. 기존 frank 입력 전체의 시간 독립성을 뜻하지 않는다.',
        '콘텐츠 결측 보충':'고정 frank 공개 게임 메타데이터로 결측 소개글을 보충한 unigram·bigram과 태그/장르 결합을 별도 평가했다. 보충 효과와 확장 메타데이터 효과를 구분했다.',
        '이웃 선택과 점수 보정 분리':'최소 공동 사용자 수, 후보/입력/합집합 방향, 절단 제거, 공동 사용자 수·lift 이웃 선택 후 코사인 점수를 각각 비교했다.',
        '비대칭 코사인':'후보 지원 지수 0.25/0.75와 지역성 지수 1/2의 네 설정을 비교했다. 전체 평균과 희귀 게임 보호 조건을 함께 보고했다.',
        'RP3β':'α=1, β=0.3/0.6을 평가했다. 사용자–아이템 경로 확률·인기도 보정이며 공동 사용자가 없는 게임 쌍을 새로 연결하는 방법이라고 설명하지 않는다.',
        '리뷰어 활동량 가중':'사용자 활동량의 역수·역제곱근 가중을 단독 비교했다. 가중 공동 사용자 수와 가중 지원 수를 함께 사용했다. 구매·무료 수령 특징은 별도 미채택 후속이다.',
        'Item-KNN·콘텐츠 결합 방식':'대조 모델 결합과 검증에서 고정한 새 계열 모델 결합을 같은 RRF 가중치 격자로 비교했다. 관측 게임을 먼저 제외하고 순위를 부여했다.',
        'EASE':'전체 37,610개 카탈로그와 singleton을 유지해 λ=100/1000의 정확 선형 해를 구했다. 원본 Gram 검사·메모리/네이티브 라이브러리 실패와 재시도를 기록했다. ELSA는 조건이 충족되지 않아 실행하지 않았다.',
        'iALS':'긍정 입력 ≥2인 train 사용자로 implicit ALS rank32/64를 학습했다. 평가 사용자는 학습한 게임 벡터를 고정하고 보이는 긍정 입력만으로 fold-in했다.',
        '태그 기반 희귀작 보강':'태그·장르에서 ALS 게임 벡터를 ridge(α=100)로 예측하고, rank32/64별 0/50/100% 보강을 비교했다. train 지원 있는 게임만 회귀 학습했다.',
        '소개글 임베딩·긴 소개글':'C 긴 소개글과 고정 다국어 MiniLM 임베딩을 실제 계산했다. 텍스트 단독과 태그·장르 50% 결합을 나누어 평가했다. 모든 문장 임베딩 모델의 상한으로 해석하지 않는다.',
        '메타데이터 채널 확장':'태그·장르에 기능 분류·개발사·유통사를 추가한 모델을 평가했다. 소개글 결측 보충과는 다른 실험이다.',
        '리뷰 본문 특징':'허용된 train 리뷰 2,250,919개(앱당 최대100)의 본문으로 TF-IDF를 만들고 단독/태그·장르 결합을 평가했다. 샘플의 사용자·원 리뷰·게임 연결 불일치와 검증·frank 리뷰 ID 일치는 모두 0건이다.',
        '부정 리뷰 신뢰도 보정':'보이는 부정 입력의 누적 플레이타임 신뢰도와 train 게임별 긍정률 지수 0.5/1.0을 비교했다. 누적 스냅샷을 작성 당시 플레이타임으로 해석하지 않는다.',
        '플레이타임 학습 가중':'허용된 train 긍정 쌍만 연결해 게임별 log1p 플레이타임 백분위의 0.5+백분위를 학습 가중으로 사용했다. 입력 가중 실험과 분리했다.',
        '2단계 재정렬':'LambdaMART 100/300트리와 게임 빈도 가중 후속을 3분할 사용자 교차 적합으로 평가했다. 두 대조군 top50 합집합에 정답을 주입하지 않았다. 전체 평균과 보호 조건 통과를 구분했다.',
        '시간 인식 학습·검증':'train 리뷰 나이 반감기 3년 가중과 별도 2022년 말 cutoff 재학습을 수행했다. 시간 평가 41,941명의 Recall 28.4113%·NDCG 0.154463을 저장 추천으로 재계산했다. 과제가 달라 기본 평가와 직접 개선량을 비교하지 않는다.',
        '기술 집계 재현 스크립트':'`diagnose_backlog_v3.py`로 분포·지원·활동량·부정 비율·동일 제목/소개글·보유 이력 커버리지를 전수 재현해 보고서 JSON으로 저장했다.'}
    text=backlog.read_text('utf-8');lines=[];changed=set()
    for line in text.splitlines():
        match=re.match(r'- \[[ x]\] \*\*(.*?):?\*\*',line)
        title=match.group(1).rstrip(':') if match else None
        if title in replacements:
            line='- [x] **v3 실험·진단 완료 — '+title+':** '+replacements[title]+' [v3 결과](../reports/model_improvement_v3.md).'
            changed.add(title)
        elif title=='후속 — 다국어 토큰화':
            line='- [ ] **부분 완료 — 다국어 토큰화:** 문자 n-gram과 다국어 문장 임베딩은 v3에서 실험했다. 언어별 사전·형태소 분절은 미실행이며, 비라틴 문서를 제거하거나 문자를 언어 라벨로 취급하지 않는다.'
        lines.append(line)
    if changed!=set(replacements):raise ValueError('Unmatched backlog titles: '+str(set(replacements)-changed))
    text='\n'.join(lines)+'\n'
    text=text.replace('정리한 **미실행 아이디어**다. 읽기 전용 기술 집계와 웹 조사만 수행했고 대안 모델의 성능은 계산하지 않았으므로 아래 효과는 모두 가설이다.',
        '정리했던 아이디어의 실행 이력이다. v3에서 실제 실험·진단한 상태를 아래에 갱신했으며, 원래 효과 가설을 측정 결과로 바꾸어 해석하지 않는다.')
    text=text.replace('백로그로 예약. 숨길 개수 또는 비율·잔여 긍정 입력 조건·새 평가 집단은 후속 설계에서 결정.',
        'v3에서 정답 2개·나머지 전체 ≥5/긍정 ≥1로 별도 평가 완료.')
    text=text.replace('Kieran-only 긍정 제외 민감도·예약 계정 Kieran 보조 평가의 채택 여부는 미정. 게임별 리뷰 수 제한 보조 검증은 보고용 아이디어로 기록했고 미실행',
        'v3에서 Kieran-only 긍정 제외·예약 계정·게임별 리뷰 100개 제한을 별도 보조 평가로 완료. 주 평가나 채택 기준을 대체하지 않음')
    text=text.replace('부정 신호·여러 정답·전체 입력 ≥3·대안 집계·인기순 보완은 계속 후속 백로그다.',
        '부정 신호·여러 정답·전체 입력 ≥3·대안 집계·인기순 보완은 v3에서 별도 실험했다.')
    text=text.replace('여러 긍정 Linux 정답 동시 숨김. 개수·비율·잔여 긍정 입력 기준은 후속 실험 설계에서 결정',
        'v3에서 긍정 Linux 정답 2개 동시 숨김·잔여 전체 ≥5/긍정 ≥1을 별도 평가 완료')
    backlog.write_text(text,encoding='utf-8')


if __name__=='__main__':run()
