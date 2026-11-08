"""내부 검증으로 선택한 v3 모델에 사용자의 게임 이력을 넣어 추천합니다."""
import argparse
import json
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiment_v3_common import ROOT,catalog,RankFusionScorer
from select_experiments_v3 import restore
from evaluate_baselines import rank_positive


def recommend(family,positive,negative,observed,top_k):
    selection=json.loads((ROOT/'selection_frozen.json').read_text('utf-8'))
    item=selection['models'][family]
    if item['query_weighted']:raise ValueError('이 모델은 입력 리뷰의 추가 정보가 필요합니다. 저장된 평가 입력을 사용하는 API로 실행하세요.')
    model=restore(item['spec']);apps,linux,support=catalog();candidates=apps[linux]
    lookup={int(a):i for i,a in enumerate(apps)}
    positives=sorted(set(positive));negatives=sorted(set(negative))
    if set(positives)&set(negatives):raise ValueError('같은 게임을 긍정 목록과 부정 목록에 동시에 넣을 수 없습니다.')
    unknown=sorted((set(positives)|set(negatives))-set(lookup))
    if unknown:raise ValueError('입력 게임 목록에 없는 Steam App ID입니다: '+str(unknown))
    if not positives:raise ValueError('긍정 평가한 게임을 1개 이상 입력하세요.')
    def vector(ids):
        columns=[lookup[a] for a in ids]
        return sparse.csr_matrix((np.ones(len(columns),dtype=np.float32),([0]*len(columns),columns)),shape=(1,len(apps)))
    q,n=vector(positives),vector(negatives)
    seen=set(positives)|set(negatives)|set(observed)
    mask=sparse.csr_matrix([[int(int(a) in seen) for a in candidates]])
    if isinstance(model,RankFusionScorer):scores=model.fuse(model.left.score(q,n),model.right.score(q,n),mask)[0]
    else:scores=model.score(q,n)[0]
    scores[mask.indices]=0
    chosen=rank_positive(scores,candidates)
    if item['popularity_fill'] and len(chosen)<10:
        excluded=set(mask.indices)|set(chosen)
        added=[j for j in np.lexsort((candidates,-support[linux])) if j not in excluded][:10-len(chosen)]
        chosen=np.r_[chosen,added]
    return {'family':family,'variant':item['variant'],
        'recommendations':[{'app_id':int(candidates[j]),'rank':rank,'score':float(scores[j]),
            'popularity_fill':bool(item['popularity_fill'] and scores[j]<=0)} for rank,j in enumerate(chosen[:top_k],1)]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['collaborative','content','hybrid'],default='collaborative',help='협업·콘텐츠·결합 추천 중 선택')
    p.add_argument('--positive-apps',nargs='+',type=int,required=True,help='긍정 평가한 게임의 Steam App ID')
    p.add_argument('--negative-apps',nargs='*',type=int,default=[],help='부정 평가한 게임의 Steam App ID')
    p.add_argument('--exclude-apps',nargs='*',type=int,default=[],help='추가로 추천에서 제외할 이미 관측한 게임')
    p.add_argument('--top-k',type=int,choices=range(1,11),default=10,help='추천 개수 (기본 10개)')
    a=p.parse_args()
    with threadpool_limits(limits=2):print(json.dumps(recommend(a.family,a.positive_apps,a.negative_apps,a.exclude_apps,a.top_k),ensure_ascii=False,indent=2))
