"""Compare direct and cached review cosine in both content configurations."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from experiment_review_kernel_v3 import prepare,CACHE
from experiment_v3_common import ROOT,V2,load_matrix,frozen_cohort,catalog,write_json
from eda_large_review_frank import digest
from evaluate_baselines import rank_positive


def run():
    base,cached=prepare();meta=load_matrix(V2/'tfidf/metadata_channel','content')
    c=frozen_cohort();apps,linux,_=catalog();ids=apps[linux]
    selected=np.unique(np.linspace(0,len(c['user_ids'])-1,1024,dtype=int))
    checks={name:{'max_absolute_error':0.,'top10_list_differences':0,'target_rank_differences':0} for name in ['text_only','text_plus_metadata']}
    for start in range(0,len(selected),128):
        idx=selected[start:start+128];q=c['positive'][idx];r,col=c['exclusions'][idx].nonzero()
        a,b=base.score(q),cached.score(q);m=meta.score(q)
        for name,left,right in [('text_only',a,b),('text_plus_metadata',.5*a+.5*m,.5*b+.5*m)]:
            if not np.allclose(left,right,atol=1e-6,rtol=1e-4):raise ValueError('Review cosine precision exceeds tolerance')
            check=checks[name];check['max_absolute_error']=max(check['max_absolute_error'],float(np.max(np.abs(left-right))))
            left[r,col]=0;right[r,col]=0
            for i,(x,y) in enumerate(zip(left,right)):
                p,s=rank_positive(x,ids),rank_positive(y,ids)
                check['top10_list_differences']+=int(not np.array_equal(p,s))
                target=c['target_ids'][idx[i]]
                def rank(v):
                    h=np.flatnonzero(ids[v]==target)
                    return int(h[0]+1) if len(h) else 0
                check['target_rank_differences']+=int(rank(p)!=rank(s))
    result={'status':'passed' if not any(c['top10_list_differences'] or c['target_rank_differences'] for c in checks.values()) else 'differences_found',
        'code_sha256':digest(__file__),'sample_users':len(selected),'checks':checks,
        'scope':'Numerical execution verification; not a model-selection metric; does not claim bitwise equivalence or inspect all validation rankings.'}
    write_json(CACHE/'two_channel_numerical_check.json',result)
    return result


if __name__=='__main__':
    with threadpool_limits(limits=2):print(json.dumps(run(),indent=2))
