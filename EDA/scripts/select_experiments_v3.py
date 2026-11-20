"""Freeze validation choices, then permit selected-model frank references."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import EDA,digest,log
from experiment_v3_common import (ROOT,PROTOCOL,V2,catalog,Scorer,CombinedScorer,NegativeScorer,
    RankFusionScorer,load_matrix,model_files,current_models,frozen_cohort,evaluate,write_json)
from run_experiments_v3 import load_factor_model,SignedScorer,baseline,train_positive_rate
from experiment_v3_query import weighted_query,build_item_kernel,MaxScorer,NegativeFilterScorer,context


def manifests():
    return [json.loads(p.read_text('utf-8')) for p in sorted((ROOT/'evaluations/validation').glob('*/manifest.json'))]


def verify_evidence(values):
    hashes={}
    for m in values:
        folder=ROOT/'evaluations/validation'/m['variant']
        if m['status']!='complete' or m['cohort']!='validation':raise ValueError('Wrong validation evidence')
        if m['provenance']['protocol_sha256']!=digest(PROTOCOL):raise ValueError('Changed validation protocol')
        for rel,info in m['files'].items():
            if digest(folder/rel)!=info['sha256']:raise ValueError('Changed evaluation '+m['variant']+'/'+rel)
        hashes[m['variant']]=digest(folder/'manifest.json')
    return hashes


def restore(spec):
    if 'directory' in spec:
        folder=EDA/spec['directory'];m=json.loads((folder/'manifest.json').read_text('utf-8'))
        if digest(folder/'manifest.json')!=spec['manifest_sha256']:raise ValueError('Chosen model manifest changed')
        if (folder/'item_factors.npy').exists():return load_factor_model(folder,m)
        if m.get('components'):
            load_matrix(folder,'content')
            children=[Scorer(sparse.load_npz(folder/x['path']),'content') for x in m['components']]
            return CombinedScorer(children,[x['weight'] for x in m['components']],spec)
        model=load_matrix(folder,'content' if m.get('family')=='content' or (folder/'tfidf_csr.npz').exists() or (folder/'embeddings.npy').exists() else 'knn')
        if folder.name.startswith('ease_lambda'):
            from cached_knn_v3 import CachedKNNScorer
            return CachedKNNScorer(model)
        return model
    method=spec.get('method')
    if method=='precomputed_item_content_cosine':
        from experiment_review_kernel_v3 import KernelContentScorer,CACHE
        path=ROOT/spec['cache_file']
        audit=json.loads((CACHE/'numerical_equivalence.json').read_text('utf-8'))
        if audit['status']!='passed' or digest(path)!=spec['cache_sha256']:
            raise ValueError('Review cosine cache is changed or unverified')
        if digest(CACHE/'execution_protocol.json')!=spec['execution_protocol_sha256']:
            raise ValueError('Review execution protocol changed')
        return KernelContentScorer(restore(spec['source']),np.load(path,mmap_mode='r'),spec)
    if method=='signed_item_relations':return SignedScorer(restore(spec['source']),spec['beta'])
    if method=='two_channel_content':
        return CombinedScorer([restore(spec['text']),restore(spec['metadata'])],spec['weights'],spec)
    if 'description' in spec and 'metadata' in spec:
        w=spec['description_weight'];return CombinedScorer([restore(spec['description']),restore(spec['metadata'])],[w,1-w],spec)
    if method=='reciprocal_rank_fusion':
        return RankFusionScorer(restore(spec['left']),restore(spec['right']),spec['content_weight'],spec['rank_constant'],spec['adaptive'],spec)
    if method=='negative_feedback_penalty':return NegativeScorer(restore(spec['base']),spec['coefficient'],spec)
    if method in ['recency','playtime','train_positive_rate','playtime_reliability_negative_penalty']:
        return CombinedScorer([restore(spec['base'])],[1.],spec)
    if method in ['max_item_similarity','negative_similarity_filter']:
        base=restore(spec['base']);prefix='knn' if isinstance(base,Scorer) and base.mode=='knn' else 'content'
        path=ROOT/'query_kernels'/f'{prefix}.npy';info=json.loads(path.with_suffix('.json').read_text('utf-8'))
        if digest(path)!=info['sha256']:raise ValueError('Item-similarity cache changed')
        kernel=np.load(path,mmap_mode='r')
        return MaxScorer(base,kernel) if method=='max_item_similarity' else NegativeFilterScorer(base,kernel,spec['threshold'])
    raise ValueError('Unsupported selected model recipe: '+str(method))


def choose(values,control,exclude_fill=False):
    candidates=[m for m in values if m['variant']!=control['variant']]
    eligible=[m for m in candidates if m['protection'].get('eligible_for_automatic_adoption')
        and not(exclude_fill and m['provenance'].get('popularity_fill'))]
    key=lambda m:(m['protection']['NDCG@10'],m['protection']['Recall@10'],m['variant'])
    return (max(eligible,key=key) if eligible else control),max(candidates+[control],key=key)


def expected_variants(include_selected_fusion=True):
    registry=json.loads((ROOT/'models/registry.json').read_text('utf-8'))
    incomplete={k:v['status'] for k,v in registry['models'].items() if v['status']!='complete'}
    if incomplete:raise ValueError('Collaborative experiments not complete: '+str(incomplete))
    expected=set(registry['models'])|{'item_knn_control','content_control',
        'lambdamart_100_trees','lambdamart_300_trees','lambdamart_300_trees_game_weighted'}
    for prefix in ['knn','content']:
        expected|={prefix+'_'+suffix for suffix in ['recency','playtime','negative_0_1','negative_0_3',
            'popularity_fill','positive_rate_0_5','positive_rate_1_0','max_similarity',
            'negative_filter_0_5','negative_filter_0_8','negative_playtime_reliability']}
    expected|={'rrf_'+adaptive+str(w).replace('.','_') for adaptive in ['','adaptive_'] for w in [.25,.5,.75]}
    from experiment_content_combinations_v3 import VARIANTS
    expected|=set(VARIANTS)|{'metadata_extended_tfidf','description_missing_fill_bigram'}
    expected|={name+'_metadata_0_5' for name in VARIANTS}|{'description_missing_fill_bigram_metadata_0_5'}
    if include_selected_fusion:
        stage=json.loads((ROOT/'family_selection_before_fusion.json').read_text('utf-8'))
        expected|=set(stage['followup_fusion_variants'])
    return expected


def family_selection():
    all_values=manifests();expected=expected_variants(False)
    path=ROOT/'family_selection_before_fusion.json'
    allowed_extra=set()
    if path.exists():
        previous=json.loads(path.read_text('utf-8'))
        allowed_extra=set(previous['followup_fusion_variants'])
    actual={m['variant'] for m in all_values}
    if actual-expected-allowed_extra:raise ValueError('Unexpected validation variants')
    # A resumed fusion run may already have some frozen follow-up evaluations.
    # Family choices always use precisely the original pre-fusion evidence.
    values=[m for m in all_values if m['variant'] in expected]
    actual={m['variant'] for m in values}
    if actual!=expected:raise ValueError('Incomplete/extra validation variants: '+str({'missing':sorted(expected-actual),'extra':sorted(actual-expected)}))
    evidence=verify_evidence(values);choices={}
    for family,controlname in [('collaborative','item_knn_control'),('content','content_control')]:
        control=next(m for m in values if m['variant']==controlname)
        chosen,best=choose([m for m in values if m['family']==family],control,exclude_fill=True)
        choices[family]={'variant':chosen['variant'],'spec':chosen['provenance']['spec'],
            'best_primary_variant':best['variant'],'validation_manifest_sha256':evidence[chosen['variant']]}
    same=(choices['collaborative']['variant']=='item_knn_control' and choices['content']['variant']=='content_control')
    names=[] if same else ['selected_rrf_'+a+str(w).replace('.','_') for a in ['','adaptive_'] for w in [.25,.5,.75]]
    result={'status':'frozen_before_followup_fusion','protocol_sha256':digest(PROTOCOL),'families':choices,
        'validation_evidence':evidence,'followup_fusion_variants':names,
        'identical_current_family_models_skip_duplicate_grid':same}
    if path.exists() and json.loads(path.read_text('utf-8'))!=result:raise ValueError('Family choices changed')
    write_json(path,result);return result


def followup_fusion():
    result=family_selection()
    if not result['followup_fusion_variants']:return
    left=restore(result['families']['collaborative']['spec']);right=restore(result['families']['content']['spec'])
    c=frozen_cohort()
    for w in [.25,.5,.75]:
        for adaptive in [False,True]:
            name='selected_rrf_'+('adaptive_' if adaptive else '')+str(w).replace('.','_')
            spec={'method':'reciprocal_rank_fusion','rank_constant':60,'content_weight':w,'adaptive':adaptive,
                'left':left.spec,'right':right.spec,'family_selection_sha256':digest(ROOT/'family_selection_before_fusion.json')}
            evaluate(name,RankFusionScorer(left,right,w,60,adaptive,spec),c,baseline('hybrid'),family='hybrid')


def freeze_selection():
    values=manifests();expected=expected_variants();actual={m['variant'] for m in values}
    if actual!=expected:raise ValueError('Final variant set mismatch '+str({'missing':sorted(expected-actual),'extra':sorted(actual-expected)}))
    evidence=verify_evidence(values);models={}
    for family,controlname in [('collaborative','item_knn_control'),('content','content_control'),('hybrid','item_knn_control')]:
        control=next(m for m in values if m['variant']==controlname)
        chosen,best=choose([m for m in values if m['family']==family],control)
        models[family]={'variant':chosen['variant'],'spec':chosen['provenance']['spec'],
            'popularity_fill':chosen['provenance']['popularity_fill'],'query_weighted':chosen['provenance']['query_weighted'],
            'best_primary_variant':best['variant'],'validation_manifest_sha256':evidence[chosen['variant']],
            'retained_control':chosen['variant']==controlname}
    pinned={p.name:digest(p) for p in (EDA/'scripts').glob('*v3*.py')}
    result={'status':'frozen_before_new_frank_reference','protocol_sha256':digest(PROTOCOL),
        'models':models,'validation_evidence':evidence,'code_sha256':pinned,
        'test_policy':'Previously observed frank reference only; no frank-based choice.'}
    for path in [ROOT/'selection_frozen.json',EDA/'data/reference/improvement_selection_v3.json']:
        if path.exists() and json.loads(path.read_text('utf-8'))!=result:raise ValueError('Refuse changed final choice')
        write_json(path,result)
    return result


def reference():
    path=ROOT/'selection_frozen.json';selected=json.loads(path.read_text('utf-8'))
    if digest(path)!=digest(EDA/'data/reference/improvement_selection_v3.json'):raise ValueError('Published choice differs')
    if verify_evidence(manifests())!=selected['validation_evidence']:raise ValueError('Validation evidence changed')
    for name,h in selected['code_sha256'].items():
        if digest(EDA/'scripts'/name)!=h:raise ValueError('Code changed after selection: '+name)
    c=frozen_cohort('test')
    for name,model in current_models().items():evaluate(name,model,c,family='content' if name.startswith('content') else 'collaborative',allow_reference=True)
    for family,item in selected['models'].items():
        if item['variant'] in ['item_knn_control','content_control']:continue
        spec=item['spec'];model=restore(spec);kwargs={}
        method=spec.get('method')
        if method in ['recency','playtime']:kwargs['query_weights']=weighted_query(c,method)
        elif method=='train_positive_rate':
            prior=train_positive_rate();e=spec['exponent']
            kwargs['score_transform']=lambda s,q,n,i,j:s*(prior**e)[None,:]
        elif method=='playtime_reliability_negative_penalty':
            raise ValueError('Selected reliability model needs its measured query feature replay')
        evaluate(item['variant'],model,c,popularity_fill=item['popularity_fill'],family=family,
            allow_reference=True,**kwargs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['families','fusion','select','reference']);a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.phase=='families':family_selection()
        elif a.phase=='fusion':followup_fusion()
        elif a.phase=='select':freeze_selection()
        else:reference()
