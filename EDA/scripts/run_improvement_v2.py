"""Run the predeclared v2 validation experiments and freeze selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from eda_large_review_frank import EDA,digest,log
from improvement_common import ROOT,PROTOCOL,V1,write_json,evaluate_variant,load_knn,CombinedCatalog
from item_knn import ItemKnnCatalog
from tfidf_content import TfidfCatalog

PUBLISHED_SELECTION=EDA/'data/reference/improvement_selection_v2.json'

def expected_text_variants():
    names={'text_v1_control','text_T1_sublinear_tf','text_T2_unigram_bigram'}
    t0=json.loads((EDA/'reports/tfidf_ranking_analysis_v2.json').read_text('utf-8'))
    if t0['diagnostics']['T3_conditional_gate']['t3_choice']=='build':names.add('text_T3_max_df_0_5')
    return names

def expected_validation_variants():
    return (expected_text_variants() | {'knn_v1_control'}
       | {'knn_k'+str(k)+'_l'+str(l) for k in [100,200,500] for l in [0,10,50]}
       | {'metadata_alpha_1_0','metadata_alpha_0_75','metadata_alpha_0_5','metadata_alpha_0_25'})

def require_variant_set(values,expected):
    actual={m['variant'] for m in values}
    if actual!=expected or len(values)!=len(expected):
        raise ValueError('Predeclared variants mismatch: missing='+str(sorted(expected-actual))+' extra='+str(sorted(actual-expected)))

def verify_model_sources(spec):
    for source in ([spec['text'],spec['metadata']] if 'text' in spec else [spec]):
        folder=EDA/source['directory']
        if digest(folder/'manifest.json')!=source['manifest_sha256']:
            raise ValueError('Model manifest changed: '+source['directory'])
        manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
        for relative,info in manifest['files'].items():
            if digest(folder/relative)!=info['sha256']:raise ValueError('Model output changed: '+relative)

def verify_validation_evidence(m):
    if (m['status']!='complete' or m['cohort']!='validation'
       or m['protocol_sha256']!=digest(PROTOCOL)
       or m['evaluator_v2_sha256']!=digest(EDA/'scripts/improvement_common.py')
       or m['evaluator_v1_sha256']!=digest(EDA/'scripts/evaluate_baselines.py')):
        raise ValueError('Validation protocol/code provenance changed: '+m['variant'])
    verify_model_sources(m['model'])
    folder=ROOT/'evaluations/validation'/m['variant']
    for relative,info in m['files'].items():
        if digest(folder/relative)!=info['sha256']:raise ValueError('Validation output changed')

def load_verified_selection():
    path=ROOT/'selection_frozen.json'
    if digest(path)!=digest(PUBLISHED_SELECTION):raise ValueError('Frozen selection differs from published snapshot')
    selected=json.loads(path.read_text('utf-8'))
    if selected['protocol_sha256']!=digest(PROTOCOL) or selected['runner_sha256']!=digest(__file__):
        raise ValueError('Frozen selection protocol/runner changed')
    values=summaries();require_variant_set(values,expected_validation_variants())
    evidence={m['variant']:digest(ROOT/'evaluations/validation'/m['variant']/'manifest.json') for m in values}
    if evidence!=selected['validation_evidence']:raise ValueError('Frozen validation evidence changed')
    for m in values:verify_validation_evidence(m)
    for item in selected['models'].values():verify_model_sources(item['model'])
    return selected

def artifact(path):
    path=Path(path)
    return {'directory':path.relative_to(EDA).as_posix(),'manifest_sha256':digest(path/'manifest.json')}

def load_text(path):
    path=Path(path); m=json.loads((path/'manifest.json').read_text('utf-8'))
    if m.get('status')!='complete':raise ValueError('Incomplete text artifact')
    for name,info in m.get('files',{}).items():
        if digest(path/name)!=info['sha256']:raise ValueError('Text artifact changed: '+name)
    rows=pq.read_table(path/'game_rows.parquet').to_pydict() if (path/'game_rows.parquet').exists() else pq.read_table(EDA/'data/processed/modeling/tfidf_content_v1/game_rows.parquet').to_pydict()
    return TfidfCatalog(rows['app_id'],rows['linux'],sparse.load_npz(path/'tfidf_csr.npz'))

def controls():
    for kind,name,path,loader in [
       ('item_knn','knn_v1_control',EDA/'data/processed/modeling/item_knn_v1',ItemKnnCatalog.load),
       ('tfidf','text_v1_control',EDA/'data/processed/modeling/tfidf_content_v1',TfidfCatalog.load)]:
        m=evaluate_variant(name,loader(path),kind,extra=artifact(path))
        a=pq.read_table(ROOT/'evaluations/validation'/name/'per_user.parquet').to_pydict()
        b=pq.read_table(V1/'validation'/kind/'per_user.parquet').to_pydict()
        if a!=b:raise ValueError('v1 control does not reproduce saved outcomes: '+kind)
        log('Exact v1 per-user control reproduced: '+kind)

def run_knn():
    diag=EDA/'reports/item_knn_failure_analysis_v2.json'
    if not diag.exists():raise ValueError('K0 must finish before K1/K2 ranking metrics')
    dirs=sorted((ROOT/'knn').glob('k*_l*'))
    if len(dirs)!=9:raise ValueError('Expected 9 predeclared K x lambda model artifacts')
    # Lambda=0 K1 first; all remaining K2 cells follow without data-driven expansion.
    dirs.sort(key=lambda p:(int(p.name.split('_l')[1]),int(p.name.split('_')[0][1:])))
    for path in dirs:
        evaluate_variant('knn_'+path.name,load_knn(path),'item_knn',extra=artifact(path))

def evaluate_text_directory(path,name=None):
    return evaluate_variant(name or 'text_'+path.name,load_text(path),'tfidf',extra=artifact(path))

def run_text():
    if not (EDA/'reports/tfidf_ranking_analysis_v2.json').exists():raise ValueError('T0 required before text ranking metrics')
    base=ROOT/'tfidf'
    # Rebuilt text control is independently checked before changed parameters.
    original=TfidfCatalog.load(EDA/'data/processed/modeling/tfidf_content_v1')
    rebuilt=load_text(base/'baseline_control')
    if not np.array_equal(original.matrix.indptr,rebuilt.matrix.indptr) or not np.array_equal(original.matrix.indices,rebuilt.matrix.indices) or not np.array_equal(original.matrix.data,rebuilt.matrix.data):
        raise ValueError('Rebuilt text control not exactly equal to v1')
    for name in ['T1_sublinear_tf','T2_unigram_bigram','T3_max_df_0_5']:
        path=base/name
        if path.exists():evaluate_text_directory(path,'text_'+name)

def run_metadata(alpha_values=None):
    alpha_values=[1.,.75,.5,.25] if alpha_values is None else list(alpha_values)
    if not alpha_values or any(alpha not in [1.,.75,.5,.25] for alpha in alpha_values):
        raise ValueError('Only predeclared metadata weights are allowed')
    values=[m for m in summaries() if m['kind']=='tfidf' and not m['variant'].startswith('metadata_')]
    require_variant_set(values,expected_text_variants())
    for m in values:verify_validation_evidence(m)
    control=next(m for m in values if m['variant']=='text_v1_control')
    chosen,best=choose([m for m in values if m['variant']!='text_v1_control'],control)
    evidence={'protocol_sha256':digest(PROTOCOL),'selected_text_variant':chosen['variant'],
       'model':chosen['model'],'best_primary_text_variant':best['variant'],
       'validation_manifests':{m['variant']:digest(ROOT/'evaluations/validation'/m['variant']/'manifest.json') for m in values}}
    target=ROOT/'text_selection_before_metadata.json'
    if target.exists():
        if json.loads(target.read_text('utf-8'))!=evidence:raise ValueError('Text selection changed')
    else:write_json(target,evidence)
    text=load_text(EDA/chosen['model']['directory']);path=ROOT/'tfidf/metadata_channel';meta=load_text(path)
    for alpha in alpha_values:
        name='metadata_alpha_'+str(alpha).replace('.','_')
        model=CombinedCatalog(text,meta,alpha)
        extra={'text':chosen['model'],'metadata':artifact(path),'alpha':alpha,
          'text_selection_sha256':digest(target)}
        evaluate_variant(name,model,'tfidf',extra=extra)
        if alpha==1:
            x=pq.read_table(ROOT/'evaluations/validation'/name/'per_user.parquet').to_pydict()
            y=pq.read_table(ROOT/'evaluations/validation'/chosen['variant']/'per_user.parquet').to_pydict()
            if x!=y:raise ValueError('Alpha=1 failed selected text control equivalence')

def run_reference():
    selected=load_verified_selection()
    for kind,item in selected['models'].items():
        extra=item['model']
        if 'text' in extra:
            model=CombinedCatalog(load_text(EDA/extra['text']['directory']),load_text(EDA/extra['metadata']['directory']),extra['alpha'])
        elif kind=='item_knn':model=load_knn(EDA/extra['directory'])
        else:model=load_text(EDA/extra['directory'])
        if digest(ROOT/'evaluations/validation'/item['variant']/'manifest.json')!=item['validation_manifest_sha256']:
            raise ValueError('Selected validation evidence changed')
        evaluate_variant(item['variant'],model,kind,'test',extra=extra)

def summaries():
    return [json.loads(p.read_text('utf-8')) for p in (ROOT/'evaluations/validation').glob('*/manifest.json')]

def choose(candidates,control):
    def order(m):
        p=m['protection'];source=m['model']
        # Exact metric ties use the predeclared preference for fewer channels,
        # smaller K and smaller feature dictionaries.
        channels=2 if 'metadata' in source else 1
        folder=EDA/source.get('directory',source.get('text',{}).get('directory',''))
        manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
        k=manifest.get('parameters',{}).get('neighbors',0)
        size=manifest.get('index',{}).get('shape',[0,0])[-1]
        return (p['NDCG@10'],p['Recall@10'],-channels,-k,-size)
    eligible=[m for m in candidates if m['protection']['eligible_for_automatic_adoption']]
    chosen=max(eligible,key=order) if eligible else control
    best=max(candidates+[control],key=order)
    return chosen,best

def freeze_selection():
    values=summaries();models={}
    require_variant_set(values,expected_validation_variants())
    for m in values:verify_validation_evidence(m)
    for kind,control_name in [('item_knn','knn_v1_control'),('tfidf','text_v1_control')]:
        control=next(m for m in values if m['variant']==control_name)
        candidates=[m for m in values if m['kind']==kind and m['variant']!=control_name]
        chosen,best=choose(candidates,control)
        models[kind]={'variant':chosen['variant'],'model':chosen['model'],
          'validation_manifest_sha256':digest(ROOT/'evaluations/validation'/chosen['variant']/'manifest.json'),
          'best_primary_variant':best['variant'],'retained_v1':chosen['variant']==control_name,
          'reason':'All automatic adoption guards passed' if chosen['variant']!=control_name else 'No candidate met all predeclared adoption guards'}
    result={'version':'improvement-v2-selection','status':'frozen_before_reference_test',
       'protocol_sha256':digest(PROTOCOL),'runner_sha256':digest(__file__),'models':models,
       'validation_evidence':{m['variant']:digest(ROOT/'evaluations/validation'/m['variant']/'manifest.json') for m in values}}
    target=ROOT/'selection_frozen.json'
    if target.exists() and json.loads(target.read_text('utf-8'))!=result:raise ValueError('Refusing changed frozen selection')
    if PUBLISHED_SELECTION.exists() and json.loads(PUBLISHED_SELECTION.read_text('utf-8'))!=result:raise ValueError('Refusing changed published selection')
    write_json(target,result)
    write_json(PUBLISHED_SELECTION,result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['controls','knn','text','metadata','metadata-one','text-one','select','reference'])
    p.add_argument('--path',type=Path);p.add_argument('--name')
    p.add_argument('--alpha',type=float,choices=[1.,.75,.5,.25])
    args=p.parse_args()
    if args.phase=='controls':controls()
    elif args.phase=='knn':run_knn()
    elif args.phase=='text':run_text()
    elif args.phase=='metadata':run_metadata()
    elif args.phase=='metadata-one':
        if args.alpha is None:raise ValueError('metadata-one requires a predeclared alpha')
        run_metadata([args.alpha])
    elif args.phase=='text-one':
        if args.name not in expected_text_variants()-{'text_v1_control'} or args.path is None:
            raise ValueError('Only predeclared text variants are allowed')
        expected=ROOT/'tfidf'/args.name.removeprefix('text_')
        if args.path.resolve()!=expected.resolve():raise ValueError('Text artifact must use the predeclared directory')
        evaluate_text_directory(args.path.resolve(),args.name)
    elif args.phase=='select':print(json.dumps(freeze_selection(),indent=2))
    elif args.phase=='reference':run_reference()

if __name__=='__main__':main()
