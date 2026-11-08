"""Three-fold out-of-fold LambdaMART; target-free retrieval at inference."""
from __future__ import annotations
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from eda_large_review_frank import digest,log
from experiment_v3_common import ROOT,PROTOCOL,catalog,current_models,frozen_cohort,evaluate,write_json,inventory
from run_experiments_v3 import baseline,train_positive_rate
from prepare_query_features_v3 import metadata_features

FEATURES=['collaborative_score','content_score','log_train_support','train_positive_rate',
          'release_year_distance','release_year_missing','language_compatible','language_unknown']
PARAMETERS={'num_leaves':15,'n_estimators':300,'learning_rate':.05,'random_state':20261108,
    'n_jobs':2,'verbosity':-1,'objective':'lambdarank','label_gain':[0,1],
    'lambdarank_truncation_level':10,'deterministic':True,'force_col_wise':True,
    'min_child_samples':50}


def freeze_plan():
    path=ROOT/'reranking/protocol.json'
    value={'status':'fixed_before_reranking_metrics','parent_protocol_sha256':digest(PROTOCOL),
        'features':FEATURES,'parameters':PARAMETERS,'tested_trees':[100,300],
        'folds':'user_key modulo 3','training_users_per_fold':50000,
        'training_user_selection':'Lowest deterministic splitmix64 hash among other-fold users; include all retrieved candidates per chosen user.',
        'all_validation_users_evaluated':True,'reference_refit_maximum_users':150000,
        'candidate_generation':'Union of positive-score top50 per frozen current model after observed-game masks.',
        'labels':'Hidden positive target=1 only if actually retrieved; all other candidates weak relevance 0, not observed dislike.',
        'no_hidden_target_injection':True,'code_sha256':digest(__file__)}
    if path.exists() and json.loads(path.read_text('utf-8'))!=value:raise ValueError('Frozen reranking plan changed')
    write_json(path,value)
    return path


def user_hash(users):
    with np.errstate(over='ignore'):
        h=np.asarray(users,dtype=np.uint64)+np.uint64(0x9e3779b97f4a7c15)
        h=(h^(h>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        h=(h^(h>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return h^(h>>np.uint64(31))


def top50(values,app_ids):
    valid=np.flatnonzero(values>0)
    if len(valid)>50:
        threshold=np.partition(values[valid],len(valid)-50)[-50]
        valid=valid[values[valid]>=threshold]
    return valid[np.lexsort((app_ids[valid],-values[valid]))[:50]]


def retrieve_features(cohort):
    folder=ROOT/'reranking'/cohort['name']
    path=folder/'candidates.parquet'
    if path.exists():
        m=json.loads((folder/'retrieval_manifest.json').read_text('utf-8'))
        if digest(path)!=m['sha256']:raise ValueError('Reranking candidates changed')
        return path
    folder.mkdir(parents=True,exist_ok=True)
    models=current_models();apps,linux,support=catalog();ids=apps[linux]
    candidate_log_support=np.log1p(support[linux])
    prior=train_positive_rate()
    years,queryyear,languages,compatibility=metadata_features(cohort)
    schema=pa.schema([('user_row',pa.int32()),('candidate_col',pa.int32()),
        ('relevance',pa.int8())]+[(n,pa.float32()) for n in FEATURES])
    hit_users=0;empty_users=0;rows_written=0
    with pq.ParquetWriter(path,schema,compression='zstd') as writer:
        for start in range(0,len(cohort['user_ids']),256):
            end=min(start+256,len(cohort['user_ids']));q=cohort['positive'][start:end]
            a=models['item_knn_control'].score(q);b=models['content_control'].score(q)
            mr,mc=cohort['exclusions'][start:end].nonzero();a[mr,mc]=0;b[mr,mc]=0
            records=[]
            for local,userrow in enumerate(range(start,end)):
                selected=np.union1d(top50(a[local],ids),top50(b[local],ids))
                hit_users+=int(cohort['target_candidates'][userrow] in selected)
                empty_users+=int(not len(selected))
                for col in selected:
                    unknown=not(np.isfinite(years[col]) and np.isfinite(queryyear[userrow]))
                    distance=0. if unknown else abs(years[col]-queryyear[userrow])
                    lang=languages[userrow]
                    languagevalue=compatibility[lang][col] if lang in compatibility else -1.
                    records.append((userrow,int(col),int(col==cohort['target_candidates'][userrow]),
                        float(a[local,col]),float(b[local,col]),float(candidate_log_support[col]),
                        float(prior[col]),distance,float(unknown),max(0.,float(languagevalue)),float(languagevalue<0)))
            writer.write_table(pa.Table.from_arrays([pa.array([r[i] for r in records],type=f.type)
                for i,f in enumerate(schema)],schema=schema))
            rows_written+=len(records)
            if start==0 or end==len(cohort['user_ids']) or start%(256*250)==0:
                log(f'Rerank retrieval {cohort["name"]}: {end:,}/{len(cohort["user_ids"]):,}')
    write_json(folder/'retrieval_manifest.json',{'status':'complete','rows':rows_written,
        'users':len(cohort['user_ids']),'users_with_target_in_candidates':hit_users,
        'empty_candidate_users':empty_users,'candidate_recall_ceiling':hit_users/len(cohort['user_ids']),
        'sha256':digest(path),'protocol_sha256':digest(ROOT/'reranking/protocol.json'),
        'feature_code_sha256':digest(__file__),'query_metadata_code_sha256':digest(__import__('prepare_query_features_v3').__file__),
        'models':{k:v.spec for k,v in models.items()}})
    return path


def arrays(path):
    d=pq.read_table(path)
    userrows=d['user_row'].to_numpy();cols=d['candidate_col'].to_numpy()
    labels=d['relevance'].to_numpy()
    x=np.column_stack([d[name].to_numpy() for name in FEATURES])
    return userrows,cols,labels,x


def normalized_group_scores(margins,userrows):
    result=np.empty(len(margins),dtype=np.float64)
    boundaries=np.r_[0,np.flatnonzero(userrows[1:]!=userrows[:-1])+1,len(userrows)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        values=margins[begin:end]
        result[begin:end]=(values-values.min()+1)/(values.max()-values.min()+1)
    return result


class SavedRankerScorer:
    def __init__(self,rows,cols,scores,spec):
        self.rows,self.cols,self.scores,self.spec=rows,cols,scores,spec
        self.cursor=0;self.n_candidates=int(catalog()[1].sum())
    def score(self,q,negative=None):
        start,end=self.cursor,self.cursor+q.shape[0];self.cursor=end
        left=np.searchsorted(self.rows,start);right=np.searchsorted(self.rows,end)
        out=np.zeros((q.shape[0],self.n_candidates))
        out[self.rows[left:right]-start,self.cols[left:right]]=self.scores[left:right]
        return out


def run_validation():
    freeze_plan();c=frozen_cohort();path=retrieve_features(c)
    rows,cols,y,x=arrays(path)
    users=c['user_ids'];fold=(users%3).astype(int)
    hashes=user_hash(users)
    predictions={trees:np.zeros(len(rows)) for trees in [100,300]}
    model_paths=[]
    for heldout in range(3):
        train_users=np.flatnonzero(fold!=heldout)
        selected=train_users[np.argsort(hashes[train_users],kind='stable')[:50000]]
        chosen=np.zeros(len(users),dtype=bool);chosen[selected]=True
        train=chosen[rows];test=fold[rows]==heldout
        counts=np.bincount(rows[train],minlength=len(users));group=counts[counts>0]
        model_path=ROOT/'reranking'/f'fold_{heldout}.txt'
        if model_path.exists():
            model=lgb.Booster(model_file=str(model_path))
        else:
            ranker=lgb.LGBMRanker(**PARAMETERS)
            ranker.fit(x[train],y[train],group=group)
            ranker.booster_.save_model(str(model_path));model=ranker.booster_
        model_paths.append(model_path)
        for trees in [100,300]:predictions[trees][test]=model.predict(x[test],num_iteration=trees,num_threads=2)
        log(f'LambdaMART fold {heldout+1}/3 fitted on {int(chosen.sum()):,} users')
    for trees,margins in predictions.items():
        name=f'lambdamart_{trees}_trees'
        scorepath=ROOT/'reranking'/f'oof_{trees}.npy'
        np.save(scorepath,normalized_group_scores(margins,rows))
        spec={'method':'LambdaMART','trees':trees,'out_of_fold':True,
            'candidate_sha256':digest(path),'score_sha256':digest(scorepath),
            'fold_models':{p.name:digest(p) for p in model_paths},
            'protocol_sha256':digest(ROOT/'reranking/protocol.json')}
        model=SavedRankerScorer(rows,cols,np.load(scorepath,mmap_mode='r'),spec)
        evaluate(name,model,c,baseline('hybrid'),family='hybrid')
    write_json(ROOT/'reranking/manifest.json',{'status':'complete','protocol_sha256':digest(ROOT/'reranking/protocol.json'),
        'cross_fitting_completed':True,'files':inventory(ROOT/'reranking')})


if __name__=='__main__':
    with threadpool_limits(limits=2):run_validation()
