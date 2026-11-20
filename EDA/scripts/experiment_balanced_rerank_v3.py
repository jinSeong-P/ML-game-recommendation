"""Validation-informed, game-frequency weighted LambdaMART cross-fitting.

The unweighted experiment improved user averages but harmed sparse games. This
follow-up tests a standard inverse-frequency query weight, with no frank tuning.
"""
import json
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest,log
from experiment_v3_common import ROOT,PROTOCOL,frozen_cohort,evaluate,write_json
from experiment_v3_rerank import (FEATURES,PARAMETERS,user_hash,arrays,
    normalized_group_scores,SavedRankerScorer)
from run_experiments_v3 import baseline


def run():
    folder=ROOT/'reranking_balanced';folder.mkdir(exist_ok=True)
    path=ROOT/'reranking/validation/candidates.parquet'
    plan={'status':'fixed_before_balanced_reranking_metrics','parent_sha256':digest(PROTOCOL),
        'previous_unweighted_result_sha256':digest(ROOT/'evaluations/validation/lambdamart_300_trees/manifest.json'),
        'candidate_sha256':digest(path),'parameters':PARAMETERS,'features':FEATURES,
        'folds':'user_key modulo3; own user always excluded from fit',
        'max_training_users_per_fold':50000,
        'weight':'Within fitting users only: sqrt(n_users/(n_target_games*target_frequency)); clip[.25,4], normalize mean1; constant weight for each query group.',
        'validation_informed_followup':True,'frank_used_for_settings':False,'code_sha256':digest(__file__)}
    target=folder/'protocol.json'
    if target.exists() and json.loads(target.read_text('utf-8'))!=plan:raise ValueError('Balanced reranking protocol changed')
    write_json(target,plan)
    c=frozen_cohort();rows,cols,y,x=arrays(path);users=c['user_ids']
    folds=(users%3).astype(int);hashes=user_hash(users);pred=np.zeros(len(rows));model_paths=[];weights_summary=[]
    for heldout in range(3):
        others=np.flatnonzero(folds!=heldout)
        selected=others[np.argsort(hashes[others],kind='stable')[:50000]]
        chosen=np.zeros(len(users),dtype=bool);chosen[selected]=True
        train=chosen[rows];test=folds[rows]==heldout
        groupcounts=np.bincount(rows[train],minlength=len(users));groups=groupcounts[groupcounts>0]
        frequency=np.bincount(c['target_candidates'][selected],minlength=6630)
        gamecount=np.count_nonzero(frequency)
        w=np.zeros(len(users))
        w[selected]=np.clip(np.sqrt(len(selected)/(gamecount*frequency[c['target_candidates'][selected]])),.25,4.)
        w[selected]/=np.mean(w[selected])
        weights_summary.append({'fold':heldout,'users':len(selected),'target_games':int(gamecount),
            'weight_min':float(w[selected].min()),'weight_max':float(w[selected].max()),'weight_mean':float(w[selected].mean())})
        model_path=folder/f'fold_{heldout}.txt'
        if model_path.exists():model=lgb.Booster(model_file=str(model_path))
        else:
            ranker=lgb.LGBMRanker(**PARAMETERS)
            ranker.fit(x[train],y[train],group=groups,sample_weight=w[rows[train]])
            ranker.booster_.save_model(str(model_path));model=ranker.booster_
        pred[test]=model.predict(x[test],num_iteration=300,num_threads=2)
        model_paths.append(model_path);log(f'Game-weighted LambdaMART fold{heldout+1}/3 complete')
    scorepath=folder/'oof_300.npy';np.save(scorepath,normalized_group_scores(pred,rows))
    spec={'method':'LambdaMART_game_frequency_weighted','trees':300,'out_of_fold':True,
        'candidate_sha256':digest(path),'score_sha256':digest(scorepath),
        'fold_models':{p.name:digest(p) for p in model_paths},'protocol_sha256':digest(target)}
    model=SavedRankerScorer(rows,cols,np.load(scorepath,mmap_mode='r'),spec)
    evaluate('lambdamart_300_trees_game_weighted',model,c,baseline('hybrid'),family='hybrid')
    write_json(folder/'manifest.json',{'status':'complete','spec':spec,'fold_weights':weights_summary})


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
