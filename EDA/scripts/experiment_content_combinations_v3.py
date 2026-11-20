"""Evaluate each new content representation with the already-selected metadata.

This is a predeclared ablation extension: preserve the current content model's
two-channel structure while changing only its description representation.
"""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from eda_large_review_frank import EDA,digest
from experiment_v3_common import (ROOT,V2,PROTOCOL,current_models,load_matrix,
    CombinedScorer,evaluate,frozen_cohort,model_files,write_json)
from run_experiments_v3 import baseline

VARIANTS=['description_missing_fill','description_long_word_tfidf',
          'description_char_wb_tfidf','description_sentence_embedding','train_review_body_tfidf']


def freeze():
    path=ROOT/'content_combination_protocol.json'
    plan={'status':'fixed_before_content_combination_metrics','parent_sha256':digest(PROTOCOL),
          'variants':VARIANTS,'weights':[.5,.5],
          'metadata_source':model_files(V2/'tfidf/metadata_channel'),
          'normalization':'Each positive-input profile normalized independently; missing channel contribution0; no weight renormalization.',
          'reason':'A new text representation alone and the current two-channel content model otherwise change different feature sets.',
          'code_sha256':digest(__file__)}
    if path.exists() and json.loads(path.read_text('utf-8'))!=plan:raise ValueError('Combination protocol changed')
    write_json(path,plan)
    return path


def run():
    plan=freeze();c=frozen_cohort();meta=load_matrix(V2/'tfidf/metadata_channel','content')
    for name in VARIANTS:
        folder=ROOT/'content'/name
        if not (folder/'manifest.json').exists():continue
        text=load_matrix(folder,'content')
        spec={'method':'two_channel_content','text':model_files(folder),
              'metadata':model_files(V2/'tfidf/metadata_channel'),'weights':[.5,.5],
              'combination_protocol_sha256':digest(plan)}
        model=CombinedScorer([text,meta],[.5,.5],spec)
        evaluate(name+'_metadata_0_5',model,c,baseline('content'),family='content')


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
