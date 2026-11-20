"""Evaluate the predeclared filled-description bigram plus metadata ablation."""
import json
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest
from experiment_v3_common import ROOT,V2,CombinedScorer,load_matrix,model_files,evaluate,frozen_cohort
from run_experiments_v3 import baseline


def run():
    plan=ROOT/'filled_bigram_protocol.json'
    p=json.loads(plan.read_text('utf-8'))
    if p['status']!='fixed_before_filled_bigram_metrics':raise ValueError('Missing frozen plan')
    text=ROOT/'content_models/description_missing_fill_bigram'
    metadata=V2/'tfidf/metadata_channel'
    spec={'method':'two_channel_content','text':model_files(text),'metadata':model_files(metadata),
          'weights':[p['description_weight'],p['metadata_weight']],
          'filled_bigram_protocol_sha256':digest(plan)}
    model=CombinedScorer([load_matrix(text,'content'),load_matrix(metadata,'content')],spec['weights'],spec)
    evaluate('description_missing_fill_bigram_metadata_0_5',model,frozen_cohort(),baseline('content'),family='content')


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
