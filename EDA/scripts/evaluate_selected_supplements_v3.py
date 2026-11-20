"""Report frozen model choices on supplemental cohorts; never select from them."""
import json
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest
from experiment_v3_common import ROOT,RankFusionScorer,evaluate,write_json
from evaluate_supplements_v3 import COHORT_ROOT,load_supplement,evaluate_multiple
from select_experiments_v3 import restore
import numpy as np


class CohortMaskedFusion:
    """Preserve exclusion-before-rank semantics in multi-target evaluations."""
    def __init__(self,model,cohort):
        self.model,self.cohort,self.cursor=model,cohort,0
        self.spec={'method':'cohort_masked_rank_fusion','source':model.spec,
                   'replay_code_sha256':digest(__file__)}
    def score(self,q,negative=None):
        begin,end=self.cursor,self.cursor+q.shape[0];self.cursor=end
        return self.model.fuse(self.model.left.score(q,negative),self.model.right.score(q,negative),
                               self.cohort['exclusions'][begin:end])


def main():
    path=ROOT/'selection_frozen.json'
    chosen=json.loads(path.read_text('utf-8'))
    models={}
    for item in chosen['models'].values():
        # These experiments select fixed item representations or rank fusion.
        # Query-feature policies need explicit cohort-specific feature replay.
        if item['query_weighted'] or item['popularity_fill']:
            raise ValueError('Supplemental query/policy replay must be defined explicitly')
        models[item['variant']]=restore(item['spec'])
    summaries=[]
    for folder in sorted(COHORT_ROOT.iterdir()):
        if not folder.is_dir():continue
        c,_=load_supplement(folder.name)
        if c is None:continue
        for name,model in models.items():
            scorer=CohortMaskedFusion(model,c) if isinstance(model,RankFusionScorer) else model
            if np.max(np.diff(c['all_targets'].indptr))>1:m=evaluate_multiple(name,scorer,c)
            else:m=evaluate(name,scorer,c,family='supplemental')
            summaries.append({'cohort':folder.name,'variant':name,
                'evaluation_manifest_sha256':digest(ROOT/'evaluations'/folder.name/name/'manifest.json')})
    write_json(ROOT/'selected_supplemental_index.json',{'status':'complete',
        'selection_sha256':digest(path),'outcomes':summaries})


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
