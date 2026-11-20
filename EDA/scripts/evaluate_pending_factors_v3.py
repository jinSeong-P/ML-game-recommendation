"""Score small fitted factor models while the larger EASE fit finishes."""
import json
from threadpoolctl import threadpool_limits
from experiment_v3_common import ROOT,frozen_cohort,evaluate
from run_experiments_v3 import load_factor_model,baseline


def run():
    c=frozen_cohort()
    for folder in sorted((ROOT/'models').iterdir()):
        path=folder/'manifest.json'
        if not path.exists() or not (folder/'item_factors.npy').exists():continue
        m=json.loads(path.read_text('utf-8'))
        if m.get('status')!='complete':continue
        model=load_factor_model(folder,m)
        evaluate(folder.name,model,c,baseline('collaborative'),family='collaborative')


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
