"""One-time precision correction, preserving old receipts and all predictions."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from eda_large_review_frank import digest,log
from improvement_common import write_json
from experiment_v4_common import ROOT,SPLIT,metrics,load_cohort,support
import experiment_v4_common as common


def run():
    c=load_cohort('selection');sup=support();code=digest(common.__file__)
    for path in sorted((ROOT/'evaluations/selection').glob('*/manifest.json')):
        m=json.loads(path.read_text('utf-8'))
        if m['provenance']['evaluator_sha256']==code:continue
        for f,h in m['files'].items():
            if digest(path.parent/f)!=h:raise ValueError('Predictions changed during correction')
        d=pq.read_table(path.parent/'per_user.parquet').to_pydict()
        if d['user_key']!=c['user_ids'].tolist() or d['target_app_id']!=c['target_ids'].tolist():
            raise ValueError('Changed user/target pairing')
        archive=ROOT/'metric_precision_correction'/path.parent.name/'manifest_initial.json'
        if archive.exists():raise ValueError('Correction has already archived a different version')
        write_json(archive,m)
        old=m['provenance']['evaluator_sha256']
        m['metrics']=metrics(np.asarray(d['target_rank'],dtype=np.int8),
            np.asarray(d['recommendation_count'],dtype=np.int8),c,sup)
        m['provenance']['evaluator_sha256']=code
        m['metric_recomputation']=dict(reason='Explicit float64 rank logarithms and averages; old NumPy int8 log2 could use float16',
            previous_evaluator_sha256=old,source_per_user_sha256=m['files']['per_user.parquet'],
            inference_repeated=False,original_manifest_archive=str(archive.relative_to(ROOT)),
            correction_code_sha256=digest(__file__))
        write_json(path,m)
        log('Corrected float64 NDCG from unchanged predictions: '+path.parent.name)


if __name__=='__main__':run()
