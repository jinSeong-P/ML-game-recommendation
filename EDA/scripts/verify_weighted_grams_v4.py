"""Compare production weighted Gram rows with the original sparse product."""
from __future__ import annotations
import gc
import json
import shutil
import numpy as np
from threadpoolctl import threadpool_limits
from eda_large_review_frank import EDA,digest,log
from improvement_common import write_json
from train_base_v4 import training_matrix
import experiment_linear_v4 as linear


def run():
    x,support=training_matrix();degree=np.diff(x.indptr).astype(np.float64)
    sources=linear._source_manifest();results={}
    for beta in (.5,1.):
        path=linear._gram_cache_path(beta);receipt=json.loads(linear._gram_cache_receipt(path).read_text('utf-8'))
        if (not receipt['complete'] or receipt['train_sha256']!=sources['train/positive_catalog.parquet']['sha256']
            or receipt['sha256']!=digest(path)):
            raise ValueError('Weighted Gram cache receipt changed')
        csc=x.astype(np.float32).tocsc()
        csc.data=np.sqrt(np.power(degree,-beta).astype(np.float32)[csc.indices])
        probes=np.unique(np.asarray([0,1,len(support)//4,len(support)//2,len(support)-1]))
        # This is the original CSR@CSC path, including its full right-operand
        # conversion. Use complete training rows, not a downsample of users.
        expected=(csc.T.tocsr()[probes]@csc).toarray()
        mapped=np.memmap(path,mode='r',dtype=np.float32,shape=(len(support),len(support)))
        actual=np.array(mapped[probes],copy=True)
        linear._close_memmap(mapped)
        if not np.array_equal(actual,expected):raise ValueError('Optimized weighted Gram differs from original production calculation')
        models=[]
        for p in linear.MODEL_ROOT.glob('dan_ease_lambda100_*/manifest.json'):
            m=json.loads(p.read_text('utf-8'))
            if m['parameters']['beta']!=beta:continue
            if m['fit_summary']['gram_sha256']!=receipt['sha256']:
                raise ValueError('DAN receipt uses a different weighted Gram')
            models.append(m['model_name'])
        if len(models)!=16:
            raise ValueError(f'Expected all 16 scheduled DAN variants for beta={beta:g}, found {len(models)}')
        results[str(beta)]=dict(status='bit_identical_complete_train_probe_rows',
            input_item_rows=probes.tolist(),checked_coefficients=int(actual.size),
            maximum_absolute_difference=0.,gram_sha256=receipt['sha256'],
            verified_model_gram_receipts=sorted(models),gram_seconds=receipt.get('gram_seconds'))
        del csc,expected,actual;gc.collect();log(f'Weighted Gram beta={beta:g} probes match original exactly')
    output=linear.MODEL_ROOT.parent/'weighted_gram_verification';output.mkdir(exist_ok=True)
    shutil.copy2(__file__,output/'builder_snapshot.py')
    result=dict(status='passed',train_sha256=sources['train/positive_catalog.parquet']['sha256'],
        code_sha256=digest(__file__),results=results,
        interpretation='Full training data for five complete item rows per beta. This verifies conversion reuse, not an independent recommendation experiment or every Gram row.')
    write_json(output/'manifest.json',result);write_json(EDA/'reports/weighted_gram_validation_v4.json',result)
    return result


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
