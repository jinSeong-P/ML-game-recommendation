"""Check the saved full-catalog EASE solution against its normal equations."""
import json
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest
from experiment_v3_common import ROOT,V2,catalog,write_json


def sampled_residual(gram,coefficients,target,regularization,rows,row_matrix=None):
    rows=np.asarray(rows);keep=rows!=target
    selected=np.asarray(gram[rows[keep]] if row_matrix is None else row_matrix[keep],dtype=np.float64)
    rows=rows[keep]
    b=np.asarray(coefficients,dtype=np.float64)
    expected=selected[:,target]
    actual=selected@b+regularization*b[rows]
    residual=actual-expected
    return {'rows':len(rows),'relative_l2_residual':float(np.linalg.norm(residual)/max(1.,np.linalg.norm(expected))),
        'max_absolute_residual':float(np.max(np.abs(residual))) if len(residual) else 0.}


def run():
    apps,linux,support=catalog();candidate_columns=np.flatnonzero(linux)
    receipt=json.loads((ROOT/'ease_gram_recovery_audit.json').read_text('utf-8'))
    selected=np.unique(np.linspace(0,len(candidate_columns)-1,16,dtype=int))
    rows=candidate_columns[np.unique(np.linspace(0,len(candidate_columns)-1,128,dtype=int))]
    row_lookup={int(apps[r]):i for i,r in enumerate(rows)}
    row_matrix=np.zeros((len(rows),len(apps)),dtype=np.float64)
    row_matrix[np.arange(len(rows)),rows]=support[rows]
    source=V2/'knn/raw_cooccurrence.parquet'
    if digest(source)!=receipt['recovery_validation']['reference_raw_cooccurrence_sha256']:
        raise ValueError('EASE co-occurrence verification source changed')
    for batch in pq.ParquetFile(source).iter_batches(batch_size=100000,
            columns=['candidate_app_id','neighbor_app_id','cooccurrence_count']):
        a=batch[0].to_numpy();keep=np.isin(a,apps[rows])
        rr=np.array([row_lookup[int(x)] for x in a[keep]],dtype=int)
        cc=np.searchsorted(apps,batch[1].to_numpy()[keep])
        row_matrix[rr,cc]=batch[2].to_numpy()[keep]
    results={}
    for lam in [100,1000]:
        folder=ROOT/'models'/f'ease_lambda{lam}_exact'
        m=json.loads((folder/'manifest.json').read_text('utf-8'))
        for rel,info in m['files'].items():
            if digest(folder/rel)!=info['sha256']:raise ValueError('EASE model changed: '+rel)
        matrix=sparse.load_npz(folder/'similarities_csr.npz')
        if matrix.shape!=(len(candidate_columns),len(apps)) or not np.all(np.isfinite(matrix.data)):
            raise ValueError('Invalid EASE matrix')
        diagonal=np.asarray(matrix[np.arange(len(candidate_columns)),candidate_columns]).ravel()
        if np.any(diagonal!=0):raise ValueError('EASE has self coefficients')
        checks=[sampled_residual(None,matrix[int(i)].toarray().ravel(),int(candidate_columns[i]),lam,rows,row_matrix) for i in selected]
        if max(c['relative_l2_residual'] for c in checks)>1e-3:raise ValueError('EASE sampled normal-equation residual exceeds fixed float32 tolerance')
        results[str(lam)]={'manifest_sha256':digest(folder/'manifest.json'),'candidate_columns_checked':len(checks),
            'equation_rows_per_candidate':len(rows),'self_coefficients_nonzero':int(np.count_nonzero(diagonal)),
            'max_relative_l2_residual':max(c['relative_l2_residual'] for c in checks),'checks':checks}
        del matrix
    result={'status':'passed','code_sha256':digest(__file__),'gram_sha256':receipt['sha256'],
        'scope':'Sampled Linux-row off-diagonal normal equations (G+lambda I)b=G[:,target] outside target; all self coefficients checked. Candidate rows from frozen raw co-occurrences, previously independently matched to the full Gram before builder temporary-file cleanup.',
        'relative_l2_tolerance':1e-3,'models':results}
    write_json(ROOT/'ease_solution_audit.json',result);return result


if __name__=='__main__':
    with threadpool_limits(limits=2):print(json.dumps(run(),indent=2))
