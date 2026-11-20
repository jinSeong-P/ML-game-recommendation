"""Recover a completed EASE Gram after a native-library crash, without refitting."""
import gc
import json
import numpy as np
import pyarrow.parquet as pq
from eda_large_review_frank import digest
from experiment_v3_common import ROOT,V2,catalog,write_json


def run():
    apps,linux,support=catalog();n=len(apps)
    work=ROOT/'models/_collaborative_work/ease'
    path=work/'full_catalog_gram_f32.dat';receipt=work/'full_catalog_gram_f32.complete.json'
    if path.stat().st_size!=n*n*4:raise ValueError('Incomplete Gram file size')
    gram=np.memmap(path,dtype=np.float32,mode='r',shape=(n,n),order='C')
    if not np.array_equal(np.diag(gram),support.astype(np.float32)):raise ValueError('Gram diagonal differs from frozen train support')
    source=V2/'knn/raw_cooccurrence.parquet'
    manifest=json.loads((V2/'knn/manifest.json').read_text('utf-8'))
    expected=manifest['raw_cooccurrence']['sha256']
    if digest(source)!=expected:raise ValueError('Changed reference co-occurrence')
    row_sums=support.astype(np.float64).copy();checked=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=100000,
            columns=['candidate_app_id','neighbor_app_id','cooccurrence_count']):
        r=np.searchsorted(apps,batch['candidate_app_id'].to_numpy())
        c=np.searchsorted(apps,batch['neighbor_app_id'].to_numpy())
        co=batch['cooccurrence_count'].to_numpy()
        if not np.array_equal(gram[r,c],co.astype(np.float32)):
            raise ValueError('EASE Gram differs from exact frozen candidate co-occurrences')
        np.add.at(row_sums,r,co);checked+=len(co)
    for r in np.flatnonzero(linux):
        if float(np.sum(gram[r],dtype=np.float64))!=row_sums[r]:raise ValueError('Extra/missing candidate co-occurrences')
    chosen=np.unique(np.linspace(0,n-1,16,dtype=int))
    if not np.array_equal(gram[chosen,:],gram[:,chosen].T):raise ValueError('Gram asymmetry')
    del gram;gc.collect()
    info={'complete':True,'shape':[n,n],'train_positive_pairs':int(support.sum()),'sha256':digest(path),
        'recovery_validation':{'all_diagonal_supports_match':True,'exact_candidate_edges_checked':checked,
            'candidate_row_sums_match':True,'symmetry_checked_rows':chosen.tolist(),
            'reference_raw_cooccurrence_sha256':expected,'verification_code_sha256':digest(__file__)},
        'reason':'Native OpenBLAS c0000005 crash occurred after Gram flush and subsequent factor-file creation; verify saved Gram before reuse.'}
    if receipt.exists():
        old=json.loads(receipt.read_text('utf-8'))
        if old['sha256']!=info['sha256']:raise ValueError('Existing Gram receipt differs')
    else:write_json(receipt,info)
    write_json(ROOT/'ease_gram_recovery_audit.json',info)
    return info


if __name__=='__main__':run()
