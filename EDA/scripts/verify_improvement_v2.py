"""Independent SQL re-ranking of raw KNN links and v2 artifact/hash checks."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from eda_large_review_frank import EDA,digest,quote
from improvement_common import ROOT,PROTOCOL,write_json

def run():
    base=ROOT/'knn'
    mapping=pq.read_table(base/'k100_l0/game_rows.parquet').to_pydict()
    apps=np.asarray(mapping['app_id']);candidates=apps[np.asarray(mapping['linux'])]
    samples=candidates[np.linspace(0,len(candidates)-1,7,dtype=int)]
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
    raw=base/'raw_cooccurrence.parquet'
    con.execute(f"CREATE TEMP TABLE links AS SELECT * FROM read_parquet({quote(raw)}) WHERE candidate_app_id IN ({','.join(map(str,samples))})")
    checked=0;variants={}
    try:
        for folder in sorted(base.glob('k*_l*')):
            m=json.loads((folder/'manifest.json').read_text('utf-8'))
            for filename,info in m['files'].items():
                if digest(folder/filename)!=info['sha256']:raise ValueError('Changed artifact '+str(folder/filename))
            if m['sources']['split_manifest_sha256']!=json.loads(PROTOCOL.read_text('utf-8'))['split_manifest_sha256']:
                raise ValueError('Changed fit split')
            mat=sparse.load_npz(folder/'similarities_csr.npz')
            k=m['parameters']['neighbors'];lam=m['parameters']['shrinkage_lambda']
            if np.diff(mat.indptr).max()>k or np.any(mat.data<=0) or np.any(mat.data>1):raise ValueError('Invalid v2 graph')
            edges=0
            for app in samples:
                result=con.execute('''SELECT neighbor_app_id,raw_cosine*(cooccurrence_count::DOUBLE/(cooccurrence_count+?)) score
                    FROM links WHERE candidate_app_id=? ORDER BY score DESC,neighbor_app_id LIMIT ?''',[lam,int(app),k]).fetchall()
                expected_ids=np.array([x[0] for x in result],dtype=np.int64)
                expected_scores=np.array([x[1] for x in result],dtype=np.float64)
                row=mat[np.searchsorted(candidates,app)]
                order=np.lexsort((apps[row.indices],-row.data))
                np.testing.assert_array_equal(apps[row.indices][order],expected_ids)
                np.testing.assert_allclose(row.data[order],expected_scores,rtol=1e-14,atol=1e-15)
                edges+=len(result);checked+=1
            variants[folder.name]={'sample_rows':len(samples),'sample_edges':edges,'nnz':mat.nnz,'passed':True}
        old=sparse.load_npz(EDA/'data/processed/modeling/item_knn_v1/similarities_csr.npz')
        control=sparse.load_npz(base/'k100_l0/similarities_csr.npz')
        exact=all(np.array_equal(getattr(old,a),getattr(control,a)) for a in ['indptr','indices','data'])
        if not exact:raise ValueError('v1 CSR control changed')
    finally:con.close()
    result={'passed':True,'code_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),
       'method':'Independent DuckDB raw-edge shrinkage and ORDER BY; no reuse of builder ranking function',
       'sample_candidate_app_ids':samples.tolist(),'candidate_model_rows_checked':checked,
       'v1_control_exact_array_equivalence':exact,'variants':variants}
    write_json(EDA/'reports/improvement_v2_artifact_audit.json',result)
    return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
