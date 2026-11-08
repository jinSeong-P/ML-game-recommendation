"""Evaluate refitted models on selection only; never dispatch confirmation."""
import argparse
import json

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import digest
from experiment_v4_common import ROOT,MatrixScorer,ContentScorer,load_cohort,evaluate
from improvement_common import write_json


class DenseLinearScorer:
    def __init__(self,folder):
        self.folder=folder
        source=folder/'weights.npy'; path=ROOT/'score_layouts'/folder.name/'input_candidate.npy'
        manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
        source_hash=digest(source)
        if manifest.get('files',{}).get('weights.npy',{}).get('sha256')!=source_hash:
            raise ValueError('Dense model hash differs from training receipt')
        receipt=path.with_suffix('.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if not receipt.exists():raise ValueError('Unverified cached dense layout')
            old=json.loads(receipt.read_text('utf-8'))
            if old['source_sha256']!=source_hash or old['layout_sha256']!=digest(path):
                raise ValueError('Cached dense scoring layout changed')
        else:
            original=np.load(source,mmap_mode='r')
            stage=path.with_suffix('.building.npy')
            output=np.lib.format.open_memmap(stage,mode='w+',dtype=np.float32,shape=original.T.shape)
            for begin in range(0,original.shape[1],128):
                output[begin:begin+128]=original[:,begin:begin+128].T
            output.flush();del output;stage.replace(path)
            write_json(receipt,dict(status='complete',source_sha256=source_hash,layout_sha256=digest(path),
                shape=list(original.T.shape),layout='input_by_candidate_row_major'))
        self.weights=np.load(path,mmap_mode='r')
        self.spec=dict(source_manifest_sha256=digest(folder/'manifest.json'),weights_sha256=digest(source),
            layout='input_by_candidate_row_major',layout_sha256=digest(path))
    def score(self,q):
        return q@self.weights/np.maximum(np.asarray(q.sum(axis=1)).ravel(),1)[:,None]


def run(names=None):
    cohort=load_cohort('selection')
    if not names or 'metadata_extended_tfidf' in names:
        evaluate('metadata_extended_tfidf',ContentScorer(),cohort)
    for path in sorted((ROOT/'models').glob('*/manifest.json')):
        name=path.parent.name
        if names and name not in names:continue
        m=json.loads(path.read_text('utf-8'))
        if m.get('status')!='complete':continue
        file=path.parent/'similarities_csr.npz'
        if not file.exists():
            if (path.parent/'weights.npy').exists():evaluate(name,DenseLinearScorer(path.parent),cohort)
            continue
        expected=m.get('matrix_sha256',m.get('files',{}).get(file.name,{}).get('sha256'))
        if expected and digest(file)!=expected:raise ValueError('Model file changed')
        scorer=MatrixScorer(sparse.load_npz(file),spec=dict(model_manifest_sha256=digest(path),matrix_sha256=digest(file)))
        evaluate(name,scorer,cohort)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='*');args=p.parse_args()
    with threadpool_limits(limits=2):run(args.names)
