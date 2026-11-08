"""Selection evaluation for frozen embeddings and train-only KNN alignment."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest
from experiment_v4_common import ROOT,catalog,load_cohort,evaluate,MatrixScorer


class TwoTowerScorer:
    def __init__(self,folder):
        self.input=np.load(folder/'input_embeddings.npy',mmap_mode='r')
        self.output=np.load(folder/'candidate_embeddings.npy',mmap_mode='r')
        self.linux=catalog()[1]
        self.spec=dict(manifest_sha256=digest(folder/'manifest.json'))
    def score(self,q):
        # Same per-user normalization cannot change one channel's ranking.
        return normalize(q@self.input)@self.output[self.linux].T


class Channels:
    def __init__(self,a,b):
        self.a,self.b=a,b;self.spec=dict(description=a.spec,metadata=b.spec,weights=[.5,.5])
    def score(self,q):return .5*self.a.score(q)+.5*self.b.score(q)


def restore(folder):
    m=json.loads((folder/'manifest.json').read_text('utf-8'))
    if m['status']!='complete':raise ValueError('Content model incomplete')
    for name,info in m['files'].items():
        if digest(folder/name)!=info['sha256']:raise ValueError('Content model changed')
    if (folder/'embeddings.npy').exists():
        return MatrixScorer(np.load(folder/'embeddings.npy',mmap_mode='r'),'content',
            spec=dict(manifest_sha256=digest(folder/'manifest.json')))
    return TwoTowerScorer(folder)


def run():
    cohort=load_cohort('selection');folders={p.parent.name:p.parent for p in (ROOT/'content').glob('*/manifest.json')}
    for name,folder in folders.items():evaluate(name,restore(folder),cohort)
    for name,folder in folders.items():
        if '_description_' not in name:continue
        other=name.replace('_description_','_metadata_')
        if other in folders:
            joint=name.replace('_description_','_channels_')
            evaluate(joint,Channels(restore(folder),restore(folders[other])),cohort)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
