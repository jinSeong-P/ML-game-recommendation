"""Reuse verified alpha=0 solves for DAN without repeating a factorization.

With beta=0 the lambda scale ratio is exactly one. EASE supplies the p=0
base; a completed alpha=0 DAN fit supplies each other beta/p/scale base.
Popularity powers are deterministic row/column multipliers. Nonzero-beta
fits preserve their lambda scale mode. No user labels are loaded.
"""
from __future__ import annotations
import json
import shutil
import numpy as np
from eda_large_review_frank import digest
import experiment_linear_v4 as linear


def rescale_rows(base, support, candidate_cols, alpha):
    left, right = linear._dan_popularity_factors(support, alpha)
    result = np.asarray(base, dtype=np.float64) * left[None, :]
    result *= right[candidate_cols, None]
    result = result.astype(np.float32)
    result[np.arange(len(candidate_cols)), candidate_cols] = 0
    return result


def derive(source_name, drop_p):
    folder = linear.MODEL_ROOT / source_name
    receipt_path = folder / 'manifest.json'
    if not receipt_path.is_file():
        return []
    source = json.loads(receipt_path.read_text('utf-8'))
    weights_path = folder / 'weights.npy'
    if source.get('status') != 'complete':
        raise ValueError('DAN reuse source is incomplete or changed')
    for file, receipt in source['files'].items():
        if digest(folder/file) != receipt['sha256']:
            raise ValueError('DAN reuse source file changed')
    parameters = source['parameters']
    beta = parameters.get('beta', 0.)
    if parameters['regularization'] != 100 or parameters.get('drop_p', 0) != drop_p:
        raise ValueError('DAN reuse must preserve the exact Gram and diagonal penalty')
    if parameters.get('alpha', 0) != 0:
        raise ValueError('DAN reuse requires an unscaled alpha=0 base')
    sources = linear._source_manifest()
    if source['sources'] != sources:
        raise ValueError('DAN reuse source belongs to a different training split')
    apps, candidates = linear._catalog()
    support = np.load(folder / 'support.npy')
    candidate_cols = np.searchsorted(apps, candidates)
    made = []
    scales = ['raw', 'scale_matched'] if beta == 0 else [parameters['lambda_scale_mode']]
    for alpha in [0., .1, .2, .3]:
        for scale in scales:
            name, params = linear._linear_names('dan', 100, alpha=alpha, beta=beta,
                drop_p=drop_p, scale_mode=scale)[0]
            stage = linear._new_stage(name, params, sources)
            if stage is None:
                continue
            output = np.lib.format.open_memmap(stage / 'weights.npy', mode='w+', dtype=np.float32,
                                               shape=(len(candidates), len(apps)))
            linear._close_memmap(output)
            # Close mappings per block so this derivation does not retain two
            # additional 1GB resident arrays beside another active exact solve.
            for begin in range(0, len(candidates), 64):
                end = min(begin + 64, len(candidates))
                original = np.load(weights_path, mmap_mode='r')
                block = np.array(original[begin:end], dtype=np.float32, copy=True)
                linear._close_memmap(original)
                block = rescale_rows(block, support, candidate_cols[begin:end], alpha)
                output = np.load(stage / 'weights.npy', mmap_mode='r+')
                output[begin:end] = block
                linear._close_memmap(output)
            shutil.copy2(__file__, stage / 'reuse_builder_snapshot.py')
            summary = dict(source['fit_summary'], beta=float(beta), drop_p=drop_p, alpha=alpha,
                regularization=100,
                factorization_reused=True, coefficient_derivation='DAN input/candidate popularity powers',
                reuse_source_manifest_sha256=digest(receipt_path),
                reuse_source_weights_sha256=digest(weights_path), reuse_builder_sha256=digest(__file__))
            linear._finish_artifact(stage, name, 'dan_ease', 'dense_candidate_weights', 'linear',
                                    params, sources, support, apps, candidates, summary)
            made.append(name)
    return made


if __name__ == '__main__':
    for name in derive('ease_lambda100_exact', 0.):
        print(name, flush=True)
    for path in sorted(linear.MODEL_ROOT.glob('dan_ease_lambda100_a000_*/manifest.json')):
        m=json.loads(path.read_text('utf-8'))
        for name in derive(path.parent.name, m['parameters']['drop_p']):
            print(name, flush=True)
