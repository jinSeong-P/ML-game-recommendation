"""Fresh RRF and LambdaMART comparisons for the v4 fusion experiments.

Historical fitted rerankers cannot be reused: reserved v4 accounts influenced
their old fits. These models use only fusion_fit labels and fixed settings.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import digest
from experiment_v4_common import (ROOT, PROTOCOL, MatrixScorer, ContentScorer,
                                  catalog, support, load_cohort, evaluate)
from improvement_common import write_json
from run_candidates_v4 import FEATURES, POLICIES


PARAMETERS = dict(num_leaves=15, n_estimators=300, learning_rate=.05,
                  random_state=20261108, n_jobs=2, verbosity=-1,
                  objective='lambdarank', label_gain=[0, 1],
                  lambdarank_truncation_level=10, deterministic=True,
                  force_col_wise=True, min_child_samples=50)
FOLDER = ROOT / 'fusion_comparisons'


def freeze_plan():
    plan = dict(status='fixed_before_fusion_comparison_outcomes',
                parent_protocol_sha256=digest(PROTOCOL),
                feature_columns=FEATURES, parameters=PARAMETERS,
                iterations=[100, 300], policies=POLICIES,
                rrf=dict(content_weight=.5, constant=60, adaptive=[False, True],
                         adaptive_weight='.5/(1+fresh_candidate_support/1000)'),
                fitting='fusion_fit only; no selection or confirmation labels in fitting',
                weak_labels='unobserved candidate label0 is not an explicit dislike',
                prior_results='First 51 v4 conditions were already inspected; comparison settings '
                              'come from the backlog and fixed v3 parameters, not tuning on these outcomes.',
                candidate_sources={role: digest(ROOT / 'candidates' / role / 'manifest.json')
                                   for role in ['fusion_fit', 'selection']},
                score_transform='exp(prediction - per-user maximum) within retrieved candidates',
                code_sha256=digest(__file__))
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / 'protocol.json'
    if path.exists() and json.loads(path.read_text('utf-8')) != plan:
        raise ValueError('Frozen fusion comparison plan changed')
    write_json(path, plan)
    return path


class RrfScorer:
    def __init__(self, adaptive=False):
        matrix_path = ROOT / 'models/activity_inverse_degree_cosine_k100/similarities_csr.npz'
        manifest = matrix_path.with_name('manifest.json')
        if json.loads(manifest.read_text('utf-8'))['matrix_sha256'] != digest(matrix_path):
            raise ValueError('RRF source matrix changed')
        self.cf = MatrixScorer(sparse.load_npz(matrix_path))
        self.content = ContentScorer()
        self.adaptive = adaptive
        apps, linux = catalog(); self.ids = apps[linux]; self.sup = support()[linux]
        self.spec = dict(method='RRF_fresh_v4', content_weight=.5, constant=60,
                         adaptive=adaptive, cf_manifest_sha256=digest(manifest),
                         content=self.content.spec, support_sha256=digest(ROOT / 'models/positive_support.npy'))

    def score(self, q):
        # Both component ranks exclude visible games before fusion. The common
        # evaluator applies the same exclusions again to the resulting scores.
        raise RuntimeError('Use the cohort-bound scorer to mask before component ranks')


class MaskedRrf(RrfScorer):
    def __init__(self, cohort, adaptive=False):
        super().__init__(adaptive); self.cohort = cohort; self.cursor = 0

    def score(self, q):
        begin, end = self.cursor, self.cursor + q.shape[0]; self.cursor = end
        a, b = np.asarray(self.cf.score(q), dtype=float), np.asarray(self.content.score(q), dtype=float)
        r, c = self.cohort['exclusions'][begin:end].nonzero(); a[r, c] = 0; b[r, c] = 0
        result = np.zeros_like(a)
        weight = .5 / (1 + self.sup / 1000) if self.adaptive else .5
        for i in range(q.shape[0]):
            for values, factor in [(a[i], 1-weight), (b[i], weight)]:
                valid = np.flatnonzero(values > 0)
                order = valid[np.lexsort((self.ids[valid], -values[valid]))]
                gain = 1 / (60 + np.arange(1, len(order) + 1))
                result[i, order] += gain * (factor[order] if np.ndim(factor) else factor)
        return result


def read_features(role):
    folder = ROOT / 'candidates' / role
    m = json.loads((folder / 'manifest.json').read_text('utf-8'))
    if digest(folder / 'features.parquet') != m['files']['features.parquet']:
        raise ValueError('Cached fusion feature rows changed')
    table = pq.read_table(folder / 'features.parquet')
    return (table['user_row'].to_numpy(), table['candidate_col'].to_numpy(),
            table['label'].to_numpy(),
            np.column_stack([table[name].to_numpy() for name in FEATURES]),
            table['policy_mask'].to_numpy())


def positive_group_scores(values, rows):
    out = np.empty(len(values), dtype=np.float64)
    if not len(values):
        return out
    bounds = np.r_[0, np.flatnonzero(np.diff(rows)) + 1, len(rows)]
    for begin, end in zip(bounds[:-1], bounds[1:]):
        v = values[begin:end]
        out[begin:end] = np.exp(np.clip(v - v.max(), -700, 0))
    return out


class CachedRanker:
    def __init__(self, rows, cols, values, spec):
        self.rows, self.cols, self.values, self.spec = rows, cols, values, spec
        self.cursor = 0

    def score(self, q):
        begin, end = self.cursor, self.cursor + q.shape[0]; self.cursor = end
        left, right = np.searchsorted(self.rows, [begin, end])
        out = np.zeros((q.shape[0], 6630))
        out[self.rows[left:right] - begin, self.cols[left:right]] = self.values[left:right]
        return out


def run(stage):
    plan = freeze_plan()
    if stage in ['fit', 'all']:
        rows, _, labels, x, masks = read_features('fusion_fit')
        for policy, name in enumerate(POLICIES):
            folder = FOLDER / name; folder.mkdir(exist_ok=True)
            model_path = folder / 'model.txt'; receipt_path = folder / 'manifest.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text('utf-8'))
                if receipt['plan_sha256'] != digest(plan) or receipt['model_sha256'] != digest(model_path):
                    raise ValueError('Fitted comparison changed')
                continue
            keep = (masks & (1 << policy)) > 0
            groups = np.bincount(rows[keep]); groups = groups[groups > 0]
            model = lgb.LGBMRanker(**PARAMETERS)
            model.fit(x[keep], labels[keep], group=groups)
            model.booster_.save_model(str(model_path))
            write_json(receipt_path, dict(status='complete', plan_sha256=digest(plan),
                model_sha256=digest(model_path), fit_users=len(groups),
                retrieved_targets=int(labels[keep].sum()), fit_rows=int(keep.sum())))
    if stage in ['evaluate', 'all']:
        cohort = load_cohort('selection')
        for adaptive in [False, True]:
            evaluate('rrf_refit_adaptive_weight0_5' if adaptive else 'rrf_refit_weight0_5',
                     MaskedRrf(cohort, adaptive), cohort)
        rows, cols, _, x, masks = read_features('selection')
        for policy, name in enumerate(POLICIES):
            folder = FOLDER / name
            receipt = json.loads((folder / 'manifest.json').read_text('utf-8'))
            if receipt['plan_sha256'] != digest(plan) or receipt['model_sha256'] != digest(folder / 'model.txt'):
                raise ValueError('Fitted comparison changed before scoring')
            model = lgb.Booster(model_file=str(folder / 'model.txt'))
            keep = (masks & (1 << policy)) > 0
            for trees in [100, 300]:
                prediction = model.predict(x[keep], num_iteration=trees, num_threads=2)
                values = positive_group_scores(prediction, rows[keep])
                spec = dict(plan_sha256=digest(plan), model_sha256=digest(folder / 'model.txt'),
                            candidate_manifest_sha256=digest(ROOT / 'candidates/selection/manifest.json'),
                            policy=name, trees=trees, features=FEATURES,
                            transform='exp(prediction - per-user maximum)')
                evaluate(f'lambdamart_refit_{trees}_{name}',
                         CachedRanker(rows[keep], cols[keep], values, spec), cohort)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['fit', 'evaluate', 'all'], default='all')
    with threadpool_limits(limits=2):
        run(parser.parse_args().stage)
