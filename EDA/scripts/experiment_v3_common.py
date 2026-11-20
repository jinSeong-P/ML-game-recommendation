"""Versioned offline recommendation experiments; v1/v2 artifacts stay immutable.

Public terminology: validation, test reference, Recall@10, NDCG@10,
item-based collaborative filtering, content-based filtering and rank fusion.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize

from eda_large_review_frank import EDA, digest, log, quote
from evaluate_baselines import load_cohort, rank_positive, inventory
from improvement_common import protective_summary, write_json

ROOT = EDA / 'data/processed/modeling/improvement_v3'
SPLIT = EDA / 'data/processed/modeling/split_v1'
PROTOCOL = EDA / 'data/reference/improvement_protocol_v3.json'
V2 = EDA / 'data/processed/modeling/improvement_v2'


def catalog():
    d = pq.read_table(EDA / 'data/processed/modeling/item_knn_v1/game_rows.parquet').to_pydict()
    return np.asarray(d['app_id']), np.asarray(d['linux']), np.asarray(d['positive_user_support'])


def frozen_cohort(name='validation'):
    apps, linux, _ = catalog()
    protocol = json.loads((EDA / 'data/reference/improvement_protocol_v2.json').read_text('utf-8'))
    c = load_cohort(SPLIT, name, apps, apps[linux], protocol)
    q = pq.read_table(SPLIT / name / 'query.parquet').to_pydict()
    rows = np.searchsorted(c['user_ids'], np.asarray(q['user_key'], dtype=np.uint64))
    cols = np.searchsorted(apps, q['app_id'])
    neg = ~np.asarray(q['positive'])
    c['negative'] = sparse.csr_matrix((np.ones(int(neg.sum()), dtype=np.float32),
        (rows[neg], cols[neg])), shape=c['positive'].shape)
    c['name'] = name
    return c


def model_files(folder):
    folder = Path(folder)
    return {'directory': str(folder.relative_to(EDA).as_posix()),
            'manifest_sha256': digest(folder / 'manifest.json')}


class Scorer:
    """A train-fitted or catalog-fitted item representation, plus query rules."""
    def __init__(self, matrix, mode='knn', spec=None):
        self.matrix = matrix
        self.mode = mode
        self.spec = spec or {}
        self.apps, self.linux, self.support = catalog()

    def score(self, q, negative=None):
        n = np.maximum(np.asarray(q.sum(axis=1)).ravel(), 1)
        if self.mode == 'knn':
            result = q @ self.matrix.T
            return (result.toarray() if sparse.issparse(result) else np.asarray(result)) / n[:, None]
        weights = q.multiply((1.0 / n.astype(np.float32))[:, None]).tocsr()
        profile = normalize(weights @ self.matrix, norm='l2', copy=False)
        result = profile @ self.matrix[self.linux].T
        return result.toarray() if sparse.issparse(result) else np.asarray(result)


class CombinedScorer:
    def __init__(self, children, weights, spec=None):
        self.children, self.weights = children, weights
        self.spec = spec or {}

    def score(self, q, negative=None):
        return sum(w * child.score(q, negative) for child, w in zip(self.children, self.weights))


class NegativeScorer:
    def __init__(self, base, penalty, spec=None):
        self.base, self.penalty, self.spec = base, penalty, spec or {}

    def score(self, q, negative=None):
        scores = self.base.score(q)
        if negative is not None and negative.nnz:
            scores -= self.penalty * self.base.score(negative)
        return scores


class RankFusionScorer:
    def __init__(self, left, right, content_weight, rank_constant=60, adaptive=False, spec=None):
        self.left, self.right, self.weight, self.constant = left, right, content_weight, rank_constant
        self.adaptive, self.spec = adaptive, spec or {}
        self.apps, self.linux, self.support = catalog()

    def score(self, q, negative=None):
        # Reciprocal rank fusion uses all positive-score candidates. Masking of
        # already-observed items occurs before ranks are assigned in the runner.
        return self.fuse(self.left.score(q, negative), self.right.score(q, negative))

    def fuse(self, a, b, exclusions=None):
        if exclusions is not None:
            r, c = exclusions.nonzero()
            a[r, c] = 0
            b[r, c] = 0
        result = np.zeros_like(a, dtype=np.float64)
        weight = self.weight
        if self.adaptive:
            weight = self.weight / (1 + self.support[self.linux] / 1000.)
        for values, w in ((a, 1 - weight), (b, weight)):
            order = np.argsort(-values, axis=1, kind='stable')
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, values.shape[1]+1), order.shape), axis=1)
            result += (values > 0) * w / (self.constant + ranks)
        return result


def load_matrix(folder, mode='knn'):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Incomplete model: ' + str(folder))
    for name, info in manifest.get('files', {}).items():
        if not (folder / name).is_file() or digest(folder / name) != info['sha256']:
            raise ValueError('Model hash differs: ' + name)
    paths = ([folder / 'similarities_csr.npz', folder / 'weights_csr.npz'] if mode == 'knn'
             else [folder / 'tfidf_csr.npz', folder / 'features_csr.npz'])
    found = next((p for p in paths if p.exists()), None)
    if found:
        matrix = sparse.load_npz(found)
    else:
        dense = next((p for p in [folder / 'embeddings.npy', folder / 'vectors.npy', folder / 'features.npy'] if p.exists()), None)
        if dense is None:
            raise FileNotFoundError('No score matrix in ' + str(folder))
        matrix = np.load(dense, mmap_mode='r')
    return Scorer(matrix, mode, model_files(folder))


def current_models():
    return {
        'item_knn_control': load_matrix(EDA / 'data/processed/modeling/item_knn_v1'),
        'content_control': CombinedScorer([
            load_matrix(V2 / 'tfidf/T2_unigram_bigram', 'content'),
            load_matrix(V2 / 'tfidf/metadata_channel', 'content')], [.5, .5],
            {'description': model_files(V2 / 'tfidf/T2_unigram_bigram'),
             'metadata': model_files(V2 / 'tfidf/metadata_channel'), 'description_weight': .5})}


def subset_cohort(cohort, indices):
    return {k: (v[indices] if k in ['user_ids','positive','negative','n_positive','n_total',
        'exclusions','target_ids','target_candidates'] or sparse.issparse(v) else v)
        for k, v in cohort.items()}


def evaluate(name, scorer, cohort, baseline=None, batch_size=256, query_weights=None,
             popularity_fill=False, score_transform=None, family='collaborative', allow_reference=False):
    """Save every user and recommendation, then audit the saved files with SQL.

    Scores need not be probabilities or bounded by one (EASE, ALS, rank fusion).
    All scores must be finite, and retained scores strictly positive.
    """
    protocol = json.loads(PROTOCOL.read_text('utf-8'))
    if protocol.get('status') != 'fixed_before_v3_metrics':
        raise ValueError('Experiment protocol must be frozen first')
    if cohort['name'] == 'test' and not allow_reference:
        raise ValueError('Reference evaluation requires a validation-frozen selection')
    apps, linux, all_support = catalog()
    candidates, support = apps[linux], all_support[linux]
    out = ROOT / 'evaluations' / cohort['name'] / name
    provenance = {'spec': scorer.spec, 'protocol_sha256': digest(PROTOCOL),
                  'evaluator_sha256': digest(__file__), 'query_weighted': query_weights is not None,
                  'popularity_fill': popularity_fill,
                  'cohort_hashes': cohort.get('hashes', {}),
                  'cohort_users': len(cohort['user_ids'])}
    if out.exists():
        m = json.loads((out / 'manifest.json').read_text('utf-8'))
        if m['provenance'] != provenance:
            raise ValueError('Existing evaluation source changed: ' + name)
        for path, info in m['files'].items():
            if digest(out / path) != info['sha256']:
                raise ValueError('Existing evaluation file changed: ' + path)
        return m
    out.mkdir(parents=True)
    start_time = time.perf_counter()
    n = len(cohort['user_ids'])
    ranks, lengths = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    target_scores = np.zeros(n, dtype=np.float64)
    fill_users, fill_slots, fill_hits = 0, 0, 0
    schema = pa.schema([('user_key',pa.uint64()),('app_id',pa.int64()),('rank',pa.int8()),('score',pa.float64())])
    pop_order = np.lexsort((candidates, -support))
    with pq.ParquetWriter(out / 'recommendations.parquet', schema, compression='zstd') as writer:
        for begin in range(0, n, batch_size):
            end = min(begin + batch_size, n)
            query = (query_weights if query_weights is not None else cohort['positive'])[begin:end]
            negative = cohort['negative'][begin:end]
            mask = cohort['exclusions'][begin:end]
            if isinstance(scorer, RankFusionScorer):
                scores = scorer.fuse(scorer.left.score(query,negative), scorer.right.score(query,negative), mask)
            else:
                scores = scorer.score(query, negative)
            if score_transform is not None:
                scores = score_transform(scores, query, negative, begin, end)
            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != (end-begin,len(candidates)) or not np.all(np.isfinite(scores)):
                raise ValueError('Invalid score dimensions or nonfinite values')
            mr, mc = mask.nonzero()
            scores[mr,mc] = 0
            target_scores[begin:end] = scores[np.arange(end-begin),cohort['target_candidates'][begin:end]]
            records = []
            for local, row in enumerate(range(begin,end)):
                chosen = rank_positive(scores[local],candidates)
                if popularity_fill and len(chosen) < 10:
                    excluded = set(mask[local].indices) | set(chosen)
                    added = [j for j in pop_order if j not in excluded][:10-len(chosen)]
                    if added:
                        fill_users += 1
                        fill_slots += len(added)
                        fill_hits += int(cohort['target_candidates'][row] in added)
                        # Preserve personalized ranking, then popularity. Store a
                        # monotone rank score; raw target score remains separate.
                        chosen = np.r_[chosen,added]
                lengths[row] = len(chosen)
                hit = np.flatnonzero(candidates[chosen] == cohort['target_ids'][row])
                if len(hit): ranks[row] = hit[0] + 1
                for rank, idx in enumerate(chosen,1):
                    value = 1. / rank if popularity_fill else float(scores[local,idx])
                    records.append((int(cohort['user_ids'][row]),int(candidates[idx]),rank,value))
            writer.write_table(pa.Table.from_arrays([
                pa.array([r[col] for r in records],type=field.type) for col,field in enumerate(schema)],schema=schema))
            if begin == 0 or end == n or begin % (batch_size * 250) == 0:
                log(f'{name} {cohort["name"]}: {end:,}/{n:,}')
    outcomes = {'user_key':cohort['user_ids'],'target_app_id':cohort['target_ids'],
        'n_positive_inputs':cohort['n_positive'],'n_total_inputs':cohort['n_total'],
        'target_train_support':support[cohort['target_candidates']],
        'target_rank':ranks,'recommendation_count':lengths,'target_score':target_scores}
    pq.write_table(pa.table(outcomes),out / 'per_user.parquet',compression='zstd')
    protection = protective_summary(out / 'per_user.parquet',out / 'recommendations.parquet',baseline)
    violations = audit_saved(out,cohort,candidates)
    if any(violations.values()):
        raise ValueError('Saved prediction violations: ' + str(violations))
    result = {'status':'complete','variant':name,'family':family,'cohort':cohort['name'],
        'elapsed_seconds':time.perf_counter()-start_time,'provenance':provenance,
        'protection':protection,'violations':violations,
        'popularity_fill':{'users':fill_users,'slots':fill_slots,'target_hits':fill_hits},
        'files':inventory(out)}
    write_json(out / 'manifest.json',result)
    return result


def audit_saved(folder,cohort,candidates):
    con = duckdb.connect()
    try:
        con.execute('SET threads=2')
        con.register('users',pa.table({'user_key':cohort['user_ids']}))
        con.register('candidates',pa.table({'app_id':candidates}))
        mask = cohort['exclusions'].tocoo()
        con.register('excluded',pa.table({'user_key':cohort['user_ids'][mask.row],'app_id':candidates[mask.col]}))
        con.execute(f'CREATE VIEW recs AS SELECT * FROM read_parquet({quote(folder/"recommendations.parquet")})')
        con.execute(f'CREATE VIEW outcomes AS SELECT * FROM read_parquet({quote(folder/"per_user.parquet")})')
        checks = {
            'nonpositive_or_invalid_scores':'SELECT count(*) FROM recs WHERE NOT isfinite(score) OR score<=0',
            'duplicate_user_game':'SELECT count(*) FROM (SELECT user_key,app_id FROM recs GROUP BY ALL HAVING count(*)<>1)',
            'invalid_ranks':'SELECT count(*) FROM (SELECT user_key FROM recs GROUP BY user_key HAVING min(rank)<>1 OR max(rank)<>count(*) OR count(DISTINCT rank)<>count(*) OR count(*)>10)',
            'noncandidate':'SELECT count(*) FROM recs ANTI JOIN candidates USING(app_id)',
            'unknown_users':'SELECT count(*) FROM recs ANTI JOIN users USING(user_key)',
            'observed_games':'SELECT count(*) FROM recs JOIN excluded USING(user_key,app_id)',
            'missing_users':'SELECT count(*) FROM users ANTI JOIN outcomes USING(user_key)',
            'duplicate_outcomes':'SELECT count(*) FROM (SELECT user_key FROM outcomes GROUP BY user_key HAVING count(*)<>1)',
            'wrong_target_rank':'SELECT count(*) FROM outcomes o LEFT JOIN recs r ON o.user_key=r.user_key AND o.target_app_id=r.app_id WHERE o.target_rank<>coalesce(r.rank,0)',
            'wrong_list_count':'SELECT count(*) FROM outcomes o LEFT JOIN (SELECT user_key,count(*) n FROM recs GROUP BY user_key) r USING(user_key) WHERE o.recommendation_count<>coalesce(r.n,0)',
            'out_of_order':'SELECT count(*) FROM (SELECT *,lag(score) OVER(PARTITION BY user_key ORDER BY rank) s,lag(app_id) OVER(PARTITION BY user_key ORDER BY rank) a FROM recs) WHERE score>s OR(score=s AND app_id<a)'}
        return {key:int(con.execute(sql).fetchone()[0]) for key,sql in checks.items()}
    finally:
        con.close()
