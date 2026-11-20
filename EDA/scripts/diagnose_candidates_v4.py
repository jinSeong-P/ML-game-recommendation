"""Report retrieval ceilings and verify the zero-residual ranking control."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from eda_large_review_frank import EDA, digest
from experiment_v4_common import ROOT, catalog, load_cohort, support
from improvement_common import macro, write_json
from run_candidates_v4 import POLICIES


def run():
    cohort = load_cohort('selection')
    folder = ROOT / 'candidates/selection'
    receipt = json.loads((folder / 'manifest.json').read_text('utf-8'))
    for name, sha in receipt['files'].items():
        if digest(folder / name) != sha:
            raise ValueError('Candidate diagnostic source changed')
    arrays = np.load(folder / 'diagnostics.npz')
    hits, counts = arrays['hits'], arrays['counts']
    apps, _ = catalog()
    target_support = support()[np.searchsorted(apps, cohort['target_ids'])]
    main = cohort['n_total'] >= 5
    groups = {'primary': main, 'history1-2': cohort['n_total'] <= 2,
              'history3-4': (cohort['n_total'] >= 3) & (cohort['n_total'] <= 4),
              'primary_support1-999': main & (target_support > 0) & (target_support < 1000)}
    for lo, hi in [(0, 1), (1, 10), (10, 100), (100, 1000),
                   (1000, 10000), (10000, 100000), (100000, np.inf)]:
        groups[f'primary_support{lo}-{hi}'] = main & (target_support >= lo) & (target_support < hi)
    ceilings = {}
    for j, name in enumerate(['cf_top100', 'content_top100', *POLICIES]):
        ceilings[name] = {}
        for label, mask in groups.items():
            n = int(mask.sum())
            record = dict(users=n, target_games=int(len(np.unique(cohort['target_ids'][mask]))),
                          recall=float(hits[mask, j].mean()) if n else None,
                          game_macro_recall=macro(hits[mask, j], cohort['target_ids'][mask]) if n else None)
            if j >= 2 and n:
                record['actual_candidates'] = dict(zip(['min', 'median', 'max'],
                    np.quantile(counts[mask, j-2], [0, .5, 1]).tolist()))
                record['actual_candidates']['mean'] = float(counts[mask, j-2].mean())
            ceilings[name][label] = record
    evaluation = ROOT / 'evaluations/selection'
    names = ['activity_inverse_degree_cosine_k100', 'metadata_extended_tfidf']
    ranks = []
    for name in names:
        table = pq.read_table(evaluation / name / 'per_user.parquet')
        if not np.array_equal(table['user_key'].to_numpy(), cohort['user_ids']):
            raise ValueError('Evaluation account order differs')
        if not np.array_equal(table['target_app_id'].to_numpy(), cohort['target_ids']):
            raise ValueError('Evaluation targets differ')
        ranks.append(table['target_rank'].to_numpy())
    overlaps = {}
    for label, mask in groups.items():
        a, b = (rank[mask] > 0 for rank in ranks)
        overlaps[label] = dict(users=int(mask.sum()), both=int((a & b).sum()),
                              cf_only=int((a & ~b).sum()), content_only=int((~a & b).sum()),
                              neither=int((~a & ~b).sum()))
    columns = ['user_key', 'app_id', 'rank']
    baseline = pq.read_table(evaluation / names[0] / 'recommendations.parquet', columns=columns)
    identity = {}
    for policy in POLICIES:
        candidate = pq.read_table(evaluation / f'residual_fixed_eta0_{policy}' /
                                  'recommendations.parquet', columns=columns)
        same = candidate.equals(baseline)
        if not same:
            raise ValueError(f'Zero residual changed actual recommendation ranks: {policy}')
        identity[policy] = dict(all_user_app_rank_rows_identical=True, rows=baseline.num_rows)
    result = dict(status='complete', candidate_manifest_sha256=digest(folder / 'manifest.json'),
                  code_sha256=digest(__file__), ceilings=ceilings, hit_overlap=overlaps,
                  zero_residual_identity=identity,
                  interpretation='Candidate recall is a retrieval ceiling; no target was injected. '
                  'Hit overlap is diagnostic and is never used as an inference feature.')
    write_json(EDA / 'reports/candidate_diagnostics_v4.json', result)
    return result


if __name__ == '__main__':
    run()
