"""Open only the single frozen challenger and controls, with model identity checks."""
from __future__ import annotations

import json

import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits

from eda_large_review_frank import EDA, digest
from experiment_v4_common import ROOT, PROTOCOL, SPLIT, ContentScorer, MatrixScorer, load_cohort, evaluate
from evaluate_models_v4 import DenseLinearScorer
from improvement_common import write_json
from summarize_experiments_v4 import check_guards


def validate_lock(lock):
    if lock.get('status') != 'frozen_after_selection_before_confirmation':
        raise ValueError('Primary settings must be frozen before confirmation')
    if lock.get('protocol_sha256') != digest(PROTOCOL):
        raise ValueError('Confirmation protocol changed after selection')
    if lock.get('split_manifest_sha256') != digest(SPLIT / 'manifest.json'):
        raise ValueError('Confirmation split changed after selection')
    if lock.get('execution_plan_sha256') != digest(EDA/'data/reference/primary_execution_plan_v4.json'):
        raise ValueError('Execution schedule changed after selection')
    names = lock.get('confirmation_models', [])
    finalist = lock.get('finalist')
    controls = ['activity_inverse_degree_cosine_k100', 'binary_item_knn_k100', 'metadata_extended_tfidf']
    expected = set(controls + [finalist]) if finalist else set()
    if set(names) != expected or len(names) != len(set(names)):
        raise ValueError('Confirmation must contain one challenger and the fixed controls, or remain closed')


def restore_selected(name, lock):
    selected = ROOT / 'evaluations/selection' / name / 'manifest.json'
    if lock['evaluation_manifests'].get(name) != digest(selected):
        raise ValueError('Selection evaluation changed after freezing')
    manifest = json.loads(selected.read_text('utf-8'))
    folder = ROOT / 'models' / name
    if name == 'metadata_extended_tfidf':
        scorer = ContentScorer()
    elif (folder / 'weights.npy').is_file():
        scorer = DenseLinearScorer(folder)
    elif (folder / 'similarities_csr.npz').is_file():
        source = folder / 'similarities_csr.npz'
        model_manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
        expected = model_manifest.get('matrix_sha256', model_manifest.get('files', {}).get(source.name, {}).get('sha256'))
        if not expected or expected != digest(source):
            raise ValueError('Frozen sparse model no longer matches its training receipt')
        scorer = MatrixScorer(sparse.load_npz(source), spec=dict(
            model_manifest_sha256=digest(folder / 'manifest.json'), matrix_sha256=digest(source)))
    else:
        # Never fall back to a similar variant after opening the confirmation data.
        raise ValueError(f'Confirmation scorer for {name} must be implemented before opening outcomes')
    if scorer.spec != manifest['provenance']['model']:
        raise ValueError('Confirmation model differs from the exact model evaluated during selection')
    return scorer


def run():
    lock_path = ROOT / 'selection_frozen.json'
    if not lock_path.is_file():
        raise ValueError('No frozen selection; confirmation remains closed')
    lock = json.loads(lock_path.read_text('utf-8'))
    validate_lock(lock)
    lock_hash=digest(lock_path)
    report_path=EDA/'reports/confirmation_v4.json'
    if report_path.exists():
        previous=json.loads(report_path.read_text('utf-8'))
        if previous.get('selection_lock_sha256') != lock_hash:
            raise ValueError('Confirmation was already recorded for a different selection lock')
    report = dict(selection_lock_sha256=lock_hash, code_sha256=digest(__file__))
    if not lock['finalist']:
        report.update(status='no_qualifying_challenger_confirmation_preserved', opened=False,
                      adopted='activity_inverse_degree_cosine_k100')
        write_json(EDA / 'reports/confirmation_v4.json', report)
        return report
    # Build and verify every scorer before reading the reserved targets.
    scorers = {name: restore_selected(name, lock) for name in lock['confirmation_models']}
    decision = dict(status='fixed_before_confirmation_outcomes', selection_lock_sha256=digest(lock_path),
                    rules='Same zero-margin guards as selection; failed confirmation retains control. '
                    'No other challenger is tried and no setting is revised.')
    decision_path = ROOT / 'confirmation_decision.json'
    if decision_path.exists() and json.loads(decision_path.read_text('utf-8')) != decision:
        raise ValueError('Confirmation decision rule changed')
    write_json(decision_path, decision)
    cohort = load_cohort('confirmation')
    outcomes = {name: evaluate(name, scorer, cohort, confirmation_lock=lock_path)
                for name, scorer in scorers.items()}
    from audit_experiments_v4 import run as audit
    from experiment_v4_common import paired_comparison
    audit('confirmation')
    folder = ROOT / 'evaluations/confirmation'
    def ranks(name):
        return pq.read_table(folder / name / 'per_user.parquet', columns=['target_rank'])['target_rank'].to_numpy()
    base = 'activity_inverse_degree_cosine_k100'; finalist = lock['finalist']
    paired = paired_comparison(ranks(base), ranks(finalist), cohort)
    guards = check_guards(outcomes[finalist], outcomes[base], paired)
    report.update(status='complete_single_frozen_comparison', opened=True, finalist=finalist,
                  guards=guards, paired=paired, outcomes=outcomes,
                  adopted=finalist if all(guards.values()) else base,
                  limitation='Prospective v4 account-disjoint refit; accounts influenced historical v1-v3 training.')
    write_json(EDA / 'reports/confirmation_v4.json', report)
    return report


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
