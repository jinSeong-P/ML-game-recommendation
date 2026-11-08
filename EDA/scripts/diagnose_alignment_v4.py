"""Train-graph diagnostic for weak KNN alignment; no reserved user labels."""
from __future__ import annotations
import json
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from eda_large_review_frank import EDA, digest
from experiment_v4_common import ROOT, catalog
from improvement_common import write_json


def distribution(values):
    return dict(mean=float(np.mean(values)), quantiles=dict(zip(
        ['min', 'p10', 'median', 'p90', 'max'], np.quantile(values, [0, .1, .5, .9, 1]).tolist())))


def run():
    name = 'knn_teacher_qwen3_embedding_0_6b_description_1024_d128_top50'
    folder = ROOT / 'content' / name
    manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
    for file, receipt in manifest['files'].items():
        if digest(folder / file) != receipt['sha256']:
            raise ValueError('Alignment model changed')
    teacher_path = ROOT / 'models/activity_inverse_degree_cosine_k100/similarities_csr.npz'
    teacher = sparse.load_npz(teacher_path)
    apps, linux = catalog(); candidate_rows = np.flatnonzero(linux)
    inp = np.load(folder / 'input_embeddings.npy', mmap_mode='r')
    out = np.load(folder / 'candidate_embeddings.npy', mmap_mode='r')
    nonempty = np.linalg.norm(inp, axis=1) > 0
    rows, cols, targets = [], [], []
    for row, global_row in enumerate(candidate_rows):
        begin, end = teacher.indptr[row:row+2]
        ix, value = teacher.indices[begin:end], teacher.data[begin:end]
        keep = nonempty[ix] & nonempty[global_row]
        ix, value = ix[keep], value[keep]
        chosen = np.lexsort((apps[ix], -value))[:50]
        rows.extend([row] * len(chosen)); cols.extend(ix[chosen]); targets.extend(value[chosen])
    rng = np.random.default_rng(20261108)
    selected = rng.choice(len(rows), size=min(20000, len(rows)), replace=False)
    rows = np.asarray(rows)[selected]; cols = np.asarray(cols)[selected]
    targets = np.asarray(targets, dtype=np.float64)[selected]
    positive = np.einsum('ij,ij->i', inp[cols], out[candidate_rows[rows]]).astype(np.float64)
    train_support = np.load(ROOT / 'models/positive_support.npy')
    allowed_inputs = np.flatnonzero(nonempty & (train_support > 0))
    weak_inputs = rng.choice(allowed_inputs, size=len(rows))
    invalid = (np.asarray(teacher[rows, weak_inputs]).ravel() > 0) | (weak_inputs == candidate_rows[rows])
    while invalid.any():
        weak_inputs[invalid] = rng.choice(allowed_inputs, size=int(invalid.sum()))
        invalid = (np.asarray(teacher[rows, weak_inputs]).ravel() > 0) | (weak_inputs == candidate_rows[rows])
    unlabelled = np.einsum('ij,ij->i', inp[weak_inputs], out[candidate_rows[rows]]).astype(np.float64)
    margin = manifest['config']['weak_unlabelled_sampling']['margin']
    result = dict(status='complete_train_graph_diagnostic', variant=name,
        alignment_manifest_sha256=digest(folder / 'manifest.json'), teacher_sha256=digest(teacher_path),
        sampled_edges=len(rows), sampling_seed=20261108, teacher_similarity=distribution(targets),
        aligned_positive_similarity=distribution(positive), aligned_unlabelled_similarity=distribution(unlabelled),
        teacher_positive_below_weak_upper_margin=float((targets < margin).mean()),
        weak_upper_margin=margin, pairwise_positive_above_unlabelled=float((positive > unlabelled).mean()),
        edge_vs_unlabelled_auc=float(roc_auc_score(np.r_[np.ones(len(rows)), np.zeros(len(rows))],
                                                  np.r_[positive, unlabelled])),
        code_sha256=digest(__file__),
        interpretation='Diagnostic on teacher edges used for fitting, not a held-out recommendation metric. '
        'Stored-edge absence is unlabelled, not dislike. A weak upper-margin loss can allow unlabelled '
        'scores above small positive teacher similarities; small regression loss alone does not establish good ranking.')
    write_json(EDA / 'reports/alignment_diagnostic_v4.json', result)
    return result


if __name__ == '__main__':
    run()
