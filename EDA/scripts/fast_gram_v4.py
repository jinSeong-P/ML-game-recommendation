"""Avoid repeated CSC-to-CSR conversion in train-only weighted Gram products."""
from __future__ import annotations
import gc
import json
import shutil
import time
import numpy as np
import scipy
from eda_large_review_frank import digest,log
import experiment_linear_v4 as linear

ORIGINAL_BUILDER=linear._build_gram_memmap


def weighted_views(x,beta):
    degree=np.diff(x.indptr).astype(np.float64)
    if np.any(degree<=0) or np.any(x.data!=1):
        raise ValueError('Weighted Gram requires nonempty binary positive user rows')
    weights=np.power(degree,-beta).astype(np.float32)
    csc=x.astype(np.float32).tocsc()
    csc.data=np.sqrt(weights[csc.indices])
    diagonal=linear._weighted_support(csc)
    # CSR @ CSC would convert the entire right operand to CSR on every block.
    # Do that identical conversion once and reuse it for all independent rows.
    csr=csc.tocsr();item_rows=csc.T.tocsr()
    return csr,item_rows,diagonal


def build(x,support,beta,source_hash,work_dir):
    n=x.shape[1];path=linear._gram_cache_path(beta)
    receipt_path=linear._gram_cache_receipt(path)
    if path.is_file() and receipt_path.is_file():
        receipt=json.loads(receipt_path.read_text('utf-8'))
        if (receipt.get('complete') is True and receipt.get('shape')==[n,n]
            and receipt.get('beta')==beta and receipt.get('train_sha256')==source_hash
            and path.stat().st_size==n*n*4 and digest(path)==receipt.get('sha256')):
            return ORIGINAL_BUILDER(x,support,beta,source_hash,work_dir)
    work_dir.mkdir(parents=True,exist_ok=True)
    if path.exists():
        linear._safe_under_model_root(path);path.unlink()
    receipt_path.unlink(missing_ok=True)
    csr,item_rows,diagonal=weighted_views(x,beta)
    linear._require_linear_memory('One-conversion weighted Gram',minimum_gib=5.)
    gram=np.memmap(path,mode='w+',dtype=np.float32,shape=(n,n))
    start=time.perf_counter();block_rows=128
    for begin in range(0,n,block_rows):
        end=min(begin+block_rows,n)
        block=(item_rows[begin:end]@csr).tocsr()
        gram[begin:end]=block.toarray()
        if begin==0 or end==n or begin%(block_rows*25)==0:
            gram.flush()
            log(f'One-conversion Gram beta={beta:g}: {end:,}/{n:,}, {time.perf_counter()-start:.1f}s')
            linear._require_linear_memory('One-conversion weighted Gram',minimum_gib=5.)
        del block
    gram.flush()
    snapshot=work_dir/f'gram_beta{linear._name_num(beta)}_builder_snapshot.py'
    shutil.copy2(__file__,snapshot)
    receipt=dict(complete=True,shape=[n,n],beta=beta,train_sha256=source_hash,
        train_positive_pairs=int(x.nnz),sha256=digest(path),block_rows=block_rows,
        optimization='Identical right-operand CSC-to-CSR conversion performed once',
        builder_snapshot=snapshot.name,builder_sha256=digest(__file__),
        numpy_version=np.__version__,scipy_version=scipy.__version__,
        gram_seconds=time.perf_counter()-start)
    linear._atomic_json(receipt_path,receipt)
    del csr,item_rows;gc.collect()
    return gram,diagonal,path,receipt
