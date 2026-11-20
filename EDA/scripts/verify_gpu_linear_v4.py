"""Independently check an in-place CUDA Cholesky against the completed CPU EASE.

This writes numerical verification artifacts only, not selection evaluations
or replacement model receipts. All inputs are the already audited train Gram.
"""
from __future__ import annotations
import gc
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from eda_large_review_frank import EDA, digest, log
from improvement_common import write_json
from experiment_v4_common import ROOT


def gpu_solve(gram_path, gamma, candidate_cols, output, batch=128):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA environment required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.9)
    n = len(gamma); bytes_matrix = n*n*4
    free, total = torch.cuda.mem_get_info()
    if free < 2*bytes_matrix + n*batch*4*4:
        raise MemoryError('Insufficient dedicated GPU memory for factor and conservative workspace')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'weights.npy'
    if path.exists():
        raise FileExistsError('Preserve the existing GPU verification; do not overwrite it')
    gram = np.memmap(gram_path, mode='r', dtype=np.float32, shape=(n,n))
    start = time.perf_counter()
    factor = torch.empty((n,n), device='cuda', dtype=torch.float32).T
    for begin in range(0,n,256):
        end = min(begin+256,n)
        factor[begin:end].copy_(torch.from_numpy(np.array(gram[begin:end], copy=True)))
    gram._mmap.close(); del gram
    factor.diagonal().add_(torch.as_tensor(gamma, device='cuda', dtype=torch.float32))
    info = torch.empty((), device='cuda', dtype=torch.int32)
    torch.linalg.cholesky_ex(factor, upper=False, check_errors=True, out=(factor, info))
    torch.cuda.synchronize()
    factor_seconds = time.perf_counter()-start
    log(f'CUDA factor complete: {factor_seconds:.2f}s')
    mapped = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                     shape=(len(candidate_cols),n))
    mapped.flush(); mapped._mmap.close(); del mapped
    diagonal = np.zeros(len(candidate_cols), dtype=np.float32)
    sample_rows = [0, len(candidate_cols)//2, len(candidate_cols)-1]
    samples = np.zeros((n,len(sample_rows)), dtype=np.float32)
    for begin in range(0,len(candidate_cols),batch):
        end=min(begin+batch,len(candidate_cols)); cols=candidate_cols[begin:end]
        rhs=torch.zeros((end-begin,n),device='cuda',dtype=torch.float32).T
        rhs[torch.as_tensor(cols,device='cuda'),torch.arange(end-begin,device='cuda')]=1
        torch.linalg.solve_triangular(factor,rhs,upper=False,out=rhs)
        torch.linalg.solve_triangular(factor.T,rhs,upper=True,out=rhs)
        block=rhs.T.contiguous().cpu().numpy()
        for j,row in enumerate(sample_rows):
            if begin<=row<end:samples[:,j]=block[row-begin]
        diag=block[np.arange(end-begin),cols].copy()
        if np.any(~np.isfinite(block)) or np.any(diag<=0):
            raise ValueError('Nonfinite CUDA solve or nonpositive inverse diagonal')
        diagonal[begin:end]=diag
        block=-block/diag[:,None];block[np.arange(end-begin),cols]=0
        mapped=np.load(path,mmap_mode='r+');mapped[begin:end]=block
        mapped.flush();mapped._mmap.close();del mapped,rhs,block
        if begin==0 or end==len(candidate_cols) or begin%1024==0:
            log(f'CUDA inverse columns: {end:,}/{len(candidate_cols):,}')
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    del factor,info;gc.collect();torch.cuda.empty_cache()
    np.save(output/'inverse_diagonal.npy',diagonal)
    np.save(output/'sample_inverse_columns.npy',samples)
    return dict(factor_seconds=factor_seconds,total_seconds=seconds,
                peak_cuda_allocated_bytes=peak,torch=torch.__version__,cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(),tf32=False,sample_rows=sample_rows,
                batch_size=batch,gpu_memory_fraction=.9)


def residual(gram_path,gamma,cols,solution):
    n=len(gamma); gram=np.memmap(gram_path,mode='r',dtype=np.float32,shape=(n,n))
    error=np.zeros(len(cols)); norm_a=0.
    for begin in range(0,n,128):
        end=min(begin+128,n);block=np.asarray(gram[begin:end],dtype=np.float64)
        product=block@solution.astype(np.float64)
        product+=gamma[begin:end,None]*solution[begin:end]
        for j,col in enumerate(cols):
            if begin<=col<end:product[col-begin,j]-=1
        error=np.maximum(error,np.max(np.abs(product),axis=0))
        norm_a=max(norm_a,float((np.abs(block).sum(axis=1)+gamma[begin:end]).max()))
    gram._mmap.close()
    return error/(norm_a*np.max(np.abs(solution),axis=0)+1)


def compare(path,reference):
    a=np.load(path,mmap_mode='r');b=np.load(reference,mmap_mode='r')
    if a.shape!=b.shape:raise ValueError('CPU/GPU coefficient shapes differ')
    square_error=square_reference=0.;maximum=0.
    for begin in range(0,len(a),64):
        left=np.asarray(a[begin:begin+64],dtype=np.float64)
        right=np.asarray(b[begin:begin+64],dtype=np.float64)
        diff=left-right
        maximum=max(maximum,float(np.abs(diff).max()))
        square_error+=float(np.square(diff).sum());square_reference+=float(np.square(right).sum())
    a._mmap.close();b._mmap.close()
    return dict(max_absolute_coefficient_difference=maximum,
                relative_frobenius_difference=float(np.sqrt(square_error/square_reference)))


def run():
    source=ROOT/'models/ease_lambda100_exact'
    m=json.loads((source/'manifest.json').read_text('utf-8'))
    if digest(source/'weights.npy')!=m['files']['weights.npy']['sha256']:
        raise ValueError('CPU reference changed')
    gram=ROOT/'models/_linear_work/gram_beta000.f32.dat'
    receipt=json.loads(gram.with_suffix('.dat.json').read_text('utf-8'))
    if not receipt['complete'] or digest(gram)!=receipt['sha256']:
        raise ValueError('Completed train Gram receipt required')
    apps=np.load(source/'input_app_ids.npy');candidates=np.load(source/'candidate_app_ids.npy')
    cols=np.searchsorted(apps,candidates);gamma=np.full(len(apps),100.)
    output=ROOT/'gpu_verification/ease_lambda100'
    runtime=gpu_solve(gram,gamma,cols,output)
    differences=compare(output/'weights.npy',source/'weights.npy')
    sample_rows=runtime['sample_rows']
    errors=residual(gram,gamma,cols[sample_rows],np.load(output/'sample_inverse_columns.npy'))
    passed=max(errors)<1e-4 and differences['relative_frobenius_difference']<1e-3
    report=dict(status='passed' if passed else 'failed',runtime=runtime,differences=differences,
        sampled_backward_errors=errors.tolist(),tolerances=dict(backward_error=1e-4,relative_frobenius=1e-3),
        gram_sha256=receipt['sha256'],cpu_model_manifest_sha256=digest(source/'manifest.json'),
        gpu_weights_sha256=digest(output/'weights.npy'),code_sha256=digest(__file__),
        interpretation='Numerical backend verification only; not an extra recommendation candidate or a performance improvement claim.')
    write_json(output/'manifest.json',report)
    write_json(EDA/'reports/linear_gpu_validation_v4.json',report)
    if not passed:raise ValueError('CUDA backend did not meet the numerical equivalence checks')
    return report


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
