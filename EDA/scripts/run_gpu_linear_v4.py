"""Run one declared linear group with the verified CUDA batch size.

The model receipt records the actual batch size and preserves this entry point
beside the underlying builder. No selection or confirmation data is opened.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from threadpoolctl import threadpool_limits
from eda_large_review_frank import digest
import experiment_linear_v4 as linear
import fast_gram_v4


def run(args):
    linear.CANDIDATE_SOLVE_BATCH=128
    finish=linear._finish_artifact
    gram_builder=linear._build_gram_memmap
    linear._build_gram_memmap=fast_gram_v4.build
    def record(stage,name,family,artifact_type,mode,parameters,sources,support,apps,candidates,summary):
        shutil.copy2(__file__,stage/'gpu_entrypoint_snapshot.py')
        shutil.copy2(fast_gram_v4.__file__,stage/'gram_builder_snapshot.py')
        summary=dict(summary,gpu_entrypoint_sha256=digest(__file__),
            gram_builder_sha256=digest(fast_gram_v4.__file__))
        return finish(stage,name,family,artifact_type,mode,parameters,sources,support,apps,candidates,summary)
    linear._finish_artifact=record
    try:
        return linear.build_linear_group(args.family,args.regularization,
            alpha=args.alpha,beta=args.beta,drop_p=args.drop_p,
            scale_mode=args.scale_mode,solver_backend='cuda')
    finally:
        linear._finish_artifact=finish
        linear._build_gram_memmap=gram_builder


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['rdlae','dan'],required=True)
    p.add_argument('--regularization',type=int,choices=[100,1000],required=True)
    p.add_argument('--alpha',type=float,choices=[0,.1,.2,.3])
    p.add_argument('--beta',type=float,choices=[0,.5,1],default=0.)
    p.add_argument('--drop-p',type=float,choices=[0,.1,.3],default=0.)
    p.add_argument('--scale-mode',choices=['raw','scale_matched'],default='raw')
    with threadpool_limits(limits=2):
        for name in run(p.parse_args()):print(name,flush=True)
