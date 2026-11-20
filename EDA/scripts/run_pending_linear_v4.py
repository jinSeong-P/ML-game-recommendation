"""Resume the declared first-stage linear fits serially, preserving receipts.

This entry point does not read ranking outcomes or confirmation targets.
Only already declared lambda100 DAN and both RDLAE lambdas are scheduled.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import psutil
from threadpoolctl import threadpool_limits

from eda_large_review_frank import log
import experiment_linear_v4 as linear
from run_gpu_linear_v4 import run as fit_gpu
from reuse_dan_v4 import derive


def wait_for_resources():
    while True:
        competing=[]
        for process in psutil.process_iter(['pid','cmdline']):
            if process.pid==os.getpid():continue
            try:
                command=' '.join(process.info['cmdline'] or [])
                if any(name in command for name in ['experiment_linear_v4.py build-linear','run_gpu_linear_v4.py --family']):
                    competing.append(process.pid)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        available=psutil.virtual_memory().available/1024**3
        if not competing and available>=10.2:return
        log(f'Waiting for serial fit window: available {available:.2f} GiB, competing fits {competing}')
        time.sleep(20)


def run():
    for regularization in (100,1000):
        for drop_p in (.1,.3):
            name=linear._linear_names('rdlae',regularization,drop_p=drop_p)[0][0]
            if not (linear.MODEL_ROOT/name/'manifest.json').exists():
                wait_for_resources();log('Resume '+name)
                fit_gpu(argparse.Namespace(family='rdlae',regularization=regularization,
                    alpha=None,beta=0.,drop_p=drop_p,scale_mode='raw'))
    # Alpha only rescales a completed base. Derive the other prescribed alphas
    # after solving alpha0, with the exact same train split/Gram/penalty.
    derive('ease_lambda100_exact',0.)
    for beta in (0.,.5,1.):
        for drop_p in (0.,.3):
            scales=('raw',) if beta==0 else ('raw','scale_matched')
            for scale in scales:
                name=linear._linear_names('dan',100,alpha=0.,beta=beta,
                    drop_p=drop_p,scale_mode=scale)[0][0]
                if not (linear.MODEL_ROOT/name/'manifest.json').exists():
                    wait_for_resources();log('Resume '+name)
                    fit_gpu(argparse.Namespace(family='dan',regularization=100,
                        alpha=0.,beta=beta,drop_p=drop_p,scale_mode=scale))
                made=derive(name,drop_p)
                log(f'DAN coefficient reuse completed {name}: {len(made)} new artifacts')
    log('Scheduled linear model artifacts complete; no ranking evaluation was dispatched')


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
