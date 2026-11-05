#!/usr/bin/env python3
"""Resumable diagnosis queue sharing the existing process-inherited GPU lease."""
from pathlib import Path
import argparse,collections,fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from diagnosis_posttraining import RUN,SPECS,SEEDS,MODELS,prepare,verify,train,location
from fm_matrix.data import save
from benchmark_gpu_lock import acquire


def tasks():
    return [dict(name=f'{source}_{model}_{target}_s{seed}_{mode}',source=source,model=model,target=target,seed=seed,mode=mode,state='queued')
            for seed in SEEDS for mode in ('external','cv') for source,target in SPECS for model in MODELS]


def run(init=False):
    prepare(); logdir=ROOT/'logs/diagnosis_posttraining_v1';logdir.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=RUN/'queue_status.json'; jobs=tasks()
        old=json.loads(path.read_text()) if path.exists() else {}
        for j in old.get('tasks',[]):
            if j['state']=='running':
                try: os.kill(j['pid'],0)
                except ProcessLookupError: pass
                else: raise RuntimeError('Previous diagnosis child is still alive')
        def update(status='running'):
            save(path,dict(status=status,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),counts=dict(collections.Counter(j['state'] for j in jobs)),automatic_shutdown=False,tasks=jobs))
        def output(j): return location(j['source'],j['model'],j['target'],j['seed'],j['mode'])/'complete.json'
        for j in jobs:
            if output(j).exists(): j['state']='complete'
        update('queued' if init else 'running')
        if init: return
        active=None
        while True:
            if active:
                child,lease,log,j=active; rc=child.poll()
                if rc is not None:
                    log.close();lease.close();j.update(state='complete' if rc==0 and output(j).exists() else 'failed',exit_code=rc,finished=time.time());active=None;update()
            for j in jobs:
                if j['state']=='queued' and output(j).exists(): j.update(state='complete',reused_canonical_training=True);update()
            if not active:
                j=next((j for j in jobs if j['state']=='queued'),None)
                if j is None:break
                lease=acquire()
                if lease is not None:
                    verify()
                    env=os.environ.copy()
                    for k in list(env):
                        if k.startswith('BRAINLM_') or k.startswith('BENCHMARK_GPU_'):env.pop(k)
                    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                    command=[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'worker']
                    for key in ('source','model','target','seed','mode'): command+=['--'+key,str(j[key])]
                    log=(logdir/(j['name']+'.log')).open('a')
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                    j.update(state='running',pid=child.pid,started=time.time(),command=command);active=(child,lease,log,j);update()
            time.sleep(5)
        update('complete' if all(j['state']=='complete' for j in jobs) else 'finished_with_failures')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','init','run','worker']);ap.add_argument('--source',choices=['adni','nacc']);ap.add_argument('--model',choices=MODELS);ap.add_argument('--target',choices=['DX','AD_DX']);ap.add_argument('--seed',type=int,choices=SEEDS);ap.add_argument('--mode',choices=['external','cv']);a=ap.parse_args()
    import torch
    torch.set_num_threads(4)
    if a.action=='worker':
        if (a.source,a.target) not in SPECS or a.model is None or a.seed is None or a.mode is None: ap.error('A complete registered classification specification is required')
        train(a.source,a.model,a.target,a.seed,a.mode)
    elif a.action=='prepare':prepare()
    else:run(init=a.action=='init')


if __name__=='__main__':main()
