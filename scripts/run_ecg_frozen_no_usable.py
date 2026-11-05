#!/usr/bin/env python3
"""Run cached four-model physiological comparison on CPU only."""
from pathlib import Path
import sys, os, time, fcntl, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import ecg_frozen_no_usable as e

def main():
    e.RUN.mkdir(exist_ok=True)
    with (e.RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time();e.audit()
        for stage,jobs,fn in [('evaluation',[(c,) for c in e.COHORTS],e.evaluate),
                              ('inference',[(c,m) for c in e.COHORTS for m in e.MODELS],e.infer)]:
            done=0;failures={}
            def status():
                e.save(e.RUN/'queue_status.json',dict(status='running',stage=stage,done=done,total=len(jobs),
                       failures=failures,pid=os.getpid(),elapsed_seconds=time.time()-start,gpu_required=False,automatic_shutdown=False))
            status()
            with ProcessPoolExecutor(max_workers=3) as pool:
                futures={pool.submit(fn,*args):str(args) for args in jobs}
                for f in as_completed(futures):
                    try: print(stage,f.result(),flush=True);done+=1
                    except Exception: failures[futures[f]]=traceback.format_exc();print(failures[futures[f]],flush=True)
                    status()
            if failures:
                e.save(e.RUN/'queue_status.json',dict(status='failed',failures=failures,automatic_shutdown=False))
                raise RuntimeError('Failed '+stage)
        e.report();e.save(e.RUN/'queue_status.json',dict(status='complete',elapsed_seconds=time.time()-start,
             gpu_required=False,automatic_shutdown=False));print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
