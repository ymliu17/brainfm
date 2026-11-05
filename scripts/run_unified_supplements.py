#!/usr/bin/env python3
"""Restartable CPU supplement queue, sharing immutable main-queue features."""
import os,sys,json,time,subprocess,fcntl,collections,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import unified_supplements as u
from fm_readout_v4.evaluate import tag
ENTRY=Path(__file__).resolve();LOG=ROOT/'logs/unified_supplements_v1'

def planned():
    jobs=[]
    for model in u.NEW:
        for state in u.states(model):
            jobs.append(dict(id='recall_'+tag(*state),action='recall',state=list(state),resource='cpu',
                gates=[str(u.feature_folder(state)/'complete.json')],output=str(u.RUN/'recall'/tag(*state)/'result.json')))
    for model in u.CONFIGS:
        for co in u.PAIR_COHORTS:
            state=('none',model,'frozen','none',0)
            jobs.append(dict(id='identity_'+model+'_'+co,action='identity',model=model,cohort=co,resource='cpu',
                gates=[str(u.feature_folder(state)/'complete.json')],output=str(u.RUN/'identity'/model/co/'result.json')))
    return jobs

def completed(p):
    return Path(p).exists() and u.read(p).get('status')=='complete'

def manifest():
    d=dict(tasks=planned(),protocol_sha256=u.sha(ROOT/'runs/unified_extension_v1/protocol.json'),
        code_sha256={str(p):u.sha(p) for p in (ENTRY,Path(u.__file__))},automatic_shutdown=False,
        scope='Recall coverage plus frozen identity absolute null, alternative readout FC comparisons, and Gaussian128 dimension controls. Other registered supplements tracked separately.',
        publication='Raw tests; joint correction deferred until all configurations complete.')
    p=u.RUN/'queue_manifest.json'
    if p.exists() and u.read(p)!=d:raise ValueError('Supplement queue changed after registration')
    if not p.exists():u.save(p,d)
    return d['tasks']

def run(workers):
    u.RUN.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    with (u.RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);jobs=manifest();active={};failed={}
        while True:
            for key,(child,stream,job) in list(active.items()):
                if child.poll() is None:continue
                stream.close()
                if child.returncode or not completed(job['output']):failed[key]=dict(returncode=child.returncode,log=str(LOG/(key+'.log')))
                del active[key]
            done={j['id'] for j in jobs if completed(j['output'])};waiting=[]
            for job in jobs:
                key=job['id']
                if key in done or key in active or key in failed:continue
                if not all(completed(p) for p in job['gates']):waiting.append(key);continue
                if len(active)>=workers:continue
                stream=(LOG/(key+'.log')).open('a');env=os.environ.copy()
                env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                cmd=[str(ROOT/'.venv/bin/python'),str(ENTRY),'task','--job',key]
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                active[key]=(child,stream,job)
            counts=dict(complete=len(done),running=len(active),failed=len(failed),waiting=len(jobs)-len(done)-len(active)-len(failed))
            terminal=counts['waiting']==0 and not active
            u.save(u.RUN/'queue_status.json',dict(status=('failed' if failed else 'complete') if terminal else 'running',controller_pid=os.getpid(),
                counts=counts,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),running=[dict(id=k,pid=v[0].pid) for k,v in active.items()],
                failed=failed,waiting_external_features=waiting,automatic_shutdown=False))
            if terminal:return
            time.sleep(10)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','run','task']);p.add_argument('--workers',type=int,default=2);p.add_argument('--job');a=p.parse_args()
    if a.action=='init':print('Registered',len(manifest()),'tasks')
    elif a.action=='run':run(a.workers)
    else:
        j=next(j for j in manifest() if j['id']==a.job)
        if completed(j['output']):return
        if j['action']=='recall':u.recall(tuple(j['state']))
        elif j['action']=='identity':u.identity(j['model'],j['cohort'])
if __name__=='__main__':main()
