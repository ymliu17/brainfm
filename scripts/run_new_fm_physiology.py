#!/usr/bin/env python3
"""Independent supplementary physiology queue with immutable external feature gates."""
import argparse,collections,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from new_fm import physiology as p
from new_fm.report import states
from new_fm.extract import folder
from new_fm.evaluate import state_name,read
from benchmark_gpu_lock import acquire
ENTRY=Path(__file__).resolve();LOG=ROOT/'logs/new_fm_window_reencoding_v1'


def planned():
    jobs=[]
    for state in states():
        name=state_name(state);f='features_'+name;phase='frozen' if state[2]=='frozen' else 'post_training'
        jobs.append(dict(id=f,action='extract',state=list(state),resource='gpu',phase=phase,
            dependencies=[] if phase=='frozen' else ['report_frozen'],
            external_output=str(folder(state)/'complete.json'),output=str(p.paths(state)/'complete.json')))
        for co in p.COHORTS:
            jobs.append(dict(id='evaluate_'+name+'_'+co,action='evaluate',state=list(state),cohort=co,
                resource='cpu',phase=phase,dependencies=[f],output=str(p.result(state,co))))
    frozen=[j['id'] for j in jobs if j['phase']=='frozen']
    jobs.append(dict(id='report_frozen',action='report',phase='frozen',resource='cpu',dependencies=frozen,output=str(p.RUN/'reports/frozen/complete.json')))
    jobs.append(dict(id='report_all',action='report',phase='all',resource='cpu',dependencies=[j['id'] for j in jobs],output=str(p.RUN/'reports/all/complete.json')))
    return jobs


def complete(job):
    path=Path(job['output'])
    return path.exists() and read(path).get('status')=='complete'


def manifest():
    jobs=planned();path=p.RUN/'queue_manifest.json'
    signatures={str(f):p.sha(f) for f in [ENTRY,Path(p.__file__),ROOT/'src/new_fm/models.py',ROOT/'src/new_fm/data.py']}
    d=dict(tasks=jobs,code_sha256=signatures,protocol_sha256=p.sha(p.RUN/'protocol.json'),automatic_shutdown=False)
    if path.exists() and read(path)!=d:raise ValueError('Window queue changed after registration')
    if not path.exists():p.save(path,d)
    return jobs


def run():
    p.prepare();LOG.mkdir(parents=True,exist_ok=True)
    with (p.RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=manifest();active={};done={j['id'] for j in jobs if complete(j)};failed={};started=time.time()
        while True:
            for key,(proc,stream,lease,tasklock,job) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                stream.close();tasklock.close()
                if lease:lease.close()
                if rc==0 and complete(job):done.add(key)
                else:failed[key]=dict(returncode=rc,log=str(LOG/(key+'.log')))
                del active[key]
            external_wait=[]
            for job in jobs:
                key=job['id']
                if key in done or key in active or key in failed:continue
                if complete(job):done.add(key);continue
                if any(d in failed for d in job['dependencies']):failed[key]=dict(reason='Failed local dependency');continue
                if not set(job['dependencies'])<=done:continue
                if 'external_output' in job:
                    gate=Path(job['external_output'])
                    if not gate.exists() or read(gate).get('status')!='complete':external_wait.append(key);continue
                gpu=job['resource']=='gpu'
                if sum(v[4]['resource']==job['resource'] for v in active.values())>=(1 if gpu else 2):continue
                tl=acquire(p.RUN/'task_locks'/(key+'.lock'))
                if tl is None:continue
                lease=acquire() if gpu else None
                if gpu and lease is None:tl.close();continue
                env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                cmd=[str(ROOT/'.venv/bin/python'),str(ENTRY),job['action']]
                if 'state' in job:cmd+=['--state',state_name(job['state'])]
                if 'cohort' in job:cmd+=['--cohort',job['cohort']]
                if job['action']=='report':cmd+=['--phase',job['phase']]
                if gpu:cmd+=['--device','cuda']
                stream=(LOG/(key+'.log')).open('a')
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(tl.fileno(),)+(() if lease is None else (lease.fileno(),)))
                active[key]=(child,stream,lease,tl,job)
            status={j['id']:'complete' if j['id'] in done else 'failed' if j['id'] in failed else 'running' if j['id'] in active else 'waiting' for j in jobs}
            terminal=len(done)+len(failed)==len(jobs) and not active
            p.save(p.RUN/'queue_status.json',dict(status=('finished_with_failures' if failed else 'complete') if terminal else 'running',controller_pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=time.time()-started,
                counts=dict(collections.Counter(status.values())),running=[dict(id=k,pid=v[0].pid,resource=v[4]['resource']) for k,v in active.items()],failed=failed,waiting_external_features=external_wait,
                tasks=[dict(id=k,state=v) for k,v in status.items()],automatic_shutdown=False))
            if terminal:return
            time.sleep(5)


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','init','run','extract','evaluate','report']);a.add_argument('--state');a.add_argument('--cohort',choices=p.COHORTS);a.add_argument('--device',default='cpu');a.add_argument('--phase',choices=['frozen','all'],default='frozen');args=a.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.action=='prepare':p.prepare()
    elif args.action=='init':p.prepare();print('Registered',len(manifest()),'tasks')
    elif args.action=='run':run()
    elif args.action=='report':p.report(args.phase)
    else:
        state=next(s for s in states() if state_name(s)==args.state)
        if args.action=='extract':p.extract(state,args.device)
        else:p.evaluate(state,args.cohort)

if __name__=='__main__':main()
