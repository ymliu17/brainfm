#!/usr/bin/env python3
"""Additive matched physiology queue; reuse all existing unshuffled embeddings."""
from pathlib import Path
import argparse,collections,fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from physiology_benchmark import RUN,MATRIX_RUN,states,prepare,verify,evaluate,shuffled_features,report
from fm_matrix.data import save
from benchmark_gpu_lock import acquire


def tasks():
    jobs=[]
    for state in states():
        dep='features_'+state['model']+'_frozen' if state['objective']=='frozen' else 'features_'+state['source']+'_'+state['model']+'_'+state['objective']+('_'+state['target'] if state['objective']=='supervised' else '')+'_s42'
        deps=[] if state['objective']=='frozen' else ['evaluate_'+state['model']+'_frozen']
        jobs.append(dict(name='evaluate_'+state['name'],action='evaluate',state_spec=state,resource='cpu',state='queued',dependencies=deps,external_dependency=dep,priority=0 if state['objective']=='frozen' else 10))
        if state['objective']=='frozen':
            name='shuffle_features_'+state['model']
            jobs.append(dict(name=name,action='shuffle_features',state_spec=state,resource='gpu',state='queued',dependencies=[],external_dependency=dep,priority=1))
            jobs.append(dict(name='shuffle_evaluate_'+state['model'],action='shuffle_evaluate',state_spec=state,resource='cpu',state='queued',dependencies=[name,'evaluate_'+state['name']],priority=2))
    jobs.append(dict(name='report',action='report',resource='cpu',state='queued',dependencies=[j['name'] for j in jobs],priority=100))
    return jobs


def run(init=False):
    prepare();logdir=ROOT/'logs/physiology_matched_v1';logdir.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=RUN/'queue_status.json';old={j['name']:j for j in json.loads(path.read_text())['tasks']} if path.exists() else {};jobs=tasks();active={}
        for j in jobs:
            previous=old.get(j['name'])
            if previous:
                for key in ('action','resource','dependencies','external_dependency','state_spec'):
                    if j.get(key)!=previous.get(key):raise ValueError('Registered physiology task changed')
                if previous['state']=='running':
                    try:os.kill(previous['pid'],0)
                    except ProcessLookupError:pass
                    else:raise ValueError('Previous physiology worker is still alive')
                if previous['state']=='complete':j.update(previous)
        def update(status='running'):
            save(path,dict(status=status,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),counts=dict(collections.Counter(j['state'] for j in jobs)),automatic_shutdown=False,tasks=jobs))
        update('queued' if init else 'running')
        if init:return
        while True:
            for resource,(child,log,lease,j) in list(active.items()):
                rc=child.poll()
                if rc is None:continue
                log.close()
                if lease is not None:lease.close()
                j.update(state='complete' if rc==0 else 'failed',exit_code=rc,finished=time.time());del active[resource];update()
            external={j['name']:j['state'] for j in json.loads((MATRIX_RUN/'queue_status.json').read_text())['tasks']};local={j['name']:j['state'] for j in jobs}
            for j in sorted(jobs,key=lambda j:j['priority']):
                if j['state']!='queued':continue
                dep=j.get('external_dependency');ds=[local[d] for d in j['dependencies']]+([] if dep is None else [external.get(dep,'missing')])
                if any(v in ('failed','blocked','missing') for v in ds):j.update(state='blocked',reason='Feature or analysis dependency failed/missing');update();continue
                if any(v!='complete' for v in ds) or j['resource'] in active:continue
                lease=None
                if j['resource']=='gpu':
                    lease=acquire()
                    if lease is None:continue
                verify();env=os.environ.copy()
                for key in list(env):
                    if key.startswith('BRAINLM_') or key.startswith('BENCHMARK_GPU_'):env.pop(key)
                env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                cmd=[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),j['action']]
                if 'state_spec' in j:cmd+=['--state',j['state_spec']['name']]
                log=(logdir/(j['name']+'.log')).open('a');child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=() if lease is None else (lease.fileno(),))
                j.update(state='running',pid=child.pid,started=time.time(),command=cmd);active[j['resource']]=(child,log,lease,j);update()
            if not active and all(j['state'] in ('complete','failed','blocked') for j in jobs):break
            time.sleep(5)
        update('complete' if all(j['state']=='complete' for j in jobs) else 'finished_with_failures')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','init','run','evaluate','shuffle_features','shuffle_evaluate','report']);ap.add_argument('--state');a=ap.parse_args()
    import torch
    torch.set_num_threads(2)
    if a.action=='prepare':prepare()
    elif a.action in ('init','run'):run(init=a.action=='init')
    elif a.action=='report':verify();report()
    else:
        state=next(s for s in states() if s['name']==a.state)
        if a.action=='shuffle_features':shuffled_features(state['model'])
        else:evaluate(state,shuffle=a.action=='shuffle_evaluate')


if __name__=='__main__':main()
