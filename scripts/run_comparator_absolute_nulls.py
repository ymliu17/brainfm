#!/usr/bin/env python3
"""Extend the frozen-model absolute tests to SF100 FC and BrainGFM.

Reuses the registered nine-FM refitting implementation and matched participant
sets. Historical files are read-only. No encoder inference or automatic shutdown.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, fcntl, json, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import pandas as pd
import run_evidence_nulls as engine

RUN=ROOT/'runs/comparator_absolute_nulls_v1'
OLD=ROOT/'runs/discussion_nulls_v1'
RECALL=ROOT/'runs/hrv_recall_pca5_v1'
SF=ROOT/'runs/sf100_benchmark_v1'
METHODS=('sf100_fc','braingfm')
read,save,sha=engine.read,engine.save,engine.sha

def activate():
    engine.RUN=RUN
    engine.payload.cache_clear()

def features(co,method,scans=None):
    folder=SF/'features'/co
    stamp=read(folder/'complete.json');p=folder/(method+'.npy')
    assert sha(p)==stamp['files_sha256'][method],str(p)
    x=np.load(p,mmap_mode='r')
    if scans is not None:
        lookup={s:i for i,s in enumerate(stamp['scan_ids'])}
        x=x[[lookup[s] for s in scans]]
    return np.asarray(x,dtype=float),p

def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'manifest.json').exists():return read(RUN/'manifest.json')
    jobs=[];sources={}
    parent=read(OLD/'manifest.json')
    plans=read(ROOT/'runs/intervention_prepost_v1/inputs.json')['plans']
    templates=[j for j in parent['jobs'] if j['model']=='brainlm13' and j['kind'] in ('local','paired')]
    assert len(templates)==34
    for template in templates:
        template=dict(template)
        if template['task']=='MEM_delayed_recall':
            template=read(RECALL/'null/job_specs'/(template['id']+'.json'))
            assert template['components']==5
        assert sha(template['input'])==template['input_sha256']
        with np.load(template['input']) as z:y=z['y'].copy();g=z['groups'].copy()
        for method in METHODS:
            j=dict(template);j.pop('archived_prediction',None)
            co=j['cohort'];task=j['task']
            if j['kind']=='local':
                x,fp=features(co,method,j['scan_ids']);sources[str(fp)]=sha(fp)
                if task=='MEM_delayed_recall':
                    ref=read(RECALL/'fits'/('hrv_pre__'+method+'.json'))
                    assert ref['subjects']==g.tolist() and np.array_equal(ref['y'],y)
                    j['archived_prediction']=ref['prediction']['fixed5']
                elif task.endswith('_followup'):
                    ref=read(OLD/'followup_references'/(co+'__'+task+'.json'))
                    assert ref['subjects']==g.tolist() and np.array_equal(ref['y'],y)
                    j['archived_prediction']=ref['references'][method]['prediction']
                else:
                    ref=read(j['reference_file'])
                    assert ref['subjects']==g.tolist() and np.array_equal(ref['y'],y)
                    j['archived_prediction']=ref['predictions'][method]
            else:
                plan=plans[co];t=plan['tasks'][task]
                assert t['subjects']==g.tolist() and np.array_equal(t['y'],y)
                a,pa=features(plan['pre_cohort'],method)
                b,pb=features(plan['post_cohort'],method)
                sources[str(pa)]=sha(pa);sources[str(pb)]=sha(pb)
                x=b[t['post_indices']]-a[t['pre_indices']]
                ref=read(ROOT/'runs/intervention_prepost_v1/jobs'/('comparator_'+method+'_frozen_none_0')/(co+'.json'))
                ref=next(r for r in ref['rows'] if r['task']==task and r['mode']=='change')
                assert ref['subjects']==g.tolist() and np.array_equal(ref['y'],y)
                j['archived_prediction']=ref['prediction']
            assert len(x)==len(y) and np.isfinite(x).all()
            j.update(model=method,id='__'.join([method,co,task]),permutations=1999,
                     parent_spec_id=template['id'],parent_spec_sha256=sha(OLD/'job_specs'/(template['id']+'.json')))
            p=RUN/'inputs'/(j['id']+'.npz');p.parent.mkdir(exist_ok=True)
            np.savez(p,x=x,y=y,groups=g)
            j.update(input=str(p),input_sha256=sha(p))
            save(RUN/'job_specs'/(j['id']+'.json'),j);jobs.append(j)
    # FC absolute Re-ID already exists. BrainGFM uses identical matched galleries.
    from readout_sensitivity import similarity
    for co in ('adni','nacc_repeated','act','cogte','breathe','psopt','hrv'):
        ref=read(SF/'identification'/(co+'.json'))
        ca,cb=('hrv_pre','hrv_post') if co=='hrv' else (co,co)
        a,pa=features(ca,'braingfm',ref['scans_a']);b,pb=features(cb,'braingfm',ref['scans_b'])
        sim=similarity(a,b);n=len(sim)
        sources[str(pa)]=sha(pa);sources[str(pb)]=sha(pb)
        j=dict(id='braingfm__'+co+'__top1',model='braingfm',cohort=co,task='top1',kind='identity',metric='accuracy',
               family='identity_absolute_null::identity_null_native_top1',permutations=20000,
               archived_score=ref['scores']['braingfm']['top1'])
        p=RUN/'inputs'/(j['id']+'.npz');np.savez(p,x=sim,y=np.arange(n),groups=np.asarray(ref['subjects']))
        j.update(input=str(p),input_sha256=sha(p));save(RUN/'job_specs'/(j['id']+'.json'),j);jobs.append(j)
    assert len(jobs)==75
    protocol=dict(permutations=1999,identity_permutations=20000,split_seeds=[0,1,2],
        scope='Fig. 2: observational AGE/SEX/DX/MEM/EF; imaging-only baseline follow-up MEM/EF; change-input MEM/EF change and arm; Re-ID. No physiology or transfer/LOSO.',
        refitting='Identical engine to the nine-FM nulls: outcome-stratified participant folds regenerated per permutation, train-only scaling/PCA and tuning refitted. Whole-participant trajectories permuted within visit-count strata. One-sided finite p.',
        components='100 observational, 5 follow-up including HRV delayed recall, 5 paired-visit change inputs.',
        correction='One Holm family per predictive task across all eleven methods and eligible cohorts. Reuse nine-FM raw p; replace HRV memory with completed 5-PC null. Re-ID retains historical 175-test native family and adds seven BrainGFM tests (182). No historical family is narrowed.',
        original_results='Read-only. Scores/heatmap colors unchanged; dots recalculated with enlarged correction. Observed and null statistics use identical refitting code.',
        automatic_shutdown=False)
    manifest=dict(jobs=jobs,protocol=protocol,created_utc=engine.utc(),source_sha256=sources,
                  code_sha256=sha(__file__),engine_sha256=sha(engine.__file__))
    save(RUN/'manifest.json',manifest);save(RUN/'protocol.json',protocol)
    print('Prepared 68 predictive tests + 7 BrainGFM Re-ID tests; reused 7 FC Re-ID tests.',flush=True)
    return manifest

def work(item):
    activate()
    return engine.work(item)

def completed_row(j,base):
    op=base/'observed'/(j['id']+'.json')
    if not op.exists():return None
    obs=read(op)
    if j['kind']=='identity':
        r=read(base/'identity'/(j['id']+'.json'));n=r['permutations'];hits=r['exceedances'];seconds=obs['seconds']
        assert abs(obs['score']-j.get('archived_score',obs['score']))<1e-12
    else:
        chunks=[read(p) for p in (base/'chunks'/j['id']).glob('*.json')]
        intervals=sorted((r['start'],r['stop']) for r in chunks)
        assert all(0<=a<b<=j['permutations'] for a,b in intervals)
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        n=sum(b-a for a,b in intervals);hits=sum(r['exceedances'] for r in chunks)
        seconds=sum(r['seconds'] for r in chunks)
    return dict(id=j['id'],model=j['model'],cohort=j['cohort'],task=j.get('display_task',j['task']),
                family=j['family'],observed_score=obs['score'],completed_permutations=n,required_permutations=j['permutations'],
                raw_p=(1+hits)/(1+n) if n==j['permutations'] else np.nan,
                status='complete' if n==j['permutations'] else 'running',seconds=seconds)

def report():
    m=read(RUN/'manifest.json');rows=[completed_row(j,RUN) for j in m['jobs']];rows=[r for r in rows if r]
    df=pd.DataFrame(rows)
    if len(df):df.to_csv(RUN/'results.csv',index=False)
    old=pd.read_csv(OLD/'results.csv');old=old[~old.task.isin(['top1','physiology'])].copy()
    # Replace historical HRV 100-PC raw tests with completed 5-PC tests.
    for j in read(RECALL/'null/manifest.json')['jobs']:
        row=completed_row(j,RECALL/'null');assert row and row['status']=='complete'
        mask=old.id.eq(j['id']);assert mask.sum()==1
        for k in ['raw_p','observed_score']:old.loc[mask,k]=row[k]
    new=df[df.task.ne('top1')] if len(df) else pd.DataFrame()
    joint=pd.concat([old,new],ignore_index=True);joint['holm_p']=np.nan;joint['family_size']=np.nan
    expected=collections.Counter(j['family'] for j in m['jobs'] if j['kind']!='identity')
    for family,ix in joint.groupby('family').groups.items():
        size=int(old.family.eq(family).sum())+expected[family]
        joint.loc[ix,'family_size']=size
        if len(ix)==size and joint.loc[ix,'raw_p'].notna().all():joint.loc[ix,'holm_p']=engine.holm(joint.loc[ix,'raw_p'])
    joint.to_csv(RUN/'predictive_joint_correction.csv',index=False)
    # Retain all prior native readouts, including FC and supplementary readouts.
    ident=pd.read_csv(OLD/'identity_retained_family.csv')
    ident=ident[['model','cohort','raw_p','candidate_score']].rename(columns={'candidate_score':'observed_score'})
    for j in read(OLD/'manifest.json')['jobs']:
        if j['kind']=='identity':
            r=read(OLD/'identity'/(j['id']+'.json'))
            ident=pd.concat([ident,pd.DataFrame([dict(model=j['model'],cohort=j['cohort'],raw_p=r['p'],observed_score=r['score'])])],ignore_index=True)
    assert len(ident)==175
    additions=df[df.task.eq('top1')][['model','cohort','raw_p','observed_score']] if len(df) else pd.DataFrame()
    ident=pd.concat([ident,additions],ignore_index=True);ident['holm_p']=np.nan;ident['family_size']=182
    if len(ident)==182 and ident.raw_p.notna().all():ident['holm_p']=engine.holm(ident.raw_p)
    ident.to_csv(RUN/'identity_joint_correction.csv',index=False)
    progress=dict(updated_utc=engine.utc(),total_jobs=75,observed_validated=len(df),
                  complete_jobs=int(df.status.eq('complete').sum()) if len(df) else 0,
                  completed_permutations=int(df.completed_permutations.sum()) if len(df) else 0,
                  total_permutations=68*1999+7*20000,
                  predictive_completed_permutations=int(df.loc[df.task.ne('top1'),'completed_permutations'].sum()) if len(df) else 0,
                  predictive_total_permutations=68*1999,automatic_shutdown=False)
    save(RUN/'progress.json',progress);return progress

def run(workers):
    manifest=prepare()
    assert manifest['engine_sha256']==sha(engine.__file__),'Parent engine changed'
    assert manifest['code_sha256']==sha(__file__),'Controller changed'
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=manifest['jobs'];todo=[]
        for j in jobs:
            if not (RUN/'observed'/(j['id']+'.json')).exists():todo.append((j['id'],))
        todo.sort(key=lambda x:(0 if 'top1' in x[0] else 2 if '__adni__' in x[0] else 1))
        pending=collections.deque(todo);active={};failures=[];phase='validate_observed';last=0;stopping=False
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while True:
                if (RUN/'STOP_AFTER_CURRENT').exists():stopping=True
                while pending and len(active)<workers and not stopping:
                    item=pending.popleft();active[pool.submit(work,item)]=(item,time.monotonic())
                if time.monotonic()-last>30:
                    progress=report();last=time.monotonic()
                    save(RUN/'queue_status.json',dict(status='running',phase=phase,controller_pid=os.getpid(),workers=workers,
                         updated_utc=engine.utc(),pending=len(pending),failures=failures,
                         running=[dict(item=i,elapsed_seconds=time.monotonic()-t) for i,t in active.values()],**{k:v for k,v in progress.items() if k!='updated_utc'}))
                if not active:
                    if stopping or failures:break
                    if phase=='validate_observed':
                        phase='permutations';items=[]
                        for j in jobs:
                            if j['kind']=='identity':continue
                            for start,end in engine.chunk_ranges(j):
                                if not (RUN/'chunks'/j['id']/f'{start:05d}_{end:05d}.json').exists():items.append((j['id'],start,end))
                        # Interleave chunks, reserving throughput for smaller intervention tasks.
                        items.sort(key=lambda v:(v[1],2 if '__adni__' in v[0] else 1 if '__nacc__' in v[0] else 0,v[0]))
                        pending=collections.deque(items);continue
                    break
                done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    item,_=active.pop(f)
                    try:r=f.result()
                    except Exception:r=dict(error=traceback.format_exc(),item=item)
                    if 'error' in r:failures.append(r)
                    print(json.dumps({k:v for k,v in r.items() if k!='observed'}),flush=True)
        progress=report();status='stopped' if stopping else 'failed' if failures else 'complete'
        if status=='complete':assert progress['complete_jobs']==75
        save(RUN/'queue_status.json',dict(status=status,failures=failures,workers=workers,**progress))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','report','pilot']);ap.add_argument('--workers',type=int,default=7);ap.add_argument('--job');ap.add_argument('--count',type=int,default=1);a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a.workers)
    elif a.action=='report':print(report())
    else:
        prepare();print(work((a.job,)),flush=True)
        if 'top1' not in a.job:print(work((a.job,0,a.count)),flush=True)
if __name__=='__main__':main()
