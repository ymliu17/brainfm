#!/usr/bin/env python3
"""Synchronous within-scan time permutations preserving static FC and amplitudes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import run_nine_model_identity_controls as ctrl
import run_connectivity_readouts as base
from fm_matrix.data import save,sha
from paper_paired_stats import holm
RUN=ROOT/'runs/temporal_order_controls_v1'
OUT=ROOT/'results/generated/temporal_order_controls_v1'
LOG=ROOT/'logs/temporal_order_controls_v1'
SEEDS=(211,212,213,214,215)


def read(p):return json.loads(Path(p).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def folder(model,seed,co):return RUN/'features'/model/str(seed)/co
def complete(p):return p.exists() and read(p).get('status')=='complete'


def register():
    sources={str(p):sha(p) for p in [Path(__file__),ROOT/'src/readout_control_encoders.py',Path(ctrl.__file__),
        Path(base.__file__),ROOT/'src/readout_sensitivity.py',ROOT/'src/unified_supplements.py']}
    for co in ctrl.COHORTS:
        p=ctrl.old.RUN/'pairs'/f'{co}.json';sources[str(p)]=sha(p)
        for model in base.MODELS:
            p=base.RUN/'identity'/model/f'{co}.json';sources[str(p)]=sha(p)
    d=dict(models=base.MODELS,cohorts=ctrl.COHORTS,shuffle_seeds=SEEDS,
        perturbation='One independent deterministic time permutation per scan and seed, identical across all observed ROIs. Permute only real frames consumed by the encoder, after model input construction, before tokenization. Preserve temporal padding, missing-region zeros, BrainHarmonix metadata and real-frame count. Visits use distinct permutations keyed by scan ID. Equal frame counts share permutations across model configurations.',
        input='Identical pretrained checkpoints, normalization, native/resampled TR, batch size and canonical batch positions as existing final-layer extraction. No weights updated. Primary and final normalized embedding connectivity only.',
        validation='Every permuted row is checked for exact preservation of untouched rows/tail and a bijective common time permutation. Full static-FC, regional-mean and standard-deviation invariance checked on first paired scan of each visit/cohort/seed; maximum errors retained. Unit tests cover every architecture input layout.',
        evaluation='Seven matched paired cohorts; native dimensionality and fixed Gaussian128 projection (seeds0/1/2). Average per-participant identification scores across five temporal permutations, not embeddings. Preserve all individual-seed scores.',
        statistics='Original ordered minus mean shuffled top1 accuracy and I_diff.5000 participant bootstrap and20000 two-sided paired query-score swaps, conditional on fixed gallery and five surrogate draws. Holm across all9 configurations,7cohorts,and both readouts separately by metric and dimensionality mode. Not an absolute null test based on five permutations.',
        interpretation='A decrease establishes sensitivity to this temporal-order perturbation, not neural specificity, temporal causality, or superiority over FC. No physiological or cognitive-outcome claims from this panel.',
        scheduling='GPU starts after existing nine-model random-control GPU queue releases the device; two GPU workers. CPU score jobs consume each completed shuffle artifact while remaining extraction proceeds.',
        automatic_shutdown=False,manuscript_modified=False,source_sha256=sources)
    from fm_matrix.data import clean
    p=RUN/'protocol.json'
    if p.exists():assert read(p)==clean(d)
    else:save(p,d)


def shuffle_row(row,model,metadata,seed,scan,validate=False):
    """CPU tensor permutation; all non-signal channels and padding remain intact."""
    import torch
    if model=='brainharmonix_f':rois,n=400,int(row[400,0])
    elif model.startswith('brainlm'):rois,n=424,min(int(metadata.native_frames),200)
    elif model=='brainjepa':rois,n=450,min(int(metadata.native_frames),160)
    else:rois,n=450,min(int(metadata.resampled_frames),100 if model=='semantoks' else 160)
    assert 3<=n<=row.shape[1]
    perm=np.random.default_rng(base.seed(f'{seed}|{scan}|{n}')).permutation(n)
    assert np.array_equal(np.sort(perm),np.arange(n))
    out=row.clone()
    out[:rois,:n]=row[:rois,torch.as_tensor(perm)]
    assert torch.equal(out[rois:],row[rois:]) and torch.equal(out[:rois,n:],row[:rois,n:])
    audit=dict(real_frames=n,regions=rois,full_fc_check=validate)
    if validate:
        a=row[:rois,:n].numpy().astype(float);b=out[:rois,:n].numpy().astype(float)
        sa,sb=a.std(1),b.std(1)
        np.testing.assert_allclose(a.mean(1),b.mean(1),atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(sa,sb,atol=1e-10,rtol=1e-10)
        def fc(x):
            x=x-x.mean(1,keepdims=True);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            return x@x.T
        fa,fb=fc(a),fc(b)
        np.testing.assert_allclose(fa,fb,atol=1e-10,rtol=1e-10)
        audit.update(fc_max_abs_error=float(abs(fa-fb).max()),std_max_abs_error=float(abs(sa-sb).max()))
    return out,audit


def extract(model,seed,device):
    import torch
    from readout_control_encoders import ControlEncoder,inputs
    torch.set_num_threads(2)
    # Cache parity is shared with the nine-model control preflight.13M was reused
    # there, so validate it explicitly before its first new shuffled extraction.
    if not complete(ctrl.RUN/'smoke'/f'{model}.json'):ctrl.smoke(model,device)
    net=ControlEncoder(model).to(device)
    batch=4 if model.startswith('fmrilm') else 1
    started=time.time()
    for co in ctrl.COHORTS:
        dest=folder(model,seed,co)
        if complete(dest/'complete.json'):continue
        pair=ctrl.pair(co);arrays={};audits=[]
        for side in ('a','b'):
            item=pair[side];x,m=inputs(model,item['cohort'])
            assert m.iloc[item['indices']].scan.tolist()==item['scans']
            vals={k:[] for k in ('native','connectivity')}
            for j,index in enumerate(item['indices']):
                chunk=dest/'chunks'/f'{side}_{j:04d}.npz'
                # Audit first paired scan even if a preceding batch already cached it.
                if j==0:
                    _,audit=shuffle_row(x[index],model,m.iloc[index],seed,item['scans'][j],True)
                    audits.append(dict(side=side,scan=item['scans'][j],**audit))
                if not chunk.exists():
                    start=index//batch*batch;end=min(start+batch,len(x))
                    xb=torch.stack([shuffle_row(x[k],model,m.iloc[k],seed,str(m.iloc[k].scan))[0] for k in range(start,end)])
                    result=ctrl.infer(net,xb,device)
                    assert all(np.isfinite(v).all() for v in result.values())
                    for other_side in ('a','b'):
                        other=pair[other_side]
                        if other['cohort']!=item['cohort']:continue
                        for q,ix in enumerate(other['indices']):
                            if start<=ix<end:
                                target=dest/'chunks'/f'{other_side}_{q:04d}.npz'
                                if not target.exists():ctrl.old.array_save(target,**{k:v[ix-start] for k,v in result.items()})
                with np.load(chunk) as z:
                    for name in vals:vals[name].append(z[name])
                if j%50==0:
                    save(RUN/'progress'/f'{model}_{seed}.json',dict(cohort=co,side=side,done=j+1,total=len(item['indices']),seconds=time.time()-started,updated=utc()))
                    print(model,seed,co,side,j+1,len(item['indices']),flush=True)
            arrays.update({k+'_'+side:np.stack(v) for k,v in vals.items()})
            del x
        ctrl.old.array_save(dest/'features.npz',**arrays)
        save(dest/'complete.json',dict(status='complete',model=model,seed=seed,cohort=co,
            sha256=sha(dest/'features.npz'),pair_sha256=sha(ctrl.old.RUN/'pairs'/f'{co}.json'),invariance_checks=audits,
            encoder_backend_sha256=sha(ROOT/'src/readout_control_encoders.py')))
        import shutil
        shutil.rmtree(dest/'chunks')


def score_seed(model,seed,co):
    dest=RUN/'scores'/model/str(seed)/f'{co}.json'
    if complete(dest):return
    f=folder(model,seed,co);stamp=read(f/'complete.json')
    assert sha(f/'features.npz')==stamp['sha256']
    assert stamp['pair_sha256']==sha(ctrl.old.RUN/'pairs'/f'{co}.json')
    with np.load(f/'features.npz') as z:arrays={k:z[k] for k in z.files}
    save(dest,dict(status='complete',scores=ctrl.old.score_arrays(arrays),source_sha256=sha(f/'complete.json')))


def evaluate(model,co):
    dest=RUN/'evaluation'/model/f'{co}.json'
    if complete(dest):return
    original=read(base.RUN/'identity'/model/f'{co}.json')
    surrogates={str(s):read(RUN/'scores'/model/str(s)/f'{co}.json')['scores'] for s in SEEDS}
    rows=[]
    for mode in ctrl.MODES:
        for metric in ctrl.METRICS:
            key='per_subject_'+metric
            for readout in ('native','connectivity'):
                reference=np.mean([surrogates[str(s)][mode][readout][key] for s in SEEDS],axis=0)
                candidate=original['scores'][mode][readout][key]
                stat=ctrl.rs.paired_identification(reference,candidate,base.seed(model+co+mode+metric+readout+'ordered'))
                rows.append(dict(model=model,cohort=co,mode=mode,metric=metric,readout=readout,
                    reference='mean_synchronously_shuffled',candidate='original_order',family=mode+'::'+metric,**stat))
    save(dest,dict(status='complete',rows=rows,subjects=original['subjects'],shuffle_seed_scores=surrogates))


def report():
    tasks=[];rows=[]
    for model in base.MODELS:
        for seed in SEEDS:
            for co in ctrl.COHORTS:
                tasks.append(dict(model=model,seed=seed,cohort=co,action='extract',done=complete(folder(model,seed,co)/'complete.json')))
        for co in ctrl.COHORTS:
            p=RUN/'evaluation'/model/f'{co}.json';ok=complete(p)
            tasks.append(dict(model=model,cohort=co,action='evaluate',done=ok))
            if ok:rows.extend(read(p)['rows'])
    done=all(t['done'] for t in tasks);OUT.mkdir(parents=True,exist_ok=True)
    if rows:
        df=pd.DataFrame(rows);df['holm_p']=np.nan
        if done:
            for family,ix in df.groupby('family').groups.items():df.loc[ix,'holm_p']=holm(df.loc[ix,'permutation_p'])
        df.to_csv(OUT/'Comparisons.csv',index=False)
        if done:
            with pd.ExcelWriter(OUT/'Temporal_order_controls.xlsx',engine='xlsxwriter') as w:
                df.to_excel(w,sheet_name='Paired tests',index=False)
                pd.DataFrame(tasks).to_excel(w,sheet_name='Coverage',index=False)
    save(RUN/'queue_status.json',dict(status='complete' if done else 'pending_or_running',updated=utc(),complete=sum(t['done'] for t in tasks),total=len(tasks),tasks=tasks,automatic_shutdown=False))
    return done


def cpu(workers):
    lock=(RUN/'cpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};failed=set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            for fut in list(active):
                if fut.done():
                    job=active.pop(fut)
                    try:fut.result()
                    except Exception:
                        failed.add(job);save(RUN/'errors'/('_'.join(map(str,job))+'.json'),dict(error=traceback.format_exc()))
            jobs=[]
            for model in base.MODELS:
                for co in ctrl.COHORTS:
                    for seed in SEEDS:
                        if complete(folder(model,seed,co)/'complete.json') and not complete(RUN/'scores'/model/str(seed)/f'{co}.json'):
                            jobs.append(('score',model,seed,co))
                    if all(complete(RUN/'scores'/model/str(s)/f'{co}.json') for s in SEEDS) and not complete(RUN/'evaluation'/model/f'{co}.json'):
                        jobs.append(('evaluate',model,co))
            for job in jobs:
                if len(active)>=workers:break
                if job not in active.values() and job not in failed:
                    fn=score_seed if job[0]=='score' else evaluate
                    active[pool.submit(fn,*job[1:])]=job
            done=report()
            save(RUN/'cpu_status.json',dict(status='complete' if done else 'running',active=list(active.values()),failed=list(failed),updated=utc()))
            if done:break
            gp=RUN/'gpu_status.json'
            if not active and gp.exists() and read(gp)['status'] in ('complete','failed'):
                save(RUN/'cpu_status.json',dict(status='failed',failed=list(failed),reason='Unfinished tasks after GPU queue exit',updated=utc()));break
            time.sleep(15)


def gpu(workers,device):
    lock=(RUN/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p,h in read(RUN/'protocol.json')['source_sha256'].items():assert sha(p)==h,p
    previous=ctrl.RUN/'gpu_status.json'
    while previous.exists() and read(previous)['status'] not in ('complete','failed'):
        save(RUN/'gpu_status.json',dict(status='waiting_for_random_controls',dependency=str(previous),updated=utc()))
        time.sleep(15)
    import torch
    assert torch.cuda.is_available()
    jobs=[(m,s) for s in SEEDS for m in base.MODELS if not all(complete(folder(m,s,c)/'complete.json') for c in ctrl.COHORTS)]
    active=[];failed=[];LOG.mkdir(parents=True,exist_ok=True)
    while jobs or active:
        for entry in active.copy():
            proc,job,handle=entry
            if proc.poll() is not None:
                active.remove(entry);handle.close()
                if proc.returncode:failed.append(job);jobs=[j for j in jobs if j[0]!=job[0]]
        while jobs and len(active)<workers:
            model,seed=jobs.pop(0);h=(LOG/f'{model}_{seed}.log').open('a')
            cmd=[sys.executable,'-u',__file__,'extract','--model',model,'--seed',str(seed),'--device',device]
            active.append((subprocess.Popen(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT),(model,seed),h))
        save(RUN/'gpu_status.json',dict(status='running',active=[dict(pid=p.pid,job=j) for p,j,_ in active],pending=len(jobs),failed=failed,updated=utc()))
        time.sleep(5)
    save(RUN/'gpu_status.json',dict(status='failed' if failed else 'complete',failed=failed,updated=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','gpu','cpu','extract','report'])
    p.add_argument('--model',choices=base.MODELS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if a.action=='register':register();report()
    elif a.action=='gpu':gpu(a.workers,a.device)
    elif a.action=='cpu':cpu(a.workers)
    elif a.action=='extract':extract(a.model,a.seed,a.device)
    else:report()
