#!/usr/bin/env python3
"""Add frozen SF100 FC/BrainGFM to the task-matched intervention LOSO panel."""
from pathlib import Path
import sys
import json
import time
import os
import fcntl
import hashlib
import shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fm_matrix.data import save,sha
from fm_readout_v4.supplement import source_projection
from benchmark_evaluation import score
from paper_paired_stats import paired,holm
import sample_alignment as sa
import sf100_benchmark as sf

RUN=ROOT/'runs/sf100_loso_v1'
COHORTS=('act','cogte','breathe','psopt','hrv_pre')
METHODS=('sf100_fc','braingfm')
REFERENCES=(*sa.MODELS,'a424_FC','a424_FC-Tucker','sch450_FC','sch450_FC-Tucker')


def populations(tables,masks,domain,held):
    cohorts=COHORTS if domain=='EF' else COHORTS[:-1]
    if held not in cohorts:raise ValueError('Unsupported held-out cohort for endpoint')
    indices={co:np.flatnonzero(masks[co]&np.isfinite(tables[co][domain].to_numpy(float))) for co in cohorts}
    return cohorts,indices


def fit_held(arrays,tables,masks,domain,held):
    cohorts,indices=populations(tables,masks,domain,held)
    train=[];targets=[];subjects=[];audit=[]
    for co in cohorts:
        if co==held:continue
        m=tables[co];ix=indices[co];y=m.iloc[ix][domain].to_numpy(float)
        train.append(arrays[co][ix]);targets.append((y-y.mean())/max(y.std(),1e-8))
        subjects.extend((co+':'+m.iloc[ix].subject.astype(str)).tolist())
        audit.append(dict(cohort=co,scan_ids=m.iloc[ix].scan.tolist(),target_mean=float(y.mean()),target_sd=float(y.std())))
    ix=indices[held];m=tables[held];x=np.concatenate(train);test=arrays[held][ix]
    if not np.isfinite(x).all() or not np.isfinite(test).all():raise ValueError('Nonfinite selected features')
    prediction=source_projection(x,test,np.concatenate(targets))
    # Held-out labels are read only for reporting after the prediction is fitted.
    y=m.iloc[ix][domain].to_numpy(float)
    return dict(cohort=held,domain=domain,subjects=m.iloc[ix].subject.tolist(),scan_ids=m.iloc[ix].scan.tolist(),
                training_subjects=subjects,training_audit=audit,y=y,prediction=prediction,n=len(ix),rho=score(y,prediction,False))


def audit():
    files=[Path(__file__),ROOT/'src/sample_alignment.py',ROOT/'src/sf100_benchmark.py',
           ROOT/'src/fm_readout_v4/supplement.py',ROOT/'src/paper_paired_stats.py',
           sa.RUN/'population_manifest.json']
    for co in COHORTS:
        files += [sf.label_file(co),sf.RUN/'matches'/(co+'.json'),sf.RUN/'features'/co/'complete.json']
        files += [sf.RUN/'features'/co/(method+'.npy') for method in METHODS]
    for method in REFERENCES:
        name=method+'_frozen' if method in sa.MODELS else method
        files.append(sa.RUN/'fm_readout_v4/supplement'/name/'loso.json')
    d=dict(status='complete',files_sha256={str(p):sha(p) for p in files},
        protocol='Exactly the existing intervention LOSO: source-cohort target z-scores; source-only StandardScaler and randomized PCA<=100 (seed 0); ridge alpha=10. Frozen encoders, no target-label fitting.',
        endpoints='MEM: ACT/CogTE/BREATHE/pSOPT. EF: those four plus HRV-ER. HRV delayed recall is not baseline memory.',
        inference='45 post hoc conditional paired contrasts against SF100 FC; 5000 participant bootstraps and 20000 prediction swaps. Holm separately across 20 MEM and 25 EF contrasts; fixed held-out predictions, not a refitted training-algorithm test.',
        automatic_shutdown=False,encoder_training=False)
    p=RUN/'input_audit.json'
    if p.exists() and sa.read(p)!=d:raise ValueError('Registered LOSO inputs or code changed')
    save(p,d)


def evaluate(method):
    tables={co:sf.labels(co) for co in COHORTS}
    masks={co:sa.common_mask(tables[co],co) for co in COHORTS}
    arrays={co:sf.features(co,include_fm=False)[method] for co in COHORTS}
    rows=[]
    for domain in ('MEM','EF'):
        for held in (COHORTS if domain=='EF' else COHORTS[:-1]):
            r=fit_held(arrays,tables,masks,domain,held);r['method']=method
            for ref in REFERENCES:
                name=ref+'_frozen' if ref in sa.MODELS else ref
                old=sa.read(sa.RUN/'fm_readout_v4/supplement'/name/'loso.json')
                q=next(q for q in old['rows'] if q['cohort']==held and q['domain']==domain)
                if r['subjects']!=q['subjects'] or r['training_subjects']!=q['training_subjects'] or not np.array_equal(r['y'],q['y']):
                    raise ValueError('Existing LOSO population mismatch: '+ref+' '+held+' '+domain)
            rows.append(r);print(method,domain,held,r['n'],r['rho'],flush=True)
    save(RUN/method/'loso.json',dict(status='complete',rows=rows,input_audit_sha256=sha(RUN/'input_audit.json'),
        inference='Descriptive held-out predictions; conditional paired inference reported separately',
        evidence_status='Checkpoint ROI-order/spatial provenance qualifications retained from SF100 intake'))


def report():
    datasets={method:sa.read(RUN/method/'loso.json')['rows'] for method in METHODS}
    for ref in REFERENCES:
        name=ref+'_frozen' if ref in sa.MODELS else ref
        datasets[ref]=sa.read(sa.RUN/'fm_readout_v4/supplement'/name/'loso.json')['rows']
    scores=[dict(method=method,**{k:r[k] for k in ('cohort','domain','n','rho')}) for method,rows in datasets.items() for r in rows]
    pd.DataFrame(scores).to_csv(RUN/'scores.csv',index=False)
    contrasts=[]
    for base in datasets['sf100_fc']:
        for method in ('braingfm',*sa.MODELS):
            q=next(r for r in datasets[method] if r['cohort']==base['cohort'] and r['domain']==base['domain'])
            ident=dict(cohort=base['cohort'],domain=base['domain'],reference='sf100_fc',candidate=method,family='LOSO_'+base['domain'])
            seed=int(hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:8],16)
            stats=paired(base['y'],base['prediction'],q['prediction'],base['subjects'],seed=seed)
            stats['uncertainty']='Participant-cluster bootstrap conditional on fixed held-out-study predictions; training cohorts and fitted models not resampled'
            contrasts.append(dict(**ident,**stats))
    for family in ('LOSO_MEM','LOSO_EF'):
        rows=[r for r in contrasts if r['family']==family]
        for r,p in zip(rows,holm([r['permutation_p'] for r in rows])):r.update(holm_p=p,family_size=len(rows))
    pd.json_normalize(contrasts).to_csv(RUN/'paired_comparisons.csv',index=False)
    save(RUN/'report.json',dict(status='complete',new_prediction_tasks=18,n_score_cells=len(scores),n_comparisons=len(contrasts),
        rows=contrasts,input_audit_sha256=sha(RUN/'input_audit.json'),automatic_shutdown=False))




def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=[('audit',audit,RUN/'input_audit.json'),*[(m,lambda m=m:evaluate(m),RUN/m/'loso.json') for m in METHODS],
               ('report',report,RUN/'report.json')]
        done=[];failed={};started=time.time()
        def status(running):
            save(RUN/'queue_status.json',dict(status='finished_with_failures' if failed else 'finished' if len(done)==len(tasks) else 'running',
                controller_pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),started=started,
                counts=dict(complete=len(done),running=len(running),failed=len(failed),waiting=len(tasks)-len(done)-len(running)-len(failed)),
                running=running,failed=failed,automatic_shutdown=False))
        for key,fn,path in tasks:
            status([key])
            try:
                if key=='audit' or not path.exists():fn()
                if sa.read(path)['status']!='complete':raise ValueError('Incomplete output '+str(path))
                done.append(key)
            except Exception as e:
                failed[key]=repr(e);status([]);raise
        status([])

if __name__=='__main__':main()
