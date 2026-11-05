"""Additive seven-cohort extension of the nine-configuration pSOPT FD protocol."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import torch
from fm_matrix.data import save, sha
import run_nine_model_identity_controls as parent
from paper_paired_stats import holm

RUN=ROOT/'runs/all_cohort_fd_controls_v1'
OUT=ROOT/'results/generated/all_cohort_fd_controls_v1'
INTAKE=ROOT/'runs/motion_covariates_v1'
MODELS=parent.MODELS
COHORTS=parent.COHORTS
THREADS=4

def read(p):return json.loads(Path(p).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def output(model,co):return RUN/'evaluation'/model/f'{co}.json'
def done(model,co):return output(model,co).exists() and read(output(model,co)).get('status')=='complete'

def pair_fd(co):
    p=parent.pair(co);values={}
    for side in ('a','b'):
        spec=p[side];m=pd.read_csv(INTAKE/'labels'/f'{spec["cohort"]}.csv',float_precision='round_trip')
        assert m.iloc[spec['indices']].scan.tolist()==spec['scans']
        values[side]=m.set_index('scan').loc[spec['scans'],'mean_fd'].to_numpy(float)
    keep=np.isfinite(values['a'])&np.isfinite(values['b'])
    assert keep.sum()>=15
    assert all((values[s][keep]>=0).all() for s in ('a','b'))
    return p,keep,np.log1p(values['a'][keep]),np.log1p(values['b'][keep])

def register():
    RUN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    intake=read(INTAKE/'protocol.json');files={str(INTAKE/'protocol.json'):sha(INTAKE/'protocol.json')}
    for co,audit in intake['sources'].items():
        for path,digest in [(ROOT/audit['source'],audit['sha256']),(ROOT/audit['canonical_manifest'],audit['canonical_sha256']),
                            (INTAKE/'labels'/f'{co}.csv',audit['output_sha256'])]:
            assert sha(path)==digest,path
            files[str(path)]=digest
    for path in [Path(__file__),Path(parent.__file__),Path(parent.old.__file__),Path(parent.rs.__file__),
                 ROOT/'src/unified_supplements.py',ROOT/'src/paper_paired_stats.py']:
        files[str(path)]=sha(path)
    coverage=[]
    for co in COHORTS:
        p,keep,fa,fb=pair_fd(co)
        pairpath=parent.old.RUN/'pairs'/f'{co}.json';files[str(pairpath)]=sha(pairpath)
        coverage.append(dict(cohort=co,existing_pairs=len(keep),fd_complete_pairs=int(keep.sum()),
                             missing_subjects=np.asarray(p['subjects'])[~keep].tolist()))
        for model in MODELS:
            for seed in parent.SEEDS:
                folder=parent.old.target(f'random{seed}',co) if model=='brainlm13' else parent.random_folder(model,seed,co)
                assert (folder/'features.npz').exists() and read(folder/'complete.json')['status']=='complete'
                files[str(folder/'complete.json')]=sha(folder/'complete.json')
            for side in ('a','b'):
                folder=parent.frozen.RUN/'features'/model/p[side]['cohort']
                stamp=read(folder/'complete.json')
                for name in ('native','connectivity'):assert (folder/f'{name}.npy').exists()
                files[str(folder/'complete.json')]=sha(folder/'complete.json')
            if co=='psopt':
                path=parent.RUN/'motion'/f'{model}.json';assert read(path)['status']=='complete';files[str(path)]=sha(path)
    protocol=dict(version=1,models=MODELS,cohorts=COHORTS,coverage=coverage,source_sha256=files,
        scope='Frozen primary and final-layer embedding connectivity only. Exact extension of archived nine-model pSOPT motion analysis; no encoder extraction/training or cognitive outcomes.',
        motion='Use scan-average FD as supplied; log1p transformation. OLS with intercept fitted jointly to both visits of calibration participants only; apply to disjoint evaluation participants.',
        splits='Participant-level train_test_split(train_size=0.2, random_state=0/1/2) after restricting to pairs with finite FD in both visits. No stratification or motion-threshold exclusions. Same galleries for all configurations and references.',
        scores='Bidirectional fractional-tie Re-ID accuracy and differential identifiability. Native feature correlation and fixed Gaussian 128-dimensional projections with seeds 0/1/2, exactly as pSOPT.',
        contrasts='FD-adjusted connectivity minus FD-adjusted primary, FD-adjusted input-window FC, FD-adjusted seed-average random connectivity, and unadjusted pretrained connectivity on the same evaluation gallery.',
        inference='5000 participant bootstraps; 20000 two-sided participant query-score swaps conditional on fitted representations, gallery, calibration split and five realized random encoders. Not absolute chance tests.',
        correction='Four new Holm families: dimensionality mode x metric. Each includes all nine configurations, seven cohorts, three splits and four contrasts (756 each). Publish corrected probabilities only after all 63 model/cohort tasks finish. Overlapping splits are not independent replications.',
        reuse='Nine original pSOPT model outputs reused after verifying FD values, participants and split membership. Their raw P values are retained and included in the new seven-cohort families; original reported corrections remain unchanged.',
        limitations='Whole-scan mean FD does not necessarily match truncated model windows; adjustment cannot establish absence of all motion confounding.',
        bootstrap=5000,swaps=20000,expected_jobs=63,expected_contrasts=3024,
        automatic_shutdown=False,manuscript_modified=False)
    from fm_matrix.data import clean
    path=RUN/'protocol.json'
    if path.exists():assert read(path)==clean(protocol),'Protocol changed'
    else:save(path,protocol)
    pd.DataFrame(coverage).to_csv(OUT/'Coverage.csv',index=False)
    return protocol

def verify():
    for path,digest in read(RUN/'protocol.json')['source_sha256'].items():
        assert sha(path)==digest,path

def state(model,co,stage,**extra):
    save(RUN/'progress'/model/f'{co}.json',dict(model=model,cohort=co,stage=stage,updated=utc(),**extra))

def load_arrays(model,co,p,keep):
    arrays={};checked={}
    for side in ('a','b'):
        item=p[side];folder=parent.frozen.RUN/'features'/model/item['cohort'];stamp=read(folder/'complete.json')
        assert [stamp['scan_ids'][i] for i in item['indices']]==item['scans']
        for name in ('native','connectivity'):
            path=folder/f'{name}.npy'
            if str(path) not in checked:checked[str(path)]=sha(path)
            assert checked[str(path)]==stamp['features_sha256'][name]
            arrays[name+'_'+side]=np.load(path,mmap_mode='r')[np.asarray(item['indices'])[keep]].copy()
        path=(ROOT/'runs/brainharmonix_f_v1/supplements/input_fc'/model/f'{item["cohort"]}.npy' if model=='brainharmonix_f'
              else ROOT/'runs/reviewer_completion_v1/input_fc'/parent.input_kind(model)/f'{item["cohort"]}.npy')
        stamp=read(path.with_suffix('.json'))
        if str(path) not in checked:checked[str(path)]=sha(path)
        assert checked[str(path)]==stamp['sha256']
        assert [stamp['scan_ids'][i] for i in item['indices']]==item['scans']
        arrays['FC_'+side]=np.load(path,mmap_mode='r')[np.asarray(item['indices'])[keep]].copy()
    for seed in parent.SEEDS:
        arrays.update({f'random{seed}_{k}':v[keep] for k,v in parent.random_arrays(model,seed,co).items()})
    assert all(np.isfinite(v).all() for v in arrays.values())
    return arrays

def compute(model,co,pilot=False):
    torch.set_num_threads(THREADS)
    started=time.time();p,keep,fa,fb=pair_fd(co);subjects=np.asarray(p['subjects'])[keep]
    if co=='psopt' and not pilot:
        previous=read(parent.RUN/'motion'/f'{model}.json')
        assert previous['n']==len(subjects)
        for seed,split in enumerate(previous['splits']):
            cal,ev=train_test_split(np.arange(len(subjects)),train_size=.2,random_state=seed)
            assert split['calibration_subjects']==subjects[cal].tolist()
            assert split['evaluation_subjects']==subjects[ev].tolist()
        motion_only=parent.rs.identification_scores(-abs(fa[:,None]-fb[None,:]))
        for k in ('per_subject_top1','per_subject_I_diff'):
            np.testing.assert_allclose(previous['motion_only'][k],motion_only[k],rtol=0,atol=1e-12)
        previous.update(cohort=co,model=model,reused_original_psopt=True,elapsed_seconds=time.time()-started)
        save(output(model,co),previous);return dict(model=model,cohort=co,reused=True,seconds=time.time()-started)
    state(model,co,'loading')
    arrays=load_arrays(model,co,p,keep);rows=[];splits=[]
    for seed in (0,1,2):
        cal,ev=train_test_split(np.arange(len(subjects)),train_size=.2,random_state=seed)
        xc=np.column_stack((np.ones(2*len(cal)),np.r_[fa[cal],fb[cal]]))
        xa=np.column_stack((np.ones(len(ev)),fa[ev]));xb=np.column_stack((np.ones(len(ev)),fb[ev]))
        adjusted={mode:{} for mode in parent.MODES}
        for name in sorted(k[:-2] for k in arrays if k.endswith('_a')):
            state(model,co,'residualize_and_score',split=seed,representation=name,elapsed_seconds=time.time()-started)
            a,b=arrays[name+'_a'],arrays[name+'_b']
            beta=np.linalg.lstsq(xc,np.r_[a[cal],b[cal]],rcond=None)[0]
            adj={name+'_a':a[ev]-xa@beta,name+'_b':b[ev]-xb@beta}
            scores=parent.old.score_arrays(adj)
            for mode in parent.MODES:adjusted[mode][name]=scores[mode][name]
            del adj,beta
        original=parent.old.score_arrays({'connectivity_a':arrays['connectivity_a'][ev],'connectivity_b':arrays['connectivity_b'][ev]})
        state(model,co,'paired_statistics',split=seed,elapsed_seconds=time.time()-started)
        for mode in parent.MODES:
            for metric in parent.METRICS:
                key='per_subject_'+metric;cand=adjusted[mode]['connectivity'][key]
                rand=np.mean([adjusted[mode][f'random{s}_connectivity'][key] for s in parent.SEEDS],axis=0)
                for ref,name in [(adjusted[mode]['native'][key],'connectivity_minus_primary'),
                                 (adjusted[mode]['FC'][key],'connectivity_minus_FC'),(rand,'connectivity_minus_random'),
                                 (original[mode]['connectivity'][key],'adjusted_minus_unadjusted')]:
                    comparison=f'split{seed}_{name}'
                    stats=parent.rs.paired_identification(ref,cand,parent.frozen.seed('|'.join((model,co,'motion',mode,metric,comparison))))
                    rows.append(dict(model=model,cohort=co,panel='motion',mode=mode,metric=metric,comparison=comparison,
                                     family=f'motion::{mode}::{metric}',**stats))
        splits.append(dict(seed=seed,calibration_subjects=subjects[cal],evaluation_subjects=subjects[ev],adjusted=adjusted,unadjusted=original))
    result=dict(status='complete',model=model,cohort=co,n=len(subjects),subjects=subjects,
                rows=rows,splits=splits,motion_only=parent.rs.identification_scores(-abs(fa[:,None]-fb[None,:])),
                elapsed_seconds=time.time()-started,reused_original_psopt=False)
    if pilot:
        assert co=='psopt'
        prior=read(parent.RUN/'motion'/f'{model}.json')
        assert len(prior['rows'])==len(rows)==48
        for old,new in zip(prior['rows'],rows):
            for k in ('reference_score','candidate_score','delta','permutation_p','ci95'):
                np.testing.assert_allclose(old[k],new[k],rtol=0,atol=1e-8,err_msg=k)
        save(RUN/'pilot'/f'{model}.json',dict(status='complete',historical_psopt_parity=True,seconds=time.time()-started,contrasts=48))
    else:save(output(model,co),result)
    state(model,co,'complete',elapsed_seconds=time.time()-started)
    return dict(model=model,cohort=co,seconds=time.time()-started,pilot=pilot)

def job(model,co,pilot=False):
    with threadpool_limits(limits=THREADS):return compute(model,co,pilot)

def export():
    rows=[];scores=[]
    for model in MODELS:
        for co in COHORTS:
            r=read(output(model,co));assert r['status']=='complete';rows.extend(r['rows'])
            for s in r['splits']:
                for setting in ('adjusted','unadjusted'):
                    for mode,names in s[setting].items():
                        for name,values in names.items():
                            scores.append(dict(model=model,cohort=co,split=s['seed'],setting=setting,mode=mode,representation=name,
                                               n_calibration=len(s['calibration_subjects']),n_evaluation=len(s['evaluation_subjects']),
                                               top1=values['top1'],I_diff=values['I_diff'],chance=values['chance']))
    d=pd.DataFrame(rows);assert len(d)==3024
    d['family']=d.family.str.replace('motion::','seven_cohort_motion::',regex=False)
    for family,ix in d.groupby('family').groups.items():
        assert len(ix)==756
        d.loc[ix,'holm_p']=holm(d.loc[ix,'permutation_p']);d.loc[ix,'family_size']=len(ix)
    d['inference_status']='final_seven_cohort_family'
    d.to_csv(OUT/'Comparisons.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'Scores.csv',index=False)
    with pd.ExcelWriter(OUT/'FD_adjusted_ReID.xlsx',engine='xlsxwriter') as writer:
        for name,table in [('Comparisons',d),('Scores',pd.DataFrame(scores)),('Coverage',pd.read_csv(OUT/'Coverage.csv')),
                           ('Protocol',pd.DataFrame([dict(item=k,value=str(v)) for k,v in read(RUN/'protocol.json').items() if k!='source_sha256']))]:
            table.to_excel(writer,sheet_name=name,index=False)
    (OUT/'README.txt').write_text('Seven-cohort FD-adjusted Re-ID supplement.\nNine frozen configurations; same protocol as pSOPT. All 3024 paired contrasts use four complete 756-test Holm families.\nOriginal pSOPT analyses and manuscript tables remain unchanged. Calibration splits overlap and are not independent replications.\nWhole-scan FD is adjusted, not window-specific FD. No new absolute-performance or cognition tests.\n')
    save(RUN/'export_status.json',dict(status='complete',contrasts=len(d),updated=utc(),output=str(OUT)))

def run(workers):
    verify();failures=[]
    lock=(RUN/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Reuse pSOPT immediately; larger ADNI tasks start early to avoid a long tail.
    for model in MODELS:
        if not done(model,'psopt'):job(model,'psopt')
    pending=[(m,co) for co in COHORTS for m in MODELS if not done(m,co)]
    active={};last=0;stopping=False
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            if (RUN/'STOP_AFTER_CURRENT').exists():stopping=True
            while pending and len(active)<workers and not stopping:
                m,co=pending.pop(0);active[pool.submit(job,m,co)]=(m,co)
            if time.time()-last>15:
                last=time.time();save(RUN/'queue_status.json',dict(status='draining' if stopping else 'running',pid=os.getpid(),workers=workers,threads_per_worker=THREADS,
                    complete=sum(done(m,co) for m in MODELS for co in COHORTS),total=63,active=list(active.values()),pending=len(pending),failures=failures,updated=utc(),automatic_shutdown=False))
            if not active:break
            finished,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in finished:
                item=active.pop(future)
                try:print(json.dumps(future.result()),flush=True)
                except Exception:
                    error=dict(job=item,error=traceback.format_exc());failures.append(error);save(RUN/'errors'/('_'.join(item)+'.json'),error);print(json.dumps(error),flush=True)
    complete=sum(done(m,co) for m in MODELS for co in COHORTS)
    if complete==63 and not failures:export()
    save(RUN/'queue_status.json',dict(status='complete' if complete==63 and not failures else 'failed' if failures else 'stopped',complete=complete,total=63,
         workers=workers,threads_per_worker=THREADS,failures=failures,updated=utc(),automatic_shutdown=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','pilot','run']);p.add_argument('--workers',type=int,default=6);p.add_argument('--model',default='brainlm13',choices=MODELS);a=p.parse_args()
    if a.action=='register':register();print('Registered 63 jobs; nine pSOPT results reusable.')
    elif a.action=='pilot':verify();print(job(a.model,'psopt',True))
    else:run(a.workers)

if __name__=='__main__':main()
