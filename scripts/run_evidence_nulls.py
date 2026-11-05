#!/usr/bin/env python3
"""Frozen primary-readout absolute tests. Additive, resumable, CPU only.

Predictive nulls refit stratified splits, scaling/PCA, tuning and heads. Repeated
observations are permuted as whole participant trajectories within visit-count
strata. Physiology uses scan-wise circular shifts and label-independent folds.
No early stopping; no manuscript edits; no shutdown.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, json, time, sys, traceback, fcntl
from pathlib import Path
from functools import lru_cache
from collections import deque
import numpy as np
from scipy.linalg import eigh
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from benchmark_evaluation import grouped_splits, predict_prepared, score
RUN=ROOT/'runs/discussion_nulls_v1'
MODELS=['brainlm13','brainlm111','brainlm650','brainjepa','semantoks','fmrilm','fmrilm_mae','fmrilm_vq_domain','brainharmonix_f']
B=1999
GRID=(.001,.01,.1,1.,10.)

def read(p):return json.loads(Path(p).read_text())
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(clean(x),indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(key):return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def feature_path(model,co):
    if model=='brainlm13':root='fm_readout_v4';name=model
    elif model in ('brainlm111','brainlm650','brainjepa'):root='fm_posttraining_matrix_v2';name=model
    elif model in ('semantoks','fmrilm'):root='new_fm_v1';name=model
    elif model in ('fmrilm_mae','fmrilm_vq_domain'):
        root='fmrilm_variants_v1/'+('mae' if model.endswith('mae') else 'vq-domain');name='fmrilm'
    else:root='brainharmonix_f_v1';name=model
    return ROOT/'runs'/root/'features'/(name+'_frozen')/(co+'.npy')

def projection(x,tr,te,k,paired=False,exact_svd=False):
    """Exact PCA via the smaller symmetric Gram matrix (same full-SVD space)."""
    sc=StandardScaler().fit(x[tr]);a=sc.transform(x[tr]).astype('float64');b=sc.transform(x[te]).astype('float64')
    k=min(k,len(tr)-1,a.shape[1])
    if exact_svd:
        pc=PCA(n_components=k,svd_solver='full').fit(a)
        return pc.transform(a),pc.transform(b)
    if paired or a.shape[0]<=a.shape[1]:
        vals,u=eigh(a@a.T,subset_by_index=[len(a)-k,len(a)-1])
        vals,u=vals[::-1],u[:,::-1]
        tol=max(float(vals[0]),1)*(1e-12 if paired else np.finfo(float).eps*max(a.shape))
        good=vals>tol
        v=(a.T@u[:,good])/np.sqrt(vals[good])[None,:]
    else:
        vals,v=eigh(a.T@a,subset_by_index=[a.shape[1]-k,a.shape[1]-1]);v=v[:,::-1]
    if not v.shape[1]:return np.zeros((len(a),1)),np.zeros((len(b),1))
    return a@v,b@v

def predictions(x,y,g,j,splits=None):
    # The archived local pipeline standardized stored float32 embeddings before
    # casting to double. Preserve that order, including its rounding behavior.
    if j['kind']=='local':x=np.asarray(x,dtype=np.float32)
    by=[];audit=[]
    for s in range(3):
        prep=[]
        folds=[f for f in splits if f['seed']==s] if splits is not None else None
        outer=[(np.asarray(f['train']),np.asarray(f['test'])) for f in folds] if folds is not None else grouped_splits(y,g,s)
        for fi,(tr,te) in enumerate(outer):
            if folds is not None:
                it,iv=np.asarray(folds[fi]['inner_train']),np.asarray(folds[fi]['validation'])
            else:
                if j['task']=='arm':
                    a,b=next(StratifiedShuffleSplit(1,test_size=.2,random_state=s).split(tr,y[tr]))
                else:a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=s).split(tr,groups=g[tr]))
                it,iv=tr[a],tr[b]
            assert not set(g[tr])&set(g[te]) and not set(g[it])&set(g[iv])
            op=projection(x,tr,te,j['components'],j['kind']=='paired',j['kind']=='local' and j['metric']=='auc')
            ip=projection(x,it,iv,j['components'],j['kind']=='paired',j['kind']=='local' and j['metric']=='auc')
            prep.append((tr,te,op,it,iv,ip))
            audit.append(dict(seed=s,fold=fi,train=tr,test=te,inner_train=it,validation=iv))
        if j['task']=='arm':
            nc=j['n_classes'];p=np.full((len(y),nc),np.nan)
            for tr,te,(a,b),it,iv,(ai,bi) in prep:
                best=GRID[0];vbest=-np.inf
                for c in GRID:
                    pp=class_predict(ai,y[it],bi,c,nc);v=metric(y[iv],pp,j)
                    if np.isfinite(v) and v>vbest:best,vbest=c,v
                p[te]=class_predict(a,y[tr],b,best,nc)
        else:p=predict_prepared(prep,y,binary=j['metric']=='auc')
        if not np.isfinite(p).all():raise ValueError('Nonfinite OOF predictions')
        by.append(p)
    return np.mean(by,axis=0),audit

def class_predict(a,y,b,c,nc):
    model=LogisticRegression(C=c,max_iter=2000,solver='lbfgs').fit(a,y)
    p=np.zeros((len(b),nc));p[:,model.classes_.astype(int)]=model.predict_proba(b)
    return p
def metric(y,p,j):
    if j['task']=='arm':
        if len(np.unique(y))!=j['n_classes']:return np.nan
        return float(roc_auc_score(y,p[:,1]) if j['n_classes']==2 else roc_auc_score(y,p,average='macro',multi_class='ovr'))
    return score(y,p,j['metric']=='auc')

def permuted(y,g,rng):
    """Whole trajectories, exchangeable within identical observation counts."""
    groups={}
    for person in np.unique(g):
        ix=np.flatnonzero(g==person);groups.setdefault(len(ix),[]).append(ix)
    out=y.copy()
    for blocks in groups.values():
        for target,source in zip(blocks,rng.permutation(len(blocks))):out[target]=y[blocks[source]]
    return out

def prepare():
    import pandas as pd
    from fm_readout_v4.data import metadata
    import sf100_benchmark as sf
    from unittest.mock import patch
    from fm_matched_v3 import physiology as ph
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'manifest.json').exists():return read(RUN/'manifest.json')
    jobs=[];sources={}
    def record(p):
        p=Path(p);sources[str(p)]=sha(p)
    def add(j,x,y,g,extra=None):
        j['id']='__'.join([j['model'],j['cohort'],j['task']])
        j['permutations']=20000 if j['kind']=='identity' else B
        j['family']='above_chance::'+j.get('display_task',j['task'])
        path=RUN/'inputs'/(j['id']+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez(path,x=np.asarray(x,dtype=np.float64),y=y,groups=np.asarray(g,dtype=str),**(extra or {}))
        j['input']=str(path);j['input_sha256']=sha(path);jobs.append(j)
    for co in ['adni','nacc','act','cogte','breathe','psopt','hrv_pre']:
        m=metadata(co)
        tasks=['AGE','SEX','DX','MEM','EF'] if co in ('adni','nacc') else ['MEM_followup','EF_followup'] if co!='hrv_pre' else ['MEM_delayed_recall','EF_followup']
        for task in tasks:
            p=sf.RUN/'evaluation'/co/(task+'.json');record(p);d=read(p)
            ix=m.set_index('scan').index.get_indexer(d['scan_ids']);assert (ix>=0).all()
            assert m.iloc[ix].subject.astype(str).tolist()==d['subjects']
            splits=d['split_audit'];y=np.asarray(d['y']);g=np.asarray(d['subjects'])
            # Keep original trajectory ordering and report it; no scan-wise shuffle.
            for model in MODELS:
                fp=feature_path(model,co);record(fp);x=np.load(fp,mmap_mode='r')[ix]
                j=dict(model=model,cohort=co,task=task,display_task='MEM_followup' if task=='MEM_delayed_recall' else task,
                    kind='local',metric='auc' if task in ('SEX','DX') else 'rho',components=5 if d['covariates'] else 100,
                    original_covariates=d['covariates'],imaging_only=True,reference_file=str(p),splits=splits,scan_ids=d['scan_ids'])
                if model in d['predictions'] and not d['covariates']:j['archived_prediction']=d['predictions'][model]
                elif not d['covariates']:
                    ep=fp.parent.parent.parent/'evaluation'/fp.parent.name/co/(task+'.json')
                    if ep.exists():record(ep);j['archived_prediction']=read(ep)['prediction']
                add(j,x,y,g)
    p=ROOT/'runs/intervention_prepost_v1/inputs.json';record(p);plans=read(p)['plans']
    for co,plan in plans.items():
        for task,t in plan['tasks'].items():
            for model in MODELS:
                pa=feature_path(model,plan['pre_cohort']);pb=feature_path(model,plan['post_cohort']);record(pa);record(pb)
                x=np.asarray(np.load(pb,mmap_mode='r')[t['post_indices']],float)-np.asarray(np.load(pa,mmap_mode='r')[t['pre_indices']],float)
                j=dict(model=model,cohort=co,task=task,kind='paired',metric='auc' if task=='arm' else 'rho',components=5,
                    n_classes=len(t['classes']),classes=t['classes'],splits=t['splits'],reference_file=str(p),input_mode='change',imaging_only=True)
                add(j,x,np.asarray(t['y']),np.asarray(t['subjects']))
    for model in MODELS:
        base=ROOT/'runs'/('brainharmonix_f_window_v1' if model=='brainharmonix_f' else 'benchmark_gaps_v1/physiology')
        for co in ['breathe','psopt','hrv']:
            fp=base/'features'/(model+'_frozen')/(co+'.npy');record(fp)
            folder=base/co;w=pd.read_csv(folder/'windows.csv');record(folder/'windows.csv')
            x=np.load(fp);scans=w.scan.to_numpy();y=ph.center_scans(w.rate.to_numpy(),scans)
            nuisance=pd.get_dummies(w.Timepoint+'_'+w.WindowIdx.astype(str),dtype=float).to_numpy();nuisance=ph.center_scans(nuisance,scans)
            cache=RUN/'physiology_cache'/model/co;cache.mkdir(parents=True,exist_ok=True)
            for name in ['windows.csv','scans.csv','protocol.json',*[f'dynamic_splits_s{s}.json' for s in range(3)]]:
                origin=folder/name;record(origin);link=cache/name
                if not link.exists():link.symlink_to(origin)
            with patch.object(ph,'RUN',cache):
                a=ph.cached_transforms(nuisance,'dynamic',nuisance.shape[1]);b=ph.cached_transforms(ph.center_scans(x,scans),'dynamic',10)
            arr={};splits=[]
            for s in range(3):
                for f,sp in enumerate(read(folder/f'dynamic_splits_s{s}.json')):
                    for k in range(4):arr[f's{s}_f{f}_{k}']=np.hstack([a[s][f][k],b[s][f][k]])
                    splits.append(dict(seed=s,fold=f,**sp))
            rp=base/'evaluation'/(model+'_frozen')/(co+'.json');record(rp)
            rr=next(r for r in read(rp)['rows'] if r['scope']=='dynamic')
            pred=rr.get('prediction',rr.get('predictions',{}).get('FM'))
            j=dict(model=model,cohort=co,task='physiology',kind='physiology',metric='rho',splits=splits,reference_file=str(rp),archived_prediction=pred,
                null='Independent uniform cyclic shifts (including identity) within each ordered retained scan; refit nuisance-plus-embedding ridge heads and tuning')
            add(j,np.empty((len(y),0)),y,w.Subject.to_numpy(),dict(scans=scans.astype(str),**arr))
    p=ROOT/'runs/brainharmonix_f_v1/evaluation/brainharmonix_f_frozen/identification.json';record(p)
    for r in read(p)['rows']:
        add(dict(model='brainharmonix_f',cohort=r['cohort'],task='top1',kind='identity',metric='accuracy',reference_file=str(p)),r['similarity'],np.arange(len(r['subjects'])),r['subjects'])
    assert len(jobs)==340,len(jobs)
    import collections
    protocol=dict(created_utc=utc(),permutations=B,identity_permutations=20000,split_seeds=[0,1,2],automatic_shutdown=False,
        scope='Frozen primary readouts only; 12 display columns; no fixed-source transfer or LOSO; follow-up recall merged in display only; paired inputs post-minus-pre only; physiology independent-window within-scan only.',
        families='One Holm family per predictive display task across all nine configurations and eligible cohorts. Native identity top1: retain all 168 historical native-readout tests and add seven BrainHarmonix-F tests; do not shrink the historical family.',
        repeated_measures='ADNI participant outcome trajectories permuted only among participants with the same retained visit count. All visits stay grouped in evaluation. Session weighting retained.',
        refitting='Outcome-stratified outer folds regenerated for every permutation; grouped inner splits regenerated (arm-stratified for arm). Scaler, exact PCA, ridge/logistic tuning and heads refitted. Scores use seed-averaged OOF predictions.',
        followup='Imaging-only above-chance tests. Existing clinical-plus-imaging scores are not evidence of embedding-only validity. Matched imaging-only FC/BrainGFM reference fits and paired contrasts will be added in this release; original results remain intact.',
        physiology='Label-independent archived participant folds permit caching unsupervised transforms. Circular shifts retain within-scan serial order up to circular origin; validity assumes cyclic exchangeability, not arbitrary white-noise labels. Includes nuisance covariates; tests temporal alignment, not uniquely neural information.',
        statistics='One-sided finite p=(1+exceedances)/(B+1). Fixed budgets, no significance-dependent stopping. Undefined scores fail explicitly. Reused old identity raw p retained before enlarged correction.',
        counts=dict(collections.Counter(j['kind'] for j in jobs)),family_sizes=dict(collections.Counter(j['family'] for j in jobs)))
    result=dict(protocol=protocol,jobs=jobs,source_sha256=sources,code_sha256=sha(__file__))
    for j in jobs:save(RUN/'job_specs'/(j['id']+'.json'),j)
    save(RUN/'manifest.json',result);save(RUN/'protocol.json',protocol)
    print('Prepared',len(jobs),protocol['counts'],flush=True)
    return result

@lru_cache(maxsize=2)
def payload(id):
    j=read(RUN/'job_specs'/(id+'.json'))
    if sha(j['input'])!=j['input_sha256']:raise ValueError('Input changed')
    with np.load(j['input']) as z:d={k:z[k] for k in z.files}
    return j,d

def centered(x,scans):
    out=np.asarray(x,float).copy()
    for s in np.unique(scans):
        ix=np.flatnonzero(scans==s);out[ix]-=out[ix].mean(0)
    return out
def corr_many(y,p):
    a=rankdata(y,axis=0);b=rankdata(p,axis=0);a-=a.mean(0);b-=b.mean(0)
    return (a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))
def ridge_many(a,y,b,c):
    z=a-a.mean(0);coef=np.linalg.solve(z.T@z+np.eye(a.shape[1])/c,z.T@(y-y.mean(0)))
    return (b-a.mean(0))@coef+y.mean(0)
def physiological(j,d,ys):
    pp=[]
    for s in range(3):
        p=np.full(ys.shape,np.nan)
        for sp in (sp for sp in j['splits'] if sp['seed']==s):
            f=sp['fold'];tr,te,it,iv=[np.asarray(sp[k]) for k in ('train','test','inner_train','inner_validation')]
            a,b,ai,bi=[d[f's{s}_f{f}_{k}'] for k in range(4)]
            vals=np.stack([corr_many(ys[iv],ridge_many(ai,ys[it],bi,c)) for c in GRID]);choice=np.argmax(np.where(np.isfinite(vals),vals,-np.inf),axis=0)
            for k,c in enumerate(GRID):
                cols=np.flatnonzero(choice==k)
                if len(cols):p[np.ix_(te,cols)]=ridge_many(a,ys[tr][:,cols],b,c)
        pp.append(centered(p,d['scans']))
    p=np.mean(pp,axis=0)
    if not np.isfinite(p).all():raise ValueError('Incomplete physiology prediction')
    return p

def observed(id):
    p=RUN/'observed'/(id+'.json')
    if p.exists():return read(p)
    j,d=payload(id);t=time.monotonic();x,y,g=d['x'],d['y'],d['groups']
    if j['kind']=='identity':
        from unified_supplements import absolute_null
        r=absolute_null(x,seed('identity-null:'+j['cohort']),20000)['top1']
        save(RUN/'identity'/(id+'.json'),r);result=dict(score=r['score'],n=len(y))
    else:
        if j['kind']=='physiology':pred=physiological(j,d,y[:,None])[:,0]
        else:
            pred,audit=predictions(x,y,g,j)
            if clean(audit)!=j['splits']:raise ValueError('Observed regenerated folds differ from archived splits')
        if j.get('archived_prediction') is not None:
            delta=float(np.max(abs(pred-np.asarray(j['archived_prediction']))))
            archived_score=metric(y,np.asarray(j['archived_prediction']),j)
            score_delta=metric(y,pred,j)-archived_score
            # Classification hyperparameter choice can change at near-tied
            # validation AUCs under different LAPACK/solver roundoff. The null
            # test MUST use its own observed refit, never an archived statistic
            # from a numerically different fit against a newly fitted null.
            if j['metric']!='auc' and delta>2e-5:raise ValueError(f'Nonclassification prediction reproduction failed: {delta}')
        else:delta=None;archived_score=None;score_delta=None
        result=dict(score=metric(y,pred,j),prediction=pred,n=len(y),n_subjects=len(np.unique(g)),archived_prediction_max_abs_difference=delta,
            archived_score=archived_score,score_difference_from_archive=score_delta,
            numerical_reproducibility='Observed and null refitted using the identical current CPU implementation; archived results never substituted or overwritten',
            distinct_followup_imaging_only=bool(j.get('original_covariates')))
    if not np.isfinite(result['score']):raise ValueError('Undefined observed metric')
    result.update(id=id,seconds=time.monotonic()-t,completed_utc=utc());save(p,result);return clean(result)

def chunk(id,start,stop):
    j,d=payload(id)
    if not 0<=start<stop<=j['permutations']:raise ValueError('Chunk exceeds the active permutation budget')
    dest=RUN/'chunks'/id/f'{start:05d}_{stop:05d}.json'
    if dest.exists():return dict(id=id,start=start,stop=stop,cached=True)
    obs=observed(id);y,g=d['y'],d['groups'];t=time.monotonic()
    if j['kind']=='physiology':
        ys=[]
        for b in range(start,stop):
            rng=np.random.default_rng(seed(j['cohort']+'::physiology::'+str(b)));yp=y.copy()
            for scan in np.unique(d['scans']):
                ix=np.flatnonzero(d['scans']==scan);yp[ix]=np.roll(y[ix],int(rng.integers(len(ix))))
            ys.append(yp)
        ys=np.stack(ys,axis=1);vals=corr_many(ys,physiological(j,d,ys))
    else:
        vals=[]
        for b in range(start,stop):
            rng=np.random.default_rng(seed(j['cohort']+'::'+j['task']+'::'+str(b)))
            yp=permuted(y,g,rng);pred,_=predictions(d['x'],yp,g,j);vals.append(metric(yp,pred,j))
    if not np.isfinite(vals).all():raise ValueError('Undefined null metric; cannot discard permutations')
    r=dict(id=id,start=start,stop=stop,values=vals,exceedances=int(np.sum(np.asarray(vals)>=obs['score']-1e-12)),seconds=time.monotonic()-t,completed_utc=utc())
    save(dest,r);return dict(id=id,start=start,stop=stop,seconds=r['seconds'])

def work(item):
    try:
        if item[0]=='followup_reference':return followup_reference(item[1],item[2])
        if len(item)==1:return dict(id=item[0],observed=observed(item[0]))
        return chunk(*item)
    except Exception:
        error=dict(item=item,error=traceback.format_exc(),time_utc=utc());save(RUN/'errors'/('__'.join(map(str,item))+'.json'),error);return error

def followup_reference(co,task):
    """Matched imaging-only controls; not the archived clinical+imaging scores."""
    dest=RUN/'followup_references'/(co+'__'+task+'.json')
    if dest.exists():return dict(id=str(dest),cached=True)
    import pandas as pd
    from fm_readout_v4.data import metadata
    from paper_paired_stats import paired
    j,d=payload('__'.join([MODELS[0],co,task]));m=metadata(co)
    ix=m.set_index('scan').index.get_indexer(j['scan_ids']);assert (ix>=0).all()
    refs={};rows=[];files={}
    for method in ['sf100_fc','braingfm']:
        fp=ROOT/'runs/sf100_benchmark_v1/features'/co/(method+'.npy');files[str(fp)]=sha(fp)
        x=np.asarray(np.load(fp,mmap_mode='r')[ix],float)
        pred,_=predictions(x,d['y'],d['groups'],j)
        refs[method]=dict(prediction=pred,score=metric(d['y'],pred,j))
        for model in MODELS:
            op=RUN/'observed'/('__'.join([model,co,task])+'.json')
            if not op.exists():raise ValueError('Missing validated FM prediction: '+str(op))
            obs=read(op);rows.append(dict(model=model,cohort=co,task=j.get('display_task',task),reference=method,
                **paired(d['y'],pred,np.asarray(obs['prediction']),d['groups'],n_boot=5000,n_perm=20000,seed=seed(co+task+model+method))))
    save(dest,dict(status='complete',cohort=co,task=task,imaging_only=True,components=j['components'],subjects=d['groups'],y=d['y'],references=refs,rows=rows,source_sha256=files))
    return dict(id=str(dest),status='complete')

def chunk_ranges(j):
    size=100 if j['kind']=='physiology' else 5 if j['cohort'] in ('adni','nacc') else 25
    return [(start,min(j['permutations'],start+size)) for start in range(0,j['permutations'],size)]

def take_next(fast,heavy,active_fast,workers,fast_workers):
    # Reserved slots keep short jobs moving while the observational tasks run.
    # When either pending lane empties, the other can use all remaining slots.
    if fast and (active_fast<min(fast_workers,workers) or not heavy):return fast.popleft(),'fast'
    if heavy:return heavy.popleft(),'heavy'
    if fast:return fast.popleft(),'fast'
    return None,None

def run(workers,fast_workers=16):
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    manifest=read(RUN/'manifest.json')
    if manifest['code_sha256']!=sha(__file__):raise ValueError('Code changed since registration')
    if manifest['protocol']['permutations']!=B or any(j['permutations']!=(20000 if j['kind']=='identity' else B) for j in manifest['jobs']):raise ValueError('Manifest budget differs from active budget')
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=manifest['jobs'];pending=[]
        # All observed validations first; a failed validation blocks that job.
        for j in jobs:
            if not (RUN/'observed'/(j['id']+'.json')).exists():pending.append((j['id'],))
        failures=[];active={};completed=0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            phase='validate_observed';fast=deque(pending);heavy=deque();stopping=False
            while True:
                if (RUN/'STOP_AFTER_CURRENT').exists():
                    stopping=True;phase='stopping'
                while not stopping and len(active)<workers:
                    item,lane=take_next(fast,heavy,sum(v[2]=='fast' for v in active.values()),workers,fast_workers)
                    if item is None:break
                    active[pool.submit(work,item)]=(item,time.monotonic(),lane)
                save(RUN/'queue_status.json',dict(status='running',phase=phase,controller_pid=os.getpid(),workers=workers,updated_utc=utc(),
                    predictive_permutations=B,fast_workers_reserved=fast_workers,pending_fast=len(fast),pending_heavy=len(heavy),
                    completed_work_units=completed,failed=failures,running=[dict(item=v[0],elapsed_seconds=time.monotonic()-v[1],lane=v[2]) for v in active.values()],automatic_shutdown=False))
                if not active:
                    if phase=='validate_observed':
                        phase='permutations';fast_items=[];heavy_items=[]
                        valid=[j for j in jobs if j['kind']!='identity' and (RUN/'observed'/(j['id']+'.json')).exists()]
                        for j in valid:
                            lane=heavy_items if j['kind']=='local' and j['cohort'] in ('adni','nacc') else fast_items
                            for start,end in chunk_ranges(j):
                                if not (RUN/'chunks'/j['id']/f'{start:05d}_{end:05d}.json').exists():lane.append((j['id'],start,end))
                        heavy_items.sort(key=lambda v:v[1])
                        fast_items.sort(key=lambda v:(0 if 'physiology' in v[0] else 1,v[1]))
                        refs=sorted({(j['cohort'],j['task']) for j in jobs if j['kind']=='local' and j.get('display_task',j['task']) in ('MEM_followup','EF_followup')})
                        fast=deque([('followup_reference',co,task) for co,task in refs]+fast_items);heavy=deque(heavy_items);continue
                    break
                done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:
                    item,_,lane=active.pop(f);r=f.result();completed+=1
                    if 'error' in r:
                        failures.append(r);print('FAILED',item,r['error'],flush=True)
                    else:print('DONE',json.dumps({k:v for k,v in r.items() if k!='observed'}),flush=True)
        report()
        save(RUN/'queue_status.json',dict(status='stopped' if phase=='stopping' else 'failed' if failures else 'complete',controller_pid=os.getpid(),workers=workers,predictive_permutations=B,failed=failures,updated_utc=utc(),automatic_shutdown=False))

def report():
    import pandas as pd
    m=read(RUN/'manifest.json');rows=[]
    reference_rows=[]
    for p in (RUN/'followup_references').glob('*.json'):reference_rows.extend(read(p)['rows'])
    if reference_rows:
        refs=pd.DataFrame(reference_rows);refs['holm_p']=np.nan
        for (task,ref),ix in refs.groupby(['task','reference']).groups.items():
            if len(ix)==45:refs.loc[ix,'holm_p']=holm(refs.loc[ix,'permutation_p'])
        refs.to_csv(RUN/'followup_imaging_only_comparisons.csv',index=False)
    for j in m['jobs']:
        op=RUN/'observed'/(j['id']+'.json')
        if not op.exists():continue
        obs=read(op);chunks=[read(p) for p in sorted((RUN/'chunks'/j['id']).glob('*.json'))]
        if j['kind']=='identity':
            r=read(RUN/'identity'/(j['id']+'.json'));n=r['permutations'];hits=r['exceedances']
        else:
            intervals=sorted((r['start'],r['stop']) for r in chunks)
            if any(a<0 or b>j['permutations'] for a,b in intervals):raise ValueError('Saved chunk outside active budget: '+j['id'])
            if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Overlapping permutation chunks: '+j['id'])
            n=sum(r['stop']-r['start'] for r in chunks);hits=sum(r['exceedances'] for r in chunks)
            if n==j['permutations'] and (intervals[0][0]!=0 or intervals[-1][1]!=j['permutations']):raise ValueError('Invalid permutation coverage')
        rows.append(dict(id=j['id'],model=j['model'],cohort=j['cohort'],task=j.get('display_task',j['task']),family=j['family'],observed_score=obs['score'],
            completed_permutations=n,required_permutations=j['permutations'],raw_p=(1+hits)/(1+n) if n==j['permutations'] else None,
            status='complete' if n==j['permutations'] else 'running',seconds=sum(r['seconds'] for r in chunks)))
    df=pd.DataFrame(rows)
    if len(df):
        df['holm_p']=np.nan
        for family,ix in df.groupby('family').groups.items():
            expected=sum(j['family']==family for j in m['jobs'])
            if len(ix)!=expected or df.loc[ix,'raw_p'].isna().any():continue
            if family.endswith('::top1'):
                old=pd.read_csv(ROOT/'results/generated/brainharmonix_f_v1/joint/Comparisons.csv',low_memory=False)
                old=old[old.panel.eq('identity_absolute_null')&old.task.eq('top1')&old['mode'].eq('native')]
                assert len(old)==168
                p=np.r_[old.raw_p.to_numpy(),df.loc[ix,'raw_p'].to_numpy()];adj=holm(p)
                old=old.copy();old['holm_p_expanded_native_family']=adj[:168];old.to_csv(RUN/'identity_retained_family.csv',index=False)
                df.loc[ix,'holm_p']=adj[168:]
            else:df.loc[ix,'holm_p']=holm(df.loc[ix,'raw_p'])
        df.to_csv(RUN/'results.csv',index=False)
    save(RUN/'progress.json',dict(updated_utc=utc(),total_jobs=len(m['jobs']),observed_validated=len(df),complete_jobs=int(df.status.eq('complete').sum()) if len(df) else 0,
        completed_permutations=int(df.completed_permutations.sum()) if len(df) else 0,total_permutations=sum(j['permutations'] for j in m['jobs']),automatic_shutdown=False))
    print(read(RUN/'progress.json'),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','report','benchmark']);ap.add_argument('--workers',type=int,default=28);ap.add_argument('--fast-workers',type=int,default=16);ap.add_argument('--job');ap.add_argument('--count',type=int,default=1);a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a.workers,a.fast_workers)
    elif a.action=='report':report()
    else:
        print(work((a.job,)),flush=True)
        if 'top1' not in a.job:print(work((a.job,0,a.count)),flush=True)
