"""Frozen scan-level FD regression, with participant-held-out evaluation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import torch
from fm_matrix.data import save,sha
from benchmark_evaluation import grouped_splits,GRID,score
from paper_paired_stats import paired,holm,weighted_scores
import run_connectivity_readouts as frozen
import run_nine_model_identity_controls as identity

RUN=ROOT/'runs/scan_embedding_fd_prediction_v1'
OUT=ROOT/'results/generated/scan_embedding_fd_prediction_v1'
FD=ROOT/'runs/motion_covariates_v1'
COHORTS=('adni','nacc','act','cogte','breathe','psopt','hrv')
KINDS=('a424','sch450','semantoks','fmrilm','harmonix')
METHODS=(*frozen.MODELS,'sf100_fc','braingfm',*('SD_'+k for k in KINDS))

def read(p):return json.loads(Path(p).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def datasets(co):return ('hrv_pre','hrv_post') if co=='hrv' else (co,)
def result_path(method,co):return RUN/'results'/co/(method+'.json')
def done(method,co):return result_path(method,co).exists()

def source(method,co):
    if method in frozen.MODELS:
        folder=frozen.RUN/'features'/method/co;stamp=read(folder/'complete.json')
        return folder/'native.npy',stamp['scan_ids'],stamp['features_sha256']['native'],folder/'complete.json'
    if method.startswith('SD_'):
        kind=method[3:];base=ROOT/'runs/sd_prediction_controls_v1'
        stamp_path=base/'results/features'/f'{co}.json';stamp=read(stamp_path)
        return base/'features'/co/(kind+'.npy'),stamp['scan_ids'],stamp['features_sha256'][kind],stamp_path
    folder=ROOT/'runs/sf100_benchmark_v1/features'/co;stamp=read(folder/'complete.json')
    return folder/(method+'.npy'),stamp['scan_ids'],stamp['files_sha256'][method],folder/'complete.json'

def register():
    RUN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);files={};coverage=[]
    for co in COHORTS:
        blocks=[]
        for ds in datasets(co):
            p=FD/'labels'/f'{ds}.csv';m=pd.read_csv(p,float_precision='round_trip');files[str(p)]=sha(p)
            m['dataset']=ds;m['row_index']=np.arange(len(m));valid=np.isfinite(m.mean_fd)
            for method in METHODS:
                fp,scans,digest,stamp=source(method,ds)
                assert scans==m.scan.tolist(),(method,ds)
                x=np.load(fp,mmap_mode='r');assert len(x)==len(m)
                valid &= np.isfinite(x).all(axis=1)
                files[str(stamp)]=sha(stamp)
            m['fd_prediction_include']=valid;blocks.append(m)
        full=pd.concat(blocks,ignore_index=True);m=full.loc[full.fd_prediction_include].reset_index(drop=True)
        assert not m.scan.duplicated().any() and m.subject.nunique()>=15
        path=RUN/'labels'/f'{co}.csv';path.parent.mkdir(exist_ok=True);m.to_csv(path,index=False)
        files[str(path)]=sha(path)
        y=np.log1p(m.mean_fd.to_numpy(float));g=m.subject.to_numpy(str);splits=[]
        for seed in (0,1,2):
            seen=np.zeros(len(m),int)
            for fold,(tr,te) in enumerate(grouped_splits(y,g,seed)):
                it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=g[tr]));it,iv=tr[it],tr[iv]
                assert not set(g[tr])&set(g[te]) and not set(g[it])&set(g[iv])
                seen[te]+=1;splits.append(dict(seed=seed,fold=fold,train=tr,test=te,inner_train=it,validation=iv))
            assert (seen==1).all()
        save(RUN/'splits'/f'{co}.json',dict(scan_ids=m.scan.tolist(),subjects=g,y_log1p=y,splits=splits))
        files[str(RUN/'splits'/f'{co}.json')]=sha(RUN/'splits'/f'{co}.json')
        coverage.append(dict(cohort=co,available_scans=len(full),fd_complete_scans=int(full.mean_fd.notna().sum()),
                             matched_scans=len(m),participants=m.subject.nunique(),additional_feature_missing=int((full.mean_fd.notna()&~full.fd_prediction_include).sum())))
    for p in [Path(__file__),Path(frozen.__file__),ROOT/'src/benchmark_evaluation.py',ROOT/'src/paper_completion.py',ROOT/'src/paper_paired_stats.py']:
        files[str(p)]=sha(p)
    protocol=dict(status='registered',models=frozen.MODELS,methods=METHODS,cohorts=COHORTS,
        target='Whole-scan mean FD; train on log1p(FD) to reduce the influence of extreme positive values. No FD-threshold exclusions. Prediction refers to concurrent motion information, not future motion or neural validity.',
        features='Nine frozen official primary scan-level readouts, frozen BrainGFM and supplied SF100 FC; five input-matched regional SD families as supplementary signal controls. No connectivity-readout sweep or post-training.',
        samples='ADNI all eligible canonical scans; NACC selected one per participant; all available canonical pre/post scans in intervention cohorts, including nonrandomized BREATHE participants for this technical outcome. HRV older-adult subgroup only. Common finite-FD/feature samples within each cohort.',
        evaluation='Five participant-grouped outer folds stratified by FD quintiles; one approximately80:20 grouped random inner split. Seeds0/1/2. Every visit of a participant stays together. Separate cohort models; no pooled-cohort or cross-cohort transfer.',
        transforms='Training-partition StandardScaler and exact PCA, capped at100 PCs and training rank, matching scan-level demographic probes. Refitted on outer-training data after ridge selection. No target-fitted feature normalization.',
        fitting='Ridge alpha=1/C, C in existing GRID; choose inner-validation Spearman correlation. Average three OOF log predictions, then inverse with expm1; negative FD predictions clipped to0 only for MAE/RMSE reporting. Spearman uses untruncated predictions.',
        references='Report train-mean FD baseline errors. Paired FM-versus-SF100 and FM-versus-input-matched-SD contrasts, plus BrainGFM-versus-SF100. SD remains a supplementary control.',
        inference='5000 participant-cluster bootstrap samples for pointwise conditional rho intervals. Paired contrasts use5000 participant bootstraps and20000 two-sided participant prediction swaps. Two separate full Holm families:70 FC-reference and63 SD-control comparisons. No refitted absolute null tests and no above-chance significance claims.',
        interpretation='Predictable FD indicates motion-related information in the representation, not that motion explains its cognitive or Re-ID performance. Whole-scan FD may not match truncated model windows.',
        automatic_shutdown=False,encoder_training=False,expected_fits=112,expected_contrasts=133,source_sha256=files)
    p=RUN/'protocol.json'
    if p.exists():
        from fm_matrix.data import clean
        assert read(p)==clean(protocol),'Registered inputs changed'
    else:save(p,protocol)
    pd.DataFrame(coverage).to_csv(OUT/'Coverage.csv',index=False)
    return protocol

def features(method,co,m):
    chunks=[]
    for ds in datasets(co):
        path,scans,digest,_=source(method,ds);assert sha(path)==digest,path
        part=m[m.dataset.eq(ds)];indices=part.row_index.to_numpy(int)
        assert [scans[i] for i in indices]==part.scan.tolist()
        chunks.append(np.load(path,mmap_mode='r')[indices])
    x=np.concatenate(chunks);assert len(x)==len(m) and np.isfinite(x).all();return x

def absolute_interval(y,pred,g,seed):
    subjects,inverse=np.unique(g,return_inverse=True);rng=np.random.default_rng(seed);draws=[]
    for start in range(0,5000,128):
        w=rng.multinomial(len(subjects),np.full(len(subjects),1/len(subjects)),size=min(128,5000-start))[:,inverse]
        draws.extend(weighted_scores(y,pred,w))
    draws=np.asarray(draws);draws=draws[np.isfinite(draws)]
    assert len(draws)>=4750
    return np.quantile(draws,[.025,.975]).tolist()

def fit(method,co):
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        started=time.time();m=pd.read_csv(RUN/'labels'/f'{co}.csv',float_precision='round_trip');spec=read(RUN/'splits'/f'{co}.json')
        assert m.scan.tolist()==spec['scan_ids'];y=np.asarray(spec['y_log1p']);g=m.subject.to_numpy(str)
        x=features(method,co,m);pred=np.full((3,len(m)),np.nan);baseline=pred.copy();choices=[]
        for f in spec['splits']:
            path=RUN/'folds'/co/method/f's{f["seed"]}f{f["fold"]}.json'
            tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
            if path.exists():rec=read(path);assert rec['test']==te.tolist()
            else:
                save(RUN/'progress'/co/(method+'.json'),dict(stage='fitting',seed=f['seed'],fold=f['fold'],updated=utc(),elapsed_seconds=time.time()-started))
                ai,bi=frozen.project_indices(x,it,iv,100,'cpu')
                best=GRID[0];value=-np.inf
                for c in GRID:
                    pr=Ridge(alpha=1/c).fit(ai,y[it]).predict(bi);v=score(y[iv],pr,False)
                    if np.isfinite(v) and v>value:best,value=c,v
                a,b=frozen.project_indices(x,tr,te,100,'cpu')
                pr=Ridge(alpha=1/best).fit(a,y[tr]).predict(b)
                rec=dict(test=te,predicted_log_fd=pr,train_mean_fd=float(m.mean_fd.iloc[tr].mean()),C=best,inner_rho=value,
                         outer_components=a.shape[1],inner_components=ai.shape[1]);save(path,rec)
            pred[f['seed'],te]=rec['predicted_log_fd'];baseline[f['seed'],te]=rec['train_mean_fd'];choices.append(dict(seed=f['seed'],fold=f['fold'],C=rec['C']))
        assert np.isfinite(pred).all();plog=pred.mean(0);praw=np.expm1(plog);yraw=m.mean_fd.to_numpy(float);clip=np.maximum(praw,0)
        metrics=dict(rho=score(yraw,praw,False),mae_mm=float(np.abs(yraw-clip).mean()),rmse_mm=float(np.sqrt(np.mean((yraw-clip)**2))),
                     train_mean_mae_mm=float(np.abs(yraw-baseline.mean(0)).mean()),train_mean_rmse_mm=float(np.sqrt(np.mean((yraw-baseline.mean(0))**2))),
                     negative_predictions_clipped_for_error=int((praw<0).sum()),rho_ci95=absolute_interval(yraw,praw,g,frozen.seed(method+'|'+co+'|FD_ci')))
        result=dict(status='complete',method=method,cohort=co,n=len(m),n_subjects=m.subject.nunique(),scan_ids=m.scan.tolist(),subjects=g,
                    y_fd=yraw,predicted_fd=praw,predicted_log_fd_by_seed=pred,metrics=metrics,selected_ridge=choices,elapsed_seconds=time.time()-started)
        save(result_path(method,co),result)
        return dict(method=method,cohort=co,seconds=time.time()-started,metrics=metrics)

def compare_cohort(co):
    rows=[]
    for model in (*frozen.MODELS,'braingfm'):
        c=read(result_path(model,co))
        references=['sf100_fc']+(['SD_'+identity.input_kind(model)] if model!='braingfm' else [])
        for reference in references:
            path=RUN/'contrasts'/co/f'{model}_vs_{reference}.json'
            if path.exists():row=read(path)
            else:
                r=read(result_path(reference,co));assert r['scan_ids']==c['scan_ids'];assert r['y_fd']==c['y_fd']
                stats=paired(c['y_fd'],r['predicted_fd'],c['predicted_fd'],c['subjects'],seed=frozen.seed(co+'|'+model+'|'+reference+'|FD'))
                row=dict(cohort=co,model=model,reference=reference,family='FD_vs_SF100' if reference=='sf100_fc' else 'FD_vs_SD',**stats);save(path,row)
            rows.append(row)
    save(RUN/'comparisons'/f'{co}.json',dict(status='complete',rows=rows));return dict(compared_cohort=co)

def job(action,method,co):
    with threadpool_limits(limits=2):return fit(method,co) if action=='fit' else compare_cohort(co)

def report_final():
    scores=[];contrasts=[]
    for co in COHORTS:
        for method in METHODS:
            d=read(result_path(method,co));scores.append(dict(cohort=co,method=method,n=d['n'],n_subjects=d['n_subjects'],**d['metrics']))
        contrasts.extend(read(RUN/'comparisons'/f'{co}.json')['rows'])
    df=pd.DataFrame(contrasts);assert len(df)==133
    for fam,ix in df.groupby('family').groups.items():
        assert len(ix)==(70 if fam=='FD_vs_SF100' else 63)
        df.loc[ix,'holm_p']=holm(df.loc[ix,'permutation_p']);df.loc[ix,'family_size']=len(ix)
    df.to_csv(OUT/'Comparisons.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'Scores.csv',index=False)
    with pd.ExcelWriter(OUT/'Scan_embedding_FD_prediction.xlsx',engine='xlsxwriter') as writer:
        for name,tab in [('Scores',pd.DataFrame(scores)),('Comparisons',df),('Coverage',pd.read_csv(OUT/'Coverage.csv')),
                         ('Protocol',pd.DataFrame([dict(item=k,value=str(v)) for k,v in read(RUN/'protocol.json').items() if k!='source_sha256']))]:
            tab.to_excel(writer,sheet_name=name,index=False)
    (OUT/'README.txt').write_text('Concurrent mean-FD prediction from frozen scan-level representations.\nNo absolute above-chance tests; rho intervals are pointwise conditional on fitted OOF predictions. Paired correction families compare methods, not a refitted no-signal null.\nAll available selected scans, including repeated intervention visits, are grouped by participant. No new model training or primary sample exclusions.\nSD is a supplementary signal control. No individual-level predictions are included in this export.\n')
    save(RUN/'export_status.json',dict(status='complete',fits=112,contrasts=133,updated=utc()))

def run():
    protocol=read(RUN/'protocol.json')
    for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,path
    lock=(RUN/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    failures=[];failed=set();active={};last=0
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
        while True:
            jobs=[('fit',m,co) for co in COHORTS for m in METHODS if not done(m,co)]
            jobs += [('compare','',co) for co in COHORTS if all(done(m,co) for m in METHODS) and not (RUN/'comparisons'/f'{co}.json').exists()]
            other=ROOT/'runs/all_cohort_fd_controls_v1/queue_status.json'
            budget=4 if other.exists() and read(other).get('status') in ('running','draining') else 12
            stopping=(RUN/'STOP_AFTER_CURRENT').exists()
            for j in jobs:
                if stopping or len(active)>=budget:break
                if j not in active.values() and j not in failed:active[pool.submit(job,*j)]=j
            count=sum(done(m,co) for co in COHORTS for m in METHODS)
            if time.time()-last>15:
                last=time.time();save(RUN/'queue_status.json',dict(status='running',pid=os.getpid(),fits_complete=count,fits_total=112,active=list(active.values()),worker_budget=budget,threads_per_worker=2,failures=failures,updated=utc(),automatic_shutdown=False))
            if not active:break
            finished,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for f in finished:
                j=active.pop(f)
                try:print(json.dumps(f.result()),flush=True)
                except Exception:
                    e=dict(job=j,error=traceback.format_exc());failures.append(e);failed.add(j);save(RUN/'errors'/('_'.join(j)+'.json'),e);print(json.dumps(e),flush=True)
    complete=all(done(m,co) for co in COHORTS for m in METHODS) and all((RUN/'comparisons'/f'{co}.json').exists() for co in COHORTS)
    if complete and not failures:report_final()
    save(RUN/'queue_status.json',dict(status='complete' if complete and not failures else 'failed' if failures else 'stopped',fits_complete=count,fits_total=112,failures=failures,updated=utc(),automatic_shutdown=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','pilot','run']);a=p.parse_args()
    if a.action=='register':register();print('Registered112 fits plus7 cohort comparison jobs.')
    elif a.action=='pilot':print(job('fit','brainlm13','psopt'))
    else:run()

if __name__=='__main__':main()
