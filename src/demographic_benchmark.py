"""Frozen/native-SSL demographic probes with unchanged model readouts.

Fits downstream probes only. Source-exposed SSL evaluations are explicitly
separated from external-cohort encoder transfer. HRV age bins are categorical.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,balanced_accuracy_score,log_loss,roc_auc_score
from benchmark_evaluation import prepare_nested,predict_prepared,score,GRID
from fm_matrix.data import ROOT,DATA as ORIGINAL_DATA,RUN as ORIGINAL_RUN,save,sha
from fm_readout_v4 import evaluate as readout
from fm_readout_v4.models import READOUTS

RUN=ROOT/'runs/demographic_benchmark_v1'
COHORTS=('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post')
MODELS=('brainlm13','brainlm111','brainlm650','brainjepa')
AGE_BINS=('54-56','57-60','61-63','64-66','67-69','70-75','76-80')


def states():
    for model in MODELS:
        yield ('none',model,'frozen','none',0)
        for source in ('adni','nacc'):
            for seed in (42,43,44):yield (source,model,'native_ssl','none',seed)


def subject_key(series):
    return series.astype(str).str.strip().str.replace(r'^sub-','',regex=True)


def labels(cohort,table,uds=None):
    m=table.copy()
    if m.scan.duplicated().any():raise ValueError('Duplicate scan identifier')
    if cohort=='nacc':
        u=uds.copy();u['_key']=subject_key(u.NACCID)
        if u._key.duplicated().any():raise ValueError('Ambiguous NACC UDS participant match')
        code=pd.to_numeric(u.SEX,errors='raise')
        if not set(code.dropna().unique())<={1,2}:raise ValueError('Unrecognized UDS sex code')
        lookup=pd.Series(code.map({1:1.,2:0.}).to_numpy(),index=u._key)
        m['sex']=subject_key(m.subject).map(lookup).to_numpy()
    if cohort.startswith('hrv'):
        raw=m.sex.astype('string').str.strip().str.lower()
        if not set(raw.dropna())<={'male','female'}:raise ValueError('Unrecognized HRV sex code')
        m['sex']=raw.map({'male':1.,'female':0.}).astype(float)
        if not set(m.age_category.dropna())<=set(AGE_BINS):raise ValueError('Unregistered HRV age interval')
        m['age_group']=m.age_category.map({s:i for i,s in enumerate(AGE_BINS)})
    else:
        m['age']=pd.to_numeric(m.age,errors='raise')
        if np.isinf(m.age.to_numpy()).any():raise ValueError('Infinite age')
    if not set(m.sex.dropna().unique())<={0,1}:raise ValueError('Sex must have audited binary codes')
    m['primary_include']=m.eligible.astype(bool)
    if cohort in ('act','cogte','breathe','psopt'):m['primary_include'] &= m.visit.eq('T1')
    return m


def prepare():
    files=[ORIGINAL_RUN/'intake.json',ROOT/'runs/fm_readout_v4/intake.json',ROOT/'data/NACC/uds_SelectedVariables.csv',
           *[ORIGINAL_DATA/(c+'.csv') for c in COHORTS]]
    signature={str(p):sha(p) for p in files};out=RUN/'intake.json'
    if out.exists():
        d=json.loads(out.read_text())
        if d['input_sha256']!=signature:raise ValueError('Registered demographic inputs changed')
        for co,info in d['cohorts'].items():
            if sha(RUN/'labels'/(co+'.csv'))!=info['label_sha256']:raise ValueError('Registered demographic labels changed')
        return d
    u=pd.read_csv(ROOT/'data/NACC/uds_SelectedVariables.csv');summary={}
    for co in COHORTS:
        table=labels(co,pd.read_csv(ORIGINAL_DATA/(co+'.csv')),u)
        dest=RUN/'labels'/(co+'.csv');dest.parent.mkdir(parents=True,exist_ok=True);table.to_csv(dest,index=False)
        keep=table.primary_include
        tasks={'SEX':'sex',('AGE_GROUP' if co.startswith('hrv') else 'AGE'):('age_group' if co.startswith('hrv') else 'age')}
        counts={task:dict(n=int((keep&table[col].notna()).sum()),n_subjects=int(table.loc[keep&table[col].notna(),'subject'].nunique())) for task,col in tasks.items()}
        if co=='nacc' and counts['SEX']['n']!=int(keep.sum()):raise ValueError('NACC sex merge left unmatched participants')
        summary[co]=dict(total_scans=len(table),primary_scans=int(keep.sum()),tasks=counts,label_sha256=sha(dest))
    d=dict(status='complete',input_sha256=signature,cohorts=summary,
        protocol='Frozen and source-native-SSL embeddings; four original model readouts (13M final CLS added); no encoder training.',
        labels='NACC sex: participant-key match to uploaded UDS SEX (1 male/2 female), encoded male=1. ADNI/HRV male=1. Other RCTs retain their audited native 0/1 codes; no cross-cohort sex pooling.',
        age='Available continuous age in ADNI/NACC/four original RCTs. HRV uses seven released age intervals as an exploratory multiclass task; no midpoint imputation or age-in-years claim.',
        eligibility='ADNI repeated scans grouped by participant; NACC selected primary scans; four RCTs eligible baseline scans; HRV baseline and post separately.',
        ssl_scope='Source-cohort probes use a source-exposed encoder and are not independent within-source encoder validation. Other cohorts are external to this adaptation; pretraining overlap is not resolved here.')
    save(out,d);return d


def feature_folder(state):return RUN/'features'/readout.tag(*state)


def source_features(state):
    source,model,objective,target,seed=state
    if model=='brainlm13':return readout.folder(*state)
    from fm_matrix.evaluate import feature_dir
    return feature_dir(*state)


def feature_registration(state,device='cpu'):
    prepare();source,model,objective,target,seed=state;p=source_features(state)
    from fm_matrix.train import location
    checkpoint=None if objective=='frozen' else location(source,model,objective,target,seed,'external')/'adapter.pt'
    if not (p/'complete.json').exists():
        if model!='brainlm13' or objective!='native_ssl':raise FileNotFoundError(p)
        if device!='cuda':raise ValueError('Missing 13M CLS exports require the GPU lease')
        readout.extraction(*state,device)
    d=json.loads((p/'complete.json').read_text());s=d['signature']
    key='checkpoint_sha256' if model=='brainlm13' else 'adapter_sha256'
    parent=ROOT/'runs/fm_readout_v4/intake.json' if model=='brainlm13' else ORIGINAL_RUN/'intake.json'
    if s['model']!=model or s[key]!=(sha(checkpoint) if checkpoint else None) or s['intake_sha256']!=sha(parent):raise ValueError('Feature checkpoint/intake mismatch')
    actual=d['encoder'].get('readout')
    if actual!=READOUTS[model]:raise ValueError('Unexpected model readout')
    signature=dict(state=list(state),intake_sha256=sha(RUN/'intake.json'),parent_feature_sha256=sha(p/'complete.json'))
    out=feature_folder(state);done=out/'complete.json'
    if done.exists():
        if json.loads(done.read_text())['signature']!=signature:raise ValueError('Stale demographic feature reference')
        return
    out.mkdir(parents=True,exist_ok=True);hashes={}
    for co in COHORTS:
        src=p/(co+'.npy');dest=out/src.name;arr=np.load(src,mmap_mode='r');m=pd.read_csv(RUN/'labels'/(co+'.csv'))
        if arr.ndim!=2 or len(arr)!=len(m) or not np.isfinite(arr).all():raise ValueError('Invalid feature array')
        if dest.exists() and dest.resolve()!=src.resolve():raise ValueError('Unexpected feature link')
        if not dest.exists():dest.symlink_to(src)
        hashes[co]=sha(src)
    save(done,dict(status='complete',signature=signature,readout=actual,source=str(p),feature_sha256=hashes,encoder_training_performed=False))


def multiclass_predict(a,y,b,c,classes):
    counts=np.bincount(y.astype(int),minlength=classes)
    if np.count_nonzero(counts)<2:return np.tile(counts/counts.sum(),(len(b),1))
    model=LogisticRegression(C=c,max_iter=5000).fit(a,y)
    out=np.zeros((len(b),classes));out[:,model.classes_.astype(int)]=model.predict_proba(b)
    return out


def predict_multiclass(prepared,y,classes):
    out=np.full((len(y),classes),np.nan)
    for tr,te,(a,b),itr,iva,(ai,bi) in prepared:
        best=GRID[0];best_score=-np.inf
        for c in GRID:
            pr=multiclass_predict(ai,y[itr],bi,c,classes)
            s=balanced_accuracy_score(y[iva],pr.argmax(1))
            if s>best_score:best,best_score=c,s
        out[te]=multiclass_predict(a,y[tr],b,best,classes)
    if not np.isfinite(out).all():raise ValueError('Incomplete multiclass predictions')
    return out


def metrics(task,y,pred):
    if task=='AGE':
        return dict(rho=score(y,pred,False),mae=mean_absolute_error(y,pred),rmse=float(np.sqrt(mean_squared_error(y,pred))))
    if task=='SEX':return dict(auc=roc_auc_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred>=.5),accuracy=accuracy_score(y,pred>=.5))
    return dict(balanced_accuracy=balanced_accuracy_score(y,pred.argmax(1)),accuracy=accuracy_score(y,pred.argmax(1)),log_loss=log_loss(y,pred,labels=list(range(len(AGE_BINS)))))


def evaluation(state):
    prepare();source,model,objective,target,seed=state;features=feature_folder(state)
    stamp=json.loads((features/'complete.json').read_text());name=readout.tag(*state);out=RUN/'evaluation'/name
    for co in COHORTS:
        m=pd.read_csv(RUN/'labels'/(co+'.csv'));fpath=features/(co+'.npy')
        if sha(fpath)!=stamp['feature_sha256'][co]:raise ValueError('Feature array changed')
        f=np.load(fpath)
        tasks=[('SEX','sex'),('AGE_GROUP','age_group')] if co.startswith('hrv') else [('SEX','sex'),('AGE','age')]
        for task,col in tasks:
            dest=out/(co+'_'+task+'.json');signature=dict(features_sha256=sha(features/'complete.json'),labels_sha256=sha(RUN/'labels'/(co+'.csv')),task=task)
            if dest.exists():
                if json.loads(dest.read_text())['signature']!=signature:raise ValueError('Stale demographic prediction')
                continue
            ix=np.flatnonzero(m.primary_include.to_numpy(bool)&m[col].notna().to_numpy());y=m.iloc[ix][col].to_numpy(float);g=m.iloc[ix].subject.astype(str).to_numpy();x=f[ix]
            if len(set(g))<20 or (task!='AGE' and len(np.unique(y))<2):raise ValueError('Insufficient demographic labels')
            pred=[];audits=[];seed_metrics=[]
            for split_seed in (0,1,2):
                prepared=prepare_nested([x],[100],y,g,split_seed)
                pp=predict_multiclass(prepared,y,len(AGE_BINS)) if task=='AGE_GROUP' else predict_prepared(prepared,y,binary=task=='SEX')
                pred.append(pp);seed_metrics.append(metrics(task,y,pp))
                audits.append(dict(seed=split_seed,folds=[dict(train_subjects=g[tr].tolist(),test_subjects=g[te].tolist(),inner_train_subjects=g[it].tolist(),inner_validation_subjects=g[iv].tolist()) for tr,te,_,it,iv,_ in prepared]))
            prediction=np.mean(pred,axis=0)
            exposure='source_exposed_SSL_probe' if objective=='native_ssl' and co==source else 'external_to_SSL_source' if objective=='native_ssl' else 'frozen_encoder_probe'
            result=dict(status='complete',signature=signature,state=name,model=model,readout=READOUTS[model],source=source,objective=objective,adapter_seed=seed,
                cohort=co,task=task,n=len(ix),n_subjects=len(set(g)),scan_ids=m.iloc[ix].scan.tolist(),subjects=g,y=y,prediction=prediction,predictions_by_probe_seed=pred,
                metrics=metrics(task,y,prediction),metrics_by_probe_seed=seed_metrics,mean_probe_seed_metrics={key:np.mean([s[key] for s in seed_metrics]) for key in seed_metrics[0]},
                evaluation_scope=exposure,split_audit=audits,classes=list(AGE_BINS) if task=='AGE_GROUP' else None,
                protocol='Participant-grouped 5-fold outer CV; grouped 20% inner validation; probe seeds 0/1/2; training-only scaling/PCA<=100; ridge regression or logistic classification; fixed tuning grid.',
                inference='Descriptive, conditional on the supplied encoder. Source-exposed SSL probes do not establish independent within-source encoder generalization.')
            save(dest,result);print('demographic',name,co,task,result['metrics'],flush=True)
    save(out/'complete.json',dict(status='complete',state=name,tasks=16,encoder_training_performed=False))


def report():
    rows=[];paired=[];averaged=[]
    for state in states():
        for p in sorted((RUN/'evaluation'/readout.tag(*state)).glob('*.json')):
            if p.name=='complete.json':continue
            d=json.loads(p.read_text());r={k:d[k] for k in ('state','model','readout','source','objective','adapter_seed','cohort','task','n','n_subjects','evaluation_scope')};r.update(d['metrics']);rows.append(r)
            if state[2]=='frozen':continue
            ref=RUN/'evaluation'/(state[1]+'_frozen')/p.name;q=json.loads(ref.read_text())
            if d['scan_ids']!=q['scan_ids'] or d['subjects']!=q['subjects'] or not np.allclose(d['y'],q['y']):raise ValueError('Frozen/SSL participant rows differ')
            paired.append(dict(**{k:r[k] for k in ('state','model','source','adapter_seed','cohort','task','n','evaluation_scope')},
                **{'delta_'+k:None if d['metrics'][k] is None or q['metrics'][k] is None else d['metrics'][k]-q['metrics'][k] for k in d['metrics']}))
    for model in MODELS:
        for source in ('adni','nacc'):
            for p in sorted((RUN/'evaluation'/readout.tag(source,model,'native_ssl','none',42)).glob('*.json')):
                if p.name=='complete.json':continue
                ds=[json.loads((RUN/'evaluation'/readout.tag(source,model,'native_ssl','none',s)/p.name).read_text()) for s in (42,43,44)];d=ds[0]
                if any(q['scan_ids']!=d['scan_ids'] or not np.allclose(q['y'],d['y']) for q in ds[1:]):raise ValueError('SSL seed rows differ')
                scores=metrics(d['task'],np.asarray(d['y']),np.mean([q['prediction'] for q in ds],axis=0))
                averaged.append(dict(model=model,source=source,cohort=d['cohort'],task=d['task'],evaluation_scope=d['evaluation_scope'],n=d['n'],**scores))
    pd.DataFrame(rows).to_csv(RUN/'results_by_state.csv',index=False)
    pd.DataFrame(paired).to_csv(RUN/'paired_frozen_SSL_differences.csv',index=False)
    pd.DataFrame(averaged).to_csv(RUN/'SSL_seed_averaged_results.csv',index=False)
    save(RUN/'report.json',dict(status='complete',n_state_results=len(rows),n_paired_results=len(paired),n_SSL_seed_averaged_results=len(averaged),
        tables=['results_by_state.csv','paired_frozen_SSL_differences.csv','SSL_seed_averaged_results.csv'],
        interpretation='Descriptive estimates only. Age MAE/RMSE differences below zero favor SSL; AUC/correlation/accuracy differences above zero favor SSL. HRV age-group classification is separate from continuous age regression. Source-exposed SSL results are not external validation.'))
