"""Common-sample S4/S5 additions using immutable cached encoder representations.

No encoder fitting is performed. Source-exposed probes are explicitly separated
from independent cohorts. Scores and conditional tests retain participant groups.
"""
from pathlib import Path
from functools import lru_cache
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import paper_completion as pc
import readout_sensitivity as rs
import sample_alignment as sa
import sf100_benchmark as sf
from fm_matrix.data import ROOT,save,sha
from fm_readout_v4.data import metadata,DATA
from fm_readout_v4.supplement import source_projection
from benchmark_evaluation import grouped_splits,predict_prepared,score
from paper_paired_stats import paired,holm

RUN=ROOT/'runs/paper_task_expansion_v1'
BASE=ROOT/'runs/paper_results_2026-10-14'
STATES=list(pc.states())
DIAG=(('adni','DX'),('nacc','DX'),('nacc','AD_DX'))
TRIALS=('act','cogte','breathe','psopt','hrv_pre')
READOUTS=('native','content_mean','emb_fc_L3')

def name(state):return pc.tag(*state)
def read(p):return json.loads(Path(p).read_text())
def scope(source,co):
    if source=='none':return 'frozen'
    return 'source_exposed_descriptive' if co==source or (source=='nacc' and co=='nacc_repeated') else 'independent_of_posttraining_source'

def exact_project(train,test,k=100):
    """Exact dual PCA for wide arrays, mathematically equivalent to full SVD."""
    if train.shape[1]<5000:return pc.project_block(train,test,k)
    sc=StandardScaler().fit(train)
    a=sc.transform(train).astype(np.float64);b=sc.transform(test).astype(np.float64)
    k=min(k,len(a)-1,a.shape[1]);gram=a@a.T
    e,u=eigh(gram,subset_by_index=(len(a)-k,len(a)-1),check_finite=False)
    e=e[::-1].clip(min=0);u=u[:,::-1]
    tolerance=np.finfo(a.dtype).eps*max(a.shape)*max(float(e.max()),1)
    inv=np.divide(1,np.sqrt(np.maximum(e,tolerance)),out=np.zeros_like(e),where=e>tolerance)
    return (gram@u)*inv,((b@a.T)@u)*inv

@lru_cache(None)
def feature(state,co):
    return np.load(pc.feature_dir(state)/(co+'.npy'),mmap_mode='r')

def readout_features(co):
    return {'native':feature(('none','brainlm13','frozen','none',0),co),
            'content_mean':np.load(rs.mean_folder('brainlm13')/(co+'.npy'),mmap_mode='r'),
            'emb_fc_L3':np.load(ROOT/'runs/fm_posttraining_matrix_v2/features/brainlm13_frozen'/(co+'.npy'),mmap_mode='r')}

def audit():
    files={}
    def register(p):files[str(p.relative_to(ROOT))]=sha(p)
    for co in (*pc.COHORTS,'nacc_repeated'):
        register(sf.label_file(co));register(sf.RUN/'matches'/(co+'.json'))
    for co in ('adni','nacc'):register(ROOT/'runs/diagnosis_posttraining_v1'/(co+'_labels.csv'))
    for state in STATES:
        f=pc.feature_dir(state);stamp=read(f/'complete.json');register(f/'complete.json')
        assert stamp['status']=='complete'
        actual=stamp.get('encoder',{}).get('readout',stamp['signature'].get('readout'))
        assert actual==('mean_final' if state[1]=='brainjepa' else 'CLS_final'),(state,actual)
        for co in (*pc.COHORTS,'nacc_repeated'):
            p=f/(co+'.npy');x=np.load(p,mmap_mode='r')
            assert x.ndim==2 and len(x)==len(metadata(co)) and np.isfinite(x).all()
            register(p)
    for folder in (rs.mean_folder('brainlm13'),ROOT/'runs/fm_posttraining_matrix_v2/features/brainlm13_frozen'):
        register(folder/'complete.json')
        for co in rs.COHORTS:
            p=folder/(co+'.npy');assert len(np.load(p,mmap_mode='r'))==len(metadata(co));register(p)
    for co,task in DIAG:register(sf.RUN/'evaluation'/co/(task+'.json'))
    for model in pc.MODELS:
        register(sa.RUN/'fm_readout_v4/supplement'/(model+'_frozen')/'loso.json')
        register(ROOT/'runs/hrv_recall_external_v1'/(model+'.json'))
    for p in (Path(__file__),ROOT/'src/readout_sensitivity.py',ROOT/'src/sample_alignment.py',ROOT/'src/benchmark_evaluation.py',ROOT/'src/paper_paired_stats.py'):register(p)
    protocol=dict(status='complete',files_sha256=files,states=STATES,encoder_training=False,gpu_required=False,automatic_shutdown=False,
        readout='13M final CLS versus final token mean, plus layer-3 embedding connectivity as a layer-and-readout pipeline contrast. ADNI/NACC MEM/EF grouped 5-fold outer, 20% grouped inner validation, split seeds 0/1/2; training-only exact PCA<=100. Seven-cohort identification with common paired scans and native-dimensional correlation similarity.',
        readout_families='Four-model final-layer families expanded from 72 to 90 comparisons. The 18 13M layer-3 contrasts form separate layer-readout families. Holm recalculated; historical adjusted P values preserved in parent artifacts.',
        diagnosis='Frozen/MEM/EF/SSL features; same task-specific participants and grouped 5-fold/20%-inner splits as S1, split seeds 0/1/2, training-only exact PCA<=100 and tuned logistic regression. Adapted predictions averaged over seeds 42/43/44. All source-exposed probes labeled descriptive, not independent encoder validation.',
        identification='Seven cohorts, same first-two-visits pairing as S1/S4. Ensemble similarity matrices averaged over seeds 42/43/44 before top-1/I_diff calculation; individual seed scores retained. Source-exposed galleries separated. No averaging of feature coordinates across encoders.',
        loso='Four-cohort MEM and five-cohort EF LOSO, plus four-RCT MEM to HRV Week-5 recall from Week-2 MRI. Cohort-specific source outcome standardization; training-only scaling/PCA<=100, ridge alpha10; average external predictions across adapter seeds. No held-out outcome fitting.',
        inference='5000 participant bootstraps, 20000 conditional swaps. New post hoc families by task type, endpoint/metric, adaptation objective and source-exposure scope; all models/sources and target cohorts in each applicable family. Existing S5 families unchanged.')
    p=RUN/'input_audit.json'
    if p.exists():assert read(p)==json.loads(json.dumps(protocol,default=list)),'Expansion input changed; use a new run directory'
    else:save(p,protocol)
    return protocol

def prepare(x,y,groups,seed):
    folds=[];audit=[]
    for fold,(tr,te) in enumerate(grouped_splits(y,groups,seed)):
        it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=groups[tr]));it,iv=tr[it],tr[iv]
        assert not set(groups[tr])&set(groups[te]) and not set(groups[it])&set(groups[iv])
        folds.append((tr,te,exact_project(x[tr],x[te]),it,iv,exact_project(x[it],x[iv])))
        audit.append(dict(seed=seed,fold=fold,train=tr,test=te,inner_train=it,validation=iv))
    return folds,audit

def readout_cognition(co):
    m=sf.labels(co);xs=readout_features(co)
    for endpoint in ('MEM','EF'):
        out=RUN/'readout/cognition'/(co+'_'+endpoint+'.json')
        if out.exists():continue
        ix=np.flatnonzero(sa.common_mask(m,co)&m[endpoint].notna().to_numpy());y=m.iloc[ix][endpoint].to_numpy(float);groups=m.iloc[ix].subject.to_numpy()
        predictions={};splits=[]
        # Reuse native S1 predictions only with exactly identical scan/outcome/split audits.
        old=read(sf.RUN/'evaluation'/co/(endpoint+'.json'))
        assert old['scan_ids']==m.iloc[ix].scan.tolist() and np.array_equal(old['y'],y)
        for method,x in xs.items():
            ps=[]
            for seed in (0,1,2):
                if method=='native':
                    ps.append(np.asarray(old['predictions_by_seed']['brainlm13'][seed]));continue
                prep,audit_rows=prepare(x[ix],y,groups,seed)
                for new,prior in zip(audit_rows,[a for a in old['split_audit'] if a['seed']==seed]):
                    for k in ('train','test','inner_train','validation'):assert np.array_equal(new[k],prior[k])
                ps.append(predict_prepared(prep,y,binary=False))
                if method=='content_mean':splits.extend(audit_rows)
                print('readout',co,endpoint,method,seed,flush=True)
            predictions[method]=ps
        mean={k:np.mean(v,axis=0) for k,v in predictions.items()};rows=[]
        for method in READOUTS[1:]:
            family=('cognition_' if method=='content_mean' else 'layer_readout_cognition_')+endpoint
            rows.append(dict(model='brainlm13',cohort=co,endpoint=endpoint,reference='native',candidate=method,family=family,
                **paired(y,mean['native'],mean[method],groups,seed=rs.seed_of('13M'+co+endpoint+method))))
        save(out,dict(status='complete',rows=rows,scan_ids=m.iloc[ix].scan.tolist(),subjects=groups,y=y,predictions=mean,predictions_by_seed=predictions,split_audit=splits,
            protocol='S4 exact evaluation; content_mean is same-final-layer readout comparison, emb_fc_L3 changes both layer and readout; same samples/probes for both.'))

def readout_identification():
    for co in rs.PAIR_COHORTS:
        out=RUN/'readout/identification'/(co+'.json')
        if out.exists():continue
        subjects,(ca,ia),(cb,ib)=sa.paired_rows(co);a=readout_features(ca);b=a if ca==cb else readout_features(cb)
        scores={k:rs.identification_scores(rs.similarity(a[k][ia],b[k][ib])) for k in READOUTS};rows=[]
        for method in READOUTS[1:]:
            for metric in ('top1','I_diff'):
                family=('identification_' if method=='content_mean' else 'layer_readout_identification_')+metric
                rows.append(dict(model='brainlm13',cohort=co,reference='native',candidate=method,metric=metric,family=family,
                    **rs.paired_identification(scores['native']['per_subject_'+metric],scores[method]['per_subject_'+metric],rs.seed_of('13M'+co+method+metric))))
        save(out,dict(status='complete',rows=rows,scores=scores,subjects=subjects,scan_ids_a=metadata(ca).iloc[ia].scan.tolist(),scan_ids_b=metadata(cb).iloc[ib].scan.tolist()))

def diagnosis(state,co):
    out=RUN/'diagnosis'/name(state)/(co+'.json')
    if out.exists():return
    m=sf.labels(co);x=feature(state,co);rows=[]
    for _,endpoint in [s for s in DIAG if s[0]==co]:
        ix=np.flatnonzero(sa.common_mask(m,co)&m[endpoint].notna().to_numpy());y=m.iloc[ix][endpoint].to_numpy(float);groups=m.iloc[ix].subject.to_numpy()
        old=read(sf.RUN/'evaluation'/co/(endpoint+'.json'))
        assert old['scan_ids']==m.iloc[ix].scan.tolist() and np.array_equal(old['y'],y)
        predictions=[];splits=[]
        for seed in (0,1,2):
            if state[2]=='frozen':
                predictions.append(np.asarray(old['predictions_by_seed'][state[1]][seed]));continue
            prep,audit_rows=prepare(x[ix],y,groups,seed)
            for new,prior in zip(audit_rows,[a for a in old['split_audit'] if a['seed']==seed]):
                for k in ('train','test','inner_train','validation'):assert np.array_equal(new[k],prior[k])
            predictions.append(predict_prepared(prep,y,binary=True));splits.extend(audit_rows)
        pred=np.mean(predictions,axis=0)
        rows.append(dict(cohort=co,task=endpoint,metric='auc',n=len(ix),n_subjects=len(set(groups)),subjects=groups,scan_ids=m.iloc[ix].scan.tolist(),y=y,prediction=pred,
            predictions_by_split_seed=predictions,score=score(y,pred,True),split_audit=splits,evaluation_scope=scope(state[0],co)))
    save(out,dict(status='complete',state=state,rows=rows))

def identify(state):
    out=RUN/'identification'/name(state)/'scores.json'
    if out.exists():return
    rows=[]
    for co in rs.PAIR_COHORTS:
        subjects,(ca,ia),(cb,ib)=sa.paired_rows(co)
        sim=rs.similarity(feature(state,ca)[ia],feature(state,cb)[ib]);scores=rs.identification_scores(sim)
        dest=RUN/'identification'/name(state)/(co+'.npy');dest.parent.mkdir(parents=True,exist_ok=True);np.save(dest,sim)
        rows.append(dict(cohort=co,subjects=subjects,scan_ids_a=metadata(ca).iloc[ia].scan.tolist(),scan_ids_b=metadata(cb).iloc[ib].scan.tolist(),scores=scores,evaluation_scope=scope(state[0],co)))
    save(out,dict(status='complete',state=state,rows=rows))

def loso(state):
    out=RUN/'loso'/name(state)/'scores.json'
    if out.exists():return
    tables={co:sf.labels(co) for co in TRIALS};rows=[]
    for domain in ('MEM','EF'):
        pool=TRIALS if domain=='EF' else TRIALS[:-1]
        for held in (*pool,*(['hrv_pre'] if domain=='MEM' else [])):
            arrays=[];ys=[];participants=[];audits=[]
            for co in pool:
                if co==held:continue
                m=tables[co];ix=np.flatnonzero(sa.common_mask(m,co)&m[domain].notna().to_numpy());y=m.iloc[ix][domain].to_numpy(float)
                arrays.append(feature(state,co)[ix]);ys.append((y-y.mean())/max(y.std(),1e-8));participants.extend((co+':'+m.iloc[ix].subject.astype(str)).tolist())
                audits.append(dict(cohort=co,scan_ids=m.iloc[ix].scan.tolist(),target_mean=y.mean(),target_sd=y.std()))
            endpoint='MEM_delayed_recall' if held=='hrv_pre' and domain=='MEM' else domain
            m=tables[held];ix=np.flatnonzero(sa.common_mask(m,held)&m[endpoint].notna().to_numpy())
            y=m.iloc[ix][endpoint].to_numpy(float)
            if state[2]=='frozen':
                if endpoint=='MEM_delayed_recall':old=read(ROOT/'runs/hrv_recall_external_v1'/(state[1]+'.json'))
                else:old=next(r for r in read(sa.RUN/'fm_readout_v4/supplement'/name(state)/'loso.json')['rows'] if r['cohort']==held and r['domain']==domain)
                pred=np.asarray(old['prediction'])
            else:pred=source_projection(np.concatenate(arrays),feature(state,held)[ix],np.concatenate(ys))
            row=dict(cohort=held,task=endpoint,metric='rho',n=len(ix),subjects=m.iloc[ix].subject.tolist(),scan_ids=m.iloc[ix].scan.tolist(),training_subjects=participants,training_audit=audits,y=y,prediction=pred,score=score(y,pred,False))
            if state[2]=='frozen':
                assert old['subjects']==row['subjects'] and old['training_subjects']==participants and np.array_equal(old['y'],y)
                row['reused_frozen_predictions']=True
            rows.append(row)
    save(out,dict(status='complete',state=state,rows=rows))

def state_rows(kind,state):
    if kind=='diagnosis':return [r for co in ('adni','nacc') for r in read(RUN/kind/name(state)/(co+'.json'))['rows']]
    return read(RUN/kind/name(state)/'scores.json')['rows']

def infer(model,source,objective,target):
    out=RUN/'inference'/('_'.join([model,source,objective,target])+'.json')
    if out.exists():return
    frozen=('none',model,'frozen','none',0);states=[(source,model,objective,target,s) for s in pc.SEEDS];results=[]
    for kind in ('diagnosis','loso','identification'):
        ref=state_rows(kind,frozen);adapt=[state_rows(kind,s) for s in states]
        for i,r in enumerate(ref):
            adapted=[a[i] for a in adapt];co=r['cohort'];exposure=scope(source,co)
            for q in adapted:
                assert q['cohort']==co and q['subjects']==r['subjects']
                if kind=='identification':
                    assert q['scan_ids_a']==r['scan_ids_a'] and q['scan_ids_b']==r['scan_ids_b']
                else:
                    assert q['task']==r['task'] and q['scan_ids']==r['scan_ids'] and np.array_equal(q['y'],r['y'])
                    if kind=='loso':assert q['training_subjects']==r['training_subjects']
            base=dict(kind=kind,model=model,source=source,objective=objective,adaptation_target=target,cohort=co,evaluation_scope=exposure,reference='frozen',candidate='adapted',aggregation='Prediction ensemble over adaptation seeds 42/43/44')
            if kind=='identification':
                sim=np.mean([np.load(RUN/kind/name(s)/(co+'.npy')) for s in states],axis=0);scores=rs.identification_scores(sim)
                for metric in ('top1','I_diff'):
                    key=json.dumps(dict(base,metric=metric),sort_keys=True)
                    stat=rs.paired_identification(r['scores']['per_subject_'+metric],scores['per_subject_'+metric],rs.seed_of(key))
                    results.append(dict(base,task=metric,metric=metric,n=r['scores']['n'],family='_'.join([kind,metric,objective,exposure]),aggregation='Mean similarity matrix over adaptation seeds before identification scoring',**stat))
            else:
                pred=np.mean([q['prediction'] for q in adapted],axis=0);key=json.dumps(dict(base,task=r['task']),sort_keys=True)
                stat=paired(r['y'],r['prediction'],pred,r['subjects'],metric=r['metric'],seed=rs.seed_of(key))
                results.append(dict(base,task=r['task'],family='_'.join([kind,r['task'],objective,exposure]),**stat))
    save(out,dict(status='complete',rows=results))

def report():
    prior=pd.read_csv(BASE/'S4_readout_comparisons.csv');rows=[]
    for p in (RUN/'readout').glob('*/*.json'):rows.extend(read(p)['rows'])
    assert len(rows)==36
    new=pd.DataFrame(rows);readouts=pd.concat([prior.drop(columns=['holm_p','family_size'],errors='ignore'),new],ignore_index=True)
    for family,ix in readouts.groupby('family').groups.items():
        readouts.loc[ix,'holm_p']=holm(readouts.loc[ix,'permutation_p']);readouts.loc[ix,'family_size']=len(ix)
    readouts.to_csv(RUN/'readout_comparisons.csv',index=False)
    comparisons=[]
    for p in (RUN/'inference').glob('*.json'):comparisons.extend(read(p)['rows'])
    assert len(comparisons)==24*(3+10+14),len(comparisons)
    frame=pd.DataFrame(comparisons)
    for family,ix in frame.groupby('family').groups.items():
        frame.loc[ix,'holm_p']=holm(frame.loc[ix,'permutation_p']);frame.loc[ix,'family_size']=len(ix)
    frame.to_csv(RUN/'posttraining_comparisons.csv',index=False)
    # Preserve the full per-state results as well as paired ensemble scores.
    scores=[]
    for state in STATES:
        for kind in ('diagnosis','loso','identification'):
            for r in state_rows(kind,state):
                base=dict(kind=kind,source=state[0],model=state[1],objective=state[2],adaptation_target=state[3],training_seed=state[4],cohort=r['cohort'],evaluation_scope=scope(state[0],r['cohort']))
                if kind=='identification':
                    for metric in ('top1','I_diff'):scores.append(dict(base,task=metric,metric=metric,score=r['scores'][metric],n=r['scores']['n']))
                else:scores.append(dict(base,**{k:r[k] for k in ('task','metric','score','n')}))
    pd.DataFrame(scores).to_csv(RUN/'scores_by_state.csv',index=False)
    save(RUN/'report.json',dict(status='complete',readout_comparisons=len(readouts),posttraining_comparisons=len(frame),state_scores=len(scores),
        readout_families=readouts.groupby('family').size().to_dict(),posttraining_families=frame.groupby('family').size().to_dict(),
        scope='Four-model cached-representation evaluation; source-exposed and independent tasks separated. New 13M layer-3 contrast controls the probe but changes layer and readout.',
        input_audit_sha256=sha(RUN/'input_audit.json'),automatic_shutdown=False))

def tasks():
    yield ('readout_adni','readout_cognition',('adni',),RUN/'readout/cognition/adni_EF.json')
    yield ('readout_nacc','readout_cognition',('nacc',),RUN/'readout/cognition/nacc_EF.json')
    yield ('readout_identification','readout_identification',(),RUN/'readout/identification/hrv.json')
    for state in STATES:
        for co in ('adni','nacc'):yield ('diagnosis_'+name(state)+'_'+co,'diagnosis',(state,co),RUN/'diagnosis'/name(state)/(co+'.json'))
        yield ('identification_'+name(state),'identify',(state,),RUN/'identification'/name(state)/'scores.json')
        yield ('loso_'+name(state),'loso',(state,),RUN/'loso'/name(state)/'scores.json')

def execute(job):
    key,action,args,out=job
    if Path(out).exists():return key,'reused'
    globals()[action](*args)
    assert Path(out).exists(),key
    return key,'complete'
