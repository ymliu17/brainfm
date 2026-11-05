"""Additional BrainLM13 CLS exports and original-readout HRV probes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from fm_readout_v4.data import ROOT, RUN, DATA, dataset, metadata, endpoints, save, sha
from fm_readout_v4.models import Encoder, readout_name
from fm_matrix.train import features, amp
from fm_matched_v3.probes import TuckerBlock, oof, oof_predictions
from fm_matrix import data as original
from fm_matrix.train import location as old_location
from transfer_baseline import fc_feature
from benchmark_evaluation import score

COHORTS = ['adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post','nacc_repeated']


def tag(source, model, objective, target, seed):
    return model+'_'+('frozen' if objective=='frozen' else f'{source}_{objective}_{target}_s{seed}')


def folder(source, model, objective, target, seed):
    return RUN/'features'/tag(source,model,objective,target,seed)


def extraction(source,model,objective,target,seed,device='cuda'):
    from fm_matrix.evaluate import feature_dir
    base='brainlm13' if model=='brainlm13_embfc' else model
    ck=None if objective=='frozen' else old_location(source,base,objective,'none' if objective=='native_ssl' else target,seed,'external')/('fold0/adapter.pt' if objective=='supervised' else 'adapter.pt')
    signature=dict(readout=readout_name(model),intake_sha256=sha(RUN/'intake.json'),
                   checkpoint_sha256=sha(ck) if ck else None,model=model)
    out=folder(source,model,objective,target,seed)
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['signature']!=signature:raise ValueError('Stale readout export')
        return
    out.mkdir(parents=True,exist_ok=True)
    request=out/'request.json'
    if request.exists() and json.loads(request.read_text())!=signature:
        raise ValueError('Partial export belongs to a different checkpoint/input')
    if not request.exists():save(request,signature)
    if model!='brainlm13':
        old=feature_dir(source,base,objective,target,seed);stamp=json.loads((old/'complete.json').read_text())
        expected=dict(intake_sha256=sha(original.RUN/'intake.json'),adapter_sha256=sha(ck) if ck else None,model=base)
        if stamp['signature']!=expected:raise ValueError('Original feature provenance mismatch')
        for cohort in COHORTS:
            src=old/(cohort+'.npy');dest=out/src.name
            if not src.is_file():raise FileNotFoundError(src)
            if dest.exists() and dest.resolve()!=src.resolve():raise ValueError('Unexpected feature reuse destination')
            if not dest.exists():dest.symlink_to(src)
        save(out/'complete.json',dict(status='complete',signature=signature,reused_from=str(old),encoder=stamp['encoder'],post_training_performed=False))
        return
    net=Encoder(model,device)
    if ck:
        state=torch.load(ck,weights_only=True,map_location='cpu')
        if state['model']!=base or state['intake_sha256']!=sha(original.RUN/'intake.json') or state['source_checkpoint_sha256']!=sha(net.checkpoint_path):raise ValueError('Adapter provenance mismatch')
        net.load_adapter(state['adapter'])
    for cohort in COHORTS:
        path=out/(cohort+'.npy')
        if path.exists():continue
        x,m=dataset(cohort,base);z=features(net,x,device)
        tmp=path.with_suffix('.tmp.npy');np.save(tmp,z);tmp.replace(path)
        print('CLS features',source,objective,target,cohort,z.shape,flush=True)
    save(out/'complete.json',dict(status='complete',signature=signature,encoder=net.provenance(),post_training_performed=False,
         checkpoint=str(ck) if ck else None,interpretation='Additional inference readout on existing encoder. Supervised adapters were trained using the original layer-3 embedding-connectivity readout, not CLS.'))


def representation(cohort, model, method, source, objective, target, seed):
    if method=='FM':
        f=folder(source,model,objective,target,seed)
        d=json.loads((f/'complete.json').read_text())
        if d['signature']['readout']!=readout_name(model):
            raise ValueError('Unmatched representation readout')
        return np.load(f/(cohort+'.npy'))
    x,_=dataset(cohort,model)
    if method=='FC':
        return fc_feature(x.numpy(),None)
    if method=='FC-Tucker':
        return TuckerBlock(x.numpy(),64)
    if method=='BrainGFM':
        raise ValueError('BrainGFM is gated on a verified compatible checkpoint/atlas; no random fallback')
    raise ValueError(method)


def clinical_features(m,cohort):
    if cohort.startswith('hrv'):
        cats=pd.get_dummies(m[['age_category','sex','arm']],drop_first=True,dtype=float).to_numpy()
        return np.column_stack([m.baseline_EF,cats])
    return m[['baseline_MEM','baseline_EF','age','sex']].to_numpy(float)


def local(source,model,objective,target,seed,cohort,method='FM'):
    if objective!='frozen' and cohort==source:
        raise ValueError('Source-fitted encoders cannot enter local source evaluation; use fold-specific CV')
    m=metadata(cohort)
    f=representation(cohort,model,method,source,objective,target,seed)
    state=tag(source,model,objective,target,seed) if method=='FM' else ('sch450' if model=='brainjepa' else 'a424')+'_'+method
    out=RUN/'evaluation'/state/(cohort+'_local.json')
    signature=dict(readout=readout_name(model) if method=='FM' else method,
                   labels=sha(DATA/f'{cohort}_domains.csv') if cohort.startswith('hrv') else sha(DATA/f'{cohort}.csv'),
                   features=sha(folder(source,model,objective,target,seed)/'complete.json') if method=='FM' else sha(RUN/'intake.json'))
    if out.exists():
        if json.loads(out.read_text())['signature']!=signature:
            raise ValueError('Stale local evaluation labels or features')
        return
    eligible=m.eligible.to_numpy(bool)
    if cohort in ['act','cogte','breathe','psopt']:
        eligible &= m.visit.eq('T1').to_numpy()
    outcomes=endpoints(cohort)
    if cohort in ('adni','nacc'):
        p=ROOT/f'runs/diagnosis_posttraining_v1/{cohort}_labels.csv'
        diag=pd.read_csv(p).set_index('scan')
        for key in ['DX','AD_DX']:
            if key in diag:
                m[key]=m.scan.map(diag[key]);outcomes.append(key)
    rows=[]
    for endpoint in outcomes:
        y=m[endpoint].to_numpy(float)
        ix=np.flatnonzero(eligible & np.isfinite(y));g=m.subject.to_numpy()[ix]
        if len(set(g))<20:
            continue
        pred=oof(f[ix],y[ix],g)
        binary=endpoint in ('DX','AD_DX')
        rows.append(dict(cohort=cohort,task=endpoint+('_concurrent_post' if cohort=='hrv_post' else '_baseline'),method=method,
                         n=len(ix),n_subjects=len(set(g)),subjects=g,y=y[ix],prediction=pred,
                         metric='auc' if binary else 'rho',score=score(y[ix],pred,binary)))
        if 'follow_'+endpoint not in m or cohort=='hrv_post':
            continue
        clinical=clinical_features(m,cohort)
        yy=m['follow_'+endpoint].to_numpy(float)
        ii=np.flatnonzero(eligible & np.isfinite(yy) & np.isfinite(clinical).all(1));gg=m.subject.to_numpy()[ii]
        if len(set(gg))<20:
            continue
        for label,blocks,dims in [('clinical',[clinical[ii]],[clinical.shape[1]]),
                                 ('clinical+'+method,[clinical[ii],f[ii]],[clinical.shape[1],5])]:
            pp=np.mean([oof_predictions(blocks,dims,yy[ii],s,gg) for s in [0,1,2]],axis=0)
            rows.append(dict(cohort=cohort,task=endpoint+'_followup',method=label,n=len(ii),
                             n_subjects=len(set(gg)),subjects=gg,y=yy[ii],prediction=pp,metric='rho',
                             score=score(yy[ii],pp,False)))
        print('local',state,cohort,endpoint,flush=True)
    if cohort.startswith('hrv') and 'MEM_delayed_recall' in m:
        y=m.MEM_delayed_recall.to_numpy(float)
        ix=np.flatnonzero(eligible & np.isfinite(y));g=m.subject.to_numpy()[ix]
        if len(set(g))>=20:
            task='MEM_delayed_recall_'+('baseline_prediction' if cohort=='hrv_pre' else 'post_scan_association')
            p=oof(f[ix],y[ix],g)
            rows.append(dict(cohort=cohort,task=task,method=method,n=len(ix),n_subjects=len(set(g)),
                             subjects=g,y=y[ix],prediction=p,metric='rho',score=score(y[ix],p,False),
                             outcome_week=5,imaging_week=2 if cohort=='hrv_pre' else 7,
                             endpoint='One-time delayed picture recall count; not a MEM composite or memory change'))
            if cohort=='hrv_pre':
                clinical=clinical_features(m,cohort)
                ii=np.flatnonzero(eligible & np.isfinite(y) & np.isfinite(clinical).all(1));gg=m.subject.to_numpy()[ii]
                for label,blocks,dims in [('clinical',[clinical[ii]],[clinical.shape[1]]),
                                         ('clinical+'+method,[clinical[ii],f[ii]],[clinical.shape[1],5])]:
                    pp=np.mean([oof_predictions(blocks,dims,y[ii],s,gg) for s in (0,1,2)],axis=0)
                    rows.append(dict(cohort=cohort,task=task,method=label,n=len(ii),n_subjects=len(set(gg)),
                        subjects=gg,y=y[ii],prediction=pp,metric='rho',score=score(y[ii],pp,False),
                        outcome_week=5,imaging_week=2,clinical_covariates='Baseline EF, age category, sex, and arm; no baseline memory available'))
    save(out,dict(status='complete',signature=signature,rows=rows,
         inference='Descriptive paired OOF predictions; no new significance claim',
         protocol='Identical grouped 5-fold/inner 80:20, split seeds 0/1/2; FC-Tucker basis fitted independently in every outer and inner training partition'))


def identification(source,model,objective,target,seed):
    # Standalone implementation uses the same pairing rule, with new features.
    f=folder(source,model,objective,target,seed);rows=[]
    for cohort in ['adni','nacc_repeated','act','cogte','breathe','psopt','hrv']:
        if objective!='frozen' and cohort.removesuffix('_repeated')==source:
            continue
        if cohort=='hrv':
            a=pd.read_csv(DATA/'hrv_pre.csv');b=pd.read_csv(DATA/'hrv_post.csv')
            common=sorted(set(a.subject)&set(b.subject));ia=a.reset_index().set_index('subject').loc[common,'index'];ib=b.reset_index().set_index('subject').loc[common,'index']
            A=np.load(f/'hrv_pre.npy')[ia];B=np.load(f/'hrv_post.npy')[ib]
        else:
            m=pd.read_csv(DATA/f'{cohort}.csv')
            pairs=[g.sort_values('visit').index[:2].to_list() for _,g in m.groupby('subject') if len(g)>=2]
            if not pairs:continue
            ia,ib=np.asarray(pairs).T;z=np.load(f/(cohort+'.npy'));A=z[ia];B=z[ib]
        A=A-A.mean(1,keepdims=True);B=B-B.mean(1,keepdims=True)
        A=A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-8)
        B=B/np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-8);sim=A@B.T
        rows.append(dict(cohort=cohort,n=len(A),chance=1/len(A),
                         pre_post=float(np.mean(sim.argmax(1)==np.arange(len(A)))),
                         post_pre=float(np.mean(sim.argmax(0)==np.arange(len(A))))))
    save(RUN/'evaluation'/tag(source,model,objective,target,seed)/'identification.json',dict(status='complete',readout=readout_name(model),rows=rows))


def axes(source,model,objective,target,seed,cohorts=None):
    """Source-fitted axes; target outcome labels never fit transforms or heads."""
    from fm_matrix.evaluate import fitted_axis
    f=folder(source,model,objective,target,seed);rows=[]
    for src in (['adni','nacc'] if objective=='frozen' else [source]):
        m=metadata(src);sf=np.load(f/(src+'.npy'))
        for domain in ('MEM','EF'):
            keep=m.eligible.to_numpy(bool)&m[domain].notna().to_numpy()
            for cohort in COHORTS:
                if cohorts is not None and cohort not in cohorts:continue
                if cohort in (src,'nacc_repeated','hrv_post'):continue
                # Primary axes preserve domain labels; no MEM->HRV-EF relabeling.
                tm=metadata(cohort);eligible=tm.eligible.to_numpy(bool)
                outcome=domain
                if cohort.startswith('hrv') and domain=='MEM' and 'MEM_delayed_recall' in tm:
                    outcome='MEM_delayed_recall'
                elif domain not in endpoints(cohort):continue
                if cohort in ('act','cogte','breathe','psopt'):
                    eligible &= tm.visit.eq('T1').to_numpy()
                ix=eligible&tm[outcome].notna().to_numpy()
                if ix.sum()<3:continue
                p=fitted_axis(sf[keep],m.loc[keep,domain].to_numpy(),np.load(f/(cohort+'.npy'))[ix])
                y=tm.loc[ix,outcome].to_numpy()
                rows.append(dict(source=src,cohort=cohort,task=domain+'_axis_to_'+outcome,subjects=tm.loc[ix,'subject'].to_numpy(),
                                 n=int(ix.sum()),y=y,prediction=p,rho=score(y,p,False)))
    save(RUN/'evaluation'/tag(source,model,objective,target,seed)/'axes.json',dict(status='complete',rows=rows,
         scope='Source-only scaling/PCA/ridge fitting; external label units differ; correlation only'))
