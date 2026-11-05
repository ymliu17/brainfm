"""Additive matched-readout LOSO, calibration and native-window exports.

The already registered core implementation is unchanged. No GFM task is fitted.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from fm_matched_v3.data import ROOT,RUN,DATA,save,sha,dataset,metadata
from fm_matched_v3.models import Encoder,READOUT
from fm_matched_v3.train import location,amp
from fm_matched_v3.evaluate import folder,tag,clinical_features
from fm_matched_v3.probes import TuckerBlock
from fm_matrix import data as legacy
from fm_matrix.train import location as old_location
from benchmark_evaluation import score
from transfer_baseline import fc_feature

COHORTS=('adni','nacc','act','cogte','breathe','psopt','hrv_pre')


def checkpoint_path(source,model,objective,target,seed):
    if objective=='frozen':return None
    if objective=='native_ssl':return old_location(source,model,objective,'none',seed,'external')/'adapter.pt'
    if objective=='supervised':return location(source,model,objective,target,seed,'external')/'fold0/adapter.pt'
    raise ValueError('Unsupported adapter objective')


def load_encoder(source,model,objective,target,seed,device):
    net=Encoder(model,device);ck=checkpoint_path(source,model,objective,target,seed)
    if ck is not None:
        state=torch.load(ck,weights_only=True,map_location='cpu')
        expected=sha((legacy.RUN if objective=='native_ssl' else RUN)/'intake.json')
        if state['model']!=model or state['intake_sha256']!=expected or state['source_checkpoint_sha256']!=sha(net.checkpoint_path):
            raise ValueError('Adapter/source provenance mismatch')
        net.load_adapter(state['adapter'])
    return net


def window_dir(source,model,objective,target,seed):
    return RUN/'window_features'/tag(source,model,objective,target,seed)


def window_signature(source,model,objective,target,seed):
    ck=checkpoint_path(source,model,objective,target,seed)
    return dict(readout=READOUT,intake_sha256=sha(RUN/'intake.json'),model=model,
                adapter_sha256=sha(ck) if ck else None,implementation_sha256=sha(__file__),
                encoder_sha256=sha(Path(__file__).with_name('models.py')))


def window_features(source,model,objective,target,seed,device='cuda'):
    from model_input_protocol import build_input
    args=(source,model,objective,target,seed);out=window_dir(*args);signature=window_signature(*args)
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['signature']!=signature:raise ValueError('Stale windows')
        return
    if (out/'request.json').exists() and json.loads((out/'request.json').read_text())!=signature:
        raise ValueError('Stale partial window export')
    save(out/'request.json',signature);net=load_encoder(*args,device)
    with torch.inference_mode():
        for name in ('hrv_pre','hrv_post','breathe','psopt'):
            if (out/(name+'_windows.npy')).exists():continue
            x,m=dataset(name,model);keep=np.flatnonzero(m.native_frames.to_numpy()>=160);values=[]
            for i in keep:
                xx=build_input(x[i,:,:160],net.length)[None].to(device)
                with amp(device):z=net.windows(xx)
                if z.shape[1]!=8 or not torch.isfinite(z).all():raise ValueError('Invalid windows')
                values.append(z.float().cpu().numpy())
            np.save(out/(name+'_window_indices.npy'),keep)
            p=out/(name+'_windows.npy');tmp=p.with_suffix('.tmp.npy');np.save(tmp,np.concatenate(values));tmp.replace(p)
            print('windows',tag(*args),name,len(keep),flush=True)
    save(out/'complete.json',dict(status='complete',signature=signature,
         protocol='Final-layer ROI/content-weighted temporal tokens, eight 20-TR windows over the first 160 real frames; common pooling operator across FMs; full offline context, not causal prediction'))


def outcome(m,cohort,domain):
    key='MEM_delayed_recall' if cohort=='hrv_pre' and domain=='MEM' else domain
    return key,m[key].to_numpy(float)


def baseline_rows(m,cohort):
    keep=m.eligible.to_numpy(bool)
    if cohort in ('act','cogte','breathe','psopt'):keep &= m.visit.eq('T1').to_numpy()
    return keep


def representation(cohort,model,method,args):
    if method=='FM':
        p=folder(*args);d=json.loads((p/'complete.json').read_text())
        if d['signature']['readout']!=READOUT or d['signature']['intake_sha256']!=sha(RUN/'intake.json'):
            raise ValueError('Unmatched scan embeddings')
        return np.load(p/(cohort+'.npy'))
    x,_=dataset(cohort,model)
    return fc_feature(x.numpy(),None) if method=='FC' else x.numpy()


def shared_features(train,test,method):
    if method!='FC-Tucker':return train,test
    from thirdorder import tucker_fc_feature
    basis=TuckerBlock(train,64).basis(np.arange(len(train)))
    return tucker_fc_feature(train,basis=basis),tucker_fc_feature(test,basis=basis)


def source_projection(train,test,y,k=100):
    sc=StandardScaler().fit(train);a=sc.transform(train)
    pc=PCA(n_components=min(k,len(a)-1,a.shape[1]),svd_solver='randomized',random_state=0).fit(a)
    return Ridge(alpha=10).fit(pc.transform(a),y).predict(pc.transform(sc.transform(test)))


def state_name(args,method):
    return tag(*args) if method=='FM' else ('sch450' if args[1]=='brainjepa' else 'a424')+'_'+method


def loso(source,model,objective,target,seed,method='FM'):
    args=(source,model,objective,target,seed);name=state_name(args,method);out=RUN/'supplement'/name/'loso.json'
    if out.exists():return
    arrays={co:representation(co,model,method,args) for co in COHORTS[2:]}
    tables={co:metadata(co) for co in arrays};rows=[]
    for domain in ('MEM','EF'):
        cohorts=list(COHORTS[2:]) if domain=='EF' else list(COHORTS[2:-1])
        for held in cohorts:
            train=[];targets=[];participants=[]
            for co in cohorts:
                if co==held:continue
                m=tables[co];y=m[domain].to_numpy(float);ix=baseline_rows(m,co)&np.isfinite(y)
                yy=y[ix];train.append(arrays[co][ix]);targets.append((yy-yy.mean())/max(yy.std(),1e-8))
                participants.extend((co+':'+m.loc[ix,'subject'].astype(str)).tolist())
            m=tables[held];y=m[domain].to_numpy(float);ix=baseline_rows(m,held)&np.isfinite(y)
            a,b=shared_features(np.concatenate(train),arrays[held][ix],method)
            p=source_projection(a,b,np.concatenate(targets));yy=y[ix]
            rows.append(dict(cohort=held,domain=domain,method=method,subjects=m.loc[ix,'subject'].tolist(),
                training_subjects=participants,y=yy,prediction=p,n=int(ix.sum()),rho=score(yy,p,False)))
            print('LOSO',name,domain,held,flush=True)
    save(out,dict(status='complete',rows=rows,readout=READOUT if method=='FM' else method,
         protocol='Training-cohort outcome z-scores; source-only FC-Tucker/scaling/PCA<=100; fixed ridge alpha=10; no held-cohort label fitting',
         scope='EF: five intervention cohorts including HRV. MEM: four original cohorts; HRV Week-5 recall is not a baseline MEM outcome.',
         inference='Descriptive external prediction; no significance claim'))


def calibration(source,model,objective,target,seed,method='FM'):
    from transfer_calibration import Align,subject_mean,subject_folds,clinical_predict
    args=(source,model,objective,target,seed);name=state_name(args,method)
    out=RUN/'supplement'/name/'calibration.json'
    if out.exists():return
    arrays={co:representation(co,model,method,args) for co in COHORTS};tables={co:metadata(co) for co in COHORTS};rows=[]
    for src in (('adni','nacc') if objective=='frozen' else (source,)):
        m=tables[src]
        for domain in ('MEM','EF'):
            y=m[domain].to_numpy(float);ok=baseline_rows(m,src)&np.isfinite(y);a=arrays[src][ok];basis=None
            if method=='FC-Tucker':
                from thirdorder import tucker_fc_feature
                basis=TuckerBlock(a,64).basis(np.arange(len(a)));a=tucker_fc_feature(a,basis=basis)
            sc=StandardScaler().fit(a);a=sc.transform(a)
            pc=PCA(n_components=min(20,len(a)-1,a.shape[1]),svd_solver='randomized',random_state=0).fit(a)
            zs=pc.transform(a);axis=Ridge(alpha=10).fit(zs,y[ok]);reference=subject_mean(zs,m.loc[ok,'subject'].to_numpy())
            for co in COHORTS:
                if co==src:continue
                tm=tables[co];select=baseline_rows(tm,co);b=arrays[co][select]
                if basis is not None:b=tucker_fc_feature(b,basis=basis)
                z=pc.transform(sc.transform(b));mm=tm.loc[select].reset_index(drop=True);groups=mm.subject.to_numpy()
                endpoint,yy=outcome(mm,co,domain)
                for mode in ('none','moments','coral'):
                    predictions=[];audits=[]
                    for split_seed in (0,1,2):
                        pred=np.full(len(groups),np.nan)
                        for tr,te in subject_folds(groups,split_seed):
                            align=Align().fit(reference,subject_mean(z[tr],groups[tr]),mode)
                            pred[te]=axis.predict(align.transform(z[te]))
                            audits.append(dict(seed=split_seed,calibration=groups[tr].tolist(),test=groups[te].tolist()))
                        predictions.append(pred)
                    p=np.mean(predictions,axis=0);ix=np.isfinite(yy)
                    rows.append(dict(source=src,cohort=co,domain=domain,endpoint=endpoint,task='source_axis',method=method,
                        alignment=mode,subjects=groups[ix],y=yy[ix],prediction=p[ix],rho=score(yy[ix],p[ix],False),fold_audit=audits))
                    # Clinical prognosis recalibrates separately in every inner/outer training partition.
                    if co not in ('adni','nacc'):
                        C=clinical_features(mm,co)
                        fy=yy if endpoint=='MEM_delayed_recall' else mm['follow_'+domain].to_numpy(float)
                        ii=np.isfinite(fy)&np.isfinite(C).all(1)
                        if len(set(groups[ii]))>=20:
                            pp=[];audit=[]
                            for split_seed in (0,1,2):
                                v,aa=clinical_predict(C[ii],fy[ii],z[ii],reference,groups[ii],mode,split_seed)
                                pp.append(v);audit.append(aa)
                            p=np.mean(pp,axis=0)
                            rows.append(dict(source=src,cohort=co,domain=domain,endpoint=endpoint,task='baseline_prognosis',method=method,
                                alignment=mode,subjects=groups[ii],y=fy[ii],prediction=p,rho=score(fy[ii],p,False),fold_audit=audit))
                print('calibration',name,src,domain,co,flush=True)
    save(out,dict(status='complete',rows=rows,
         protocol='Fixed source scaling/PCA20 and ridge; target-to-source mean/SD or shrinkage CORAL fitted to training participants only; label-independent target calibration folds; clinical heads use nested recalibration',
         scope='Embedding-space alignment sensitivity; may remove biological cohort differences. Input-space gain is not implemented here.',
         inference='Descriptive sensitivity; no target outcome used for encoder, source axis or alignment fitting'))
