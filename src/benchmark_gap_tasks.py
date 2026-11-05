"""Shared FC references and independent-window physiology for all configurations."""
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from fm_matrix.data import ROOT,save,sha
from fm_readout_v4.data import dataset,metadata,DATA
from fm_readout_v4.evaluate import tag
from benchmark_evaluation import predict_prepared,score
from fm_matched_v3.probes import TuckerBlock
from paper_task_expansion import exact_project
import paper_completion as pc
import sf100_benchmark as sf
import unified_supplements as u

RUN=ROOT/'runs/benchmark_gaps_v1'
PARENT=ROOT/'runs/physiology_frozen_crosscohort_v2_no_usable'
PHYS=('psopt','breathe','hrv')

def read(p):return json.loads(Path(p).read_text())

def fc_reference(co,atlas):
    model='brainlm13' if atlas=='a424' else 'brainjepa'
    xx,_=dataset(co,model);x=xx.numpy();m=sf.labels(co)
    from transfer_baseline import fc_feature
    fc=fc_feature(x,None);tucker=TuckerBlock(x,64);rows=[]
    for p in sorted((sf.RUN/'evaluation'/co).glob('*.json')):
        if p.name=='complete.json':continue
        old=read(p)
        if old.get('status')!='complete' or 'split_audit' not in old:continue
        dest=RUN/'references'/atlas/co/p.name
        if dest.exists():continue
        ix=m.set_index('scan').index.get_indexer(old['scan_ids']);assert (ix>=0).all()
        y=np.asarray(old['y'],float);g=m.iloc[ix].subject.astype(str).tolist();assert g==old['subjects']
        cov=old['covariates'];binary=old['metrics']['sf100_fc']['metric']=='auc';predictions={}
        for name in ('FC','FC-Tucker'):
            prep={s:[] for s in range(3)}
            for split in old['split_audit']:
                tr,te,it,iv=[np.asarray(split[k],int) for k in ('train','test','inner_train','validation')]
                if set(np.asarray(g)[tr])&set(np.asarray(g)[te]):raise ValueError('Subject leakage')
                ca,cb=sf.covariate_pair(m,cov,ix[tr],ix[te]);ci,cj=sf.covariate_pair(m,cov,ix[it],ix[iv])
                if name=='FC':aa,bb=fc[ix[tr]],fc[ix[te]];ai,bi=fc[ix[it]],fc[ix[iv]]
                else:aa,bb=tucker.fit_pair(ix[tr],ix[te]);ai,bi=tucker.fit_pair(ix[it],ix[iv])
                aa,bb=exact_project(aa,bb,5 if cov else 100);ai,bi=exact_project(ai,bi,5 if cov else 100)
                prep[split['seed']].append((tr,te,(np.c_[ca,aa],np.c_[cb,bb]),it,iv,(np.c_[ci,ai],np.c_[cj,bi])))
            predictions[name]=[predict_prepared(prep[s],y,binary=binary) for s in range(3)]
        pp={k:np.mean(v,axis=0) for k,v in predictions.items()}
        save(dest,dict(status='complete',cohort=co,atlas=atlas,task=old['task'],scan_ids=old['scan_ids'],subjects=g,y=y,covariates=cov,
            predictions=pp,predictions_by_seed=predictions,metric='auc' if binary else 'rho',scores={k:score(y,v,binary) for k,v in pp.items()},
            split_audit=old['split_audit'],parent_sha256=sha(p),input_sha256=sha(DATA/f'{co}_{atlas}.pt'),
            protocol='Existing canonical atlas input, unchanged SF100 samples/labels/splits/covariates/ridge-logistic grid; exact training-only PCA; Tucker rank64 fit in each outer/inner training partition.',
            caveat='Atlas-matched reference; model-specific TR/context differences retained. No claim of time-context-matched architecture ablation.'))
        print('FC references',co,atlas,old['task'],flush=True)
    save(RUN/'references'/atlas/co/'complete.json',dict(status='complete',cohort=co,atlas=atlas))

def phys_prepare():
    """Frozen observed window rows; identical timing, folds, and no usable filtering."""
    out=RUN/'physiology';out.mkdir(parents=True,exist_ok=True);files={}
    from model_input_protocol import build_input
    for co in PHYS:
        dest=out/co;dest.mkdir(exist_ok=True)
        for name in ['windows.csv','scans.csv',*[f'{s}_splits_s{k}.json' for s in ('dynamic','scan_mean') for k in range(3)]]:
            p=PARENT/co/name;files[str(p)]=sha(p);shutil.copyfile(p,dest/name)
        w=pd.read_csv(dest/'windows.csv');sm=pd.read_csv(dest/'scans.csv')
        for scope,meta in [('dynamic',w),('scan_mean',sm)]:
            for s in range(3):
                for f in read(dest/f'{scope}_splits_s{s}.json'):
                    g=meta.Subject.to_numpy()
                    for a,b in [('train','test'),('inner_train','inner_validation')]:
                        if set(g[f[a]])&set(g[f[b]]):raise ValueError('Physiology split leakage')
        for atlas,model in [('a424','brainlm13'),('sch450','brainjepa')]:
            result=[None]*len(w)
            for ds in w.dataset.unique():
                x,m=dataset(ds,model);files[str(DATA/f'{ds}_{atlas}.pt')]=sha(DATA/f'{ds}_{atlas}.pt')
                for i,r in w[w.dataset==ds].iterrows():
                    if m.iloc[int(r.matrix_index)].scan!=r.scan:raise ValueError('Window canonical scan mismatch')
                    start=int(r.WindowIdx)*20;end=start+20
                    if end>min(x.shape[2],int(r.native_frames)):raise ValueError('Incomplete real window')
                    # Parent input only subtracts an ROI-wise constant. Re-centering
                    # the crop removes that constant exactly up to float rounding.
                    result[i]=build_input(x[int(r.matrix_index),:,start:end],200 if atlas=='a424' else 160)
            path=dest/(atlas+'_inputs.pt');torch.save(torch.stack(result),path);files[str(path)]=sha(path)
            p=PARENT/co/(atlas+'_FC.npy');files[str(p)]=sha(p)
        save(dest/'protocol.json',dict(status='complete',parent_sha256=sha(PARENT/co/'protocol.json'),
            scope='Independent 20-native-frame encoding; same row order, labels, covariates and participant folds across models; heavy padding sensitivity, separate from native contextual tokens.',
            probes='One shared refit implementation for FM, FC and nuisance, no archived prediction substitution.',
            native_TR_seconds={'breathe':1.01,'psopt':1.375,'hrv':2.4}[co],usable_exclusion=False))
    save(out/'prepared.json',dict(status='complete',files_sha256=files,code_sha256=sha(__file__)))

def adapter(state,stamp):
    source,model,obj,target,seed=state
    if obj=='frozen':return None
    if model in pc.MODELS:
        if 'checkpoint' in stamp:return Path(stamp['checkpoint'])
        from fm_matrix.train import location
        return location(source,model,obj,target,seed,'external')/('fold0/adapter.pt' if obj=='supervised' else 'adapter.pt')
    folder=u.feature_folder(state);main=folder.parent.parent
    base='fmrilm' if model.startswith('fmrilm_') else model
    return main/'training'/source/base/f'{obj}_{target}_s{seed}_external'/('fold0/adapter.pt' if obj=='supervised' else 'adapter.pt')

def phys_features(state,device='cuda'):
    source,model,obj,target,seed=state;out=RUN/'physiology/features'/tag(*state);out.mkdir(parents=True,exist_ok=True)
    stamp=read(u.feature_folder(state)/'complete.json');ck=adapter(state,stamp)
    expected=stamp['signature'].get('checkpoint_sha256',stamp['signature'].get('adapter_sha256'))
    if (sha(ck) if ck else None)!=expected:raise ValueError('Adapter does not match scan-level primary export')
    # Previously encoded independent windows for the original two new models are reusable.
    if model in ('semantoks','fmrilm'):
        src=ROOT/'runs/new_fm_window_reencoding_v1/features'/tag(*state);prior=read(src/'complete.json')
        if prior['signature']['adapter_sha256']!=expected:raise ValueError('Wrong cached physiological adapter')
        for co in PHYS:
            for f in ('windows.csv','scans.csv'):
                if sha(ROOT/'runs/new_fm_window_reencoding_v1'/co/f)!=sha(RUN/'physiology'/co/f):raise ValueError('Cached window population differs')
            p=src/(co+'.npy')
            if sha(p)!=prior['features_sha256'][str(p)]:raise ValueError('Cached window feature changed')
            dest=out/p.name
            if not dest.exists():dest.symlink_to(p)
        save(out/'complete.json',dict(status='complete',state=state,reused_from=str(src),parent_sha256=sha(src/'complete.json'),
            files_sha256={co:sha(out/(co+'.npy')) for co in PHYS}));return
    if model in pc.MODELS:
        from fm_readout_v4.models import Encoder
        net=Encoder(model,device)
    else:
        from fmrilm_variants.runtime import configure
        _,Encoder=configure('mae' if model=='fmrilm_mae' else 'vq-domain');net=Encoder('fmrilm',device)
    if ck:
        d=torch.load(ck,map_location='cpu',weights_only=True)
        if d['source_checkpoint_sha256']!=sha(net.checkpoint_path):raise ValueError('Wrong base checkpoint')
        net.load_adapter(d['adapter'])
    net.eval();net.requires_grad_(False)
    from fm_matrix.train import amp
    hashes={}
    for co in PHYS:
        if model in pc.MODELS:
            atlas='sch450' if model=='brainjepa' else 'a424';p=RUN/'physiology'/co/(atlas+'_inputs.pt')
            prepared=read(RUN/'physiology/prepared.json')
            if sha(p)!=prepared['files_sha256'][str(p)]:raise ValueError('Window input changed')
        else:
            p=ROOT/'data/derived/new_fm_window_reencoding_v1'/f'{co}_fmrilm.pt'
            protocol=read(ROOT/'runs/new_fm_window_reencoding_v1/protocol.json')
            if sha(p)!=protocol['artifacts_sha256'][str(p)]:raise ValueError('Window input changed')
        hashes[str(p)]=sha(p);x=torch.load(p,weights_only=True);values=[]
        with torch.inference_mode():
            for start in range(0,len(x),4):
                with amp(device):v=net(x[start:start+4].to(device))
                if not torch.isfinite(v).all():raise ValueError('Nonfinite window representation')
                values.append(v.float().cpu().numpy())
        arr=np.concatenate(values);assert len(arr)==len(pd.read_csv(RUN/'physiology'/co/'windows.csv'))
        np.save(out/(co+'.npy'),arr);print('Independent windows',tag(*state),co,arr.shape,flush=True)
    save(out/'complete.json',dict(status='complete',state=state,input_sha256=hashes,adapter_sha256=expected,
        primary_feature_stamp_sha256=sha(u.feature_folder(state)/'complete.json'),encoder=net.provenance(),
        files_sha256={co:sha(out/(co+'.npy')) for co in PHYS}))

def phys_probe(co,state=None):
    """Refit nuisance and FC with exactly the FM probe, using shared fold files."""
    from fm_matched_v3 import physiology as engine
    root=RUN/'physiology';name='references' if state is None else tag(*state);out=root/'evaluation'/name;out.mkdir(parents=True,exist_ok=True)
    cache=root/'probes'/name/co;cache.mkdir(parents=True,exist_ok=True)
    for f in ['windows.csv','scans.csv','protocol.json',*[f'{s}_splits_s{k}.json' for s in ('dynamic','scan_mean') for k in range(3)]]:
        p=cache/f
        if not p.exists():p.symlink_to(root/co/f)
    w=pd.read_csv(root/co/'windows.csv');sm=pd.read_csv(root/co/'scans.csv');scan=w.scan.to_numpy();rows=[]
    if state is None:arrays={a+'_FC':np.load(PARENT/co/(a+'_FC.npy'),mmap_mode='r') for a in ('a424','sch450')}
    else:
        p=root/'features'/name/(co+'.npy');stamp=read(p.parent/'complete.json')
        if sha(p)!=stamp['files_sha256'][co]:raise ValueError('Window embedding hash mismatch')
        arrays={'FM':np.load(p)}
    for k,x in arrays.items():
        if len(x)!=len(w):raise ValueError('Feature/window row count mismatch')
    with patch.object(engine,'RUN',cache):
        for scope in ('dynamic','scan_mean'):
            dyn=scope=='dynamic';meta=w if dyn else sm
            nuisance=pd.get_dummies(w.Timepoint+'_'+w.WindowIdx.astype(str) if dyn else sm.Timepoint,dtype=float).to_numpy()
            if dyn:nuisance=engine.center_scans(nuisance,scan)
            nf=engine.cached_transforms(nuisance,scope,nuisance.shape[1])
            y=engine.center_scans(w.rate.to_numpy(),scan) if dyn else np.array([w.rate.to_numpy()[scan==s].mean() for s in sm.scan])
            predictions={}
            if state is None:predictions['nuisance']=engine.predict_many(nf,y,scope,meta.scan.to_numpy())[:,0]
            for key,x in arrays.items():
                z=engine.center_scans(x,scan) if dyn else np.stack([x[scan==s].mean(0) for s in sm.scan])
                fit=engine.cached_transforms(z,scope,10)
                combined=[[tuple(np.hstack([nf[s][i][j],fit[s][i][j]]) for j in range(4)) for i in range(5)] for s in range(3)]
                predictions[key]=engine.predict_many(combined,y,scope,meta.scan.to_numpy())[:,0]
            old=next(r for r in read(PARENT/co/'evaluation.json')['rows'] if r['scope']==scope)
            np.testing.assert_allclose(y,old['y'],rtol=0,atol=1e-12)
            rows.append(dict(scope=scope,subjects=meta.Subject.tolist(),scans=meta.scan.tolist(),y=y,predictions=predictions,
                scores={k:engine.summarize(y,p) for k,p in predictions.items()}))
    save(out/(co+'.json'),dict(status='complete',state=state,cohort=co,rows=rows,protocol_sha256=sha(root/co/'protocol.json'),
        interpretation='Supplementary independent-window reencoding; not native temporal-token analysis. Controls and embeddings use same freshly fitted prediction implementation.'))

def reference_comparisons(model,source,obj,target):
    from paper_paired_stats import paired
    from readout_sensitivity import seed_of
    seeds=(0,) if obj=='frozen' else (42,43,44);states=[(source,model,obj,target,s) for s in seeds]
    rows=[];hashes={}
    for co in ('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post'):
        if obj=='supervised' and co==source:continue
        for p in sorted((RUN/'references/sch450'/co).glob('*.json')):
            if p.name=='complete.json':continue
            ref=read(p);task=ref['task']
            if obj=='supervised' and task in ('AGE','SEX','AGE_GROUP'):continue
            candidates=[]
            for state in states:
                folder=u.feature_folder(state);path=folder.parent.parent/'evaluation'/folder.name/co/p.name
                q=read(path)
                if q['scan_ids']!=ref['scan_ids'] or q['subjects']!=ref['subjects']:raise ValueError('FC/FM population mismatch')
                np.testing.assert_array_equal(q['y'],ref['y']);candidates.append(q['prediction']);hashes[str(path)]=sha(path)
            pred=np.mean(candidates,axis=0)
            for method,baseline in ref['predictions'].items():
                stat=paired(ref['y'],baseline,pred,ref['subjects'],metric=ref['metric'],seed=seed_of('atlas'+model+source+obj+target+co+task+method))
                rows.append(dict(model=model,source=source,objective=obj,target=target,cohort=co,task=task,reference='sch450_'+method,
                    metric=ref['metric'],scope='source_exposed_SSL' if obj=='native_ssl' and source==co else 'external_or_frozen',**stat))
            hashes[str(p)]=sha(p)
            print('Atlas comparison',model,source,obj,target,co,task,flush=True)
    save(RUN/'reference_comparisons'/('_'.join((model,source,obj,target))+'.json'),dict(status='complete',rows=rows,files_sha256=hashes,
        inference='Raw participant bootstrap/prediction swaps; final all-configuration Holm correction pending. Adaptation seed predictions averaged before inference.'))
