"""Separate window-reencoding sensitivity; never pooled with native temporal tokens."""
import json, shutil, hashlib
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from new_fm.data import ROOT,RUN as MAIN,DATA,MODELS,source_rows,metadata,transform,save,sha
from new_fm.models import Encoder
from new_fm.extract import folder as main_features
from new_fm.train import location
from new_fm.report import states,average,adjusted
from new_fm.evaluate import read,state_name
from fm_matrix.train import amp
from fm_matched_v3 import physiology as engine
from paper_paired_stats import paired
from readout_sensitivity import seed_of

RUN=ROOT/'runs/new_fm_window_reencoding_v1'
PARENT=ROOT/'runs/physiology_frozen_crosscohort_v2_no_usable'
COHORTS=('psopt','breathe','hrv')
WIDTH={'psopt':27.5,'breathe':20.2,'hrv':48.}
INPUTS=ROOT/'data/derived/new_fm_window_reencoding_v1'


def paths(state):return RUN/'features'/state_name(state)
def result(state,co):return RUN/'evaluation'/state_name(state)/(co+'.json')

def window_input(x,window,tr,model,order):
    start=int(window)*20;stop=start+20
    if start<0 or stop>x.shape[1]:raise ValueError('Incomplete real physiological window')
    # Crop BEFORE resampling or normalization: no neighboring window is supplied
    # to the encoder, though the upstream denoising remains whole-scan processing.
    return transform(x[:,start:stop],tr,model,order)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True);INPUTS.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():verify();return
    main=read(MAIN/'intake.json');order=np.asarray(read(DATA/'roi_mapping.json')['permutation'])
    files={};artifacts={};counts={};verified_sources=set()
    def record(p):files[str(p)]=sha(p)
    record(MAIN/'intake.json');record(PARENT/'protocol.json');record(DATA/'roi_mapping.json')
    for p in [Path(__file__),ROOT/'scripts/run_new_fm_physiology.py',ROOT/'src/new_fm/models.py',ROOT/'src/new_fm/data.py',Path(engine.__file__),ROOT/'src/paper_paired_stats.py']:record(p)
    for co in COHORTS:
        out=RUN/co;out.mkdir(exist_ok=True);old=PARENT/co
        for name in ['windows.csv','scans.csv',*[f'{scope}_splits_s{s}.json' for scope in ('dynamic','scan_mean') for s in (0,1,2)]]:
            record(old/name);shutil.copyfile(old/name,out/name);artifacts[str(out/name)]=sha(out/name)
        record(old/'evaluation.json')
        w=pd.read_csv(out/'windows.csv');m=pd.read_csv(out/'scans.csv');ref=read(old/'evaluation.json')
        if w.duplicated(['scan','WindowIdx']).any():raise ValueError('Duplicate window')
        for r in ref['rows']:
            meta=w if r['scope']=='dynamic' else m
            if r['subjects']!=meta.Subject.tolist() or r['scans']!=meta.scan.tolist():raise ValueError('Reference population changed')
            for seed in (0,1,2):
                for f in read(out/f"{r['scope']}_splits_s{seed}.json"):
                    g=meta.Subject.to_numpy()
                    if set(g[f['train']])&set(g[f['test']]) or set(g[f['inner_train']])&set(g[f['inner_validation']]):raise ValueError('Participant split overlap')
        outputs={model:[None]*len(w) for model in MODELS};lengths={model:np.zeros(len(w),int) for model in MODELS}
        for ds in w.dataset.unique():
            meta=metadata(ds);lookup={str(s):i for i,s in enumerate(meta.scan)}
            byindex=defaultdict(list)
            for i,r in w[w.dataset==ds].iterrows():
                if r.scan not in lookup:raise ValueError('Missing registered scan '+r.scan)
                k=lookup[r.scan]
                if k!=int(r.matrix_index):raise ValueError('Parent/new-model scan index mismatch')
                byindex[k].append(i)
            for k,(x,tr,path) in enumerate(source_rows(ds)):
                if k not in byindex:continue
                p=Path(path);expected=main['sources_sha256'].get(str(p))
                if str(p) not in verified_sources:
                    if expected is None or sha(p)!=expected:raise ValueError('Changed original time series '+str(p))
                    verified_sources.add(str(p))
                if not np.isclose(20*tr,WIDTH[co]):raise ValueError('Physiological timing mismatch')
                for i in byindex[k]:
                    for model in MODELS:
                        v,n=window_input(x,w.iloc[i].WindowIdx,tr,model,order)
                        outputs[model][i]=v;lengths[model][i]=n
        if any(v is None for vv in outputs.values() for v in vv):raise ValueError('Incomplete window coverage; never silently drop samples')
        for model in MODELS:
            p=INPUTS/f'{co}_{model}.pt';tmp=p.with_suffix('.tmp.pt');torch.save(torch.stack(outputs[model]),tmp);tmp.replace(p);artifacts[str(p)]=sha(p)
        counts[co]=dict(subjects=int(w.Subject.nunique()),scans=int(w.scan.nunique()),windows=len(w),window_seconds=WIDTH[co],resampled_real_frames={m:sorted(set(map(int,lengths[m]))) for m in MODELS})
        save(out/'protocol.json',dict(cohort=co,parent_sha256=sha(old/'protocol.json'),kind='window_reencoding',input_hashes={m:sha(INPUTS/f'{co}_{m}.pt') for m in MODELS}))
        artifacts[str(out/'protocol.json')]=sha(out/'protocol.json')
        print('Prepared window reencoding',co,counts[co],flush=True)
    save(RUN/'protocol.json',dict(status='registered',files_sha256=files,artifacts_sha256=artifacts,counts=counts,
        states=list(states()),models=list(MODELS),cohorts=list(COHORTS),
        label='Exploratory independent-window reencoding; not native temporal-token physiology',
        authorization='User explicitly selected separate per-window reencoding after architectural incompatibility was explained.',
        input='Each retained20-native-frame window is cropped from original-length Schaefer450 signals before polyphase TR2 resampling and window-local normalization. Semantoks uses observed cortical/subcortical tokens with missing cerebellar network masked; fMRI-LM uses Tian-first/Schaefer7 order and scan-local robust-scaling variant. Here scan-local scaling is applied independently to each cropped window.',
        padding='Right zero-padding after normalization to model input length100 Semantoks/160 fMRI-LM. Only about11/14/24 real TR2 samples exist per window. No stretching, full-scan embedding replication, or neighboring-window encoder context. Extensive padding is an explicit out-of-distribution limitation.',
        readout='Same primary scan readout as new-model frozen/post-training panel, applied independently to every window: Semantoks teacher CLS; fMRI-LM mean final brain-only Qwen tokens.',
        endpoint='ECG mean heart rate in BREATHE/pSOPT; PPG mean pulse rate in HRV-ER. No raw waveform/beat-to-beat reconstruction or new RMSSD endpoint.',
        sample='Exact current no-usable-mark reference windows/scans; finite and time-aligned retained windows, >=6windows/scan. No cognitive-eligibility restriction or new QC exclusion.',
        dynamic='Within-scan centering of retained features, rate and predictions; window-position-by-visit covariates. This is offline within-scan variation, not prospective online rate prediction.',
        scan_mean='Average embeddings and rates across identical retained windows; visit covariates. No whole-recording interpretation.',
        evaluation='Exact archived outcome-independent five-fold participant splits, inner grouped20% tuning, seeds0/1/2; train-fold scaler/PCA<=10 using shared<=20 basis; ridge C=.001,.01,.1,1,10. Frozen and post-training use identical readout/input/probes. Average predictions across split seeds and adaptation42/43/44 before inference.',
        references='Reuse verified nuisance-only and Schaefer450 window-FC OOF predictions on identical outcomes/rows/folds. FC uses original20native-frame input; not a duration/parcellation/encoder-mechanism matched ablation. No amplitude arm added.',
        inference='5000 participant-cluster bootstraps;20000 two-sided participant prediction swaps, conditional on fitted predictions. Separate supplementary Holm families by scope, objective/target and comparison type across bothnew models, sources and3cohorts. This panel does not enlarge or replace native temporal-token comparison families.',
        adaptations='Use existing external ADNI/NACC MEM,EF,SSL checkpoints only; no encoder training on physiological labels. All3adaptationseeds. No new source fitting.',
        limitations=['Short-window heavily padded model inputs differ from pretraining context','Whole-scan upstream denoising remains; no causal or online interpretation','Inherited atlas/normalization and synchronization qualifications retained'],
        main_queue_policy='Independent supplementary queue shares GPU lock; never interrupts a training job. New-model frozen window report precedes adapted window extraction; main queue unchanged.',automatic_shutdown=False))


def verify():
    d=read(RUN/'protocol.json')
    for p,h in {**d['files_sha256'],**d['artifacts_sha256']}.items():
        if sha(p)!=h:raise ValueError('Changed registered window input '+p)
    return d


def extract(state,device='cuda'):
    verify();out=paths(state);out.mkdir(parents=True,exist_ok=True)
    source,model,obj,target,seed=state
    ck=None if obj=='frozen' else location(source,model,obj,target,seed,'external')/('fold0/adapter.pt' if obj=='supervised' else 'adapter.pt')
    parent_export=main_features(state)/'complete.json';parent=read(parent_export)
    if parent['signature']['intake_sha256']!=sha(MAIN/'intake.json'):raise ValueError('Stale parent encoder export')
    signature=dict(state=list(state),protocol_sha256=sha(RUN/'protocol.json'),adapter_sha256=sha(ck) if ck else None,parent_export_sha256=sha(parent_export))
    if (out/'request.json').exists() and read(out/'request.json')!=signature:raise ValueError('Changed window export request')
    save(out/'request.json',signature)
    if (out/'complete.json').exists():return
    net=Encoder(model,device)
    if sha(net.checkpoint_path)!=parent['source_checkpoint_sha256']:raise ValueError('Parent/base checkpoint mismatch')
    if ck:
        d=torch.load(ck,map_location='cpu',weights_only=True)
        if d['model']!=model or d['intake_sha256']!=sha(MAIN/'intake.json') or d['source_checkpoint_sha256']!=sha(net.checkpoint_path):raise ValueError('Wrong adapter provenance')
        net.load_adapter(d['adapter'])
    files={}
    for co in COHORTS:
        p=out/(co+'.npy')
        if not p.exists():
            x=torch.load(INPUTS/f'{co}_{model}.pt',weights_only=True);values=[]
            with torch.no_grad():
                for start in range(0,len(x),4):
                    with amp(device):v=net(x[start:start+4].to(device))
                    if not torch.isfinite(v).all():raise ValueError('Nonfinite window embedding')
                    values.append(v.float().cpu().numpy())
            tmp=p.with_suffix('.tmp.npy');np.save(tmp,np.concatenate(values));tmp.replace(p)
            print('Window features',state_name(state),co,len(x),flush=True)
        files[str(p)]=sha(p)
    save(out/'complete.json',dict(status='complete',signature=signature,features_sha256=files,encoder=net.provenance(),source_checkpoint_sha256=sha(net.checkpoint_path)))


def evaluate(state,co):
    dest=result(state,co)
    if dest.exists():return
    verify();out=RUN/co;f=paths(state);stamp=read(f/'complete.json');p=f/(co+'.npy')
    if stamp['signature']['protocol_sha256']!=sha(RUN/'protocol.json') or sha(p)!=stamp['features_sha256'][str(p)]:raise ValueError('Changed window embeddings')
    x=np.load(p);w=pd.read_csv(out/'windows.csv');sm=pd.read_csv(out/'scans.csv');scan=w.scan.to_numpy()
    if len(x)!=len(w):raise ValueError('Window feature ordering mismatch')
    refs=read(PARENT/co/'evaluation.json')['rows'];rows=[]
    # Different states can be evaluated concurrently. Keep fit caches per worker
    # to avoid two processes replacing the same nuisance-transform temporary file.
    cache=RUN/'probes'/state_name(state)/co;cache.mkdir(parents=True,exist_ok=True)
    for name in ['windows.csv','scans.csv','protocol.json',*[f'{s}_splits_s{k}.json' for s in ('dynamic','scan_mean') for k in (0,1,2)]]:
        path=cache/name
        if not path.exists():path.symlink_to(out/name)
        if path.resolve()!=(out/name).resolve():raise ValueError('Unexpected physiology cache input')
    with patch.object(engine,'RUN',cache):
        for scope in ('dynamic','scan_mean'):
            dyn=scope=='dynamic';meta=w if dyn else sm
            nuisance=pd.get_dummies(w.Timepoint+'_'+w.WindowIdx.astype(str) if dyn else sm.Timepoint,dtype=float).to_numpy()
            if dyn:nuisance=engine.center_scans(nuisance,scan)
            nfit=engine.cached_transforms(nuisance,scope,nuisance.shape[1])
            z=engine.center_scans(x,scan) if dyn else np.stack([x[scan==s].mean(0) for s in sm.scan])
            y=engine.center_scans(w.rate.to_numpy(),scan) if dyn else np.array([w.rate.to_numpy()[scan==s].mean() for s in sm.scan])
            ref=next(r for r in refs if r['scope']==scope)
            if ref['subjects']!=meta.Subject.tolist() or ref['scans']!=meta.scan.tolist():raise ValueError('Reference order changed')
            np.testing.assert_allclose(y,ref['y'],rtol=0,atol=1e-12)
            # Verify identical training-fold nuisance designs. Archived controls
            # remain fixed: numerical refits need not reproduce their predictions.
            h=hashlib.sha256(np.ascontiguousarray(nuisance).tobytes())
            h.update(f'{scope}/{nuisance.shape[1]}/False'.encode())
            h.update(sha(PARENT/co/'protocol.json').encode())
            parent_cache=PARENT/co/'transform_cache'/h.hexdigest()
            for seed in range(3):
                for fold in range(5):
                    with np.load(parent_cache/f's{seed}_f{fold}.npz') as saved:
                        for value,key in zip(nfit[seed][fold],('a','b','ai','bi')):
                            np.testing.assert_array_equal(value,saved[key])
            npred=engine.predict_many(nfit,y,scope,meta.scan.to_numpy())[:,0]
            refit_delta=float(np.max(abs(npred-np.asarray(ref['predictions']['nuisance']))))
            fit=engine.cached_transforms(z,scope,10)
            combined=[[tuple(np.hstack([nfit[s][i][j],fit[s][i][j]]) for j in range(4)) for i in range(5)] for s in range(3)]
            prediction=engine.predict_many(combined,y,scope,meta.scan.to_numpy())[:,0]
            rows.append(dict(scope=scope,subjects=meta.Subject.tolist(),scans=meta.scan.tolist(),y=y,prediction=prediction,scores=engine.summarize(y,prediction),
                reference_reuse_audit=dict(nuisance_training_design='Bitwise identical to archived outer/inner fitted designs',nuisance_refit_max_abs_difference=refit_delta,
                    reference_predictions='Immutable archived nuisance/FC predictions used; diagnostic nuisance refit not substituted.')))
            print('Window probe',state_name(state),co,scope,rows[-1]['scores'],flush=True)
    save(dest,dict(status='complete',state=state,cohort=co,rows=rows,protocol_sha256=sha(RUN/'protocol.json'),feature_sha256=sha(p),reference_sha256=sha(PARENT/co/'evaluation.json')))


def report(phase):
    verify();out=RUN/'reports'/phase;out.mkdir(parents=True,exist_ok=True);groups=defaultdict(list)
    for state in states(phase):
        for co in COHORTS:
            for r in read(result(state,co))['rows']:groups[(*state[:4],co,r['scope'])].append(r)
    scores=[];predictions=[];contrasts=[]
    for k,rr in groups.items():
        source,model,obj,target,co,scope=k
        if len(rr)!=(1 if obj=='frozen' else 3):raise ValueError('Incomplete adaptation seeds')
        for r in rr[1:]:
            if r['scans']!=rr[0]['scans']:raise ValueError('Adaptation seed scan order mismatch')
        pred=average(rr);r=rr[0]
        info=dict(source=source,model=model,objective=obj,target=target,cohort=co,scope=scope)
        scores.append(dict(info,n=len(pred),subjects=len(set(r['subjects'])),**engine.summarize(r['y'],pred)))
        predictions.append(dict(info,subjects=r['subjects'],scans=r['scans'],y=r['y'],prediction=pred))
        ref=next(a for a in read(PARENT/co/'evaluation.json')['rows'] if a['scope']==scope)
        if ref['subjects']!=r['subjects'] or ref['scans']!=r['scans']:raise ValueError('Reference participant mismatch')
        np.testing.assert_array_equal(ref['y'],r['y'])
        references={n:ref['predictions'][n] for n in ('nuisance','sch450_FC')}
        if obj!='frozen':references['same_model_frozen']=groups[('none',model,'frozen','none',co,scope)][0]['prediction']
        for name,a in references.items():
            label='_'.join(map(str,(*k,name)));cache=RUN/'inference'/(label+'.json')
            if cache.exists():stat=read(cache)['statistics']
            else:
                stat=paired(r['y'],a,pred,r['subjects'],seed=seed_of(label))
                save(cache,dict(status='complete',statistics=stat,protocol_sha256=sha(RUN/'protocol.json')))
            contrasts.append(dict(info,reference=name,family='_'.join((scope,obj,target,name)),**stat))
    for co in COHORTS:
        for r in read(PARENT/co/'evaluation.json')['rows']:
            for method in ('nuisance','sch450_FC'):
                scores.append(dict(source='none',model=method,objective='reference',target='none',cohort=co,scope=r['scope'],n=len(r['y']),subjects=len(set(r['subjects'])),**r['scores'][method]))
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False);save(out/'predictions.json',predictions)
    save(out/'paired_comparisons.json',adjusted(contrasts));pd.DataFrame(contrasts).to_csv(out/'paired_comparisons.csv',index=False)
    save(out/'complete.json',dict(status='complete',phase=phase,models=list(MODELS),score_rows=len(scores),comparisons=len(contrasts),protocol_sha256=sha(RUN/'protocol.json'),panel='Exploratory independent-window reencoding; separate from native temporal-token physiology',automatic_shutdown=False))
