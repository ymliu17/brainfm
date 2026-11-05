#!/usr/bin/env python3
"""Additive, CPU-only regional BOLD SD controls for seven-cohort cognition.

Reuses registered samples, partitions and fitted FM predictions where available.
No encoder extraction/training, absolute null refits, manuscript edits or shutdown.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, copy, fcntl, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from fm_matrix.data import save,sha
from fm_readout_v4.data import metadata,dataset
from benchmark_evaluation import predict_prepared,score
from paper_paired_stats import paired,holm
import run_connectivity_readouts as frozen
import run_nine_model_identity_controls as identity
import run_intervention_readouts as longitudinal
import intervention_prepost as ip
import run_evidence_nulls as null
import sf100_benchmark as sf

RUN=ROOT/'runs/sd_prediction_controls_v1'
OUT=ROOT/'results/generated/sd_prediction_controls_v1'
LOG=ROOT/'logs/sd_prediction_controls_v1'
KINDS=('a424','sch450','semantoks','fmrilm','harmonix')
COHORTS=('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post')
MODELS=frozen.MODELS
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def features_path(kind,co):return RUN/'features'/co/(kind+'.npy')
def model_kind(model):return identity.input_kind(model)
def output(j):return RUN/'results'/(j['id']+'.json')
def local_tasks(co):
    if co in ('adni','nacc'):return ('MEM','EF')
    if co=='hrv_pre':return ('EF','EF_followup','MEM_delayed_recall','MEM_recall_clinical')
    return ('MEM','EF','MEM_followup','EF_followup')

def jobs():
    result=[dict(id='features/'+co,action='features',cohort=co) for co in COHORTS]
    for co in COHORTS[:-1]:
        for kind in KINDS:
            for task in local_tasks(co):
                result.append(dict(id=f'local/{kind}/{co}/{task}',action='local',kind=kind,cohort=co,task=task,dependencies=['features/'+co]))
    for co in ip.COHORTS:
        for kind in KINDS:
            deps=['features/hrv_pre','features/hrv_post'] if co=='hrv' else ['features/'+co]
            result.append(dict(id=f'longitudinal/{kind}/{co}',action='longitudinal',kind=kind,cohort=co,dependencies=deps))
    return result

def register():
    paths=[Path(__file__),Path(frozen.__file__),Path(identity.__file__),Path(longitudinal.__file__),Path(ip.__file__),
        ROOT/'src/paper_paired_stats.py',ROOT/'src/benchmark_evaluation.py',ROOT/'src/paper_completion.py',
        ROOT/'src/new_fm/data.py',ROOT/'src/temporal.py',ip.RUN/'inputs.json',ROOT/'runs/new_fm_v1/intake.json',
        ROOT/'data/derived/new_fm_v1/roi_mapping.json']
    for co in COHORTS[:-1]:
        for task in local_tasks(co):
            if task!='MEM_recall_clinical':paths.append(sf.RUN/'evaluation'/co/(task+'.json'))
        for model in MODELS:paths.append(frozen.RUN/'features'/model/co/'complete.json')
    for model in MODELS:
        for co in ip.COHORTS:
            paths.extend([longitudinal.sensitivity.frozen_path(model,co),longitudinal.RUN/'results'/model/(co+'.json')])
    protocol=dict(scope='Seven-cohort cognitive-level prediction; intervention follow-up, cognitive change and arm discrimination. Frozen primary and final-layer embedding connectivity; no post-training/readout/alignment Cartesian expansion.',
        unavailable='No HRV baseline MEM or paired MEM change. Single delayed recall uses baseline imaging and is labeled separately. No invented endpoint.',
        SD='Per-region population SD (ddof=0), observed model-matched window only, no padding or cerebellar missing rows. Canonical 200 A424 /160 Sch450 frames; Semantoks first100 and fMRI-LM first160 TR2-resampled frames; Harmonix first18*round(35.28/TR) native frames. Before model-specific regional scaling, exactly matching existing Re-ID SD construction.',
        reuse='Five unique input families shared by nine configurations. SD extracted for all eligible scans, not limited to the paired Re-ID sample. Verify overlap against completed Re-ID SD features.',
        fitting='Reuse original scan IDs/outcomes/grouped outer and inner partitions and ridge/logistic grids. Baseline cognition PCA<=100; follow-up/recall PCA<=5; longitudinal PCA<=5. Three split seeds averaged before scoring. HRV clinical recall follows its existing separate clinical-block probe.',
        longitudinal='Baseline, post-minus-pre SD, concatenated baseline+change. Imaging-only and clinical-plus-imaging for change; imaging-only for arm. Same paired-visit engine and class order.',
        references='Primary and final-layer connectivity FMs, Schaefer100 FC, actual-input-window FC for local tasks; clinical-only where applicable. Existing primary/readout results remain unchanged. New local connectivity follow-up fits are explicitly supplementary.',
        inference='5000 participant-cluster bootstrap samples; 20000 two-sided conditional prediction swaps. Arm probability vectors swapped together. No SD absolute-null test, no label-permutation refitting, no above-chance claims.',
        families='Local: endpoint x setting x FM-versus-SD (pool configurations and readouts) / FC-versus-SD (pool input families and FC types) / SD clinical increment. Longitudinal: endpoint x setting x contrast class, pooling configurations, readouts, cohorts and input modes. Separate additive exploratory families; publish corrected values only when the full planned family is complete.',
        priority='32 reserved single-thread CPU workers; original FC/GFM null queue runs separately on48. No GPU required.',automatic_shutdown=False,original_manuscript_modified=False)
    manifest=dict(jobs=jobs(),protocol=protocol,files_sha256={str(p):sha(p) for p in sorted(set(paths))})
    p=RUN/'manifest.json'
    if p.exists():assert read(p)==manifest,'Registered inputs/code changed'
    else:save(p,manifest);save(RUN/'protocol.json',protocol)
    return manifest

def verify():
    manifest=read(RUN/'manifest.json')
    assert sha(__file__)==manifest['files_sha256'][str(Path(__file__).resolve())]
    return manifest

def extract(co):
    from new_fm.data import source_rows
    from temporal import resample_time
    import torch
    torch.set_num_threads(1)
    m=metadata(co);values={};sources={};parity=[]
    for kind,model,width in [('a424','brainlm13',200),('sch450','brainjepa',160)]:
        x,im=dataset(co,model);assert im.scan.tolist()==m.scan.tolist()
        values[kind]=np.stack([identity.old.input_summaries(z.numpy(),min(int(r.native_frames),width))[0] for z,r in zip(x,im.itertuples())])
        del x
    order=np.asarray(read(ROOT/'data/derived/new_fm_v1/roi_mapping.json')['permutation'])
    expected=read(ROOT/'runs/new_fm_v1/intake.json')['sources_sha256']
    rows={k:[] for k in KINDS[2:]}
    for ix,(x,tr,path) in enumerate(source_rows(co)):
        path=str(path)
        if path not in sources:
            sources[path]=sha(path);assert sources[path]==expected[path],path
        z=resample_time(x.float(),tr,2.).numpy()
        rows['semantoks'].append(identity.old.input_summaries(z[order],min(z.shape[1],100))[0])
        rows['fmrilm'].append(identity.old.input_summaries(z[np.r_[order[400:],order[:400]]],min(z.shape[1],160))[0])
        rows['harmonix'].append(identity.old.input_summaries(x[:400].numpy(),min(x.shape[1],18*round(35.28/tr)))[0])
    for k,v in rows.items():values[k]=np.stack(v)
    # Verify already completed paired-scan controls on every overlapping scan.
    pairco='hrv' if co.startswith('hrv') else 'nacc_repeated' if co=='nacc' else co
    pair=identity.pair(pairco);lookup={s:i for i,s in enumerate(m.scan)}
    for kind,x in values.items():
        assert len(x)==len(m) and np.isfinite(x).all()
        oldfile=identity.RUN/'summaries'/pairco/kind/'features.npz'
        with np.load(oldfile) as old:
            for side in ('a','b'):
                indices=[(j,lookup[s]) for j,s in enumerate(pair[side]['scans']) if s in lookup]
                for j,i in indices:np.testing.assert_array_equal(x[i],old['roi_std_'+side][j])
                parity.append(dict(kind=kind,side=side,matched=len(indices)))
        p=features_path(kind,co);p.parent.mkdir(parents=True,exist_ok=True);np.save(p,x)
    return dict(status='complete',cohort=co,scan_ids=m.scan.tolist(),features_sha256={k:sha(features_path(k,co)) for k in KINDS},sources_sha256=sources,ReID_parity=parity)

def load_sd(kind,co):
    stamp=read(output(dict(id='features/'+co)))
    assert stamp['scan_ids']==metadata(co).scan.tolist()
    p=features_path(kind,co);assert sha(p)==stamp['features_sha256'][kind]
    return np.load(p,mmap_mode='r')

def check_template(t,co):
    m=sf.labels(co);ix=m.set_index('scan').index.get_indexer(t['scan_ids']);g=np.asarray(t['subjects']).astype(str)
    assert (ix>=0).all() and m.iloc[ix].subject.astype(str).tolist()==list(g)
    for s in (0,1,2):
        seen=np.zeros(len(g),int)
        for f in [f for f in t['split_audit'] if f['seed']==s]:
            tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
            assert set(tr)|set(te)==set(range(len(g))) and not set(tr)&set(te)
            assert set(it)|set(iv)==set(tr) and not set(it)&set(iv)
            assert not set(g[tr])&set(g[te]) and not set(g[it])&set(g[iv])
            seen[te]+=1
        assert (seen==1).all()
    return m,ix

def fit_local(x,t,co,k):
    from paper_robustness import covariate_pair
    m,ix=check_template(t,co);y=np.asarray(t['y']);cov=t.get('covariates',[])
    modes=['imaging_only']+(['clinical_plus_imaging','clinical'] if cov else [])
    prepared={mode:{s:[] for s in (0,1,2)} for mode in modes}
    for f in t['split_audit']:
        tr,te,it,iv=[np.asarray(f[q],int) for q in ('train','test','inner_train','validation')]
        a,b=frozen.project_indices(x,ix[tr],ix[te],k,'cpu');ai,bi=frozen.project_indices(x,ix[it],ix[iv],k,'cpu')
        prepared['imaging_only'][f['seed']].append((tr,te,(a,b),it,iv,(ai,bi)))
        if cov:
            ca,cb=covariate_pair(m,cov,ix[tr],ix[te]);ci,cj=covariate_pair(m,cov,ix[it],ix[iv])
            prepared['clinical_plus_imaging'][f['seed']].append((tr,te,(np.c_[ca,a],np.c_[cb,b]),it,iv,(np.c_[ci,ai],np.c_[cj,bi])))
            prepared['clinical'][f['seed']].append((tr,te,(ca,cb),it,iv,(ci,cj)))
    by={mode:[predict_prepared(prepared[mode][s],y,binary=False) for s in (0,1,2)] for mode in modes}
    pp={mode:np.mean(v,axis=0) for mode,v in by.items()}
    assert all(np.isfinite(v).all() for v in pp.values())
    return dict(predictions=pp,predictions_by_seed=by,subjects=t['subjects'],scan_ids=t['scan_ids'],y=y,split_audit=t['split_audit'])

def cached_local(name,x,t,co,task,k):
    p=RUN/'local_fits'/name/co/(task+'.json')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if p.exists():
            d=read(p);assert d['scan_ids']==t['scan_ids'] and d['split_audit']==t['split_audit'];return d
        d=fit_local(x,t,co,k);save(p,d);return d

def add_contrast(rows,y,a,b,g,info,classes=None):
    key=json.dumps(info,sort_keys=True);token=frozen.seed(key);p=RUN/'contrasts'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
    if p.exists():row=read(p)
    else:
        if classes:st=longitudinal.compare(np.asarray(y),np.asarray(a),np.asarray(b),classes,info);row=st
        else:row=dict(**info,**paired(y,a,b,g,seed=token))
        save(p,row)
    for k,v in info.items():assert row[k]==v
    rows.append(row)

def fc_path(kind,co):
    if kind=='harmonix':return ROOT/'runs/brainharmonix_f_v1/supplements/input_fc/brainharmonix_f'/(co+'.npy')
    return ROOT/'runs/reviewer_completion_v1/input_fc'/kind/(co+'.npy')

def local(kind,co,task):
    if task=='MEM_recall_clinical':return recall_clinical(kind)
    tp=sf.RUN/'evaluation'/co/(task+'.json');t=read(tp);check_template(t,co)
    k=5 if t['covariates'] or task=='MEM_delayed_recall' else 100
    sd=cached_local('SD_'+kind,load_sd(kind,co),t,co,task,k)
    preds={'SD':sd['predictions']};rows=[];provenance={str(tp):sha(tp)}
    for model in [m for m in MODELS if model_kind(m)==kind]:
        old=frozen.RUN/'cognition'/model/(co+'_'+task+'.json')
        if co in ('adni','nacc') and old.exists():
            d=read(old);assert d['scan_ids']==t['scan_ids'] and d['split_audit']==t['split_audit'];np.testing.assert_array_equal(d['y'],t['y'])
            for rd in ('native','connectivity'):preds[model+'_'+rd]={'imaging_only':d['predictions'][rd]}
            provenance[str(old)]=sha(old)
        else:
            for rd in ('native','connectivity'):
                fp=null.feature_path(model,co) if rd=='native' else frozen.RUN/'features'/model/co/'connectivity.npy'
                x=np.load(fp,mmap_mode='r');assert len(x)==len(metadata(co))
                provenance[str(fp)]=sha(fp)
                d=cached_local(model+'_'+rd,x,t,co,task,k);preds[model+'_'+rd]=d['predictions']
                # Validate primary predictions against archived outputs when available.
                if rd=='native' and model in t['predictions'] and task!='MEM_delayed_recall':
                    setting='clinical_plus_imaging' if t['covariates'] else 'imaging_only'
                    np.testing.assert_allclose(d['predictions'][setting],t['predictions'][model],rtol=0,atol=2e-5)
                if rd=='native' and task=='MEM_delayed_recall':
                    from fm_readout_v4.evaluate import tag
                    rp=ROOT/'runs/hrv_recall_pca5_v1/fits'/('hrv_pre__'+tag('none',model,'frozen','none',0)+'.json')
                    r=read(rp);assert r['scan_ids']==t['scan_ids'];np.testing.assert_allclose(d['predictions']['imaging_only'],r['prediction']['fixed5'],rtol=0,atol=2e-5)
    # The same-window FC arm uses the same projection and tuning as SD.
    fp=fc_path(kind,co);provenance[str(fp)]=sha(fp)
    preds['input_window_FC']=cached_local('input_FC_'+kind,np.load(fp,mmap_mode='r'),t,co,task,k)['predictions']
    sfp=sf.RUN/'features'/co/'sf100_fc.npy'
    preds['sf100_fc']=cached_local('sf100_fc',np.load(sfp,mmap_mode='r'),t,co,task,k)['predictions']
    y=t['y'];g=t['subjects'];scores=[]
    ep='MEM_followup' if task=='MEM_delayed_recall' else task
    for setting,b in preds['SD'].items():
        if setting=='clinical':continue
        for name,arms in preds.items():
            if setting not in arms:continue
            a=arms[setting];scores.append(dict(kind=kind,cohort=co,task=task,setting=setting,representation=name,score=score(y,a,False),metric='rho',n=len(y),participants=len(set(g))))
            if name=='SD':continue
            group='FC_vs_SD' if name in ('sf100_fc','input_window_FC') else 'FM_vs_SD'
            info=dict(kind=kind,cohort=co,task=task,setting=setting,reference='SD',candidate=name,family='local|'+ep+'|'+setting+'|'+group)
            add_contrast(rows,y,b,a,g,info)
        if setting=='clinical_plus_imaging':
            add_contrast(rows,y,preds['SD']['clinical'],b,g,dict(kind=kind,cohort=co,task=task,setting=setting,reference='clinical',candidate='SD+clinical',family='local|'+ep+'|'+setting+'|SD_increment'))
    return dict(status='complete',scores=scores,contrasts=rows,sources_sha256=provenance,scan_ids=t['scan_ids'],subjects=g,split_audit=t['split_audit'])

def recall_clinical(kind):
    from fm_readout_v4.evaluate import clinical_features
    from fm_matched_v3.probes import oof_predictions
    import sample_alignment as sa
    co='hrv_pre';m=sf.labels(co);c=clinical_features(m,co);yy=m.MEM_delayed_recall.to_numpy(float)
    ix=np.flatnonzero(sa.common_mask(m,co)&np.isfinite(yy)&np.isfinite(c).all(1));y=yy[ix];g=m.iloc[ix].subject.astype(str).to_numpy()
    xs={'SD':load_sd(kind,co),'input_window_FC':np.load(fc_path(kind,co),mmap_mode='r'),'sf100_fc':np.load(sf.RUN/'features'/co/'sf100_fc.npy',mmap_mode='r')}
    for model in [m for m in MODELS if model_kind(m)==kind]:
        for rd in ('native','connectivity'):
            p=null.feature_path(model,co) if rd=='native' else frozen.RUN/'features'/model/co/'connectivity.npy'
            xs[model+'_'+rd]=np.load(p,mmap_mode='r')
    clinical=np.mean([oof_predictions([c[ix]],[c.shape[1]],y,s,g) for s in (0,1,2)],axis=0)
    preds={};rows=[];scores=[]
    for name,x in xs.items():
        p=RUN/'recall_clinical_fits'/kind/(name+'.json')
        if p.exists():d=read(p);assert d['scan_ids']==m.iloc[ix].scan.tolist();preds[name]=d['prediction']
        else:
            by=[oof_predictions([c[ix],x[ix]],[c.shape[1],5],y,s,g) for s in (0,1,2)];preds[name]=np.mean(by,axis=0)
            save(p,dict(scan_ids=m.iloc[ix].scan.tolist(),subjects=g,y=y,prediction=preds[name],by_seed=by))
        scores.append(dict(kind=kind,cohort=co,task='MEM_delayed_recall',setting='clinical_plus_imaging',representation=name,score=score(y,preds[name],False),metric='rho',n=len(y),participants=len(set(g))))
    for name,pred in preds.items():
        if name=='SD':continue
        group='FC_vs_SD' if name in ('sf100_fc','input_window_FC') else 'FM_vs_SD'
        add_contrast(rows,y,preds['SD'],pred,g,dict(kind=kind,cohort=co,task='MEM_delayed_recall',setting='clinical_plus_imaging',reference='SD',candidate=name,family='local|MEM_followup|clinical_plus_imaging|'+group))
    add_contrast(rows,y,clinical,preds['SD'],g,dict(kind=kind,cohort=co,task='MEM_delayed_recall',setting='clinical_plus_imaging',reference='clinical',candidate='SD+clinical',family='local|MEM_followup|clinical_plus_imaging|SD_increment'))
    return dict(status='complete',scores=scores,contrasts=rows,subjects=g,scan_ids=m.iloc[ix].scan.tolist(),protocol='Existing separate HRV clinical recall block/PCA5 probe, not the old imaging-only PCA100 probe.')

def paired_visits(kind,co):
    parent=read(ip.RUN/'inputs.json');plan=parent['plans'][co];state=('none','SD_'+kind,'frozen','none',0)
    fitroot=RUN/'paired_fits'/kind/co;data=copy.deepcopy(parent);data['states']=[state];data['files_sha256']={}
    for ds in {plan['pre_cohort'],plan['post_cohort']}:
        load_sd(kind,ds);p=features_path(kind,ds);data['files_sha256'][str(p)]=sha(p)
    data['signature']=frozen.seed(json.dumps(dict(parent=parent['signature'],kind=kind,files=data['files_sha256']),sort_keys=True))
    save(fitroot/'inputs.json',data)
    with patch.object(ip,'RUN',fitroot),patch.object(ip,'feature_path',lambda s,ds:features_path(kind,ds)):
        fit=read(ip.run_job((state,co)))
    sd={(r['task'],r['mode']):r for r in fit['rows']};comparators={}
    for model in [m for m in MODELS if model_kind(m)==kind]:
        d=read(longitudinal.sensitivity.frozen_path(model,co));longitudinal.sensitivity.validate_rows(d['rows'],co)
        comparators[model+'_native']={(r['task'],r['mode']+('+clinical' if r['setting']=='clinical_plus_imaging' else '')):dict(r,prediction=r['predictions']['fixed5']) for r in d['rows']}
        d=read(read(longitudinal.RUN/'results'/model/(co+'.json'))['fit'])
        comparators[model+'_connectivity']={(r['task'],r['mode']):r for r in d['rows']}
    d=read(longitudinal.sensitivity.frozen_path('sf100_fc',co));longitudinal.sensitivity.validate_rows(d['rows'],co)
    comparators['sf100_fc']={(r['task'],r['mode']+('+clinical' if r['setting']=='clinical_plus_imaging' else '')):dict(r,prediction=r['predictions']['fixed5']) for r in d['rows']}
    rows=[];scores=[]
    for (task,mode),r in sd.items():
        if mode in ('clinical','null'):continue
        longitudinal.validate_row(r,plan);y=r['y'];g=r['subjects'];classes=r['classes'];a=r['prediction']
        setting='clinical_plus_imaging' if '+clinical' in mode else 'imaging_only'
        base=dict(kind=kind,cohort=co,task=task,mode=mode,setting=setting)
        scores.append(dict(**base,representation='SD',n=len(y),participants=len(g),**ip.metrics(y,np.asarray(a),classes)))
        for name,rr in comparators.items():
            b=rr[(task,mode)];longitudinal.validate_row(b,plan)
            scores.append(dict(**base,representation=name,n=len(y),participants=len(g),**ip.metrics(y,np.asarray(b['prediction']),classes)))
            group='FC_vs_SD' if name=='sf100_fc' else 'FM_vs_SD'
            add_contrast(rows,y,a,b['prediction'],g,dict(**base,reference='SD',candidate=name,family='longitudinal|'+task+'|'+setting+'|'+group),classes)
        if setting=='clinical_plus_imaging':
            add_contrast(rows,y,sd[(task,'clinical')]['prediction'],a,g,dict(**base,reference='clinical',candidate='SD+clinical',family='longitudinal|'+task+'|'+setting+'|SD_increment'))
    return dict(status='complete',scores=scores,contrasts=rows,fit=str(fitroot),input_signature=data['signature'])

def execute(j):
    verify();dest=output(j)
    if dest.exists():return
    start=time.time()
    if j['action']=='features':d=extract(j['cohort'])
    elif j['action']=='local':d=local(j['kind'],j['cohort'],j['task'])
    else:d=paired_visits(j['kind'],j['cohort'])
    d.update(job=j,seconds=time.time()-start,completed_utc=utc(),manifest_sha256=sha(RUN/'manifest.json'))
    save(dest,d);print('complete',j['id'],'seconds',round(d['seconds'],1),flush=True)

def report():
    manifest=read(RUN/'manifest.json');coverage=[];scores=[];rows=[]
    for j in manifest['jobs']:
        p=output(j);coverage.append(dict(id=j['id'],action=j['action'],status='complete' if p.exists() else 'pending'))
        if p.exists():
            d=read(p);scores.extend(d.get('scores',[]));rows.extend(d.get('contrasts',[]))
    done=all(c['status']=='complete' for c in coverage);OUT.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    if len(df):
        df['holm_p']=np.nan
        if done:
            for _,ix in df.groupby('family').groups.items():df.loc[ix,'holm_p']=holm(df.loc[ix,'permutation_p'].fillna(1).to_numpy())
        df['inference_status']='complete_registered_families' if done else 'pending_complete_panel'
    tables={'Coverage':pd.DataFrame(coverage),'Scores':pd.DataFrame(scores),'Comparisons':df,'Protocol':pd.DataFrame([{'item':k,'description':str(v)} for k,v in manifest['protocol'].items()])}
    for name,table in tables.items():table.to_csv(OUT/(name+'.csv'),index=False)
    tmp=OUT/'SD_prediction_controls.tmp.xlsx'
    with pd.ExcelWriter(tmp,engine='xlsxwriter') as w:
        for name,table in tables.items():table.to_excel(w,sheet_name=name,index=False)
    tmp.replace(OUT/'SD_prediction_controls.xlsx')
    summary=dict(total=len(coverage),completed=sum(c['status']=='complete' for c in coverage),comparisons=len(rows),all_complete=done,automatic_shutdown=False,updated_utc=utc())
    save(RUN/'summary.json',summary)
    if done:save(RUN/'complete.json',summary)
    return summary

def run(workers):
    manifest=register();LOG.mkdir(parents=True,exist_ok=True)
    lock=open(RUN/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};failed=set();last=0
    ordered=sorted(manifest['jobs'],key=lambda j:(j['action']!='features',j['cohort'] in ('adni','nacc'),j['action']!='longitudinal',j['id']))
    while True:
        for key,(proc,fh) in list(active.items()):
            if proc.poll() is None:continue
            fh.close();del active[key]
            if proc.returncode:failed.add(key)
            print('finished',key,proc.returncode,flush=True)
        done={j['id'] for j in ordered if output(j).exists()}
        for j in ordered:
            if len(active)>=workers:break
            if j['id'] in done|set(active)|failed or not set(j.get('dependencies',[]))<=done:continue
            fh=open(LOG/(j['id'].replace('/','_')+'.log'),'a')
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'work','--job',j['id']],stdout=fh,stderr=subprocess.STDOUT)
            active[j['id']]=(proc,fh);print('started',j['id'],proc.pid,flush=True)
        save(RUN/'status.json',dict(total=len(ordered),completed=len(done),running=list(active),failed=sorted(failed),pending=len(ordered)-len(done)-len(active)-len(failed),workers=workers,automatic_shutdown=False,updated_utc=utc()))
        if time.time()-last>120:report();last=time.time()
        if not active:break
        time.sleep(5)
    print(report(),flush=True)
    if failed:raise SystemExit('Failed jobs: '+str(sorted(failed)))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','work','run','report']);p.add_argument('--job');p.add_argument('--workers',type=int,default=32);a=p.parse_args()
    if a.action=='prepare':print({'jobs':len(register()['jobs'])})
    elif a.action=='work':execute(next(j for j in read(RUN/'manifest.json')['jobs'] if j['id']==a.job))
    elif a.action=='run':run(a.workers)
    else:print(report())
if __name__=='__main__':main()
