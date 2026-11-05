#!/usr/bin/env python3
"""Add paired-visit fixed80-native-frame sensitivity, reusing baseline caches.

Frozen encoders only; same original paired participants, primary readouts and probes.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import torch
import intervention_prepost as ip
import run_intervention_readouts as readouts
import paper_robustness as robustness
from fm_matrix.data import save,sha
from fm_matrix.train import amp
from fm_readout_v4.data import metadata,dataset,DATA
from paper_paired_stats import holm

RUN=ROOT/'runs/intervention_input_length_v1'
OUT=ROOT/'results/generated/intervention_input_length_v1'
MODELS=readouts.MODELS
COHORTS=ip.COHORTS


def read(p):return json.loads(Path(p).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def cache_root(model):
    if model in MODELS[:4]:return ROOT/'runs/paper_robustness_v1/short_features'/model
    if model in ('semantoks','fmrilm'):return ROOT/'runs/new_fm_v1/sensitivity/short_features'/model
    if model.startswith('fmrilm_'):
        return ROOT/'runs/fmrilm_variants_v1'/('mae' if model=='fmrilm_mae' else 'vq-domain')/'sensitivity/short_features/fmrilm'
    return ROOT/'runs/brainharmonix_f_v1/sensitivity/short_features/brainharmonix_f'


def checkpoint(model):
    if model in MODELS[:4]:
        return Path(next(r['checkpoint'] for r in read(robustness.RUN/'pretraining_audit.json')['models'] if r['model']==model))
    if model=='brainharmonix_f':return ROOT/'checkpoints/pretrained/brainharmonix_f/model.pth'
    base=ROOT/'checkpoints/pretrained/new_fm_v1'
    if model=='semantoks':return base/'semantoks/brainsemantoks_ckpt_epoch_100.pth'
    variant={'fmrilm':'vq-contrastive','fmrilm_mae':'mae','fmrilm_vq_domain':'vq-domain'}[model]
    return base/'fmrilm'/variant/'stage2-pretrain-Qwen3-0.6B.pt'


def required(plan,visit):
    return sorted({i for t in plan['tasks'].values() for i in t[visit+'_indices']})


def jobs():
    out=[]
    for model in MODELS:
        out.append(dict(id='extract/'+model,action='extract',model=model,output=str(RUN/'features'/model/'complete.json')))
        for co in COHORTS:
            out.append(dict(id='evaluate/'+model+'/'+co,action='evaluate',model=model,cohort=co,
                dependency=str(RUN/'features'/model/(co+'_pair.json')),output=str(RUN/'results'/model/(co+'.json'))))
    return out


def prepare():
    parent=read(ip.RUN/'inputs.json');files={};counts=[]
    def record(p):
        p=Path(p)
        if str(p) not in files:files[str(p)]=sha(p)
        return files[str(p)]
    record(ip.RUN/'inputs.json');record(robustness.RUN/'pretraining_audit.json')
    record(ROOT/'runs/new_fm_v1/intake.json');record(ROOT/'data/derived/new_fm_v1/roi_mapping.json')
    for co in COHORTS:
        plan=parent['plans'][co]
        for visit in ('pre','post'):
            ds=plan[visit+'_cohort'];m=metadata(ds);ix=required(plan,visit)
            record(DATA/(ds+'_domains.csv' if ds.startswith('hrv') else ds+'.csv'))
            if not (m.iloc[ix].native_frames>=80).all():raise ValueError('Existing paired sample contains a scan shorter than80 native frames')
            for t in plan['tasks'].values():
                assert m.iloc[t[visit+'_indices']].scan.tolist()==t[visit+'_scans']
                assert m.iloc[t[visit+'_indices']].subject.astype(str).tolist()==list(map(str,t['subjects']))
            for atlas in ('a424','sch450'):record(DATA/f'{ds}_{atlas}.pt')
        for model in MODELS:
            root=cache_root(model);ds=plan['pre_cohort'];stamp=read(root/(ds+'.json'));record(root/(ds+'.json'))
            assert stamp['status']=='complete'
            assert stamp.get('observed_frames',stamp.get('observed_native_frames'))==80
            assert record(root/(ds+'.npy'))==stamp['feature_sha256']
            required_ids=metadata(ds).iloc[required(plan,'pre')].scan.tolist()
            if not set(required_ids)<=set(stamp['scan_ids']):raise ValueError('Baseline cache misses requested participants')
            for name in (model,'sf100_fc'):
                p=readouts.sensitivity.frozen_path(name,co);record(p);readouts.sensitivity.validate_rows(read(p)['rows'],co)
            record(ip.RUN/'jobs/none_brainlm13_frozen_none_0'/(co+'.json'))
            counts.append(dict(model=model,cohort=co,baseline_reused=len(required_ids),post_to_extract=len(required(plan,'post'))))
    for model in MODELS:
        record(checkpoint(model))
        record(readouts.sensitivity.feature_folder(('none',model,'frozen','none',0))/'complete.json')
        if model.startswith('fmrilm'):
            record(checkpoint(model).parent/'stage1-tokenizer.pt')
    for p in ('checkpoints/pretrained/new_fm_v1/fmrilm/qwen_config.json','checkpoints/pretrained/new_fm_v1/semantoks/config_used.yaml','checkpoints/pretrained/new_fm_v1/semantoks/network_mapping.npz'):
        record(ROOT/p)
    codes=[Path(__file__),Path(ip.__file__),Path(robustness.__file__),Path(readouts.__file__),
        ROOT/'scripts/intervention_prepost_inference.py',ROOT/'src/benchmark_evaluation.py',ROOT/'src/paper_paired_stats.py',
        ROOT/'src/new_fm/data.py',ROOT/'src/new_fm/models.py',ROOT/'src/brainharmonix/data.py',ROOT/'src/brainharmonix/models.py',
        ROOT/'src/fmrilm_variants/models.py',ROOT/'src/fm_matrix/models.py',ROOT/'src/fm_readout_v4/models.py']
    protocol=dict(models=list(MODELS),cohorts=list(COHORTS),state='Frozen primary readouts only',native_observed_frames=80,
        samples='Exact registered paired-visit subjects, outcomes/classes and splits. Both visits have>=80 native frames; no additional exclusions. HRV-ER MEM change unavailable.',
        inputs='Baseline, post-minus-pre, baseline+change; float64 subtraction before target training-only PCA<=5. Clinical-plus-imaging for cognitive change; same tuning and split seeds0/1/2.',
        construction='Original four models: first80 frames of canonical input, recenter, zero-pad to original length using the inherited shortened() recipe. Added models: first80 original native frames then their inherited resampling, normalization, atlas order, padding and masks. Actual seconds differ by TR; not an isolated TR/duration experiment.',
        reuse='Reuse audited baseline short caches. Reencode up to one original batch of baseline scans per cohort as a provenance/numerical check; require cosine>=.9999 and relativeRMSE<=.005. Only post predictions newly extracted for analysis.',
        comparison='Short versus archived original-input primary fixed5 predictions under identical participants/folds; additional SF100 and clinical-only references and short change-input versus short baseline comparisons.',
        inference='5000 participant bootstraps and20000 two-sided conditional prediction swaps, entire multiclass probability vectors together. No absolute null tests. Average predictions over split seeds before scoring.',
        families='Contrast type x endpoint x imaging-only/clinical-plus-imaging, pooling all nine configurations, cohorts and input modes. Final Holm only when complete; parent results remain unchanged.',
        encoder_training=False,automatic_shutdown=False,manuscript_modified=False)
    manifest=dict(jobs=jobs(),protocol=protocol,coverage=counts,files_sha256=files,code_sha256={str(p):sha(p) for p in codes})
    p=RUN/'manifest.json'
    if p.exists() and read(p)!=manifest:raise ValueError('Immutable protocol changed')
    save(p,manifest)
    return dict(extraction_jobs=9,evaluation_jobs=45,post_embeddings=sum(c['post_to_extract'] for c in counts))


def verify():
    m=read(RUN/'manifest.json')
    for p,h in m['code_sha256'].items():
        if sha(p)!=h:raise ValueError('Registered code changed '+p)
    return m


def check(p,m):
    if sha(p)!=m['files_sha256'][str(p)]:raise ValueError('Changed input '+str(p))


def encoder(model,device):
    if model in MODELS[:4]:
        from fm_readout_v4.models import Encoder
        return Encoder(model,device)
    if model=='brainharmonix_f':
        from brainharmonix.models import Encoder
        return Encoder(model,device)
    if model in ('fmrilm_mae','fmrilm_vq_domain'):
        from fmrilm_variants.models import VariantEncoder
        class Encoder(VariantEncoder):pass
        Encoder.variant='mae' if model=='fmrilm_mae' else 'vq-domain'
        return Encoder('fmrilm',device)
    from new_fm.models import Encoder
    return Encoder(model,device)


def short_inputs(model,ds,indices,manifest):
    if model in MODELS[:4]:
        check(DATA/f'{ds}_{robustness.atlas_for(model)}.pt',manifest)
        x,_=dataset(ds,model)
        return robustness.shortened(x[indices])
    from new_fm.data import source_rows,transform,DATA as NEWDATA
    if model=='brainharmonix_f':from brainharmonix.data import transform
    check(ROOT/'runs/new_fm_v1/intake.json',manifest)
    check(NEWDATA/'roi_mapping.json',manifest)
    source_hashes=read(ROOT/'runs/new_fm_v1/intake.json')['sources_sha256']
    order=np.asarray(read(NEWDATA/'roi_mapping.json')['permutation'])
    wanted=set(indices);values={};checked=set()
    for i,(x,tr,path) in enumerate(source_rows(ds)):
        if i not in wanted:continue
        if str(path) not in checked:
            if sha(Path(path))!=source_hashes[str(path)]:raise ValueError('Changed original timeseries '+str(path))
            checked.add(str(path))
        if x.shape[-1]<80:raise ValueError('Too few observed frames')
        name='fmrilm' if model.startswith('fmrilm') else model
        values[i]=transform(x[:,:80],tr,name,order)[0]
    if set(values)!=wanted:raise ValueError('Missing requested source rows')
    return torch.stack([values[i] for i in indices])


def reproduction(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid replay')
    cosine=np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
    relative=float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(b*b)),1e-12))
    if cosine.min()<.9999 or relative>.005:raise ValueError('Baseline replay differs from registered short-input cache')
    return dict(min_cosine=float(cosine.min()),relative_rmse=relative,max_absolute_error=float(np.max(abs(a-b))))


def extract(model,device='cuda'):
    manifest=verify();check(checkpoint(model),manifest);check(ip.RUN/'inputs.json',manifest)
    if model.startswith('fmrilm'):check(checkpoint(model).parent/'stage1-tokenizer.pt',manifest)
    torch.set_num_threads(1);torch.manual_seed(0)
    net=encoder(model,device);net.eval();net.requires_grad_(False)
    assert Path(net.checkpoint_path)==checkpoint(model)
    batch=2 if model=='brainlm13' else 8 if model in MODELS[:4] else 4
    root=RUN/'features'/model;root.mkdir(parents=True,exist_ok=True)
    parent=read(ip.RUN/'inputs.json')
    for co in COHORTS:
        mark=root/(co+'_pair.json')
        if mark.exists():continue
        plan=parent['plans'][co];ca,cb=plan['pre_cohort'],plan['post_cohort']
        baseline=cache_root(model)/(ca+'.npy');side=baseline.with_suffix('.json')
        check(baseline,manifest);check(side,manifest)
        stamp=read(side);cached=np.load(baseline,mmap_mode='r');lookup={s:i for i,s in enumerate(stamp['scan_ids'])}
        pre_ix=required(plan,'pre');post_ix=required(plan,'post');ma,mb=metadata(ca),metadata(cb)
        cache_ix=[lookup[s] for s in ma.iloc[pre_ix].scan.tolist()]
        # Reproduce the original first batch, rather than changing baseline batch composition.
        replay_ids=stamp['scan_ids'][:batch];replay_ix=ma.set_index('scan').index.get_indexer(replay_ids).tolist()
        if min(replay_ix)<0:raise ValueError('Replay scan missing')
        xx=short_inputs(model,ca,replay_ix,manifest)
        with torch.inference_mode(),amp(device):replayed=net(xx.to(device)).float().cpu().numpy()
        audit=reproduction(replayed,cached[:len(replay_ix)])
        del xx
        xx=short_inputs(model,cb,post_ix,manifest);parts=root/(co+'_chunks');parts.mkdir(exist_ok=True)
        values=[]
        for start in range(0,len(xx),batch):
            p=parts/(str(start)+'.npy')
            if p.exists():z=np.load(p)
            else:
                with torch.inference_mode(),amp(device):z=net(xx[start:start+batch].to(device)).float().cpu().numpy()
                if not np.isfinite(z).all():raise ValueError('Nonfinite post embedding')
                tmp=p.with_suffix('.tmp.npy');np.save(tmp,z);tmp.replace(p)
            values.append(z)
        post=np.concatenate(values);del xx
        arrays={ds:np.full((len(metadata(ds)),cached.shape[1]),np.nan,dtype=np.float32) for ds in {ca,cb}}
        arrays[ca][pre_ix]=cached[cache_ix];arrays[cb][post_ix]=post
        hashes={}
        for ds,x in arrays.items():
            p=root/(ds+'.npy');tmp=p.with_suffix('.tmp.npy');np.save(tmp,x);tmp.replace(p);hashes[ds]=sha(p)
        save(mark,dict(status='complete',model=model,cohort=co,pre_indices=pre_ix,post_indices=post_ix,
            pre_scans=ma.iloc[pre_ix].scan.tolist(),post_scans=mb.iloc[post_ix].scan.tolist(),baseline_reproduction=audit,
            checkpoint_sha256=manifest['files_sha256'][str(checkpoint(model))],files_sha256=hashes,
            unused_rows='NaN; intentionally outside registered paired tasks',manifest_sha256=sha(RUN/'manifest.json')))
        print('features complete',model,co,'post',len(post_ix),'replay',audit,flush=True)
    save(root/'complete.json',dict(status='complete',model=model,manifest_sha256=sha(RUN/'manifest.json')))


def evaluate(model,co):
    dest=RUN/'results'/model/(co+'.json')
    if dest.exists():return
    manifest=verify();check(ip.RUN/'inputs.json',manifest);parent=read(ip.RUN/'inputs.json');plan=parent['plans'][co]
    root=RUN/'features'/model;stamp=read(root/(co+'_pair.json'));assert stamp['manifest_sha256']==sha(RUN/'manifest.json')
    state=('none',model,'frozen','short80',0);fitroot=RUN/'fits'/model/co;d=copy.deepcopy(parent);d['states']=[state];d['files_sha256']={}
    for ds in {plan['pre_cohort'],plan['post_cohort']}:
        p=root/(ds+'.npy');h=sha(p);assert h==stamp['files_sha256'][ds];d['files_sha256'][str(p)]=h
        for visit in ('pre','post'):
            if ds==plan[visit+'_cohort']:
                assert np.isfinite(np.load(p,mmap_mode='r')[required(plan,visit)]).all()
    d['signature']=readouts.frozen.seed(json.dumps(dict(parent=parent['signature'],files=d['files_sha256'],model=model),sort_keys=True))
    save(fitroot/'inputs.json',d)
    with patch.object(ip,'RUN',fitroot),patch.object(ip,'feature_path',lambda s,ds:root/(ds+'.npy')):
        fit=ip.run_job((state,co))
    rr={(r['task'],r['mode']):r for r in read(fit)['rows']}
    pp=readouts.sensitivity.frozen_path(model,co);sp=readouts.sensitivity.frozen_path('sf100_fc',co)
    cp=ip.RUN/'jobs/none_brainlm13_frozen_none_0'/(co+'.json')
    for p in (pp,sp,cp):check(p,manifest)
    primary,sf=readouts.original_rows(pp),readouts.original_rows(sp)
    clinical={r['task']:r for r in read(cp)['rows'] if r['mode']=='clinical'}
    if set(rr)-{k for k in rr if k[1] in ('clinical','null')}!=set(primary):raise ValueError('Incomplete task coverage')
    scores=[];contrasts=[]
    for key in sorted(primary):
        row=rr[key];native=primary[key];fc=sf[key]
        for r in (row,native,fc):readouts.validate_row(r,plan)
        task,mode=key;classes=row['classes'];y=np.asarray(row['y']);b=np.asarray(row['prediction'])
        setting='clinical_plus_imaging' if '+clinical' in mode else 'imaging_only'
        info=dict(model=model,cohort=co,task=task,mode=mode,setting=setting,objective='frozen')
        preds={'short80':b,'primary':np.asarray(native['predictions']['fixed5']),'sf100_fc':np.asarray(fc['predictions']['fixed5'])}
        refs=[('short_vs_primary','primary'),('short_vs_sf100','sf100_fc')]
        if '+clinical' in mode:
            preds['clinical']=np.asarray(clinical[task]['prediction'])
            np.testing.assert_allclose(rr[(task,'clinical')]['prediction'],preds['clinical'],atol=1e-12,rtol=1e-12)
            refs.append(('short_added_to_clinical','clinical'))
        if mode.startswith(('change','baseline_plus_change')):
            preds['short_baseline']=np.asarray(rr[(task,'baseline'+('+clinical' if '+clinical' in mode else ''))]['prediction'])
            refs.append(('short_input_vs_baseline','short_baseline'))
        for name,pred in preds.items():scores.append(dict(**info,input=name,n_subjects=len(y),**ip.metrics(y,pred,classes)))
        for contrast,reference in refs:
            part=RUN/'parts'/model/co/(task+'_'+mode+'_'+contrast+'.json')
            if part.exists():stat=read(part)
            else:
                stat=readouts.compare(y,preds[reference],b,classes,dict(**info,contrast=contrast,reference=reference,candidate='short80',family='|'.join([contrast,task,setting])))
                save(part,stat)
            contrasts.append(stat)
    save(dest,dict(status='complete',scores=scores,contrasts=contrasts,manifest_sha256=sha(RUN/'manifest.json'),fit=fit,fit_sha256=sha(Path(fit))))


def report():
    m=read(RUN/'manifest.json');coverage=[];scores=[];contrasts=[]
    for j in m['jobs']:
        p=Path(j['output']);coverage.append(dict(id=j['id'],stage=j['action'],status='complete' if p.exists() else 'pending'))
        if p.exists() and j['action']=='evaluate':
            d=read(p);assert d['manifest_sha256']==sha(RUN/'manifest.json');scores+=d['scores'];contrasts+=d['contrasts']
    done=all(c['status']=='complete' for c in coverage);a=pd.DataFrame(scores);b=pd.DataFrame(contrasts);cov=pd.DataFrame(coverage)
    if len(b):
        b['holm_p']=np.nan;b['inference_status']='pending_complete_panel'
        if done:
            for _,ix in b.groupby('family').groups.items():b.loc[ix,'holm_p']=holm(b.loc[ix,'permutation_p'].fillna(1).tolist())
            b['inference_status']='final_nine_configuration_family'
    OUT.mkdir(parents=True,exist_ok=True)
    for name,d in [('Scores',a),('Comparisons',b),('Coverage',cov)]:d.to_csv(OUT/(name+'.csv'),index=False)
    tmp=OUT/'Intervention_input_length.tmp.xlsx'
    with pd.ExcelWriter(tmp,engine='xlsxwriter') as w:
        for name,d in [('Scores',a),('Comparisons',b),('Coverage',cov)]:d.to_excel(w,sheet_name=name,index=False)
        pd.DataFrame([dict(item=k,description=str(v)) for k,v in m['protocol'].items()]).to_excel(w,sheet_name='Protocol',index=False)
    tmp.replace(OUT/'Intervention_input_length.xlsx')
    result=dict(total=len(coverage),completed=sum(c['status']=='complete' for c in coverage),all_complete=done,
        extraction_complete=sum(c['stage']=='extract' and c['status']=='complete' for c in coverage),
        evaluation_complete=sum(c['stage']=='evaluate' and c['status']=='complete' for c in coverage),comparisons=len(b),automatic_shutdown=False,updated_utc=utc())
    save(RUN/'summary.json',result)
    if done:save(RUN/'complete.json',result)
    return result


def run(cpu_workers):
    m=verify();lock=open(RUN/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=ROOT/'logs/intervention_input_length_v1';logs.mkdir(parents=True,exist_ok=True)
    active={};failed=set();last_report=0
    ordered=sorted(m['jobs'],key=lambda j:(j['action']!='evaluate',MODELS.index(j['model'])))
    while True:
        for key,(proc,j,fh) in list(active.items()):
            if proc.poll() is None:continue
            fh.close();del active[key]
            if proc.returncode:failed.add(key)
            print('finished',key,proc.returncode,flush=True)
        done={j['id'] for j in ordered if Path(j['output']).exists()}
        for j in ordered:
            if j['id'] in done|set(active)|failed:continue
            if j.get('dependency') and not Path(j['dependency']).exists():continue
            if j['action']=='extract':
                if any(v[1]['action']=='extract' for v in active.values()):continue
                q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(q.strip().splitlines()[0])<10000:continue
            elif sum(v[1]['action']=='evaluate' for v in active.values())>=cpu_workers:continue
            cmd=[sys.executable,str(Path(__file__).resolve()),j['action'],'--model',j['model']]
            if 'cohort' in j:cmd+=['--cohort',j['cohort']]
            fh=open(logs/(j['id'].replace('/','_')+'.log'),'a');active[j['id']]=(subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT),j,fh)
            print('started',j['id'],flush=True)
        save(RUN/'status.json',dict(total=len(ordered),completed=len(done),running=list(active),failed=sorted(failed),pending=len(ordered)-len(done)-len(active)-len(failed),
            gpu_workers=1,cpu_workers=cpu_workers,automatic_shutdown=False,updated_utc=utc()))
        if time.time()-last_report>180:report();last_report=time.time()
        if len(done)==len(ordered) or (not active and failed):break
        time.sleep(5)
    print(report(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','extract','evaluate','run','report'])
    p.add_argument('--model',choices=MODELS);p.add_argument('--cohort',choices=COHORTS);p.add_argument('--cpu-workers',type=int,default=4)
    a=p.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='extract':extract(a.model)
    elif a.action=='evaluate':evaluate(a.model,a.cohort)
    elif a.action=='run':run(a.cpu_workers)
    else:print(report())


if __name__=='__main__':main()
