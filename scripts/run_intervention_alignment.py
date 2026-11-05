#!/usr/bin/env python3
"""Bounded frozen CORAL sensitivity for paired-visit intervention tasks.

Source-only PCA; baseline-only target training calibration shared across visits.
Existing primary predictions and all parent analyses remain immutable.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import embedding_harmonization as harmonization
import run_intervention_readouts as readouts
import intervention_prepost as ip
import paper_completion as pc
from fm_matrix.data import save,sha
from fm_readout_v4.data import metadata,DATA
from paper_paired_stats import holm

RUN=ROOT/'runs/intervention_alignment_v1'
OUT=ROOT/'results/generated/intervention_alignment_v1'
MODELS=readouts.MODELS
SOURCES=('adni','nacc')
COHORTS=ip.COHORTS
METHODS=('source_pca_unaligned','coral')


def read(p):return json.loads(Path(p).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def folder(model):return readouts.sensitivity.feature_folder(('none',model,'frozen','none',0))
def source_dir(model,source):return RUN/'sources'/model/source


def jobs():
    out=[]
    for model in MODELS:
        for source in SOURCES:
            root=source_dir(model,source)
            out.append(dict(id='source/'+model+'/'+source,action='source',model=model,source=source,
                            output=str(root/'complete.json')))
            for co in COHORTS:
                out.append(dict(id='evaluate/'+model+'/'+source+'/'+co,action='evaluate',model=model,source=source,
                                cohort=co,dependency=str(root/'complete.json'),output=str(RUN/'results'/model/source/(co+'.json'))))
    return out


def prepare():
    parent=read(ip.RUN/'inputs.json');files={}
    def record(p):
        p=Path(p)
        if str(p) not in files:files[str(p)]=sha(p)
        return files[str(p)]
    record(ip.RUN/'inputs.json')
    datasets=set(SOURCES)
    for plan in parent['plans'].values():datasets.update((plan['pre_cohort'],plan['post_cohort']))
    for co in datasets:record(DATA/(co+'_domains.csv' if co.startswith('hrv') else co+'.csv'))
    for model in MODELS:
        stamp=read(folder(model)/'complete.json');assert stamp['status']=='complete'
        record(folder(model)/'complete.json')
        for co in datasets:
            p=folder(model)/(co+'.npy');h=record(p)
            if model in MODELS[:4]:pc.verify_input_files([p])
            else:assert stamp['feature_sha256'][co]==h
            x=np.load(p,mmap_mode='r');assert len(x)==len(metadata(co)) and np.isfinite(x).all()
        for co in COHORTS:
            for name in (model,'sf100_fc'):
                p=readouts.sensitivity.frozen_path(name,co);record(p)
                readouts.sensitivity.validate_rows(read(p)['rows'],co)
            record(ip.RUN/'jobs/none_brainlm13_frozen_none_0'/(co+'.json'))
            plan=parent['plans'][co]
            for t in plan['tasks'].values():
                for visit in ('pre','post'):
                    m=metadata(plan[visit+'_cohort']).iloc[t[visit+'_indices']]
                    assert m.scan.tolist()==t[visit+'_scans']
                    assert m.subject.astype(str).tolist()==list(map(str,t['subjects']))
                for f in t['splits']:
                    tr,te,it,iv=[set(f[k]) for k in ('train','test','inner_train','validation')]
                    assert not tr&te and not it&iv and it|iv==tr
    codes=[Path(__file__),Path(ip.__file__),Path(harmonization.__file__),Path(readouts.__file__),
           ROOT/'scripts/intervention_prepost_inference.py',ROOT/'src/benchmark_evaluation.py',ROOT/'src/paper_paired_stats.py']
    protocol=dict(models=list(MODELS),sources=list(SOURCES),cohorts=list(COHORTS),states='Frozen only; original primary scan-level readouts.',
        tasks=['MEM_change','EF_change','arm'],inputs=list(ip.MODES),
        unavailable='HRV-ER memory change excluded: no paired memory assessments.',
        source_pool='All eligible source-cohort scans under existing primary_mask, without any cognitive/diagnostic outcome requirement. One task-independent pool per source/model. Source outcomes unused; unlike the earlier cognitive-axis panel, no source cognitive head is fitted.',
        source_projection='Float64 source-only StandardScaler and randomized PCA up to100 components, random_state0. Covariance reference uses participant means of source coordinates, so repeated source visits receive equal participant weight in CORAL statistics.',
        target_calibration='Baseline embeddings from inner-training or outer-training participants only. No target post scans, validation/test participants or outcome values estimate CORAL parameters. The same fitted affine transform is applied to both visits.',
        transformation='Ledoit-Wolf source/target covariance, inherited eigenvalue floor. First transform pre/post in common source coordinates, then compute post-minus-pre; baseline+change concatenated before one target-fold PCA.',
        fitting='Exact paired-visit samples, classes, clinical variables, outer/inner partitions and split seeds0/1/2. Target training-only standardization/PCA<=5 and existing ridge/logistic grid. Clinical-plus-imaging only for cognitive tasks.',
        controls='Unaligned source-PCA control shares source projection and downstream procedure with CORAL. Archived raw-primary fixed5 and SF100 fixed5 predictions retained separately; never treat the source-projected control as numerically identical to archived primary.',
        comparisons=['coral_vs_source_pca_unaligned','coral_vs_archived_primary','coral_vs_sf100','coral_added_to_clinical'],
        families='Contrast type x endpoint x imaging-only/clinical-plus-imaging, pooling nine configurations, both source cohorts, five target cohorts and all relevant input modes. New exploratory families; parent corrections unchanged. Nonestimable planned contrasts use P=1.',
        inference='5000 participant bootstraps,20000 two-sided conditional prediction swaps; entire multiclass probability vectors swapped; seed-averaged predictions scored. No absolute-null fitting.',
        moments='Not rerun: positive coordinate-wise mean/SD transforms cancel under the downstream training-fitted standardization; invariance is a pipeline property.',
        scope='Paired-visit cognition change and arm discrimination only; no new encoder, readout, adaptation, LOSO or physiology factorial.',
        encoder_training=False,encoder_inference=False,automatic_shutdown=False,manuscript_modified=False)
    manifest=dict(jobs=jobs(),protocol=protocol,files_sha256=files,code_sha256={str(p):sha(p) for p in codes})
    p=RUN/'manifest.json'
    if p.exists() and read(p)!=manifest:raise ValueError('Registered protocol changed; create a new version')
    save(p,manifest)
    return dict(source_jobs=18,evaluation_jobs=90,input_files=len(files))


def verify():
    m=read(RUN/'manifest.json')
    for p,h in m['code_sha256'].items():
        if sha(p)!=h:raise ValueError('Changed code '+p)
    return m


def check(p,manifest):
    if sha(p)!=manifest['files_sha256'][str(p)]:raise ValueError('Changed input '+str(p))


def source_fit(model,source):
    root=source_dir(model,source)
    if (root/'complete.json').exists():return
    manifest=verify();p=folder(model)/(source+'.npy');check(p,manifest)
    m=metadata(source);keep=pc.primary_mask(m,source)
    x=np.asarray(np.load(p,mmap_mode='r')[keep],float)
    sc=StandardScaler().fit(x);a=sc.transform(x)
    proj=PCA(n_components=min(100,len(x)-1,x.shape[1]),svd_solver='randomized',random_state=0).fit(a)
    zs=proj.transform(a);ref=harmonization.subject_mean(zs,m.loc[keep,'subject'].astype(str).to_numpy())
    root.mkdir(parents=True,exist_ok=True);outputs={};scans={}
    arrays={'reference':ref,'scaler_mean':sc.mean_,'scaler_scale':sc.scale_,'pca_mean':proj.mean_,'components':proj.components_}
    plan=read(ip.RUN/'inputs.json')['plans']
    for ds in sorted({v[k] for v in plan.values() for k in ('pre_cohort','post_cohort')}):
        p=folder(model)/(ds+'.npy');check(p,manifest)
        arrays[ds]=proj.transform(sc.transform(np.asarray(np.load(p,mmap_mode='r'),float)))
        scans[ds]=metadata(ds).scan.tolist()
    for name,arr in arrays.items():
        p=root/(name+'.npy');temp=p.with_suffix('.tmp.npy');np.save(temp,arr);temp.replace(p);outputs[name]=sha(p)
    save(root/'complete.json',dict(status='complete',model=model,source=source,source_scans=m.loc[keep,'scan'].tolist(),
        source_subjects=m.loc[keep,'subject'].astype(str).tolist(),source_participants=len(ref),components=proj.n_components_,
        target_scan_ids=scans,files_sha256=outputs,manifest_sha256=sha(RUN/'manifest.json')))


def aligned_inputs(pre,post,train,reference,method):
    """Baseline training calibration only; share one transform across visits."""
    if method=='source_pca_unaligned':a,b=pre.copy(),post.copy()
    elif method=='coral':
        align=harmonization.Alignment(reference).fit(pre[train],'coral')
        a,b=align.transform(pre),align.transform(post)
    else:raise ValueError(method)
    delta=b-a
    return {'baseline':a,'change':delta,'baseline_plus_change':np.c_[a,delta]}


def fit_task(pre,post,reference,t):
    y=np.asarray(t['y']);classes=t['classes'];n=len(y)
    modes=list(ip.MODES)+([] if classes else [m+'+clinical' for m in ip.MODES])
    shape=(3,n,len(classes)) if classes else (3,n)
    predictions={method:{mode:np.full(shape,np.nan) for mode in modes} for method in METHODS}
    audit=[];selected=[]
    for split in t['splits']:
        seed=split['seed'];tr,te,it,iv=[np.asarray(split[k],int) for k in ('train','test','inner_train','validation')]
        if set(tr)&set(te) or set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise ValueError('Invalid participant split')
        if not classes:oc=ip.clinical_transform(t,tr,te);ic=ip.clinical_transform(t,it,iv)
        for method in METHODS:
            outer=aligned_inputs(pre,post,tr,reference,method)
            inner=aligned_inputs(pre,post,it,reference,method)
            for mode in ip.MODES:
                op=ip.project(outer[mode],tr,te);inp=ip.project(inner[mode],it,iv)
                pp,c=ip.tuned(y,tr,te,it,iv,op,inp,classes)
                predictions[method][mode][seed,te]=pp
                selected.append(dict(seed=seed,fold=split['fold'],method=method,mode=mode,C=c))
                if not classes:
                    op=tuple(np.c_[a,b] for a,b in zip(oc,op));inp=tuple(np.c_[a,b] for a,b in zip(ic,inp))
                    pp,c=ip.tuned(y,tr,te,it,iv,op,inp,classes)
                    predictions[method][mode+'+clinical'][seed,te]=pp
                    selected.append(dict(seed=seed,fold=split['fold'],method=method,mode=mode+'+clinical',C=c))
        audit.append(dict(seed=seed,fold=split['fold'],train=tr.tolist(),test=te.tolist(),inner_train=it.tolist(),validation=iv.tolist(),calibration_visit='pre_only'))
    for methods in predictions.values():
        for p in methods.values():
            if not np.isfinite(p).all():raise ValueError('Incomplete OOF predictions')
            if classes:np.testing.assert_allclose(p.sum(2),1,atol=1e-8)
    return dict(predictions={k:{m:p.mean(0) for m,p in v.items()} for k,v in predictions.items()},
        predictions_by_seed=predictions,split_audit=audit,selected_C=selected)


def evaluate(model,source,co):
    dest=RUN/'results'/model/source/(co+'.json')
    if dest.exists():return
    manifest=verify();check(ip.RUN/'inputs.json',manifest)
    plan=read(ip.RUN/'inputs.json')['plans'][co];root=source_dir(model,source);stamp=read(root/'complete.json')
    assert stamp['manifest_sha256']==sha(RUN/'manifest.json')
    arrays={}
    for ds in {'reference',plan['pre_cohort'],plan['post_cohort']}:
        p=root/(ds+'.npy');assert sha(p)==stamp['files_sha256'][ds];arrays[ds]=np.load(p)
        if ds!='reference':assert stamp['target_scan_ids'][ds]==metadata(ds).scan.tolist()
    paths=[readouts.sensitivity.frozen_path(m,co) for m in (model,'sf100_fc')]
    clinical_path=ip.RUN/'jobs/none_brainlm13_frozen_none_0'/(co+'.json')
    for p in [*paths,clinical_path]:check(p,manifest)
    primary,sf=[readouts.original_rows(p) for p in paths]
    clinical={r['task']:r for r in read(clinical_path)['rows'] if r['mode']=='clinical'}
    scores=[];contrasts=[];fit_paths=[]
    for task,t in plan['tasks'].items():
        fitpath=RUN/'fits'/model/source/co/(task+'.json')
        if fitpath.exists():fitted=read(fitpath);assert fitted['manifest_sha256']==sha(RUN/'manifest.json')
        else:
            pre=arrays[plan['pre_cohort']][t['pre_indices']];post=arrays[plan['post_cohort']][t['post_indices']]
            fitted=fit_task(pre,post,arrays['reference'],t)
            fitted.update(status='complete',task=task,subjects=t['subjects'],y=t['y'],classes=t['classes'],manifest_sha256=sha(RUN/'manifest.json'))
            save(fitpath,fitted)
        fit_paths.append(str(fitpath));y=np.asarray(t['y']);classes=t['classes']
        for mode in fitted['predictions']['coral']:
            key=(task,mode);native=primary[key];sfr=sf[key]
            for row in (native,sfr):readouts.validate_row(row,plan)
            pp={m:np.asarray(fitted['predictions'][m][mode]) for m in METHODS}
            pp.update(archived_primary=np.asarray(native['predictions']['fixed5']),sf100_fc=np.asarray(sfr['predictions']['fixed5']))
            setting='clinical_plus_imaging' if '+clinical' in mode else 'imaging_only'
            info=dict(model=model,source=source,cohort=co,task=task,mode=mode,setting=setting,objective='frozen')
            refs=[('coral_vs_source_pca_unaligned','source_pca_unaligned'),('coral_vs_archived_primary','archived_primary'),('coral_vs_sf100','sf100_fc')]
            if '+clinical' in mode:
                readouts.validate_row(clinical[task],plan);pp['clinical']=np.asarray(clinical[task]['prediction'])
                refs.append(('coral_added_to_clinical','clinical'))
            for method,pred in pp.items():scores.append(dict(**info,method=method,n_subjects=len(y),**ip.metrics(y,pred,classes)))
            for contrast,reference in refs:
                part=RUN/'parts'/model/source/co/(task+'_'+mode+'_'+contrast+'.json')
                if part.exists():stat=read(part)
                else:
                    stat=readouts.compare(y,pp[reference],pp['coral'],classes,dict(**info,contrast=contrast,reference=reference,candidate='coral',family='|'.join([contrast,task,setting])))
                    save(part,stat)
                contrasts.append(stat)
    save(dest,dict(status='complete',scores=scores,contrasts=contrasts,fit_paths=fit_paths,
        manifest_sha256=sha(RUN/'manifest.json'),source_fit_sha256=sha(root/'complete.json')))


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
            b['inference_status']='final_nine_configuration_two_source_family'
    OUT.mkdir(parents=True,exist_ok=True)
    for name,d in [('Scores',a),('Comparisons',b),('Coverage',cov)]:d.to_csv(OUT/(name+'.csv'),index=False)
    tmp=OUT/'Intervention_CORAL.tmp.xlsx'
    with pd.ExcelWriter(tmp,engine='xlsxwriter') as w:
        for name,d in [('Scores',a),('Comparisons',b),('Coverage',cov)]:d.to_excel(w,sheet_name=name,index=False)
        pd.DataFrame([dict(item=k,description=str(v)) for k,v in m['protocol'].items()]).to_excel(w,sheet_name='Protocol',index=False)
    tmp.replace(OUT/'Intervention_CORAL.xlsx')
    summary=dict(total=len(coverage),completed=sum(c['status']=='complete' for c in coverage),
        evaluations_completed=sum(c['stage']=='evaluate' and c['status']=='complete' for c in coverage),evaluations_total=90,
        all_complete=done,comparisons=len(b),automatic_shutdown=False,updated_utc=utc())
    save(RUN/'summary.json',summary)
    if done:save(RUN/'complete.json',summary)
    return summary


def run(workers):
    m=verify();lock=open(RUN/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=ROOT/'logs/intervention_alignment_v1';logs.mkdir(parents=True,exist_ok=True)
    ordered=sorted(m['jobs'],key=lambda j:(j['action']!='evaluate',j['model']!='semantoks',j.get('cohort')!='psopt',j['id']))
    active={};failed=set();last_report=0
    while True:
        for key,(proc,fh) in list(active.items()):
            if proc.poll() is None:continue
            fh.close();del active[key]
            if proc.returncode:failed.add(key)
            print('finished',key,proc.returncode,flush=True)
        done={j['id'] for j in ordered if Path(j['output']).exists()}
        for j in ordered:
            if len(active)>=workers:break
            if j['id'] in done|set(active)|failed:continue
            if j.get('dependency') and not Path(j['dependency']).exists():continue
            cmd=[sys.executable,str(Path(__file__).resolve()),j['action'],'--model',j['model'],'--source',j['source']]
            if 'cohort' in j:cmd+=['--cohort',j['cohort']]
            fh=open(logs/(j['id'].replace('/','_')+'.log'),'a');active[j['id']]=(subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT),fh)
            print('started',j['id'],flush=True)
        save(RUN/'status.json',dict(total=len(ordered),completed=len(done),running=list(active),failed=sorted(failed),
            pending=len(ordered)-len(done)-len(active)-len(failed),workers=workers,automatic_shutdown=False,updated_utc=utc()))
        if time.time()-last_report>180:report();last_report=time.time()
        if len(done)==len(ordered) or (not active and failed):break
        time.sleep(5)
    print(report(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','source','evaluate','run','report'])
    p.add_argument('--model',choices=MODELS);p.add_argument('--source',choices=SOURCES);p.add_argument('--cohort',choices=COHORTS);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='source':source_fit(a.model,a.source)
    elif a.action=='evaluate':evaluate(a.model,a.source,a.cohort)
    elif a.action=='run':run(a.workers)
    else:print(report())


if __name__=='__main__':main()
