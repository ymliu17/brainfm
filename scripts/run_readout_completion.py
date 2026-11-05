#!/usr/bin/env python3
"""Bounded readout completion: matched FC controls, frozen transfer, audit tables.

No encoder training/inference, outcome-based readout selection, or shutdown.
All new artifacts are additive; existing scientific protocols remain immutable.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import run_connectivity_readouts as frozen
import run_readout_task_extension as extension
import run_adapted_readouts as adapted
from fm_matrix.data import save,sha
from benchmark_evaluation import score
from paper_paired_stats import paired,holm

RUN=ROOT/'runs/readout_completion_v1'
OUT=ROOT/'results/generated/readout_completion_v1'
ENDPOINTS={'adni':('MEM','EF','AGE','SEX','DX'),'nacc':('MEM','EF','AGE','SEX','DX','AD_DX')}


def read(p):return json.loads(Path(p).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def atlas(model):return 'a424' if model.startswith('brainlm') else 'sch400' if model=='brainharmonix_f' else 'sch450'
def reference_root(model):
    return ROOT/('runs/brainharmonix_f_v1/references' if model=='brainharmonix_f' else 'runs/benchmark_gaps_v1/references')
def window_root(model):
    return ROOT/('runs/brainharmonix_f_v1/supplements' if model=='brainharmonix_f' else 'runs/reviewer_completion_v1')
def axis_path(model):
    if model in frozen.MODELS[:4]:return ROOT/'runs/sample_alignment_v1/paper_completion_v1/canonical'/(model+'_frozen.json')
    if model=='brainharmonix_f':return ROOT/'runs/brainharmonix_f_v1/evaluation/brainharmonix_f_frozen/axes.json'
    if model.startswith('fmrilm_'):
        variant='mae' if model=='fmrilm_mae' else 'vq-domain'
        return ROOT/'runs/fmrilm_variants_v1'/variant/'evaluation/fmrilm_frozen/axes.json'
    return ROOT/'runs/new_fm_v1/evaluation'/(model+'_frozen')/'axes.json'
def local_path(model,co,ep):
    if ep in ('MEM','EF'):return frozen.RUN/'cognition'/model/(co+'_'+ep+'.json')
    return extension.RUN/'evaluation'/model/co/ep/'complete.json'


def jobs():
    out=[]
    for model in frozen.MODELS:
        for co,eps in ENDPOINTS.items():
            for ep in eps:
                out.append(dict(id=f'local/{model}/{co}/{ep}',action='local',model=model,cohort=co,task=ep,
                    dependency=str(local_path(model,co,ep)),output=f'local/{model}/{co}_{ep}.json'))
        for co in frozen.PAIRS:
            out.append(dict(id=f'identity/{model}/{co}',action='identity',model=model,cohort=co,
                dependency=str(frozen.RUN/'identity'/model/(co+'.json')),output=f'identity/{model}/{co}.json'))
        for source in ('adni','nacc'):
            for ep in ('MEM','EF'):
                out.append(dict(id=f'transfer/{model}/{source}/{ep}',action='transfer',model=model,cohort=source,task=ep,
                    dependency=str(axis_path(model)),output=f'transfer/{model}/{source}_{ep}.json'))
    return out


def register():
    paths={Path(__file__),Path(frozen.__file__),Path(extension.__file__),ROOT/'src/paper_paired_stats.py',
           ROOT/'src/paper_completion.py',frozen.RUN/'manifest.json',extension.RUN/'manifest.json',adapted.RUN/'manifest.json'}
    for model in frozen.MODELS:
        paths.add(axis_path(model))
        for co,eps in ENDPOINTS.items():
            paths.add(frozen.RUN/'features'/model/co/'complete.json')
            for ep in eps:
                paths.add(reference_root(model)/atlas(model)/co/(ep+'.json'))
                paths.add(extension.sf.RUN/'evaluation'/co/(ep+'.json'))
            if model not in frozen.MODELS[:4]:paths.add(window_root(model)/'input_comparison'/model/(co+'.json'))
        for co in frozen.PAIRS:
            paths.add(window_root(model)/'identity'/model/(co+'.json'))
    missing=[str(p) for p in paths if not p.exists()]
    if missing:raise ValueError('Missing reference artifacts: '+str(missing))
    d=dict(version=1,jobs=jobs(),input_sha256={str(p):sha(p) for p in sorted(paths)},automatic_shutdown=False,
        encoder_training=False,encoder_inference=False,
        local='Reuse fixed frozen-readout and FC/FC-Tucker OOF predictions only after exact sample, outcome and outer/inner split audit. All supported final-layer alternatives; primary is not selected/replaced.',
        references='Canonical A424200/Schaefer450160/Schaefer400160 atlas-matched FC and FC-Tucker; the latter windows are NOT matched to all new architectures. Additional actual-model-input-window FC for Semantoks, three fMRI-LM checkpoints and BrainHarmonix-F. Original four models already use their canonical windows. No invented same-window Tucker reference.',
        identification='Existing common participant pairs, same-window FC, native and fixed128 projections; top1 and I_diff. No new Tucker identification protocol or absolute-null tests.',
        transfer='Nine frozen configurations; ADNI->NACC and NACC->ADNI; MEM/EF. Primary versus final normalized embedding connectivity ONLY. Source-only StandardScaler, randomized PCA<=100 random_state0, ridge alpha10, matching existing source-axis implementation; no target labels in fitting or selection. Previously explored readouts make this a retrospective sensitivity, not an untouched confirmatory test.',
        inference='5000 participant-cluster bootstraps and20000 two-sided conditional swaps. Holm local by endpoint/reference kind across models/readouts/cohorts; identity by metric/mode across models/readouts/cohorts; transfer by endpoint across all nine models and both directions. Adjustment published only when every planned contrast in its panel is complete.',
        audit='Report source-validation checkpoint selection, zero-adapter/epoch0 retention, learning histories; alignment algebraic invariance and nine-model complete-context coverage. No new training or acquisition experiment.')
    p=RUN/'manifest.json'
    if p.exists() and read(p)!=d:raise ValueError('Registered completion protocol changed')
    if not p.exists():save(p,d)
    return d


def verify():
    d=read(RUN/'manifest.json')
    for p,h in d['input_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('Registered dependency changed: '+p)
    return d


def assert_aligned(a,b,check_splits=False):
    for k in ('scan_ids','subjects'):
        if list(a[k])!=list(b[k]):raise ValueError('Unmatched '+k)
    np.testing.assert_array_equal(a['y'],b['y'])
    if check_splits and a['split_audit']!=b['split_audit']:raise ValueError('Different evaluation partitions')


def local(model,co,ep):
    p=local_path(model,co,ep);d=read(p)
    template=extension.sf.RUN/'evaluation'/co/(ep+'.json');old=read(template)
    assert_aligned(d,old,True)
    refpath=reference_root(model)/atlas(model)/co/(ep+'.json');ref=read(refpath)
    assert_aligned(d,ref,True)
    if ref.get('parent_sha256')!=sha(template):raise ValueError('FC template provenance changed')
    refs={'atlas_FC':ref['predictions']['FC'],'atlas_FC_Tucker':ref['predictions']['FC-Tucker']}
    paths=[p,template,refpath]
    if model not in frozen.MODELS[:4]:
        wp=window_root(model)/'input_comparison'/model/(co+'.json');w=read(wp)
        w=next(r for r in w['scores'] if r['task']==ep)
        assert_aligned(d,w)
        # The window-FC implementation uses the registered template split audit.
        refs['input_window_FC']=w['predictions']['FC'];paths.append(wp)
    rows=[]
    for candidate in frozen.readout_names(model):
        if candidate=='native':continue
        for reference,pred in refs.items():
            cache=RUN/'parts/local'/model/co/ep/(candidate+'_'+reference+'.json')
            if cache.exists():row=read(cache)
            else:
                st=paired(d['y'],pred,d['predictions'][candidate],d['subjects'],metric='rho' if ep in ('MEM','EF','AGE') else 'auc',seed=frozen.seed(model+co+ep+candidate+reference))
                row=dict(model=model,cohort=co,task=ep,reference=reference,candidate=candidate,
                         family=f'local::{ep}::{reference}',**st);save(cache,row)
            rows.append(row)
    save(RUN/'local'/model/(co+'_'+ep+'.json'),dict(status='complete',rows=rows,
        sources_sha256={str(p):sha(p) for p in paths},scan_ids=d['scan_ids'],subjects=d['subjects']))


def identity(model,co):
    import readout_sensitivity as rs
    p=frozen.RUN/'identity'/model/(co+'.json');d=read(p)
    rpath=window_root(model)/'identity'/model/(co+'.json');r=read(rpath)
    if list(d['subjects'])!=list(r['subjects']):raise ValueError('Identification participants differ')
    refs={x['mode']:x['FC'] for x in r['scores']};rows=[]
    for mode in ('native','projected128'):
        for candidate in frozen.readout_names(model):
            if candidate=='native':continue
            for metric in ('top1','I_diff'):
                cache=RUN/'parts/identity'/model/co/(mode+'_'+candidate+'_'+metric+'.json')
                if cache.exists():row=read(cache)
                else:
                    st=rs.paired_identification(refs[mode]['per_subject_'+metric],d['scores'][mode][candidate]['per_subject_'+metric],frozen.seed(model+co+mode+candidate+metric+'inputFC'))
                    row=dict(model=model,cohort=co,task='identification',mode=mode,metric=metric,
                        reference='input_window_FC',candidate=candidate,family=f'identity::{mode}::{metric}',**st);save(cache,row)
                rows.append(row)
    save(RUN/'identity'/model/(co+'.json'),dict(status='complete',rows=rows,subjects=d['subjects'],
        sources_sha256={str(p):sha(p),str(rpath):sha(rpath)}))


def fitted_transfer(train,y,test):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    sc=StandardScaler().fit(train);a=sc.transform(train)
    pc=PCA(n_components=min(100,len(a)-1,a.shape[1]),svd_solver='randomized',random_state=0).fit(a)
    head=Ridge(alpha=10).fit(pc.transform(a),y)
    return head.predict(pc.transform(sc.transform(test)))


def transfer(model,source,ep):
    import paper_completion as pc
    from fm_readout_v4.data import metadata
    target='nacc' if source=='adni' else 'adni'
    parent=read(axis_path(model));old=next(r for r in parent.get('axes',parent.get('rows',[]))
        if r['source']==source and r['cohort']==target and r['task']==ep+'_axis_to_'+ep)
    m=metadata(source);tm=metadata(target)
    si=np.flatnonzero(pc.primary_mask(m,source)&m[ep].notna().to_numpy())
    ti=tm.set_index('scan').index.get_indexer(old['scan_ids'])
    if (ti<0).any() or tm.iloc[ti].subject.astype(str).tolist()!=old['subjects']:raise ValueError('Transfer target sample mismatch')
    np.testing.assert_array_equal(tm.iloc[ti][ep].to_numpy(),old['y'])
    pp={};files={str(axis_path(model)):sha(axis_path(model))}
    for name in ('native','connectivity'):
        cache=RUN/'parts/transfer'/model/source/ep/(name+'.json')
        arrays=[]
        for co in (source,target):
            stamp=read(frozen.RUN/'features'/model/co/'complete.json')
            if stamp['scan_ids']!=metadata(co).scan.tolist():raise ValueError('Feature metadata mismatch')
            p=frozen.RUN/'features'/model/co/(name+'.npy');h=sha(p)
            if h!=stamp['features_sha256'][name]:raise ValueError('Feature hash mismatch')
            files[str(p)]=h;arrays.append(np.load(p,mmap_mode='r'))
        if cache.exists():pp[name]=read(cache)['prediction']
        else:
            pp[name]=fitted_transfer(arrays[0][si],m.iloc[si][ep].to_numpy(),arrays[1][ti])
            save(cache,dict(prediction=pp[name],source_indices=si,target_scan_ids=old['scan_ids'],source_only_fit=True))
        print('transfer',model,source,target,ep,name,score(old['y'],pp[name],False),flush=True)
    stats=paired(old['y'],pp['native'],pp['connectivity'],old['subjects'],seed=frozen.seed(model+source+ep+'transfer'))
    row=dict(model=model,source=source,cohort=target,task=ep,reference='native',candidate='connectivity',family='transfer::'+ep,**stats)
    save(RUN/'transfer'/model/(source+'_'+ep+'.json'),dict(status='complete',rows=[row],
        source_scan_ids=m.iloc[si].scan.tolist(),source_subjects=m.iloc[si].subject.astype(str).tolist(),
        scan_ids=old['scan_ids'],subjects=old['subjects'],y=old['y'],predictions=pp,
        scores={k:score(old['y'],v,False) for k,v in pp.items()},sources_sha256=files,
        primary_archived_prediction_max_abs_difference=float(np.max(np.abs(np.asarray(pp['native'])-old['prediction']))),
        primary_archived_score=score(old['y'],old['prediction'],False)))


def audits():
    """Metadata only: external checkpoints, no extra training or validation fitting."""
    d=read(adapted.RUN/'manifest.json');rows=[];history=[];files={}
    for r in d['states']:
        s=r['state'];folder=Path(r['adapter_path']).parent
        p=folder/('result.json' if s[2]=='supervised' else 'complete.json');a=read(p);files[str(p)]=sha(p)
        hp=folder/'history.json';h=read(hp) if hp.exists() else []
        if hp.exists():files[str(hp)]=sha(hp)
        before=a.get('validation_frozen',a.get('base_validation'))
        after=a.get('validation_selected',a.get('best_validation'))
        if before is None or after is None:raise ValueError('Unrecognized validation schema: '+str(p))
        metric='Spearman' if s[2]=='supervised' else 'objective_specific_validation_loss'
        rows.append(dict(state=r['id'],model=s[1],source=s[0],objective=s[3] if s[2]=='supervised' else 'SSL',
            seed=s[4],best_epoch=r['best_epoch'],initial_selected=r['best_epoch']==0,zero_adapter=r['zero_adapter'],
            validation_before=before,validation_selected=after,validation_metric=metric,
            validation_improvement=after-before if s[2]=='supervised' else before-after,
            epochs_run=max([v['epoch'] for v in h],default=0),trainable_parameters=a.get('protocol',{}).get('trainable_parameters'),
            training_participants=len(set(a.get('train_subjects',[]))) or None,
            validation_participants=len(set(a.get('validation_subjects',a.get('val_subjects',[])))) or None,
            scope='External checkpoint selection; not independent within-source encoder evaluation'))
        history.extend(dict(state=r['id'],model=s[1],source=s[0],objective=s[2],**v) for v in h)
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['model','source','objective']).agg(checkpoints=('state','size'),initial_selected=('initial_selected','sum'),
        zero_adapter=('zero_adapter','sum'),best_epoch_median=('best_epoch','median'),validation_improvement_mean=('validation_improvement','mean')).reset_index()
    context=[]
    for co in frozen.COHORTS:
        p=ROOT/'runs/brainharmonix_f_v1/complete_context_nine/context_probes/brainharmonix_f_frozen'/(co+'.json')
        if not p.exists():continue
        a=read(p);files[str(p)]=sha(p)
        context.extend(dict(cohort=co,**r) for r in a.get('coverage',[]))
        if not a.get('coverage'):context.append(dict(cohort=co,status='estimable',reason='See complete-context prediction and comparison tables for endpoint counts'))
    notes=[dict(item='Alignment coverage',detail='Nine configurations; frozen and existing MEM/EF/SSL checkpoints; fixed-source cognitive axes and local follow-up/recall. No diagnosis, demographic, identification or physiology alignment extension.'),
        dict(item='Moment matching invariance',detail='In the registered local pipeline, subsequent training-fitted standardization cancels coordinate-wise positive mean/SD transforms. Identical predictions are a mathematical property, not empirical evidence that domain shift is absent.'),
        dict(item='Target calibration',detail='Calibration uses participant-mean embeddings without target outcomes and excludes evaluated participants. Local prediction refits calibration within training partitions.'),
        dict(item='Duration interpretation',detail='80 observed native frames, followed by model-specific input construction, also changes padding and sometimes normalization. This is not an isolated TR or physical-duration experiment.'),
        dict(item='Complete context',detail='Nine-configuration common full-context restriction is estimable in pSOPT only; other cohorts are explicitly not estimable. Existing original-model and eight-configuration subsets have different coverage.'),
        dict(item='Checkpoint interpretation',detail='External selection diagnostics cover162 existing adapters, not the separate fold-specific within-source encoders. Native SSL loss scales differ across model families and must not be ranked against each other.')]
    OUT.mkdir(parents=True,exist_ok=True)
    with pd.ExcelWriter(OUT/'Technical_coverage_and_checkpoint_audit.xlsx',engine='xlsxwriter') as w:
        for name,tab in [('Checkpoint_selection',frame),('Selection_summary',summary),('Training_histories',pd.DataFrame(history)),('Complete_context_coverage',pd.DataFrame(context)),('Interpretation',pd.DataFrame(notes))]:
            tab.to_excel(w,sheet_name=name,index=False);tab.to_csv(OUT/(name+'.csv'),index=False)
    save(RUN/'audit_complete.json',dict(status='complete',checkpoints=len(frame),initial_selected=int(frame.initial_selected.sum()),zero_adapter=int(frame.zero_adapter.sum()),sources_sha256=files))


def report():
    d=read(RUN/'manifest.json');coverage=[];rows=[];scores=[]
    for j in d['jobs']:
        p=RUN/j['output'];coverage.append(dict(id=j['id'],panel=j['action'],status='complete' if p.exists() else 'pending'))
        if p.exists():
            a=read(p);rows.extend(dict(panel=j['action'],**r) for r in a['rows'])
            if j['action']=='transfer':scores.extend(dict(model=j['model'],source=j['cohort'],task=j['task'],readout=k,score=v) for k,v in a['scores'].items())
    cov=pd.DataFrame(coverage);df=pd.DataFrame(rows);OUT.mkdir(parents=True,exist_ok=True)
    panel_complete=cov.groupby('panel').status.apply(lambda s:s.eq('complete').all()).to_dict()
    cov.to_csv(OUT/'Coverage.csv',index=False)
    if len(df):
        df['holm_p']=np.nan;df['inference_status']='pending_complete_panel'
        for _,ix in df.groupby('family').groups.items():
            if panel_complete[df.loc[ix[0],'panel']]:
                df.loc[ix,'holm_p']=holm(df.loc[ix,'permutation_p'].fillna(1).tolist());df.loc[ix,'inference_status']='final_registered_family'
        df.to_csv(OUT/'Comparisons.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'Transfer_scores.csv',index=False)
        tmp=OUT/'Readout_completion.tmp.xlsx'
        with pd.ExcelWriter(tmp,engine='xlsxwriter') as w:
            df.to_excel(w,sheet_name='Comparisons',index=False);cov.to_excel(w,sheet_name='Coverage',index=False)
            pd.DataFrame(scores).to_excel(w,sheet_name='Transfer_scores',index=False)
            pd.DataFrame([dict(item=k,description=d[k]) for k in ('local','references','identification','transfer','inference')]).to_excel(w,sheet_name='Protocol',index=False)
        tmp.replace(OUT/'Readout_completion.xlsx')
    result=dict(total=len(cov),completed=int(cov.status.eq('complete').sum()),panel_complete=panel_complete,
                all_complete=all(panel_complete.values()),automatic_shutdown=False,updated_utc=utc())
    save(RUN/'summary.json',result)
    if result['all_complete']:save(RUN/'complete.json',result)
    return result


def run(workers):
    d=register();lock=open(RUN/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=ROOT/'logs/readout_completion_v1';logs.mkdir(parents=True,exist_ok=True)
    active={};failed=set();last_report=0
    # Finish informative transfer first, then cheap prediction-reuse controls.
    ordered=sorted(d['jobs'],key=lambda j:({'transfer':0,'local':1,'identity':2}[j['action']],j['model']))
    while True:
        for key,(proc,j,fh) in list(active.items()):
            if proc.poll() is None:continue
            fh.close();del active[key]
            if proc.returncode or not (RUN/j['output']).exists():failed.add(key)
            print('finished',key,proc.returncode,flush=True)
        # Preserve the user's priority: existing frozen supplement finishes first.
        slots=workers if (extension.RUN/'complete.json').exists() else 1
        done={j['id'] for j in ordered if (RUN/j['output']).exists()}
        for j in ordered:
            if len(active)>=slots:break
            if j['id'] in done|set(active)|failed or not Path(j['dependency']).exists():continue
            cmd=[sys.executable,str(Path(__file__).resolve()),j['action'],'--model',j['model'],'--cohort',j['cohort']]
            if 'task' in j:cmd+=['--task',j['task']]
            fh=open(logs/(j['id'].replace('/','_')+'.log'),'a')
            active[j['id']]=(subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT),j,fh)
            print('started',j['id'],flush=True)
        save(RUN/'status.json',dict(total=len(ordered),completed=len(done),running=list(active),failed=sorted(failed),
            pending=len(ordered)-len(done)-len(active)-len(failed),effective_workers=slots,
            automatic_shutdown=False,updated_utc=utc()))
        if time.time()-last_report>180:
            report();last_report=time.time()
        if len(done)==len(ordered):break
        if not active and failed:break
        time.sleep(5)
    print(report(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','local','identity','transfer','audit','report','run'])
    p.add_argument('--model',choices=frozen.MODELS);p.add_argument('--cohort');p.add_argument('--task');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.action=='init':print(len(register()['jobs']))
    elif a.action=='run':run(a.workers)
    elif a.action=='report':print(report())
    elif a.action=='audit':audits()
    else:
        verify()
        dest=RUN/(f'identity/{a.model}/{a.cohort}.json' if a.action=='identity' else f'{a.action}/{a.model}/{a.cohort}_{a.task}.json')
        if dest.exists():return
        if a.action=='local':local(a.model,a.cohort,a.task)
        elif a.action=='identity':identity(a.model,a.cohort)
        else:transfer(a.model,a.cohort,a.task)


if __name__=='__main__':main()
