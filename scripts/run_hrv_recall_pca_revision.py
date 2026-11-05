#!/usr/bin/env python3
"""CPU-only, resumable HRV delayed-recall PCA revision; original results read-only.

Primary: 5 PCs. Sensitivity: inner selection among 5/10/20 PCs and ridge C.
Frozen and all 162 source/objective/seed adaptations reuse cached embeddings.
The nine baseline frozen absolute tests refit the entire pipeline 1,999 times.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys, time, json, traceback, fcntl
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from benchmark_evaluation import _predict, score, predict_prepared
import run_evidence_nulls as null
from paper_paired_stats import paired, holm

RUN = ROOT/'runs/hrv_recall_pca5_v1'
OUT = ROOT/'results/generated/hrv_recall_pca5_v1'
OLD_NULL = ROOT/'runs/discussion_nulls_v1'
MODELS = null.MODELS
read, save, sha = null.read, null.save, null.sha
COHORTS = ('hrv_pre', 'hrv_post')

def prepare():
    import unified_supplements as u
    from fm_readout_v4.data import metadata, dataset
    from fm_readout_v4.evaluate import tag
    if (RUN/'manifest.json').exists():
        return read(RUN/'manifest.json')
    jobs = []
    templates = {}
    for co in COHORTS:
        p = ROOT/'runs/sf100_benchmark_v1/evaluation'/co/'MEM_delayed_recall.json'
        t = read(p); m = metadata(co)
        ix = m.set_index('scan').index.get_indexer(t['scan_ids'])
        assert (ix >= 0).all() and m.iloc[ix].subject.astype(str).tolist() == t['subjects']
        assert not t['covariates'] and len(set(t['subjects'])) == len(t['y'])
        templates[co] = dict(**{k:t[k] for k in ('scan_ids','subjects','y','split_audit')}, source=str(p), source_sha256=sha(p))
        def add(name, model, source, objective, target, seed, path, kind='array'):
            path=Path(path)
            if kind == 'array':
                x=np.load(path, mmap_mode='r')[ix]
                assert len(x)==len(t['y']) and np.isfinite(x).all()
                dest=RUN/'inputs'/co/(name+'.npy');dest.parent.mkdir(parents=True,exist_ok=True)
                np.save(dest,x)
            else: dest=path
            j=dict(id=co+'__'+name,cohort=co,model=model,source=source,objective=objective,target=target,
                   seed=seed,input=str(dest),input_sha256=sha(dest),source_file=str(path),source_sha256=sha(path),kind=kind)
            jobs.append(j)
        for model in MODELS:
            for state in u.states(model):
                folder=(ROOT/'runs/brainharmonix_f_v1/features'/tag(*state) if model=='brainharmonix_f' else u.feature_folder(state))
                source,_,obj,target,seed=state
                add(tag(*state),model,source,obj,target,seed,folder/(co+'.npy'))
        for method in ('sf100_fc','braingfm'):
            add(method,method,'none','frozen','none',0,ROOT/'runs/sf100_benchmark_v1/features'/co/(method+'.npy'))
        for atlas in ('a424','sch450','sch400'):
            x,_=dataset(co,'brainlm13' if atlas=='a424' else 'brainjepa')
            if atlas=='sch400':x=x[:,:400]
            x=x.numpy()[ix]
            p=RUN/'inputs'/co/(atlas+'_timeseries.npy');p.parent.mkdir(parents=True,exist_ok=True);np.save(p,x)
            from transfer_baseline import fc_feature
            fp=RUN/'inputs'/co/(atlas+'_FC.npy');np.save(fp,fc_feature(x,None))
            # These arrays are already in task order, unlike full-cohort caches.
            for name,path,kind in [(atlas+'_FC',fp,'ordered_array'),(atlas+'_FC-Tucker',p,'tucker')]:
                add(name,name,'none','frozen','none',0,path,kind)
        for kind in ('semantoks','fmrilm','brainharmonix_f'):
            base=ROOT/('runs/brainharmonix_f_v1/supplements' if kind=='brainharmonix_f' else 'runs/reviewer_completion_v1')
            add(kind+'_input_FC',kind+'_input_FC','none','frozen','none',0,base/'input_fc'/kind/(co+'.npy'))
    old=read(OLD_NULL/'manifest.json')
    nj=[]
    for j in old['jobs']:
        if j['task']!='MEM_delayed_recall':continue
        j=dict(j);j['components']=5;j.pop('archived_prediction',None)
        j['revision']='5-PC primary; 100-PC result retained as historical sensitivity'
        save(RUN/'null/job_specs'/(j['id']+'.json'),j);nj.append(j)
    assert len(nj)==9 and len(jobs)==364
    save(RUN/'null/manifest.json',dict(jobs=nj))
    protocol=dict(primary_PCs=5,sensitivity_PCs=[5,10,20],split_seeds=[0,1,2],adaptation_seeds=[42,43,44],
        ridge_C=list(null.GRID),outer_folds=5,inner_validation_fraction=.2,permutations=1999,
        paired_bootstraps=5000,paired_swaps=20000,selection='Joint PC/ridge selection within inner validation only',
        rationale='Uniform follow-up complexity, authorized before computing this revision; historical results were already known.',
        scope='Local imaging-only delayed recall from baseline and post-assessment scans. Clinical-added probes already used 5 PCs. Fixed-source transfer and LOSO are unchanged.',
        immutable_parent='All old 100-PC outputs remain intact. Recalculate affected complete Holm families, never only significant rows.',
        encoder_training=False,automatic_shutdown=False)
    manifest=dict(jobs=jobs,templates=templates,protocol=protocol,created_utc=null.utc(),code_sha256=sha(__file__))
    save(RUN/'manifest.json',manifest);save(RUN/'protocol.json',protocol)
    return manifest

def project_pair(a,b,k):
    sc=StandardScaler().fit(a);a=sc.transform(a).astype(float);b=sc.transform(b).astype(float)
    k=min(k,len(a)-1,a.shape[1]);vals,u=eigh(a@a.T,subset_by_index=[len(a)-k,len(a)-1])
    vals,u=vals[::-1],u[:,::-1];good=vals>max(float(vals[0]),1)*np.finfo(float).eps*max(a.shape)
    v=(a.T@u[:,good])/np.sqrt(vals[good])[None,:]
    assert v.shape[1]
    return a@v,b@v

def fit(j):
    dest=RUN/'fits'/(j['id']+'.json')
    if dest.exists():return dict(id=j['id'],cached=True)
    started=time.monotonic();m=read(RUN/'manifest.json');t=m['templates'][j['cohort']]
    assert sha(j['input'])==j['input_sha256']
    x=np.load(j['input']);y=np.asarray(t['y']);g=np.asarray(t['subjects'])
    if j['kind']=='tucker':
        from fm_matched_v3.probes import TuckerBlock
        block=TuckerBlock(x,64)
    predictions={'fixed5':[],'selected':[]};choices=[]
    for s in range(3):
        pp={k:np.full(len(y),np.nan) for k in predictions}
        for fi,f in enumerate(z for z in t['split_audit'] if z['seed']==s):
            tr,te,it,iv=[np.asarray(f[k]) for k in ('train','test','inner_train','validation')]
            assert not set(g[tr])&set(g[te]) and not set(g[it])&set(g[iv]) and set(it)|set(iv)==set(tr)
            if j['kind']=='tucker':
                a,b=block.fit_pair(tr,te);ai,bi=block.fit_pair(it,iv)
                a,b=project_pair(a,b,20);ai,bi=project_pair(ai,bi,20)
            else:
                x=np.asarray(x,dtype=np.float32)
                a,b=null.projection(x,tr,te,20);ai,bi=null.projection(x,it,iv,20)
            for mode,budgets in [('fixed5',[5]),('selected',[5,10,20])]:
                best=(5,null.GRID[0]);vbest=-np.inf
                for k in budgets:
                    for c in null.GRID:
                        v=score(y[iv],_predict(ai[:,:k],y[it],bi[:,:k],c,False),False)
                        if np.isfinite(v) and v>vbest:best=(k,c);vbest=v
                k,c=best;pp[mode][te]=_predict(a[:,:k],y[tr],b[:,:k],c,False)
                choices.append(dict(split_seed=s,fold=fi,mode=mode,requested_PCs=k,effective_PCs=min(k,a.shape[1]),C=c))
        for mode in predictions:
            assert np.isfinite(pp[mode]).all();predictions[mode].append(pp[mode])
    avg={k:np.mean(v,axis=0) for k,v in predictions.items()}
    # Guard the primary against a numerically distinct null/refit implementation.
    if j['model'] in MODELS and j['objective']=='frozen' and j['cohort']=='hrv_pre':
        nj=read(RUN/'null/job_specs'/(j['model']+'__hrv_pre__MEM_delayed_recall.json'))
        checked,_=null.predictions(x,y,g,nj,splits=t['split_audit'])
        np.testing.assert_allclose(avg['fixed5'],checked,rtol=0,atol=2e-5)
    result=dict(status='complete',job=j,subjects=t['subjects'],scan_ids=t['scan_ids'],y=y,prediction=avg,
        predictions_by_split=predictions,choices=choices,scores={k:score(y,v,False) for k,v in avg.items()},
        seconds=time.monotonic()-started,completed_utc=null.utc())
    save(dest,result);return dict(id=j['id'],seconds=result['seconds'])

def work(item):
    kind,payload=item
    if kind=='fit':return fit(payload)
    null.RUN=RUN/'null';null.payload.cache_clear()
    if kind=='observed':return null.observed(payload)
    if kind=='null':return null.chunk(*payload)
    if kind=='contrast':
        dest=RUN/'contrasts'/(payload['id']+'.json')
        if dest.exists():return dict(id=payload['id'],cached=True)
        stat=paired(payload['y'],payload['a'],payload['b'],payload['subjects'],seed=null.seed(payload['id']))
        result={k:v for k,v in payload.items() if k not in ('a','b','y','subjects')}
        result.update(stat);save(dest,result);return dict(id=payload['id'])
    raise ValueError(kind)

def stage(items,workers,name):
    started=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(work,item):item for item in items}
        for future in as_completed(futures):
            item=futures[future]
            try:future.result()
            except Exception:
                save(RUN/'error.json',dict(stage=name,item=item,error=traceback.format_exc(),utc=null.utc()))
                raise
            done+=1
            save(RUN/'status.json',dict(stage=name,completed=done,total=len(items),workers=workers,elapsed_seconds=time.monotonic()-started,updated_utc=null.utc(),automatic_shutdown=False))
            if done%20==0 or done==len(items):print(name,done,'/',len(items),flush=True)

def run(workers):
    m=prepare()
    stage([('fit',j) for j in m['jobs']],workers,'fits')
    ids=[j['id'] for j in read(RUN/'null/manifest.json')['jobs']]
    stage([('observed',i) for i in ids],min(workers,9),'observed')
    stage([('null',(i,start,min(start+25,1999))) for i in ids for start in range(0,1999,25)],workers,'nulls')
    # Reporting is a separate resumable step and must succeed before a final release.
    report(workers)

def report(workers):
    m=read(RUN/'manifest.json');OUT.mkdir(parents=True,exist_ok=True)
    grouped={}
    for j in m['jobs']:
        r=read(RUN/'fits'/(j['id']+'.json'));assert r['status']=='complete'
        key=(j['cohort'],j['model'],j['source'],j['objective'],j['target'])
        grouped.setdefault(key,[]).append(r)
    def get(co,model,source='none',obj='frozen',target='none',mode='fixed5'):
        rs=grouped[(co,model,source,obj,target)]
        assert len(rs)==(1 if obj=='frozen' else 3)
        return np.mean([r['prediction'][mode] for r in rs],axis=0)
    scores=[];selections=[]
    for (co,model,source,obj,target),rs in grouped.items():
        for mode in ('fixed5','selected'):
            p=get(co,model,source,obj,target,mode);y=np.asarray(m['templates'][co]['y'])
            scores.append(dict(cohort=co,model=model,source=source,objective=obj,target=target,mode=mode,n=len(y),score=score(y,p,False),mae=float(np.mean(abs(y-p)))))
        for r in rs:
            selections.extend(dict(model=model,cohort=co,source=source,objective=obj,target=target,adaptation_seed=r['job']['seed'],**c) for c in r['choices'])
    pd.DataFrame(scores).to_csv(OUT/'Scores.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'PC_selections.csv',index=False)
    joint=pd.read_csv(ROOT/'results/generated/brainharmonix_f_v1/joint/Comparisons.csv',low_memory=False)
    task=joint.task.fillna('').str.contains('delayed_recall')
    mask=task & (joint.panel.isin(['local_frozen','atlas_frozen_reference','atlas_adapted_reference','same_window_FC']) |
        (joint.panel.eq('cognitive_adaptation') & joint.analysis_method.eq('FM')))
    affected=joint[mask];assert len(affected)==294
    contrast_jobs=[]
    def contrast(ident,co,a,b,**meta):
        t=m['templates'][co]
        contrast_jobs.append(('contrast',dict(id=ident,cohort=co,y=t['y'],subjects=t['subjects'],a=a,b=b,**meta)))
    for ix,r in affected.iterrows():
        source=r.source if r.objective!='frozen' else 'none'
        target=r.adaptation_target if r.objective!='frozen' else 'none'
        b=get(r.cohort,r.model,source,r.objective,target)
        if r.panel=='cognitive_adaptation':a=get(r.cohort,r.model)
        elif r.panel=='local_frozen':a=get(r.cohort,'sf100_fc')
        elif r.panel=='same_window_FC':
            kind='fmrilm' if r.model.startswith('fmrilm') else r.model
            a=get(r.cohort,kind+'_input_FC')
        else:
            atlas='a424' if r.model.startswith('brainlm') else 'sch400' if r.model=='brainharmonix_f' else 'sch450'
            a=get(r.cohort,atlas+('_FC-Tucker' if 'Tucker' in r.reference else '_FC'))
        contrast('joint_'+str(ix),r.cohort,a,b,parent_row=int(ix),family=r.family)
    follow=pd.read_csv(OLD_NULL/'followup_imaging_only_comparisons.csv')
    fm=follow.cohort.eq('hrv_pre') & follow.task.eq('MEM_followup');assert fm.sum()==18
    for ix,r in follow[fm].iterrows():
        contrast('follow_'+str(ix),'hrv_pre',get('hrv_pre',r.reference),get('hrv_pre',r.model),parent_row=int(ix))
    # Predeclared sensitivity families pool all configurations and both visits,
    # separating the question of dimension choice from adaptation and FC gains.
    for key in grouped:
        co,model,source,obj,target=key
        if model not in MODELS:continue
        base='__'.join(key)
        contrast('sensitivity_'+base,co,get(*key),get(*key,mode='selected'),
                 model=model,source=source,objective=obj,target=target,contrast='selected_vs_fixed5',family='dimension::'+obj+'::'+target)
        if obj!='frozen':
            contrast('sensitivity_adapt_'+base,co,get(co,model,mode='selected'),get(*key,mode='selected'),
                     model=model,source=source,objective=obj,target=target,contrast='adapted_vs_frozen_selected',family='adaptation::'+obj+'::'+target)
        for reference in ('sf100_fc','braingfm'):
            contrast('sensitivity_ref_'+base+'__'+reference,co,get(co,reference,mode='selected'),get(*key,mode='selected'),
                     model=model,source=source,objective=obj,target=target,reference=reference,contrast='selected_vs_reference',family='reference::'+reference+'::'+obj+'::'+target)
    save(RUN/'contrast_plan.json',dict(count=len(contrast_jobs),jobs=[{k:v for k,v in item[1].items() if k not in ('a','b','y','subjects')} for item in contrast_jobs]))
    stage(contrast_jobs,workers,'paired_statistics')
    for ix,r in affected.iterrows():
        stat=read(RUN/'contrasts'/('joint_'+str(ix)+'.json'))
        for field in ('reference_score','candidate_score','delta','bootstrap_valid','bootstrap_se','permutations_valid','permutation_exceedances','permutation_p','resampling_seed'):
            joint.at[ix,field]=stat[field]
        joint.at[ix,'ci95']=json.dumps(stat['ci95']);joint.at[ix,'ci_low']=stat['ci95'][0];joint.at[ix,'ci_high']=stat['ci95'][1]
        joint.at[ix,'raw_p']=stat['permutation_p'];joint.at[ix,'score']=stat['candidate_score']
        joint.at[ix,'revision_protocol']='HRV local imaging-only fixed5 PCs; shared samples/splits; 3 adaptation seeds'
    families=set(affected.family)
    for family in families:
        ix=joint.index[joint.family.eq(family)]
        joint.loc[ix,'holm_p_joint.2']=holm(joint.loc[ix,'raw_p'])
        joint.loc[ix,'family_size_joint.2']=len(ix)
        joint.loc[ix,'significant_joint.2']=joint.loc[ix,'holm_p_joint.2']<.05
        joint.loc[ix,'status_joint.2']=np.where(joint.loc[ix,'holm_p_joint.2']<.05,np.where(joint.loc[ix,'delta']>0,'gain','loss'),'not_significant')
    joint.to_csv(OUT/'Comparisons.csv',index=False)
    joint[mask].to_csv(OUT/'Revised_primary_contrasts.csv',index=False)
    for ix,r in follow[fm].iterrows():
        stat=read(RUN/'contrasts'/('follow_'+str(ix)+'.json'))
        for key in follow.columns:
            if key in stat:follow.at[ix,key]=json.dumps(stat[key]) if key=='ci95' else stat[key]
    for (task,ref),ix in follow.groupby(['task','reference']).groups.items():
        assert len(ix)==45
        follow.loc[ix,'holm_p']=holm(follow.loc[ix,'permutation_p'])
    follow.to_csv(OUT/'followup_imaging_only_comparisons.csv',index=False)
    sens=pd.DataFrame([read(RUN/'contrasts'/(item[1]['id']+'.json')) for item in contrast_jobs if item[1]['id'].startswith('sensitivity_')])
    for family,ix in sens.groupby('family').groups.items():
        sens.loc[ix,'holm_p']=holm(sens.loc[ix,'permutation_p']);sens.loc[ix,'family_size']=len(ix)
    sens.to_csv(OUT/'Sensitivity_comparisons.csv',index=False)
    results=pd.read_csv(OLD_NULL/'results.csv')
    for j in read(RUN/'null/manifest.json')['jobs']:
        obs=read(RUN/'null/observed'/(j['id']+'.json'))
        chunks=[read(p) for p in sorted((RUN/'null/chunks'/j['id']).glob('*.json'))]
        slots=[i for c in chunks for i in range(c['start'],c['stop'])]
        assert sorted(slots)==list(range(1999)),j['id']
        ix=results.index[results.id.eq(j['id'])];assert len(ix)==1
        results.loc[ix,'observed_score']=obs['score']
        results.loc[ix,'raw_p']=(1+sum(c['exceedances'] for c in chunks))/2000
        results.loc[ix,'seconds']=obs['seconds']+sum(c['seconds'] for c in chunks)
        results.loc[ix,'revision_protocol']='fixed5'
    ix=results.index[results.family.eq('above_chance::MEM_followup')];assert len(ix)==45
    results.loc[ix,'holm_p']=holm(results.loc[ix,'raw_p'])
    results.to_csv(OUT/'Absolute_tests.csv',index=False)
    save(OUT/'complete.json',dict(status='complete',finished_utc=null.utc(),fits=len(m['jobs']),revised_primary_contrasts=int(mask.sum()),
        affected_joint_families=sorted(families),absolute_recomputed=9,permutations_each=1999,followup_absolute_family=45,
        unchanged_predictions_in_other_cohorts=True,old_results_overwritten=False,files={p.name:sha(p) for p in OUT.glob('*.csv')}))
    save(RUN/'status.json',dict(status='complete',stage='complete',updated_utc=null.utc(),automatic_shutdown=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','report']);p.add_argument('--workers',type=int,default=64);a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='prepare':print(json.dumps(dict(jobs=len(prepare()['jobs']),run=str(RUN))))
        elif a.action=='run':run(a.workers)
        else:report(a.workers)
