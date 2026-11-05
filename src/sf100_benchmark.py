"""Frozen SF100 FC/BrainGFM panel on explicitly matched canonical scans."""
from pathlib import Path
from collections import defaultdict, Counter
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from fm_matrix.data import ROOT, save, sha
from fm_readout_v4.data import metadata, DATA
from braingfm import CHECKPOINT, validate_fc, fc_edges, load_braingfm, braingfm_fc_readout
from benchmark_evaluation import grouped_splits, predict_prepared, score, GRID
from paper_completion import project_block, feature_dir
from paper_paired_stats import paired, holm
from paper_robustness import covariate_pair
from readout_sensitivity import paired_rows, similarity, identification_scores, paired_identification, seed_of

RUN=ROOT/'runs/sf100_benchmark_v1'
ARCHIVE=ROOT/'data/derived/sf100_fc_v1/archive'
COHORTS=('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post','nacc_repeated')
MODELS=('brainlm13','brainlm111','brainlm650','brainjepa')
DATASETS={'adni':'ADNI','nacc':'NACC','nacc_repeated':'NACC','act':'ACT','cogte':'CogTE','breathe':'BREATHE','psopt':'PSOPT','hrv_pre':'HRV-ER','hrv_post':'HRV-ER'}


def read(p):return json.loads(Path(p).read_text())


def archive_key(d):
    co=d['dataset'];s=d['subject'];v=d['visit']
    if co=='ADNI':return co,s,d['session'].removeprefix('ses-')
    if co=='NACC':return co,s,d['session'].removeprefix('ses-')
    return co,s,v


def canonical_key(co,r):
    s=str(r.subject);v=str(r.visit)
    if co=='adni':s='sub-ADNI'+s.replace('_','')
    if co=='act':v={'T1':'T1_baseline','T2':'T3_6mo'}[v]
    if co.startswith('hrv'):v={'pre':'T1','post':'T2'}[v]
    return DATASETS[co],s,v


def choose_equivalent(paths):
    """Collapse only numerically identical duplicate exports; never choose a run."""
    paths=sorted(paths)
    first=np.load(paths[0],allow_pickle=False)
    if any(not np.allclose(first,np.load(p,allow_pickle=False),rtol=0,atol=1e-6) for p in paths[1:]):return None
    return paths[0]


def label_file(co):return DATA/(co+'_domains.csv' if co.startswith('hrv') else co+'.csv')


def audit():
    RUN.mkdir(parents=True,exist_ok=True)
    index=defaultdict(list);files={};invalid=[];counts=Counter()
    for p in sorted((ARCHIVE/'results').rglob('*_FC.npy')):
        try:validate_fc(np.load(p,allow_pickle=False)[None],'schaefer')
        except ValueError as e:invalid.append(dict(path=str(p),reason=str(e)));continue
        d=read(p.with_suffix('.json'));index[archive_key(d)].append(p);counts[d['dataset']]+=1
        files[str(p)]=sha(p);files[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
    if invalid:save(RUN/'invalid.json',invalid);raise ValueError('Invalid archive FCs')
    summary={};conflicts=[]
    for co in COHORTS:
        m=metadata(co);rows=[]
        for i,r in m.iterrows():
            all_candidates=index.get(canonical_key(co,r),[])
            candidates=[p for p in all_candidates if '_run-01_' in p.name] if co=='breathe' else all_candidates
            chosen=choose_equivalent(candidates) if candidates else None
            reason='matched' if chosen else 'conflicting_exports' if candidates else 'missing_or_unconfirmed_visit'
            if reason=='conflicting_exports':conflicts.append(dict(cohort=co,scan=r.scan,paths=list(map(str,candidates))))
            rows.append(dict(index=int(i),subject=str(r.subject),scan=str(r.scan),visit=str(r.visit),available=chosen is not None,
                reason=reason,path=str(chosen) if chosen else None,duplicates=list(map(str,candidates)),
                other_runs_excluded=[str(p) for p in all_candidates if p not in candidates]))
        save(RUN/'matches'/(co+'.json'),dict(status='complete',rows=rows))
        summary[co]=dict(canonical=len(m),matched=sum(r['available'] for r in rows),reasons=dict(Counter(r['reason'] for r in rows)))
        files[str(label_file(co))]=sha(label_file(co))
        for p in [ROOT/'runs/demographic_benchmark_v1/labels'/(co+'.csv'),ROOT/'runs/diagnosis_posttraining_v1'/(co+'_labels.csv')]:
            if p.exists():files[str(p)]=sha(p)
        for model in MODELS:
            p=feature_dir(('none',model,'frozen','none',0))/(co+'.npy')
            x=np.load(p,mmap_mode='r')
            if len(x)!=len(m) or x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Canonical FM feature mismatch')
            files[str(p)]=sha(p)
        files[str(RUN/'matches'/(co+'.json'))]=sha(RUN/'matches'/(co+'.json'))
    files[str(CHECKPOINT)]=sha(CHECKPOINT)
    for n in ('roi_labels.tsv','README.txt','validation.json','FC_inventory.tsv'):
        p=ARCHIVE/n;files[str(p)]=sha(p)
    save(RUN/'input_audit.json',dict(status='complete',archive_sha256=sha(ROOT/'data/Schaefer100_noGSR_FC_PARTIAL.tar.gz'),
        counts=dict(counts),coverage=summary,conflicts=conflicts,files_sha256=files,
        package_validation=read(ARCHIVE/'validation.json'),evidence_status='preliminary',
        roi_order='Archive labels establish Schaefer100 7Networks LH50 then RH50; checkpoint-specific network/order provenance remains unverified.',
        preprocessing='NoGSR according to source filenames; inherited estimators and denoising for reused FCs are not independently established. Full available scans; not matched 200/160-frame FCs.',
        limitations=['Cross-template registration unresolved in newly extracted subsets.','BrainGFM participant-level pretraining overlap unresolved.'],
        user_confirmed_mapping={'ACT_T2':'T3_6mo','BREATHE':'run-01'},
        tucker=dict(status='omitted_by_user',reason='SF100 time series not available; co-skewness Tucker cannot be recovered from second-order FC.')))
    print(json.dumps(summary,indent=2),flush=True)


def verify(p):
    if read(RUN/'input_audit.json')['files_sha256'].get(str(p))!=sha(p):raise ValueError('Registered input changed: '+str(p))


def matches(co):
    p=RUN/'matches'/(co+'.json');verify(p);return read(p)['rows']


def extract(device='cuda'):
    verify(CHECKPOINT);model=load_braingfm(device=device,atlas='schaefer');cache={};unique=sorted({r['path'] for co in COHORTS for r in matches(co) if r['available']})
    # Unique source files are embedded once, including NACC repeated/primary overlap.
    for start in range(0,len(unique),32):
        ps=unique[start:start+32]
        for p in ps:verify(Path(p))
        fc=np.stack([np.load(p,allow_pickle=False) for p in ps]);g=braingfm_fc_readout(model,fc,batch=8);e=fc_edges(fc,'schaefer')
        for p,gg,ee in zip(ps,g,e):cache[p]=(gg,ee)
        print('SF100 frozen embeddings',start+len(ps),len(unique),flush=True)
    for co in COHORTS:
        rows=matches(co);folder=RUN/'features'/co;folder.mkdir(parents=True,exist_ok=True)
        for name,dim,col in [('braingfm',512,0),('sf100_fc',4950,1)]:
            x=np.full((len(rows),dim),np.nan,dtype=np.float32)
            for i,r in enumerate(rows):
                if r['available']:x[i]=cache[r['path']][col]
            np.save(folder/(name+'.npy'),x)
        save(folder/'complete.json',dict(status='complete',scan_ids=[r['scan'] for r in rows],
            files_sha256={n:sha(folder/(n+'.npy')) for n in ('braingfm','sf100_fc')},model=model.benchmark_metadata))
    save(RUN/'features/complete.json',dict(status='complete',unique_fc_files=len(unique),evidence_status='preliminary'))


def features(co,include_fm=True):
    folder=RUN/'features'/co;stamp=read(folder/'complete.json');m=metadata(co)
    if stamp['scan_ids']!=m.scan.tolist():raise ValueError('Scan order changed')
    out={}
    for n,h in stamp['files_sha256'].items():
        p=folder/(n+'.npy')
        if sha(p)!=h:raise ValueError('Feature changed')
        out[n]=np.load(p,mmap_mode='r')
    if include_fm:
        for model in MODELS:
            p=feature_dir(('none',model,'frozen','none',0))/(co+'.npy');verify(p);out[model]=np.load(p,mmap_mode='r')
    return out


def labels(co):
    verify(label_file(co));m=metadata(co).copy()
    for p,cols in [(ROOT/'runs/demographic_benchmark_v1/labels'/(co+'.csv'),['age','sex']),
                   (ROOT/'runs/diagnosis_posttraining_v1'/(co+'_labels.csv'),['DX','AD_DX'])]:
        if p.exists():
            verify(p);t=pd.read_csv(p).set_index('scan')
            if t.index.duplicated().any():raise ValueError('Duplicate labels')
            for c in cols:
                if c in t:m[c]=m.scan.map(t[c])
    return m


def task_specs(co,m):
    specs=[(e,e,False,[]) for e in ('MEM','EF') if e in m and m[e].notna().any()]
    specs += [('AGE','age',False,[])] if not co.startswith('hrv') else []
    specs += [('SEX','sex',True,[])]
    specs += [(e,e,True,[]) for e in ('DX','AD_DX') if e in m]
    if co in ('act','cogte','breathe','psopt','hrv_pre'):
        cov=['baseline_EF','age_category','sex','arm'] if co=='hrv_pre' else ['baseline_MEM','baseline_EF','age','sex']
        specs += [(e+'_followup','follow_'+e,False,cov) for e in ('MEM','EF') if 'follow_'+e in m and m['follow_'+e].notna().any()]
    if co.startswith('hrv'):specs += [('MEM_delayed_recall','MEM_delayed_recall',False,[])]
    return specs


def evaluate(co,device='cpu'):
    m=labels(co);xs=features(co);available=np.array([r['available'] for r in matches(co)])
    base=available&m.eligible.to_numpy(bool)
    if co in ('act','cogte','breathe','psopt'):base &= m.visit.eq('T1').to_numpy()
    folder=RUN/'evaluation'/co;folder.mkdir(parents=True,exist_ok=True)
    for task,col,binary,covs in task_specs(co,m):
        dest=folder/(task+'.json')
        if dest.exists():continue
        y0=pd.to_numeric(m[col],errors='coerce').to_numpy(float);mask=base&np.isfinite(y0)
        for c in covs:mask &= m[c].notna().to_numpy()
        ix=np.flatnonzero(mask);y=y0[ix];g=m.iloc[ix].subject.astype(str).to_numpy()
        if len(set(g))<20 or len(np.unique(y))<2:
            save(dest,dict(status='not_applicable',reason='Fewer than 20 subjects or constant target',task=task,n=len(ix)));continue
        if binary and not set(np.unique(y))=={0.,1.}:raise ValueError('Invalid binary labels')
        pred={n:[] for n in xs};folds=[]
        if covs:pred['clinical']=[]
        for seed in (0,1,2):
            prep={n:[] for n in pred}
            for fi,(tr,te) in enumerate(grouped_splits(y,g,seed)):
                it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=g[tr]));it,iv=tr[it],tr[iv]
                ca,cb=covariate_pair(m,covs,ix[tr],ix[te]);cia,cib=covariate_pair(m,covs,ix[it],ix[iv])
                for name,x in xs.items():
                    a,b=project_block(x[ix[tr]],x[ix[te]],5 if covs else 100,device)
                    ai,bi=project_block(x[ix[it]],x[ix[iv]],5 if covs else 100,device)
                    prep[name].append((tr,te,(np.c_[ca,a],np.c_[cb,b]),it,iv,(np.c_[cia,ai],np.c_[cib,bi])))
                if covs:prep['clinical'].append((tr,te,(ca,cb),it,iv,(cia,cib)))
                folds.append(dict(seed=seed,fold=fi,train=tr,test=te,inner_train=it,validation=iv))
            for n in pred:pred[n].append(predict_prepared(prep[n],y,binary=binary))
            print('SF100 probes',co,task,seed,flush=True)
        pp={n:np.mean(p,axis=0) for n,p in pred.items()};rows=[]
        for n in xs:
            if n=='sf100_fc':continue
            rows.append(dict(cohort=co,task=task,reference='sf100_fc',candidate=n,family='local_'+task,
                **paired(y,pp['sf100_fc'],pp[n],g,metric='auc' if binary else 'rho',seed=seed_of(co+task+n))))
        if covs:
            for n in ('sf100_fc','braingfm'):
                rows.append(dict(cohort=co,task=task,reference='clinical',candidate='clinical+'+n,family='incremental_'+task,
                    **paired(y,pp['clinical'],pp[n],g,seed=seed_of(co+task+n+'clinical'))))
        save(dest,dict(status='complete',cohort=co,task=task,rows=rows,scan_ids=m.iloc[ix].scan.tolist(),subjects=g,y=y,
            predictions=pp,predictions_by_seed=pred,split_audit=folds,covariates=covs,
            metrics={n:dict(score=score(y,p,binary),metric='auc' if binary else 'rho',mae=None if binary else float(np.mean(np.abs(y-p)))) for n,p in pp.items()},
            feature_dimension={n:x.shape[1] for n,x in xs.items()},protocol='Grouped outer 5-fold, inner grouped 20% validation, seeds 0/1/2; training-only scaling and exact PCA <=100 (<=5 follow-up); frozen features. Follow-up arms include identical baseline clinical covariates.',
            evidence_status='preliminary',cross_model_caveat='SF100 full-scan FC versus cached model-specific 200/160-frame inputs; atlas, context and preprocessing are not held constant.'))
    save(folder/'complete.json',dict(status='complete',cohort=co))


def identification():
    rows=[];coverage={}
    for co in ('adni','nacc_repeated','act','cogte','breathe','psopt','hrv'):
        subjects,(ca,ia),(cb,ib)=paired_rows(co)
        ma=matches(ca);mb=matches(cb);keep=np.array([ma[a]['available'] and mb[b]['available'] for a,b in zip(ia,ib)])
        ia=ia[keep];ib=ib[keep];subjects=np.asarray(subjects)[keep];coverage[co]=len(subjects)
        if len(subjects)<2:continue
        fa=features(ca);fb=fa if ca==cb else features(cb);scores={}
        for n in fa:scores[n]=identification_scores(similarity(fa[n][ia],fb[n][ib]))
        rr=[]
        for n in fa:
            if n=='sf100_fc':continue
            for metric in ('top1','I_diff'):
                rr.append(dict(cohort=co,task='identification',metric=metric,reference='sf100_fc',candidate=n,family='identification_'+metric,
                    **paired_identification(scores['sf100_fc']['per_subject_'+metric],scores[n]['per_subject_'+metric],seed=seed_of(co+n+metric))))
        rows.extend(rr)
        save(RUN/'identification'/(co+'.json'),dict(status='complete',rows=rr,subjects=subjects,scores=scores,
            scans_a=[ma[i]['scan'] for i in ia],scans_b=[mb[i]['scan'] for i in ib]))
    save(RUN/'identification/complete.json',dict(status='complete',rows=rows,coverage=coverage,
        act='User-confirmed canonical T2 maps to archive T3_6mo; BREATHE uses user-confirmed run-01'))


def transfer(source):
    """Source-only cognitive axes; no target label fitting or model selection."""
    sm=labels(source);sx=features(source,False);available=np.array([r['available'] for r in matches(source)]);rows=[]
    for endpoint in ('MEM','EF'):
        y=sm[endpoint].to_numpy(float);ix=np.flatnonzero(available&sm.eligible.to_numpy(bool)&np.isfinite(y))
        groups=sm.iloc[ix].subject.astype(str).to_numpy();fits={};splits=[]
        for name,x in sx.items():
            models=[]
            for seed in (0,1,2):
                tr,va=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(ix,groups=groups))
                tr,va=ix[tr],ix[va];sc=StandardScaler().fit(x[tr]);pc=PCA(n_components=min(100,len(tr)-1,x.shape[1]),svd_solver='full').fit(sc.transform(x[tr]))
                a=pc.transform(sc.transform(x[tr]));b=pc.transform(sc.transform(x[va]));best=GRID[0];best_score=-np.inf
                for c in GRID:
                    s=score(y[va],Ridge(alpha=1/c).fit(a,y[tr]).predict(b),False)
                    if np.isfinite(s) and s>best_score:best,best_score=c,s
                sc=StandardScaler().fit(x[ix]);pc=PCA(n_components=min(100,len(ix)-1,x.shape[1]),svd_solver='full').fit(sc.transform(x[ix]));model=Ridge(alpha=1/best).fit(pc.transform(sc.transform(x[ix])),y[ix]);models.append((sc,pc,model))
                splits.append(dict(representation=name,seed=seed,train_scan_ids=sm.iloc[tr].scan.tolist(),validation_scan_ids=sm.iloc[va].scan.tolist(),selected_c=best))
            fits[name]=models
        for co in COHORTS[:-1]:
            if co==source or (co.startswith('hrv') and endpoint=='MEM'):continue
            m=labels(co);xs=features(co,False);v=m[endpoint].to_numpy(float)
            mask=m.eligible.to_numpy(bool)&np.array([r['available'] for r in matches(co)])&np.isfinite(v)
            if co in ('act','cogte','breathe','psopt'):mask &= m.visit.eq('T1').to_numpy()
            ids=np.flatnonzero(mask)
            if len(ids)<20:continue
            pp={n:np.mean([mdl.predict(pc.transform(sc.transform(xs[n][ids]))) for sc,pc,mdl in ms],axis=0) for n,ms in fits.items()}
            r=dict(source=source,cohort=co,task=endpoint,reference='sf100_fc',candidate='braingfm',family='transfer_'+endpoint,
                **paired(v[ids],pp['sf100_fc'],pp['braingfm'],m.iloc[ids].subject,seed=seed_of(source+co+endpoint)))
            rows.append(r);save(RUN/'transfer'/source/(co+'_'+endpoint+'.json'),dict(status='complete',rows=[r],scan_ids=m.iloc[ids].scan.tolist(),subjects=m.iloc[ids].subject.tolist(),y=v[ids],predictions=pp,source_splits=splits,
                protocol='Source-only grouped 80:20 tuning, seeds 0/1/2, then source-only refit on all labeled source participants. No target tuning. Cross-instrument scores report correlation only.'))
    save(RUN/'transfer'/source/'complete.json',dict(status='complete',rows=rows))


def report():
    rows=[]
    for co in COHORTS[:-1]:
        for p in (RUN/'evaluation'/co).glob('*.json'):rows.extend(read(p).get('rows',[]))
    rows.extend(read(RUN/'identification/complete.json')['rows'])
    for source in ('adni','nacc'):rows.extend(read(RUN/'transfer'/source/'complete.json')['rows'])
    families=defaultdict(list)
    for i,r in enumerate(rows):families[r['family']].append(i)
    for family,ix in families.items():
        for i,p in zip(ix,holm([rows[i].get('permutation_p') for i in ix])):rows[i]['holm_p']=p
    pd.DataFrame(rows).to_csv(RUN/'comparisons.csv',index=False)
    save(RUN/'report.json',dict(status='complete',rows=rows,n_comparisons=len(rows),coverage=read(RUN/'input_audit.json')['coverage'],evidence_status='preliminary',
        tucker='Omitted by user: no SF100 time series',post_training=False,automatic_shutdown=False,
        interpretation='BrainGFM versus FC shares exactly the same SF100 matrices and CV samples. Cross-FM comparisons match samples/folds but not atlas, scan duration or all preprocessing. Checkpoint ROI-order and pretraining-overlap provenance unresolved.',
        inference='Conditional subject-cluster bootstrap 5000 and prediction swaps 20000; Holm within each declared task family across cohorts/candidates. Identification uses fixed-gallery query-score inference. Not a single benchmark-wide correction.'))
