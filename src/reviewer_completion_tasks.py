"""Bounded reviewer-requested sensitivity analyses; immutable parents, no null refits."""
import json
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from fm_matrix.data import ROOT,save,sha
import benchmark_completion_tasks as core
import unified_supplements as u
import sf100_benchmark as sf
import intervention_prepost as ip
from benchmark_evaluation import predict_prepared,score,GRID
from paper_task_expansion import exact_project
from paper_paired_stats import paired

RUN=ROOT/'runs/reviewer_completion_v1'
OUT=ROOT/'results/generated/reviewer_completion_v1'
COHORTS=core.COHORTS
MODELS=u.CONFIGS
METHODS=(*MODELS,'sf100_fc','braingfm')
DIAGNOSTIC_METHODS=(*METHODS,'a424_FC','a424_FC-Tucker','sch450_FC','sch450_FC-Tucker')
SEEDS=(0,1,2)

def read(p):return json.loads(Path(p).read_text())
def seed(s):return core.stable_seed(s)
def state(model):return ('none',model,'frozen','none',0)
def family(model):return 'a424' if model.startswith('brainlm') else 'sch450' if model=='brainjepa' else 'semantoks' if model=='semantoks' else 'fmrilm'
def configure():core.RUN=RUN

def prepare():
    configure();checks=[];files={};coverage=[]
    parent=read(ip.RUN/'inputs.json')
    for co in COHORTS:
        m=sf.labels(co)
        assert not m.scan.duplicated().any()
        for r in core.templates(co):
            ix=m.set_index('scan').index.get_indexer(r['scan_ids']);assert (ix>=0).all()
            assert m.iloc[ix].subject.astype(str).tolist()==r['subjects']
            g=np.asarray(r['subjects']);counts={s:np.zeros(len(g),int) for s in SEEDS}
            for f in r['split_audit']:
                tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
                assert not set(g[tr])&set(g[te]) and not set(g[it])&set(g[iv])
                assert set(it)|set(iv)==set(tr)
                counts[f['seed']][te]+=1
            assert all(np.all(x==1) for x in counts.values())
            checks.append(dict(cohort=co,task=r['task'],n=len(g),participants=len(set(g)),split_audit='passed'))
            p=sf.RUN/'evaluation'/co/(r['task']+'.json');files[str(p)]=sha(p)
    for co,plan in parent['plans'].items():
        for task,t in plan['tasks'].items():
            assert len(t['subjects'])==len(set(t['subjects']))
            for f in t['splits']:
                tr,te,it,iv=[set(f[k]) for k in ('train','test','inner_train','validation')]
                assert not tr&te and not it&iv and it|iv==tr
            coverage.append(dict(cohort=co,task=task,participants=len(t['subjects']),scans=2*len(t['subjects'])))
    files[str(ip.RUN/'inputs.json')]=sha(ip.RUN/'inputs.json')
    save(RUN/'audit/complete.json',dict(status='complete',checks=checks,paired_visit_coverage=coverage,files_sha256=files,
        action='Audit parent templates; refit only bounded reviewer sensitivities. No blanket primary-probe refit and no encoder training.',
        protocol='Same task-specific subjects, outer/inner partitions, split seeds and estimator grids. All new analyses retrospective.',
        manuscript_editing=False,absolute_nulls=False))
    core.context_masks()
    from new_fm.data import DATA
    padding=[]
    for co in COHORTS:
        m=sf.labels(co);n=pd.read_csv(DATA/(co+'.csv'))
        for model in MODELS:
            length=200 if model.startswith('brainlm') else 100 if model=='semantoks' else 160
            real=m.native_frames if model in MODELS[:4] else n.resampled_frames
            groups=[('all',np.arange(len(m)))]
            if 'DX' in m:groups += [(str(k),np.flatnonzero(m.DX.eq(k))) for k in m.DX.dropna().unique()]
            for group,ix in groups:
                frac=1-np.minimum(real.to_numpy()[ix],length)/length
                padding.append(dict(cohort=co,model=model,diagnosis_group=group,n=len(ix),padded=int((frac>0).sum()),mean_padding=float(frac.mean()),median_padding=float(np.median(frac)),max_padding=float(frac.max())))
    pd.DataFrame(padding).to_csv(RUN/'padding_summary.csv',index=False)
    pd.DataFrame(checks+coverage).to_csv(RUN/'task_sample_counts.csv',index=False)
    # Group distributions are descriptive; diagnosis is not used to select covariates.
    from paper_robustness import assemble_labels
    m,_=assemble_labels('nacc');lab=sf.labels('nacc');assert m.scan.tolist()==lab.scan.tolist()
    rows=[];sites=[]
    for task in ('DX','AD_DX'):
        r=next(t for t in core.templates('nacc') if t['task']==task);ix=lab.set_index('scan').index.get_indexer(r['scan_ids'])
        t=m.iloc[ix].copy();t['diagnosis']=r['y']
        for dx,g in t.groupby('diagnosis'):
            for col in ('age','sex','education','native_tr','native_frames','mean_fd'):
                if col in g:
                    z=pd.to_numeric(g[col],errors='coerce');rows.append(dict(task=task,diagnosis=dx,variable=col,n=int(z.notna().sum()),mean=z.mean(),sd=z.std(),median=z.median()))
        sites.extend(t.groupby(['site','diagnosis']).size().reset_index(name='n').assign(task=task).to_dict('records'))
    pd.DataFrame(rows).to_csv(RUN/'nacc_group_characteristics.csv',index=False)
    pd.DataFrame(sites).to_csv(RUN/'nacc_site_class_counts.csv',index=False)
    save(RUN/'prepare.json',dict(status='complete',absolute_nulls=False,manuscript_editing=False,
        unavailable='Complete NACC/ADNI FD metadata not established; no fabricated FD or motion exclusion. Site holdout is not inferred from covariate adjustment.',
        context='Common full observed temporal context, not equal physical duration. Not-estimable subsets remain explicitly recorded.'))

def covariates(m,cols,tr,te):
    aa=[];bb=[]
    for col in cols:
        if col in ('site','sex','arm','age_category'):
            a=m.iloc[tr][[col]].fillna('missing').astype(str);b=m.iloc[te][[col]].fillna('missing').astype(str)
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False);a=enc.fit_transform(a);b=enc.transform(b)
        else:
            a=m.iloc[tr][[col]].astype(float);b=m.iloc[te][[col]].astype(float)
            fill=a.median().fillna(0);a=a.fillna(fill).to_numpy();b=b.fillna(fill).to_numpy()
        sc=StandardScaler().fit(a);aa.append(sc.transform(a));bb.append(sc.transform(b))
    return np.column_stack(aa),np.column_stack(bb)

def project_k(x,tr,te,k):
    # Same PCA implementation as the registered paired-visit primary analysis.
    with patch.object(ip,'K',k):return ip.project(x,tr,te)

def selected_pca_prediction(x,y,tr,te,it,iv,classes,clinical=None):
    inner={k:project_k(x,it,iv,k) for k in (5,10,20)}
    if clinical is not None:
        ci=ip.clinical_transform(clinical,it,iv)
        inner={k:tuple(np.c_[c,a] for c,a in zip(ci,pair)) for k,pair in inner.items()}
    best=(5,GRID[0]);value=-np.inf
    for k,pair in inner.items():
        for c in GRID:
            p=ip.fit_predict(pair[0],y[it],pair[1],c,classes);v=ip.primary(y[iv],p,classes)
            if np.isfinite(v) and v>value:value=v;best=(k,c)
    outer=project_k(x,tr,te,best[0])
    if clinical is not None:
        oc=ip.clinical_transform(clinical,tr,te);outer=tuple(np.c_[c,a] for c,a in zip(oc,outer))
    return ip.fit_predict(outer[0],y[tr],outer[1],best[1],classes),best

def pca(model,co):
    plan=read(ip.RUN/'inputs.json')['plans'][co];a=core.feature(state(model),plan['pre_cohort']);b=core.feature(state(model),plan['post_cohort']);rows=[]
    for task,t in plan['tasks'].items():
        y=np.asarray(t['y']);classes=t['classes'];n=len(y)
        pre=np.asarray(a[t['pre_indices']],float);post=np.asarray(b[t['post_indices']],float);delta=post-pre
        for mode,x in [('baseline',pre),('change',delta),('baseline_plus_change',np.c_[pre,delta])]:
            for clinical in (False,True) if not classes else (False,):
                shape=(3,n,len(classes)) if classes else (3,n)
                pred={k:np.full(shape,np.nan) for k in ('fixed5','inner_selected')};selected=[]
                for f in t['splits']:
                    tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')];s=f['seed']
                    op=project_k(x,tr,te,5);inp=project_k(x,it,iv,5)
                    if clinical:
                        op=tuple(np.c_[c,z] for c,z in zip(ip.clinical_transform(t,tr,te),op));inp=tuple(np.c_[c,z] for c,z in zip(ip.clinical_transform(t,it,iv),inp))
                    pred['fixed5'][s,te],_=ip.tuned(y,tr,te,it,iv,op,inp,classes)
                    pred['inner_selected'][s,te],choice=selected_pca_prediction(x,y,tr,te,it,iv,classes,t if clinical else None)
                    selected.append(dict(seed=s,fold=f.get('fold'),requested_PCs=choice[0],C=choice[1],rank_cap=min(choice[0],len(it)-1,x.shape[1])))
                assert all(np.isfinite(v).all() for v in pred.values())
                pp={k:v.mean(0) for k,v in pred.items()}
                rows.append(dict(cohort=co,model=model,task=task,mode=mode,setting='clinical_plus_imaging' if clinical else 'imaging_only',subjects=t['subjects'],y=y,classes=classes,
                    predictions=pp,predictions_by_seed=pred,selected=selected,scores={k:ip.primary(y,v,classes) for k,v in pp.items()}))
    save(RUN/'pca'/model/(co+'.json'),dict(status='complete',rows=rows,parent_sha256=sha(ip.RUN/'inputs.json'),scope='Frozen; fixed5 versus joint inner-validation PCA/ridge selection; unchanged outer tests; no encoder fitting.'))

def pca_inference(model,co):
    from intervention_prepost_inference import bootstrap_metrics,swaps
    rows=[]
    for r in read(RUN/'pca'/model/(co+'.json'))['rows']:
        y=np.asarray(r['y']);a,b=[np.asarray(r['predictions'][k]) for k in ('fixed5','inner_selected')];classes=r['classes'];metric='auc' if classes else 'rho'
        info={k:r[k] for k in ('cohort','model','task','mode','setting')};key=json.dumps(info,sort_keys=True)
        rng=np.random.default_rng(seed(key));diff=[]
        for start in range(0,5000,128):
            w=rng.multinomial(len(y),np.full(len(y),1/len(y)),size=min(128,5000-start))
            diff.extend(bootstrap_metrics(y,b,w,classes)[metric]-bootstrap_metrics(y,a,w,classes)[metric])
        stat=swaps(y,a,b,classes,seed(key+'swap'))
        rows.append(dict(info,metric=metric,reference='fixed5',candidate='inner_selected',objective='frozen',
            family='PCA_dimension_'+r['task']+'_'+r['setting'],reference_score=ip.primary(y,a,classes),candidate_score=ip.primary(y,b,classes),
            delta=ip.primary(y,b,classes)-ip.primary(y,a,classes),ci95=np.nanquantile(diff,[.025,.975]),n_subjects=len(y),bootstrap_requested=5000,permutations_requested=20000,**stat))
    save(RUN/'pca_inference'/model/(co+'.json'),dict(status='complete',rows=rows))

def diagnostic(model,task):
    from paper_robustness import assemble_labels
    from fm_readout_v4.data import dataset
    from fm_matched_v3.probes import TuckerBlock
    from transfer_baseline import fc_feature
    m,_=assemble_labels('nacc');lab=sf.labels('nacc');assert m.scan.tolist()==lab.scan.tolist()
    r=next(t for t in core.templates('nacc') if t['task']==task);ix=lab.set_index('scan').index.get_indexer(r['scan_ids']);m=m.iloc[ix].reset_index(drop=True)
    cols=[k for k in ('age','sex','education','site','native_tr','native_frames') if k in m and m[k].notna().any()]
    y=np.asarray(r['y']);x=None;tucker=None
    if model.startswith(('a424_','sch450_')):
        xx,_=dataset('nacc','brainlm13' if model.startswith('a424') else 'brainjepa');xx=xx.numpy()[ix]
        if model.endswith('Tucker'):tucker=TuckerBlock(xx,64)
        else:x=fc_feature(xx,None)
    else:x=np.asarray(core.feature(state(model),'nacc')[ix],float)
    prep={k:{s:[] for s in SEEDS} for k in ('covariates','imaging','combined')}
    for f in r['split_audit']:
        tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')];pairs=[]
        for train,test in ((tr,te),(it,iv)):
            c=covariates(m,cols,train,test)
            a,b=tucker.fit_pair(train,test) if tucker else (x[train],x[test]);a,b=exact_project(a,b,100)
            pairs.append(dict(covariates=c,imaging=(a,b),combined=(np.c_[c[0],a],np.c_[c[1],b])))
        for kind in prep:prep[kind][f['seed']].append((tr,te,pairs[0][kind],it,iv,pairs[1][kind]))
    pp={k:np.mean([predict_prepared(v[s],y,binary=True) for s in SEEDS],axis=0) for k,v in prep.items()};rows=[]
    for ref,cand in [('covariates','combined'),('imaging','combined')]:
        stat=paired(y,pp[ref],pp[cand],r['subjects'],metric='auc',seed=seed(model+task+ref))
        rows.append(dict(model=model,cohort='nacc',task=task,objective='frozen',reference=ref,candidate=cand,family='NACC_diagnosis_'+task+'_'+ref,**stat))
    save(RUN/'diagnosis'/model/(task+'.json'),dict(status='complete',rows=rows,subjects=r['subjects'],scan_ids=r['scan_ids'],y=y,predictions=pp,
        scores={k:score(y,p,True) for k,p in pp.items()},covariates=cols,split_audit=r['split_audit'],
        missing='Numeric median imputation fitted independently within each training partition; missing categorical level and unknown-site-safe coding. No diagnosis imputation.',
        interpretation='Covariate prediction and incremental information; not removal of all confounding or causal attribution.'))

def fc_features(kind,co):
    from fm_readout_v4.data import dataset
    from new_fm.data import dataset as new_dataset
    from transfer_baseline import fc_feature
    if kind in ('a424','sch450'):
        xx,m=dataset(co,'brainlm13' if kind=='a424' else 'brainjepa');x=xx.numpy();valid=None
    else:
        xx,m=new_dataset(co,kind);length=100 if kind=='semantoks' else 160;x=xx[:,:450,:length].numpy();valid=np.minimum(m.resampled_frames.to_numpy(),length)
    f=fc_feature(x,valid)
    path=RUN/'input_fc'/kind/(co+'.npy');path.parent.mkdir(parents=True,exist_ok=True);np.save(path,f)
    save(path.with_suffix('.json'),dict(status='complete',kind=kind,cohort=co,scan_ids=m.scan.tolist(),sha256=sha(path),
        protocol='Fisher-z FC from same temporal input window as encoder. New-model temporal zero padding excluded; absent Semantoks cerebellum excluded. Original canonical centered zero padding retained (does not alter Pearson FC).',
        duration_frames=x.shape[-1],valid_frames=valid))

def load_fc(kind,co):
    p=RUN/'input_fc'/kind/(co+'.npy');stamp=read(p.with_suffix('.json'));assert stamp['sha256']==sha(p)
    return np.load(p,mmap_mode='r')

def input_comparison(model,co):
    configure();m=sf.labels(co);x=core.feature(state(model),co);fc=load_fc(family(model),co);rows=[];scores=[]
    for r in core.templates(co):
        ix=m.set_index('scan').index.get_indexer(r['scan_ids']);y=np.asarray(r['y']);binary=r['metrics']['sf100_fc']['metric']=='auc';pp={}
        for key,values in [('FM',x),('FC',fc)]:
            prep,_=core.prepare(np.asarray(values[ix],float),y,r['subjects'],m,ix,r['covariates'],r['split_audit'])
            pp[key]=np.mean([predict_prepared(prep[s],y,binary=binary) for s in SEEDS],axis=0)
        stat=paired(y,pp['FC'],pp['FM'],r['subjects'],metric='auc' if binary else 'rho',seed=seed(model+co+r['task']+'contextFC'))
        info=dict(model=model,cohort=co,task=r['task'],objective='frozen',reference=family(model)+'_same_window_FC',candidate='FM',family='same_window_'+r['task'])
        rows.append(dict(info,**stat));scores.append(dict(info,n=len(y),subjects=r['subjects'],scan_ids=r['scan_ids'],y=y,predictions=pp,reference_score=stat['reference_score'],candidate_score=stat['candidate_score']))
    save(RUN/'input_comparison'/model/(co+'.json'),dict(status='complete',rows=rows,scores=scores,protocol='Both probes refitted under identical training-only exact PCA, samples, splits, covariates and hyperparameter grid; supplementary sensitivity, original primary scores preserved.'))

def identity(model,co):
    import sample_alignment as sa
    import readout_sensitivity as rs
    subjects,(ca,ia),(cb,ib)=sa.paired_rows(co)
    def verified_features(dataset):
        if dataset=='nacc_repeated' and model in MODELS[:4]:
            # The scan-level clinical audit omits repeated NACC visits. These
            # arrays were independently registered by the expanded task audit.
            path=u.feature_folder(state(model))/(dataset+'.npy')
            audit=read(ROOT/'runs/paper_task_expansion_v1/input_audit.json')
            if audit['files_sha256'].get(str(path.relative_to(ROOT)))!=sha(path):
                raise ValueError('Repeated-visit feature differs from registered expansion input')
            x=np.load(path,mmap_mode='r')
            if x.ndim!=2 or len(x)!=len(u.metadata(dataset)) or not np.isfinite(x).all():
                raise ValueError('Invalid registered repeated-visit features')
            return x
        return core.feature(state(model),dataset)
    a=verified_features(ca)[ia];b=verified_features(cb)[ib]
    fa=load_fc(family(model),ca)[ia];fb=load_fc(family(model),cb)[ib];rows=[];scores=[]
    for mode in ('native','projected128'):
        sim=u.projected_similarity(a,b) if mode=='projected128' else rs.similarity(a,b)
        ref=u.projected_similarity(fa,fb) if mode=='projected128' else rs.similarity(fa,fb)
        p,q=rs.identification_scores(sim),rs.identification_scores(ref)
        for metric in ('top1','I_diff'):
            stat=rs.paired_identification(q['per_subject_'+metric],p['per_subject_'+metric],seed(model+co+mode+metric))
            rows.append(dict(model=model,cohort=co,task='identification',metric=metric,mode=mode,objective='frozen',reference=family(model)+'_same_window_FC',candidate='FM',family='same_window_identity_'+mode+'_'+metric,**stat))
        scores.append(dict(mode=mode,FM=p,FC=q))
    save(RUN/'identity'/model/(co+'.json'),dict(status='complete',subjects=subjects,rows=rows,scores=scores,absolute_null=False,conditional_on_gallery=True))

def context(kind,co,model=None):
    configure()
    if kind=='probe':core.local(state(model),co,context=True)
    elif kind=='references':core.references(co,context=True)
    elif kind=='inference':core.local_inference(model,co,context=True)
    else:raise ValueError(kind)

def variant_finish(model):
    """Reuse the existing scientific family mapping for both additional checkpoints."""
    import joint_inference as p
    variant='mae' if model=='fmrilm_mae' else 'vq-domain'
    directory=ROOT/'runs/fmrilm_variants_v1'/variant/'reports/all'
    cache=RUN/'variant_cache'/model;cache.mkdir(parents=True,exist_ok=True)
    clinical=p.RUN/'original_adapted_clinical_inference.json'
    def archived_clinical():return clinical,read(clinical)['rows']
    with patch.multiple(p,NEW=directory,RUN=cache,ADDED=('fmrilm',),original_adapted_clinical=archived_clinical):
        allrows=p.build_rows()
    rows=[]
    for r in allrows:
        path=ROOT/r['input_file_provenance']
        if not (path.is_relative_to(directory) or path.is_relative_to(cache)):continue
        if r['model']!='fmrilm' or r['objective']=='frozen':continue
        r=dict(r);panel=r['panel'];r['model']=model
        if r['candidate']=='fmrilm':r['candidate']=model
        r['family']=r['family'].removeprefix(panel+'::')
        p.add(rows,r,panel,path,r['input_row'])
    save(RUN/'variant_finish'/(model+'.json'),dict(status='complete',rows=rows,parent_sha256=sha(directory/'complete.json')))

def adapted_supplements(model):
    """Three-seed prediction ensembles, independent-window physiology and recall."""
    configure();rows=[];scores=[];states=list(u.states(model));atlas='a424' if model.startswith('brainlm') else 'sch450'
    groups=defaultdict(list)
    for s in states:
        if s[2]!='frozen':groups[s[:4]].append(s)
    for co in ('psopt','breathe','hrv'):
        base=ROOT/'runs/benchmark_gaps_v1/physiology/evaluation'
        frozen=read(base/u.tag(*state(model))/(co+'.json'))['rows']
        controls=read(base/'references'/(co+'.json'))['rows']
        for key,ss in groups.items():
            files=[base/u.tag(*s)/(co+'.json') for s in ss];rs=[read(p)['rows'] for p in files]
            for f in frozen:
                rr=[next(r for r in x if r['scope']==f['scope']) for x in rs]
                ref=next(x for x in controls if x['scope']==f['scope'])
                for r in [*rr,ref]:
                    for field in ('subjects','scans','y'):assert r[field]==f[field]
                pp=np.mean([r['predictions']['FM'] for r in rr],0)
                info=dict(model=model,source=key[0],objective=key[2],target=key[3],cohort=co,task='heart_or_pulse_rate',scope=f['scope'],candidate='FM',metric='rho')
                scores.append(dict(info,score=score(f['y'],pp,False),n=len(pp)))
                for reference,a in [('frozen',f['predictions']['FM']),('nuisance',ref['predictions']['nuisance']),(atlas+'_FC',ref['predictions'][atlas+'_FC'])]:
                    fam='independent_window_'+f['scope']+'_'+key[2]+'_'+key[3]+'_'+('adaptation' if reference=='frozen' else 'FC' if reference.endswith('_FC') else 'nuisance')
                    stat=core.paired_test(f['y'],a,pp,f['subjects'],'rho',json.dumps([info,reference],sort_keys=True))
                    rows.append(dict(info,reference=reference,family=fam,panel='physiology_common_window',**{k:v for k,v in stat.items() if k!='metric'}))
    if model in MODELS[4:]:
        parent=ROOT/'runs/unified_supplements_v1/recall';f=read(parent/u.tag(*state(model))/'result.json')
        fc=read(ROOT/'runs/hrv_recall_external_v1/sf100_fc.json')
        for key,ss in groups.items():
            rr=[read(parent/u.tag(*s)/'result.json') for s in ss]
            for r in [*rr,fc]:
                for field in ('subjects','scan_ids','y'):assert r[field]==f[field]
            pp=np.mean([r['prediction'] for r in rr],0)
            for ref,a in [('frozen',f['prediction']),('sf100_fc',fc['prediction'])]:
                info=dict(model=model,source=key[0],objective=key[2],target=key[3],cohort='hrv_pre',task='MEM_delayed_recall',reference=ref,candidate='adapted')
                stat=core.paired_test(f['y'],a,pp,f['subjects'],'rho',json.dumps(info,sort_keys=True)+'recall')
                rows.append(dict(info,panel='loso_adaptation',family='HRV_recall_'+key[2]+'_'+key[3]+'_'+ref,**stat))
    save(RUN/'adapted_supplements'/(model+'.json'),dict(status='complete',rows=rows,scores=scores))

def collect_sensitivity():
    import joint_inference as p
    rows=[];inputs={}
    sources=[('pca_inference','PCA_sensitivity'),('diagnosis','diagnostic_sensitivity'),('input_comparison','same_window_FC'),('identity','same_window_identification'),('context_inference','complete_context')]
    for folder,panel in sources:
        for path in sorted((RUN/folder).glob('*/*.json')):
            inputs[str(path)]=sha(path)
            for i,r in enumerate(read(path)['rows']):p.add(rows,r,panel,path,i,protocol='reviewer_sensitivity_v1')
    return rows,inputs

def export_tables(rows,folder,label,extra=None):
    import joint_inference as p
    folder.mkdir(parents=True,exist_ok=True)
    retained,duplicates=p.deduplicate(rows);df=p.adjust(retained)
    p.MODELS=MODELS;consistency=p.consistency(df)
    # Existing consistency labels are descriptive, not absolute working/no-working decisions.
    consistency['criterion_status']='Retrospective; no absolute prediction null added; independent cohorts only.'
    for _,g in df.groupby('family'):
        vals=g.raw_p.to_numpy(float);order=np.argsort(vals);expected=np.empty(len(vals));expected[order]=np.minimum(1,np.maximum.accumulate(vals[order]*(len(vals)-np.arange(len(vals)))))
        np.testing.assert_allclose(expected,g.holm_p_six_models,atol=1e-14,rtol=0)
    df=df.rename(columns={k:k.replace('six_models','joint') for k in df if 'six_models' in k})
    assert df.hypothesis_id.is_unique
    df.to_csv(folder/'Comparisons.csv',index=False);consistency.to_csv(folder/'Consistency.csv',index=False)
    pd.DataFrame(duplicates).to_csv(folder/'Duplicate_audit.csv',index=False)
    families=df.groupby('family').agg(tests=('raw_p','size'),models=('model',lambda x:';'.join(sorted(set(x))))).reset_index()
    with pd.ExcelWriter(folder/(label+'.xlsx'),engine='xlsxwriter') as w:
        df.to_excel(w,sheet_name='All comparisons',index=False);families.to_excel(w,sheet_name='Correction families',index=False);consistency.to_excel(w,sheet_name='Consistency',index=False)
        for panel,g in df.groupby('panel'):g.to_excel(w,sheet_name=panel[:31],index=False)
        for file in ('task_sample_counts','padding_summary','nacc_group_characteristics','nacc_site_class_counts'):
            pd.read_csv(RUN/(file+'.csv')).to_excel(w,sheet_name=file[:31],index=False)
        pd.DataFrame(read(RUN/'context/complete.json')['coverage']).to_excel(w,sheet_name='Complete-context coverage',index=False)
        if extra:
            for name,t in extra.items():t.to_excel(w,sheet_name=name[:31],index=False)
    # Compact supplementary plots, leaving manuscript figures untouched.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sub=df[df.panel.isin(['PCA_sensitivity','diagnostic_sensitivity'])]
    if len(sub):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,panel in zip(axes,['PCA_sensitivity','diagnostic_sensitivity']):
            t=sub[sub.panel.eq(panel)]
            for i,(model,g) in enumerate(t.groupby('model')):ax.scatter(np.full(len(g),i),g.delta,s=12,alpha=.5)
            names=sorted(t.model.unique());ax.set_xticks(range(len(names)),names,rotation=70,ha='right',fontsize=7);ax.axhline(0,color='black',lw=.6);ax.set_title(panel.replace('_',' '));ax.set_ylabel('Paired performance difference')
        fig.tight_layout();fig.savefig(folder/'Supplementary_sensitivity_overview.pdf');plt.close(fig)
    return dict(status='complete',comparisons=len(df),families=len(families),models=sorted(set(df.model)&set(MODELS)),absolute_nulls=False,manuscript_editing=False,
        outputs_sha256={str(x):sha(x) for x in folder.iterdir() if x.is_file()})

def sensitivity_package():
    rows,inputs=collect_sensitivity();d=export_tables(rows,OUT/'sensitivity','Reviewer_sensitivity');d['inputs_sha256']=inputs;save(RUN/'sensitivity_package.json',d)

def final_package():
    import joint_inference as p
    rows=pd.read_csv(ROOT/'runs/paper_priority_v1/release/Comparisons.csv').replace({np.nan:None}).to_dict('records')
    inputs={}
    for model in MODELS[-2:]:
        path=RUN/'variant_finish'/(model+'.json');rows+=read(path)['rows'];inputs[str(path)]=sha(path)
    for model in MODELS:
        path=RUN/'adapted_supplements'/(model+'.json');inputs[str(path)]=sha(path)
        for i,r in enumerate(read(path)['rows']):p.add(rows,r,r['panel'],path,i,protocol='independent_windows_shared_refitted_probe' if r['panel']=='physiology_common_window' else 'primary_scan_level')
    # Include completed external atlas-FC/Tucker contrasts for added configurations.
    for path in sorted((ROOT/'runs/benchmark_gaps_v1/reference_comparisons').glob('*.json')):
        inputs[str(path)]=sha(path)
        for i,r in enumerate(read(path)['rows']):
            if r['objective']=='frozen':continue
            p.add(rows,r,'atlas_adapted_reference',path,i,family='atlas_adapted_'+r['task']+'_'+r['objective']+'_'+r['target']+'_'+r['scope'],protocol='shared_exact_local_probe')
    sens,files=collect_sensitivity();rows+=sens;inputs.update(files)
    d=export_tables(rows,OUT/'joint','All_configuration_SI');d['inputs_sha256']=inputs
    d['coverage']='Eight frozen and adapted configurations, source and external panels; bounded reviewer sensitivities are frozen-only.'
    save(RUN/'final_package.json',d)
