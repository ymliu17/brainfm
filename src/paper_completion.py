"""Post hoc paired inference and matched frozen fusion, using existing encoders."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from fm_matrix.data import ROOT, save, sha
from fm_readout_v4.data import metadata, endpoints, DATA
from fm_readout_v4.evaluate import clinical_features, tag
from fm_matched_v3.probes import oof, oof_predictions
from benchmark_evaluation import grouped_splits, predict_prepared, score
from paper_paired_stats import paired, holm

RUN = ROOT/'runs/paper_completion_v1'
MODELS = ('brainlm13','brainlm111','brainlm650','brainjepa')
COHORTS = ('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post')
TRIALS = ('act','cogte','breathe','psopt','hrv_pre')
SEEDS = (42,43,44)


def read(p):
    return json.loads(Path(p).read_text())


def states():
    for model in MODELS:
        yield ('none',model,'frozen','none',0)
        for source in ('adni','nacc'):
            for objective,target in [('supervised','MEM'),('supervised','EF'),('native_ssl','none')]:
                for seed in SEEDS:
                    yield source,model,objective,target,seed


def feature_dir(state):
    source,model,objective,target,seed=state
    version = ('fm_cls_posttraining_v5' if model=='brainlm13' and objective=='supervised'
               else 'fm_readout_v4' if model=='brainlm13' else 'fm_posttraining_matrix_v2')
    return ROOT/'runs'/version/'features'/tag(*state)


def primary_mask(m,cohort):
    keep=m.eligible.to_numpy(bool).copy()
    if cohort in TRIALS[:-1]:keep &= m.visit.eq('T1').to_numpy()
    return keep


def verify_input_files(paths):
    registered=read(RUN/'input_audit.json')['files_sha256']
    for p in paths:
        if str(p) not in registered or sha(p)!=registered[str(p)]:
            raise ValueError('Registered analysis input changed: '+str(p))


def row(cohort,task,method,m,ix,y,prediction,**extra):
    return dict(cohort=cohort,task=task,method=method,scan_ids=m.iloc[ix].scan.astype(str).tolist(),
        subjects=m.iloc[ix].subject.astype(str).tolist(),y=np.asarray(y,float),prediction=np.asarray(prediction,float),
        n=len(ix),metric='rho',score=score(y,prediction,False),**extra)


def normalize_cached(r,m,cohort,path):
    task=r['task'];keep=primary_mask(m,cohort)
    if task.endswith('_baseline') and task.split('_')[0] in ('MEM','EF'):
        y=m[task.split('_')[0]].to_numpy(float)
    elif task.endswith('_followup') and task.split('_')[0] in ('MEM','EF'):
        y=m['follow_'+task.split('_')[0]].to_numpy(float)
        keep &= np.isfinite(clinical_features(m,cohort)).all(1)
    elif task=='EF_concurrent_post':y=m.EF.to_numpy(float)
    elif task.startswith('MEM_delayed_recall'):
        y=m.MEM_delayed_recall.to_numpy(float)
        if r['method'].startswith('clinical'):keep &= np.isfinite(clinical_features(m,cohort)).all(1)
    else:return None
    ix=np.flatnonzero(keep & np.isfinite(y))
    if m.iloc[ix].subject.astype(str).tolist()!=[str(s) for s in r['subjects']] or not np.allclose(y[ix],r['y']):
        raise ValueError('Cached prediction/metadata row mismatch: '+str(path)+' '+task)
    return row(cohort,task,r['method'],m,ix,y[ix],r['prediction'],reused_from=str(path))


def local_rows(state,cohort):
    source,model,objective,target,seed=state
    m=metadata(cohort);name=tag(*state)
    candidates=[]
    if model=='brainlm13':
        version='fm_cls_posttraining_v5' if objective=='supervised' else 'fm_readout_v4'
        candidates.append(ROOT/'runs'/version/'evaluation'/name/(cohort+'_local.json'))
    else:
        candidates.append(ROOT/'runs/fm_readout_v4/evaluation'/name/(cohort+'_local.json'))
        if not cohort.startswith('hrv'):
            candidates.append(ROOT/'runs/fm_posttraining_matrix_v2/evaluation'/name/'cognitive.json')
    for p in candidates:
        if p.exists():
            rows=[]
            for r in read(p)['rows']:
                if r['cohort']==cohort and r['method'] in ('FM','clinical','clinical+FM'):
                    rr=normalize_cached(r,m,cohort,p)
                    if rr is not None:rows.append(rr)
            if rows:return rows
    # Missing 43/44 CLS-SSL probes and updated HRV composites reuse frozen features.
    f=np.load(feature_dir(state)/(cohort+'.npy'));keep=primary_mask(m,cohort);rows=[]
    for endpoint in endpoints(cohort):
        y=m[endpoint].to_numpy(float);ix=np.flatnonzero(keep & np.isfinite(y));g=m.iloc[ix].subject.to_numpy()
        if len(set(g))<20:continue
        pp=oof(f[ix],y[ix],g)
        rows.append(row(cohort,endpoint+('_concurrent_post' if cohort=='hrv_post' else '_baseline'),'FM',m,ix,y[ix],pp))
        if 'follow_'+endpoint not in m or cohort=='hrv_post':continue
        clinical=clinical_features(m,cohort);y=m['follow_'+endpoint].to_numpy(float)
        ix=np.flatnonzero(keep & np.isfinite(y) & np.isfinite(clinical).all(1));g=m.iloc[ix].subject.to_numpy()
        for label,blocks,dims in [('clinical',[clinical[ix]],[clinical.shape[1]]),('clinical+FM',[clinical[ix],f[ix]],[clinical.shape[1],5])]:
            pp=np.mean([oof_predictions(blocks,dims,y[ix],s,g) for s in (0,1,2)],axis=0)
            rows.append(row(cohort,endpoint+'_followup',label,m,ix,y[ix],pp))
    if cohort.startswith('hrv'):
        y=m.MEM_delayed_recall.to_numpy(float);ix=np.flatnonzero(keep & np.isfinite(y));g=m.iloc[ix].subject.to_numpy()
        task='MEM_delayed_recall_'+('baseline_prediction' if cohort=='hrv_pre' else 'post_scan_association')
        rows.append(row(cohort,task,'FM',m,ix,y[ix],oof(f[ix],y[ix],g)))
        if cohort=='hrv_pre':
            clinical=clinical_features(m,cohort);ix=np.flatnonzero(keep & np.isfinite(y) & np.isfinite(clinical).all(1));g=m.iloc[ix].subject.to_numpy()
            for label,blocks,dims in [('clinical',[clinical[ix]],[clinical.shape[1]]),('clinical+FM',[clinical[ix],f[ix]],[clinical.shape[1],5])]:
                pp=np.mean([oof_predictions(blocks,dims,y[ix],s,g) for s in (0,1,2)],axis=0)
                rows.append(row(cohort,task,label,m,ix,y[ix],pp))
    return rows


def axes(state):
    source,model,objective,_,_=state;f=feature_dir(state);rows=[]
    for src in (('adni','nacc') if objective=='frozen' else (source,)):
        m=metadata(src);x=np.load(f/(src+'.npy'))
        for domain in ('MEM','EF'):
            keep=primary_mask(m,src) & m[domain].notna().to_numpy()
            sc=StandardScaler().fit(x[keep]);a=sc.transform(x[keep])
            pc=PCA(n_components=min(100,len(a)-1,a.shape[1]),svd_solver='randomized',random_state=0).fit(a)
            head=Ridge(alpha=10).fit(pc.transform(a),m.loc[keep,domain])
            for co in COHORTS:
                if co in (src,'hrv_post'):continue
                tm=metadata(co);endpoint='MEM_delayed_recall' if co=='hrv_pre' and domain=='MEM' else domain
                y=tm[endpoint].to_numpy(float);ix=np.flatnonzero(primary_mask(tm,co)&np.isfinite(y))
                pp=head.predict(pc.transform(sc.transform(np.load(f/(co+'.npy'))[ix])))
                rows.append(row(co,domain+'_axis_to_'+endpoint,'source_axis',tm,ix,y[ix],pp,source=src))
    return rows


def canonical(state):
    source,_,objective,_,_=state
    verify_input_files([feature_dir(state)/(co+'.npy') for co in COHORTS]+
                      [DATA/(co+'_domains.csv' if co.startswith('hrv') else co+'.csv') for co in COHORTS])
    rows=[]
    for co in COHORTS:
        if objective!='frozen' and co==source:continue
        rows.extend(local_rows(state,co))
    save(RUN/'canonical'/(tag(*state)+'.json'),dict(status='complete',state=state,rows=rows,axes=axes(state),
        interpretation='Locally fitted probes and directly transferred axes are separate; current HRV EF/recall labels; original encoder features reused'))


def align(a,b):
    if a['subjects']!=b['subjects'] or not np.array_equal(a['y'],b['y']):
        raise ValueError('Paired participants/outcomes differ')
    if 'scan_ids' in a and 'scan_ids' in b and a['scan_ids']!=b['scan_ids']:
        raise ValueError('Paired scan order differs')


def average(rows):
    for r in rows[1:]:align(rows[0],r)
    out=dict(rows[0]);out['prediction']=np.mean([r['prediction'] for r in rows],axis=0)
    out['aggregation']='Score of predictions averaged across adaptation seeds 42/43/44 (each local probe averages split seeds 0/1/2)'
    return out


def contrast(a,b,family,reference,candidate,**extra):
    align(a,b)
    return dict(family=family,reference=reference,candidate=candidate,cohort=a['cohort'],task=a['task'],
                y=a['y'],a=a['prediction'],b=b['prediction'],subjects=a['subjects'],metric=a.get('metric','rho'),**extra)


def setting(r):
    if r['cohort']=='hrv_post' or 'delayed_recall' in r['task']:return 'hrv_exploratory'
    if 'axis_to' in r['task']:return 'direct_transfer'
    if r['task'].endswith('followup'):return 'followup'
    return 'observational' if r['cohort'] in ('adni','nacc') else 'intervention_baseline'


def infer_bundle(key,contrasts):
    results=[]
    for i,c in enumerate(contrasts):
        ident={k:v for k,v in c.items() if k not in ('y','a','b','subjects')}
        seed=int(hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:8],16)
        stat=paired(c['y'],c['a'],c['b'],c['subjects'],c['metric'],seed=seed)
        results.append(dict(**ident,statistics=stat))
        print(key,i+1,len(contrasts),c['cohort'],c['task'],stat['delta'],flush=True)
    save(RUN/'inference'/(key+'.json'),dict(status='complete',rows=results,
         inference='Post hoc conditional paired inference. Family definitions fixed before these new tests; not prospective confirmation.'))


def frozen_inference(model):
    rows=read(RUN/'canonical'/(model+'_frozen.json'))['rows'];comparisons=[]
    atlas='sch450' if model=='brainjepa' else 'a424'
    for a in rows:
        if a['method']=='clinical':continue
        domain=a['task'].split('_')[0];group=setting(a)
        for method in ('FC','FC-Tucker'):
            p=ROOT/'runs/fm_matched_v3/evaluation'/(atlas+'_'+method)/(a['cohort']+'_local.json')
            label=method if a['method']=='FM' else 'clinical+'+method
            matches=[r for r in read(p)['rows'] if r['task']==a['task'] and r['method']==label]
            if len(matches)!=1:raise ValueError('Missing matched baseline '+str(p)+' '+a['task'])
            b=normalize_cached(matches[0],metadata(a['cohort']),a['cohort'],p)
            comparisons.append(contrast(b,a,'frozen_reference_'+group+'_'+domain,label,a['method'],model=model))
        if a['method']=='clinical+FM':
            b=next(r for r in rows if r['cohort']==a['cohort'] and r['task']==a['task'] and r['method']=='clinical')
            comparisons.append(contrast(b,a,'clinical_increment_'+group+'_'+domain,'clinical','clinical+FM',model=model))
    infer_bundle('frozen_'+model,comparisons)


def adaptation_inference(model,source,objective,target):
    frozen=read(RUN/'canonical'/(model+'_frozen.json'))
    ds=[read(RUN/'canonical'/(tag(source,model,objective,target,s)+'.json')) for s in SEEDS]
    comparisons=[]
    for section in ('rows','axes'):
        def key(r):return r['cohort'],r['task'],r['method'],r.get('source')
        ref={key(r):r for r in frozen[section]}
        others=[{key(r):r for r in d[section]} for d in ds]
        if any(set(o)!=set(others[0]) for o in others):raise ValueError('Incomplete seed coverage')
        for k,r in others[0].items():
            if r['method']=='clinical':continue
            adapted=average([o[k] for o in others]);base=ref[k]
            family='adaptation_'+objective+'_'+setting(r)+'_'+r['task'].split('_')[0]
            comparisons.append(contrast(base,adapted,family,'frozen','adapted',model=model,source=source,objective=objective,adaptation_target=target,
                aggregation=adapted['aggregation']))
    infer_bundle('_'.join(['adaptation',model,source,objective,target]),comparisons)


def source_inference(model):
    comparisons=[]
    for source in ('adni','nacc'):
        for target in ('MEM','EF'):
            seedrows=[];seed_metrics=[]
            for seed in SEEDS:
                version='fm_cls_posttraining_v5' if model=='brainlm13' else 'fm_posttraining_matrix_v2'
                p=ROOT/'runs'/version/'training'/source/model/f'supervised_{target}_s{seed}_cv/complete.json'
                d=read(p);folds=d['folds']
                for fold in folds:
                    if set(fold['test_subjects']) & (set(fold['train_subjects'])|set(fold['validation_subjects'])):raise ValueError('Encoder fold leakage')
                frame=pd.DataFrame({k:np.concatenate([f[k] for f in folds]) for k in ('test_indices','test_subjects','y','frozen_prediction','adapted_prediction')}).sort_values('test_indices')
                if frame.test_indices.duplicated().any():raise ValueError('Duplicate test scans')
                seedrows.append(frame)
                seed_metrics.append(dict(seed=seed,frozen=score(frame.y,frame.frozen_prediction,False),adapted=score(frame.y,frame.adapted_prediction,False)))
            for frame in seedrows[1:]:
                if not frame[['test_indices','test_subjects','y']].reset_index(drop=True).equals(seedrows[0][['test_indices','test_subjects','y']].reset_index(drop=True)):raise ValueError('CV seed alignment')
            frame=seedrows[0];a=dict(cohort=source,task=target,subjects=frame.test_subjects.astype(str).tolist(),y=frame.y.to_numpy(),prediction=np.mean([s.frozen_prediction.to_numpy() for s in seedrows],axis=0),metric='rho')
            b=dict(a,prediction=np.mean([s.adapted_prediction.to_numpy() for s in seedrows],axis=0))
            comparisons.append(contrast(a,b,'source_supervised_'+target,'frozen','adapted',model=model,seed_metrics=seed_metrics,
                aggregation='Score of seed-averaged OOF predictions; mean seed scores provided separately',evaluation_scope='Independent fold-specific encoders'))
    infer_bundle('source_'+model,comparisons)


def demographic_inference(model):
    comparisons=[];folder=ROOT/'runs/demographic_benchmark_v1/evaluation'
    for source in ('adni','nacc'):
        for co in COHORTS:
            for task,metric in [('SEX','auc')]+([] if co.startswith('hrv') else [('AGE','rho')]):
                name=co+'_'+task+'.json';base=read(folder/(model+'_frozen')/name)
                ds=[read(folder/tag(source,model,'native_ssl','none',s)/name) for s in SEEDS]
                adapted=average(ds);scope=ds[0]['evaluation_scope']
                a=dict(base,metric=metric);b=dict(adapted,metric=metric)
                comparisons.append(contrast(a,b,'demographic_SSL_'+scope+'_'+task,'frozen','SSL',model=model,source=source,evaluation_scope=scope))
    infer_bundle('demographics_'+model,comparisons)


def project_block(train,test,k,device='cpu'):
    sc=StandardScaler().fit(train);a=sc.transform(train).astype('float64');b=sc.transform(test).astype('float64')
    k=min(k,len(a)-1,a.shape[1])
    if device=='cuda':
        import torch
        aa=torch.as_tensor(a,device='cuda');bb=torch.as_tensor(b,device='cuda')
        eigen,u=torch.linalg.eigh(aa@aa.T)
        eigen=eigen[-k:].clamp(min=0).flip(0);u=u[:,-k:].flip(1)
        tolerance=torch.finfo(aa.dtype).eps*max(aa.shape)*eigen.max().clamp(min=1)
        inv=torch.where(eigen>tolerance,1/torch.sqrt(eigen.clamp(min=tolerance)),torch.zeros_like(eigen))
        v=(aa.T@u)*inv
        return (aa@v).cpu().numpy(),(bb@v).cpu().numpy()
    pc=PCA(n_components=k,svd_solver='full').fit(a)
    return pc.transform(a),pc.transform(b)


def fusion_sample(cohort,endpoint):
    m=metadata(cohort);keep=primary_mask(m,cohort)
    if cohort in ('adni','nacc'):
        y=m[endpoint].to_numpy(float);clinical=None
    else:
        y=m['follow_'+endpoint].to_numpy(float);clinical=clinical_features(m,cohort)
        keep &= np.isfinite(clinical).all(1)
    ix=np.flatnonzero(keep&np.isfinite(y))
    return m,ix,y[ix],m.iloc[ix].subject.to_numpy(),None if clinical is None else clinical[ix]


def projections(atlas,cohort,device='cuda'):
    cache=ROOT/'runs/fm_posttraining_matrix_v2/baselines'/atlas/(cohort+'_FC.npy')
    verify_input_files([cache,DATA/(cohort+'_domains.csv' if cohort.startswith('hrv') else cohort+'.csv')])
    fc=np.load(cache,mmap_mode='r');destination=RUN/'projections'/atlas/cohort;destination.mkdir(parents=True,exist_ok=True)
    for endpoint in endpoints(cohort):
        m,ix,y,g,c=fusion_sample(cohort,endpoint);x=np.asarray(fc[ix],dtype='float64');budget=100 if c is None else 10
        arrays={};splits=[]
        for seed in (0,1,2):
            for fold,(tr,te) in enumerate(grouped_splits(y,g,seed)):
                it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=g[tr]));it,iv=tr[it],tr[iv]
                key=f's{seed}f{fold}'
                for label,a,b in [('outer',tr,te),('inner',it,iv)]:
                    aa,bb=project_block(x[a],x[b],budget,device);arrays[key+label+'a']=aa;arrays[key+label+'b']=bb
                splits.append(dict(seed=seed,fold=fold,key=key,tr=tr,te=te,it=it,iv=iv))
                print('FC projection',atlas,cohort,endpoint,seed,fold,flush=True)
        path=destination/(endpoint+'.npz');tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**arrays);tmp.replace(path)
        save(destination/(endpoint+'.json'),dict(status='complete',cohort=cohort,endpoint=endpoint,indices=ix,subjects=g,y=y,scan_ids=m.iloc[ix].scan.tolist(),splits=splits,budget=budget,
            fc_sha256=sha(cache),labels_sha256=sha(DATA/(cohort+'_domains.csv' if cohort.startswith('hrv') else cohort+'.csv')),
            numerical_method='Exact standardized dual PCA on GPU in float64; fitted separately in inner/outer training partitions'))
    save(destination/'complete.json',dict(status='complete',atlas=atlas,cohort=cohort))


def fusion(model,cohort):
    atlas='sch450' if model=='brainjepa' else 'a424';folder=RUN/'projections'/atlas/cohort;results=[];contrasts=[]
    verify_input_files([feature_dir(('none',model,'frozen','none',0))/(cohort+'.npy'),
                       DATA/(cohort+'_domains.csv' if cohort.startswith('hrv') else cohort+'.csv')])
    fm=np.load(feature_dir(('none',model,'frozen','none',0))/(cohort+'.npy'))
    for endpoint in endpoints(cohort):
        m,ix,y,g,c=fusion_sample(cohort,endpoint);meta=read(folder/(endpoint+'.json'));fc=np.load(folder/(endpoint+'.npz'))
        if meta['indices']!=ix.tolist() or meta['subjects']!=g.tolist() or not np.array_equal(meta['y'],y):raise ValueError('Projection population mismatch')
        x=fm[ix];budget=meta['budget'];predictions={}
        recipes={'FC':(budget,0),'FM':(0,budget),'FC+FM':(budget//2,budget//2)}
        if c is not None:recipes={'clinical':(0,0),'clinical+FC':(10,0),'clinical+FM':(0,10),'clinical+FC+FM':(5,5),'clinical+FC5':(5,0)}
        prepared={name:{s:[] for s in (0,1,2)} for name in recipes}
        audit=[]
        for split in meta['splits']:
            tr,te,it,iv=[np.asarray(split[k]) for k in ('tr','te','it','iv')];key=split['key'];seed=split['seed']
            outer=project_block(x[tr],x[te],budget);inner=project_block(x[it],x[iv],budget)
            ct=[]
            for aa,bb in [(tr,te),(it,iv)]:
                if c is None:ct.append((np.empty((len(aa),0)),np.empty((len(bb),0))))
                else:
                    sc=StandardScaler().fit(c[aa]);ct.append((sc.transform(c[aa]),sc.transform(c[bb])))
            for name,(kf,km) in recipes.items():
                pairs=[]
                for i,(label,ff) in enumerate([('outer',outer),('inner',inner)]):
                    pairs.append(tuple(np.column_stack([ct[i][j],fc[key+label+('a' if j==0 else 'b')][:,:kf],ff[j][:,:km]]) for j in (0,1)))
                prepared[name][seed].append((tr,te,pairs[0],it,iv,pairs[1]))
            audit.append(dict(seed=seed,train_subjects=g[tr],test_subjects=g[te],inner_train_subjects=g[it],validation_subjects=g[iv]))
        task=endpoint+('_baseline' if c is None else '_followup')
        rows={}
        for name,byseed in prepared.items():
            pp=[predict_prepared(byseed[s],y) for s in (0,1,2)]
            rows[name]=row(cohort,task,name,m,ix,y,np.mean(pp,axis=0),predictions_by_seed=pp,dimensions=recipes[name])
        results.extend(rows.values())
        if c is None:
            pairs=[('FC','FM'),('FC','FC+FM')]
        else:pairs=[('clinical+FC','clinical+FC+FM'),('clinical','clinical+FC+FM'),('clinical+FC5','clinical+FC+FM')]
        for a,b in pairs:
            kind='dimension_matched' if a in ('FC','clinical+FC') else 'clinical_reference' if a=='clinical' else 'additive_sensitivity'
            contrasts.append(contrast(rows[a],rows[b],'fusion_'+kind+'_'+setting(rows[a])+'_'+endpoint,a,b,model=model,dimensions=recipes))
        save(RUN/'fusion'/model/(cohort+'_'+endpoint+'.json'),dict(status='complete',rows=list(rows.values()),split_audit=audit,
             protocol='Frozen native readout. Shared participant splits and training-only PCA; total imaging dimensions 100 observational or 10 follow-up; independent regularization tuning per recipe.'))
    infer_bundle('fusion_'+model+'_'+cohort,contrasts)
    save(RUN/'fusion'/model/(cohort+'_complete.json'),dict(status='complete',rows=results))


def audit():
    files={};features=[]
    for co in COHORTS:
        m=metadata(co)
        if m.scan.duplicated().any():raise ValueError('Duplicate scan IDs')
        path=DATA/(co+'_domains.csv' if co.startswith('hrv') else co+'.csv');files[str(path)]=sha(path)
    for state in states():
        folder=feature_dir(state);stamp=read(folder/'complete.json');files[str(folder/'complete.json')]=sha(folder/'complete.json')
        model=state[1];readout=stamp.get('encoder',{}).get('readout')
        expected='mean_final' if model=='brainjepa' else 'CLS_final'
        if readout!=expected:raise ValueError('Readout mismatch '+str(folder)+' '+str(readout))
        for co in COHORTS:
            p=folder/(co+'.npy');z=np.load(p,mmap_mode='r');m=metadata(co)
            if z.ndim!=2 or len(z)!=len(m) or not np.isfinite(z).all():raise ValueError('Invalid embedding '+str(p))
            files[str(p)]=sha(p)
        features.append(dict(state=state,readout=readout,path=str(folder)))
    import torch
    from transfer_baseline import fc_feature
    from fm_matrix.data import DATA as ORIGINAL_DATA,RUN as ORIGINAL_RUN
    intake=read(ORIGINAL_RUN/'intake.json')
    baseline_checks=[]
    for atlas in ('a424','sch450'):
        for co in ('adni','nacc',*TRIALS):
            p=ORIGINAL_RUN/'baselines'/atlas/(co+'_FC.npy');z=np.load(p,mmap_mode='r')
            tensor=ORIGINAL_DATA/(co+'_'+atlas+'.pt')
            if sha(tensor)!=intake['counts'][co][atlas+'_sha256']:raise ValueError('Original time-series file changed')
            x=torch.load(tensor,weights_only=True,mmap=True)
            sample=np.unique([0,len(x)//2,len(x)-1])
            if len(z)!=len(x) or not np.isfinite(z).all():raise ValueError('Invalid FC cache')
            np.testing.assert_allclose(z[sample],fc_feature(x[sample].numpy(),None),rtol=1e-6,atol=1e-6)
            files[str(p)]=sha(p)
            baseline_checks.append(dict(atlas=atlas,cohort=co,shape=list(z.shape),sampled_scan_indices=sample.tolist(),sampled_fc_recalculation='passed'))
    p=ROOT/'data/HRV_biofeedback/label/phenotype.zip'
    with zipfile.ZipFile(p) as z:
        d=pd.read_csv(z.open('Demographics.tsv'),sep='\t');older=d[d.age_group.eq('OA')]
        age=dict(source=str(p),member='Demographics.tsv',n_older=len(older),released_age_categories=older.age_category.value_counts().to_dict(),
                 exact_age_available=False,interpretation='Older-adult ages released as intervals; NIH ageScore is age-adjusted cognitive score, not chronological age. No midpoint imputation.')
    save(RUN/'input_audit.json',dict(status='complete',files_sha256=files,features=features,fc_checks=baseline_checks,hrv_age=age,
        exclusions='Existing common-atlas primary inclusion rules retained. No additional QC exclusions.',
        outstanding='Pretraining overlap, spatial registration and full FD coverage are not resolved by this analysis.'))


def report():
    rows=[]
    for p in sorted((RUN/'inference').glob('*.json')):
        for r in read(p)['rows']:
            stat=r.pop('statistics')
            for key in r.keys() & stat.keys():
                if r[key] != stat[key]:
                    raise ValueError(f'Conflicting report field {key} in {p}')
            rows.append({**r,**stat,'input_file':str(p)})
    families={}
    for r in rows:families.setdefault(r['family'],[]).append(r)
    for name,group in families.items():
        corrected=holm([r['permutation_p'] for r in group])
        for r,p in zip(group,corrected):r.update(holm_p=p,family_size=len(group),minimum_attainable_holm_p=len(group)/20001)
        if len(group)/20001>.05:raise ValueError('Permutation resolution insufficient for family '+name)
    save(RUN/'report.json',dict(status='complete',n_comparisons=len(rows),families={k:len(v) for k,v in families.items()},rows=rows,
        scope='Post hoc conditional inference for current readouts and labels; frozen fusion adds no encoder training.',
        interpretation='Two-sided participant-cluster prediction-swap tests conditional on fitted predictions; marginal percentile CIs, not simultaneous CIs. Holm across predefined scientific blocks, not the whole benchmark. No population-level training-algorithm superiority claim.',
        hrv_age=read(RUN/'input_audit.json')['hrv_age']))
    pd.json_normalize(rows).to_csv(RUN/'paired_comparisons.csv',index=False)
    for prefix in ('source','frozen','adaptation','demographic','fusion'):
        pd.json_normalize([r for r in rows if r['family'].startswith(prefix)]).to_csv(RUN/(prefix+'_comparisons.csv'),index=False)
