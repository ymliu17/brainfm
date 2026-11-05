"""Final common-protocol probes, temporal eligibility, and support tests.

The support panel uses label-independent participant folds for both its observed
and permuted scores. Its P values must not be attached to primary stratified-CV
scores. Encoders are never refitted here. Parent runs remain immutable.
"""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
from fm_matrix.data import ROOT,save,sha
from fm_readout_v4.evaluate import tag
from fm_readout_v4.data import metadata
from benchmark_evaluation import grouped_splits,predict_prepared,score,GRID
from paper_task_expansion import exact_project
import sf100_benchmark as sf
import unified_supplements as u

RUN=ROOT/'runs/benchmark_completion_v1'
COHORTS=('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post')
METHODS=(*u.CONFIGS,'sf100_fc','braingfm')
SEEDS=(0,1,2)
NULL_BUDGET=1999

def read(p):return json.loads(Path(p).read_text())
def stable_seed(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)

def templates(co):
    return [read(p) for p in sorted((sf.RUN/'evaluation'/co).glob('*.json'))
            if p.name!='complete.json' and read(p).get('status')=='complete' and 'split_audit' in read(p)]

def feature(state,co):
    if state[1] in ('sf100_fc','braingfm'):return sf.features(co,False)[state[1]]
    return u.feature(state,co)

def label_independent_splits(groups,seed):
    g=np.asarray(groups);unique=np.unique(g)
    if len(unique)<5:raise ValueError('Five participant folds require five participants')
    order=np.random.default_rng(seed).permutation(unique)
    for fold,subjects in enumerate(np.array_split(order,5)):
        te=np.flatnonzero(np.isin(g,subjects));tr=np.flatnonzero(~np.isin(g,subjects))
        a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=g[tr]))
        yield dict(seed=seed,fold=fold,train=tr,test=te,inner_train=tr[a],validation=tr[b])

def blocks(groups):
    groups=np.asarray(groups);out=defaultdict(list)
    for subject in np.unique(groups):
        ix=np.flatnonzero(groups==subject);out[len(ix)].append(ix)
    return dict(out)

def permute_participants(y,groups,rng):
    """Exchange whole visit trajectories only between equal-size participant blocks."""
    y=np.asarray(y);result=y.copy()
    for rows in blocks(groups).values():
        for target,source in zip(rows,rng.permutation(len(rows))):result[target]=y[rows[source]]
    return result

def prepare(x,y,g,m,ix,covs,split_audit=None,k=None):
    if k is None:k=5 if covs else 100
    audits=split_audit if split_audit is not None else [f for s in SEEDS for f in label_independent_splits(g,s)]
    prepared={s:[] for s in SEEDS}
    for f in audits:
        tr,te,it,iv=[np.asarray(f[key],int) for key in ('train','test','inner_train','validation')]
        if set(np.asarray(g)[tr])&set(np.asarray(g)[te]) or set(np.asarray(g)[it])&set(np.asarray(g)[iv]):raise ValueError('Participant overlap')
        ca,cb=sf.covariate_pair(m,covs,ix[tr],ix[te]);ci,cj=sf.covariate_pair(m,covs,ix[it],ix[iv])
        a,b=exact_project(x[tr],x[te],k);ai,bi=exact_project(x[it],x[iv],k)
        prepared[f['seed']].append((tr,te,(np.c_[ca,a],np.c_[cb,b]),it,iv,(np.c_[ci,ai],np.c_[cj,bi])))
    return prepared,audits

def correlation_columns(a,b):
    a=rankdata(a,axis=0);b=rankdata(b,axis=0)
    a-=a.mean(0);b-=b.mean(0);den=np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return np.divide((a*b).sum(0),den,out=np.full_like(den,np.nan),where=den>0)

def ridge_operator(a,b,alpha):
    # Exact linear smoother including Ridge's fitted intercept.
    n=len(a);mean=a.mean(0);z=a-mean
    if z.shape[1]<=n:
        return (b-mean)@np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T)+np.ones((len(b),n))/n
    return (b-mean)@z.T@np.linalg.solve(z@z.T+alpha*np.eye(n),np.eye(n))+np.ones((len(b),n))/n

def regression_operators(prepared):
    return {s:[(tr,te,it,iv,[(ridge_operator(ai,bi,1/c),ridge_operator(a,b,1/c)) for c in GRID])
               for tr,te,(a,b),it,iv,(ai,bi) in folds] for s,folds in prepared.items()}

def regression_many(operators,targets):
    targets=np.asarray(targets,float);result=np.zeros_like(targets)
    for folds in operators.values():
        pred=np.empty_like(targets)
        for tr,te,it,iv,ops in folds:
            validation=np.stack([correlation_columns(targets[iv],inner@targets[it]) for inner,outer in ops])
            best=np.argmax(np.nan_to_num(validation,nan=-np.inf),axis=0)
            for k,(inner,outer) in enumerate(ops):
                columns=np.flatnonzero(best==k)
                if len(columns):pred[np.ix_(te,columns)]=outer@targets[np.ix_(tr,columns)]
        result+=pred/len(operators)
    return result

def null_test(co,method,task,permutations=NULL_BUDGET,pilot=False):
    policy=RUN/'execution_policy.json'
    if policy.exists() and not read(policy).get('absolute_nulls_enabled',True):
        raise RuntimeError('Absolute null tests are deferred by the user; explicit resumption is required.')
    dest=RUN/('timing' if pilot else 'nulls')/method/co/(task+'.json')
    if dest.exists() and read(dest).get('status')=='complete':return
    old=next(r for r in templates(co) if r['task']==task)
    if old['covariates']:raise ValueError('Absolute FM support uses imaging-only endpoints; clinical increment has paired tests')
    m=sf.labels(co);ix=m.set_index('scan').index.get_indexer(old['scan_ids']);assert (ix>=0).all()
    state=('none',method,'frozen','none',0);x=np.asarray(feature(state,co)[ix],float)
    y=np.asarray(old['y'],float);g=np.asarray(old['subjects']);binary=old['metrics']['sf100_fc']['metric']=='auc'
    start=time.monotonic();prep,audit=prepare(x,y,g,m,ix,[]);prepared_seconds=time.monotonic()-start
    operators=None if binary else regression_operators(prep)
    def predict(yy):
        if binary:return np.column_stack([np.mean([predict_prepared(prep[s],yy[:,i],binary=True) for s in SEEDS],axis=0) for i in range(yy.shape[1])])
        return regression_many(operators,yy)
    observed_prediction=predict(y[:,None])[:,0]
    observed=score(y,observed_prediction,binary)
    if not np.isfinite(observed):observed=.5 if binary else 0.
    signature=hashlib.sha256(json.dumps(dict(co=co,method=method,task=task,y=y.tolist(),g=g.tolist(),x_sha=hashlib.sha256(x.tobytes()).hexdigest(),B=permutations,code=sha(__file__)),sort_keys=True).encode()).hexdigest()
    progress=dest.with_name(task+'_progress.json');values=[]
    if progress.exists():
        d=read(progress)
        if d['signature']!=signature:raise ValueError('Null progress signature mismatch')
        values=d['null_scores']
    start_fit=time.monotonic()
    # Index-specific RNG permits exact restart and identical shuffles across models.
    while len(values)<permutations:
        first=len(values);n=min(8 if binary else 64,permutations-first)
        yy=np.column_stack([permute_participants(y,g,np.random.default_rng(stable_seed(f'{co}/{task}/{i}'))) for i in range(first,first+n)])
        pp=predict(yy)
        for i in range(n):
            v=score(yy[:,i],pp[:,i],binary);values.append(v if np.isfinite(v) else (.5 if binary else 0.))
        save(progress,dict(status='running',signature=signature,null_scores=values,completed=len(values),requested=permutations))
        print('null',co,method,task,len(values),permutations,flush=True)
    hits=sum(v>=observed-1e-12 for v in values)
    save(dest,dict(status='complete',cohort=co,model=method,task=task,objective='frozen',metric='auc' if binary else 'rho',
        subjects=g,scan_ids=old['scan_ids'],observed=observed,observed_prediction=observed_prediction,y=y,
        null_scores=values,permutations=permutations,exceedances=hits,permutation_p=(1+hits)/(1+permutations),
        split_audit=audit,signature=signature,preparation_seconds=prepared_seconds,permutation_seconds=time.monotonic()-start_fit,
        exchangeable_participants=sum(len(rows) for rows in blocks(g).values() if len(rows)>1),
        protocol='Label-independent grouped outer five-fold, grouped inner20%, split seeds0/1/2; train-only exact scaling/PCA; refitted Ridge/logistic heads and tuning. Same folds for observed/null, same participant-block permutations across models. Frozen encoders only.',
        interpretation='Separate support panel; do not attach this P value to primary stratified-CV scores. Unconditional association, not incremental value over FC/clinical.'))

def timing():
    for task in ('MEM','DX'):null_test('nacc','brainlm13',task,permutations=8,pilot=True)
    save(RUN/'timing/complete.json',dict(status='complete',rows=[read(RUN/'timing/brainlm13/nacc'/(t+'.json')) for t in ('MEM','DX')]))

def audit():
    """Executable protocol audit; all corrected primary probes use one engine."""
    checks=[];hashes={}
    for co in COHORTS:
        m=sf.labels(co)
        if m.scan.duplicated().any():raise ValueError('Duplicate scan identifiers')
        for r in templates(co):
            ix=m.set_index('scan').index.get_indexer(r['scan_ids'])
            assert (ix>=0).all() and m.iloc[ix].subject.astype(str).tolist()==r['subjects']
            g=np.asarray(r['subjects']);tests={s:[] for s in SEEDS}
            for f in r['split_audit']:
                tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
                assert not set(g[tr])&set(g[te]) and not set(g[it])&set(g[iv])
                assert set(it)|set(iv)==set(tr)
                tests[f['seed']].extend(te)
            assert all(sorted(v)==list(range(len(g))) for v in tests.values())
            checks.append(dict(cohort=co,task=r['task'],n=len(g),n_subjects=len(set(g)),split_check='passed',covariates=r['covariates']))
            p=sf.RUN/'evaluation'/co/(r['task']+'.json');hashes[str(p)]=sha(p)
    for p in [Path(__file__),ROOT/'src/benchmark_evaluation.py',ROOT/'src/paper_completion.py',ROOT/'src/paper_task_expansion.py',ROOT/'src/new_fm/evaluate.py',ROOT/'scripts/intervention_prepost.py']:
        hashes[str(p)]=sha(p)
    save(RUN/'audit/complete.json',dict(status='complete',checks=checks,code_and_templates_sha256=hashes,
        action='Refit all eligible common primary scan-level probes under one exact-projection implementation; preserve existing transfer/paired-visit protocols and audit their raw outputs before release.',
        equivalence='Full-SVD and dual exact PCA are mathematically equivalent; different implementation alone is not evidence of an error. Historical randomized or different-dimensional probes are not silently mixed.',
        source_rule='No source-cohort cognitive claims from externally selected supervised encoders; independent source CV retained separately. Source-exposed SSL is descriptive.',
        primary_covariates='Inherited task-specific clinical columns and train-only coding; same for every configuration.',
        limitations=['Participant-level pretraining overlap and missing metadata cannot be established by this computational audit.'],
        manuscript_editing=False))

def context_masks():
    from new_fm.data import DATA as ND
    rows=[]
    for co in COHORTS:
        m=sf.labels(co);p=ND/(co+'.csv');n=pd.read_csv(p)
        assert n.scan.tolist()==m.scan.tolist()
        masks={k:(m.native_frames.to_numpy()>= (160 if k=='brainjepa' else 200)) for k in u.CONFIGS[:4]}
        masks.update({k:(n.resampled_frames.to_numpy()>=(100 if k=='semantoks' else 160)) for k in u.CONFIGS[4:]})
        keep=np.logical_and.reduce(list(masks.values()))
        save(RUN/'context'/co/'mask.json',dict(status='complete',cohort=co,scan_ids=m.scan.tolist(),eligible=keep,
            per_configuration=masks,n_complete=int(keep.sum()),resampling_metadata_sha256=sha(p),
            definition='Intersection: BrainLM >=200 real native frames; Brain-JEPA >=160 real native frames; Semantoks >=100 real TR2 frames; all fMRI-LM variants >=160 real TR2 frames.',
            interpretation='Complete temporal inputs, not equal physical duration or elimination of Semantoks missing spatial tokens. Same within-task intersection for all methods. No input or encoder change.'))
        for t in templates(co):
            ix=m.set_index('scan').index.get_indexer(t['scan_ids']);rows.append(dict(cohort=co,task=t['task'],original_n=len(ix),complete_n=int(keep[ix].sum()),complete_subjects=int(m.iloc[ix[keep[ix]]].subject.nunique())))
    save(RUN/'context/complete.json',dict(status='complete',coverage=rows))

def selected_template(co,r,context=False):
    import copy
    m=sf.labels(co);r=copy.deepcopy(r)
    if not context:return r
    d=read(RUN/'context'/co/'mask.json');assert d['scan_ids']==m.scan.tolist()
    ix=m.set_index('scan').index.get_indexer(r['scan_ids']);keep=np.asarray(d['eligible'],bool)[ix]
    for k in ('scan_ids','subjects','y'):r[k]=np.asarray(r[k])[keep].tolist()
    y=np.asarray(r['y']);g=np.asarray(r['subjects'])
    if len(set(g))<20 or len(np.unique(y))<2:return None
    audit=[]
    for s in SEEDS:
        for f,(tr,te) in enumerate(grouped_splits(y,g,s)):
            a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=s).split(tr,groups=g[tr]))
            audit.append(dict(seed=s,fold=f,train=tr,test=te,inner_train=tr[a],validation=tr[b]))
    r['split_audit']=audit
    return r

def local(state,co,context=False):
    name=tag(*state);dest=RUN/('context_probes' if context else 'local')/name/(co+'.json')
    if dest.exists():return
    source,model,obj,target,seed=state;m=sf.labels(co);xall=feature(state,co);rows=[];coverage=[]
    for original in templates(co):
        task=original['task']
        if obj=='supervised' and task in ('AGE','SEX','AGE_GROUP'):continue
        if obj=='supervised' and co==source:
            coverage.append(dict(task=task,status='excluded_source_exposed_supervised',reason='Independent source encoder CV is reported separately'));continue
        r=selected_template(co,original,context)
        if r is None:coverage.append(dict(task=task,status='not_applicable',reason='Complete-context subset <20 participants or constant target'));continue
        ix=m.set_index('scan').index.get_indexer(r['scan_ids']);assert (ix>=0).all()
        y=np.asarray(r['y'],float);g=np.asarray(r['subjects']);binary=original['metrics']['sf100_fc']['metric']=='auc'
        prep,audits=prepare(np.asarray(xall[ix],float),y,g,m,ix,r['covariates'],r['split_audit'])
        predictions=[predict_prepared(prep[s],y,binary=binary) for s in SEEDS];pp=np.mean(predictions,axis=0)
        rows.append(dict(state=state,cohort=co,task=task,scan_ids=r['scan_ids'],subjects=g,y=y,prediction=pp,predictions_by_seed=predictions,
            covariates=r['covariates'],metric='auc' if binary else 'rho',score=score(y,pp,binary),split_audit=audits,
            scope='source_exposed_SSL' if obj=='native_ssl' and source==co else 'external_or_frozen',
            method='clinical+FM' if r['covariates'] else 'FM',protocol='common_exact_primary_v1',
            mae=None if binary else float(np.mean(abs(y-pp))),rmse=None if binary else float(np.sqrt(np.mean((y-pp)**2)))))
        print('unified probe',name,co,task,context,flush=True)
    save(dest,dict(status='complete',state=state,cohort=co,context=context,rows=rows,coverage=coverage,
        feature_sha256=sha(u.feature_folder(state)/(co+'.npy')) if model in u.CONFIGS else sha(sf.RUN/'features'/co/(model+'.npy')),
        engine_sha256=sha(__file__),encoder_refit=False,code_audit_sha256=sha(RUN/'audit/complete.json')))

def references(co,context=False):
    """Reuse verified full-sample controls; fit restricted-context controls afresh."""
    from fm_readout_v4.data import dataset
    from transfer_baseline import fc_feature
    from fm_matched_v3.probes import TuckerBlock
    dest=RUN/('context_references' if context else 'references')/(co+'.json');out=[]
    m=sf.labels(co)
    for original in templates(co):
        r=selected_template(co,original,context)
        if r is None:continue
        row={k:r[k] for k in ('cohort','task','scan_ids','subjects','y','covariates','split_audit')};row['predictions']={}
        row['metric']=original['metrics']['sf100_fc']['metric'];binary=row['metric']=='auc'
        if not context:
            for k in ('sf100_fc','braingfm','clinical'):
                if k in original['predictions']:row['predictions'][k]=original['predictions'][k]
            for atlas in ('a424','sch450'):
                p=ROOT/'runs/benchmark_gaps_v1/references'/atlas/co/(r['task']+'.json');a=read(p)
                for k in ('scan_ids','subjects','y','split_audit'):assert a[k]==r[k]
                for k,v in a['predictions'].items():row['predictions'][atlas+'_'+k]=v
        else:
            ix=m.set_index('scan').index.get_indexer(r['scan_ids']);y=np.asarray(r['y']);g=np.asarray(r['subjects'])
            for method in ('sf100_fc','braingfm'):
                x=feature(('none',method,'frozen','none',0),co)[ix]
                prep,_=prepare(x,y,g,m,ix,r['covariates'],r['split_audit'])
                row['predictions'][method]=np.mean([predict_prepared(prep[s],y,binary=binary) for s in SEEDS],axis=0)
            for atlas,model in [('a424','brainlm13'),('sch450','brainjepa')]:
                xx,_=dataset(co,model);x=xx.numpy();fc=fc_feature(x,None);tucker=TuckerBlock(x,64)
                for method in ('FC','FC-Tucker'):
                    prep={s:[] for s in SEEDS}
                    for f in r['split_audit']:
                        tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
                        pairs=[]
                        for a,b in ((tr,te),(it,iv)):
                            ca,cb=sf.covariate_pair(m,r['covariates'],ix[a],ix[b])
                            z,w=(fc[ix[a]],fc[ix[b]]) if method=='FC' else tucker.fit_pair(ix[a],ix[b])
                            z,w=exact_project(z,w,5 if r['covariates'] else 100);pairs.append((np.c_[ca,z],np.c_[cb,w]))
                        prep[f['seed']].append((tr,te,pairs[0],it,iv,pairs[1]))
                    row['predictions'][atlas+'_'+method]=np.mean([predict_prepared(prep[s],y,binary=binary) for s in SEEDS],axis=0)
            if r['covariates']:
                prep={s:[] for s in SEEDS}
                for f in r['split_audit']:
                    tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
                    prep[f['seed']].append((tr,te,sf.covariate_pair(m,r['covariates'],ix[tr],ix[te]),it,iv,sf.covariate_pair(m,r['covariates'],ix[it],ix[iv])))
                row['predictions']['clinical']=np.mean([predict_prepared(prep[s],np.asarray(r['y']),binary=binary) for s in SEEDS],axis=0)
        out.append(row)
    save(dest,dict(status='complete',cohort=co,context=context,rows=out))

def paired_test(y,a,b,g,metric,key):
    from paper_paired_stats import paired
    path=RUN/'inference_cache'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
    sig=hashlib.sha256(json.dumps([list(y),np.asarray(a).tolist(),np.asarray(b).tolist(),list(g),metric],default=str).encode()).hexdigest()
    if path.exists():
        d=read(path)
        if sig!=d['signature']:raise ValueError('Inference input changed')
        return d['statistics']
    d=paired(y,a,b,g,metric=metric,seed=stable_seed(key));save(path,dict(status='complete',signature=sig,statistics=d));return d

def local_inference(model,co,context=False):
    kind='context_probes' if context else 'local';dest=RUN/('context_inference' if context else 'local_inference')/model/(co+'.json')
    refs=read(RUN/('context_references' if context else 'references')/(co+'.json'))['rows'];groups=defaultdict(list);scores=[];comparisons=[]
    states=[('none',model,'frozen','none',0)] if context else list(u.states(model))
    for state in states:
        p=RUN/kind/tag(*state)/(co+'.json')
        for r in read(p)['rows']:groups[(*state[:4],r['task'])].append(r)
    for key,rr in groups.items():
        source,_,obj,target,task=key;r=rr[0]
        assert len(rr)==(1 if obj=='frozen' else 3)
        for a in rr[1:]:
            for field in ('subjects','scan_ids','y','split_audit'):assert a[field]==r[field]
        pred=np.mean([a['prediction'] for a in rr],axis=0);ref=next(a for a in refs if a['task']==task)
        for field in ('subjects','scan_ids','y'):assert r[field]==ref[field]
        info=dict(model=model,cohort=co,source=source,objective=obj,target=target,task=task,context=context,method=r['method'],scope=r['scope'],metric=r['metric'])
        scores.append(dict(info,score=score(r['y'],pred,r['metric']=='auc'),n=len(pred),n_subjects=len(set(r['subjects'])),subjects=r['subjects'],scan_ids=r['scan_ids'],y=r['y'],prediction=pred))
        atlas='a424' if model.startswith('brainlm') else 'sch450'
        baseline={k:v for k,v in ref['predictions'].items() if k in ('sf100_fc','braingfm','clinical',atlas+'_FC',atlas+'_FC-Tucker')}
        if obj!='frozen':baseline['frozen']=groups[('none',model,'frozen','none',task)][0]['prediction']
        for method,reference in baseline.items():
            family=('context_' if context else 'local_')+task+'_'+obj+'_'+target+'_'+('adaptation' if method=='frozen' else 'clinical_increment' if method=='clinical' else 'reference')+'_'+r['scope']
            stat=paired_test(r['y'],reference,pred,r['subjects'],r['metric'],str((model,co,context,key,method)))
            comparisons.append(dict(info,reference=method,candidate=r['method'],family=family,**{k:v for k,v in stat.items() if k!='metric'}))
    save(dest,dict(status='complete',model=model,cohort=co,context=context,rows=comparisons,scores=scores,
        seed_aggregation='Mean split-seed predictions then mean adaptation-seed predictions; never average feature coordinates.'))
