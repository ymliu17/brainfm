"""Paired-visit probes of cached frozen/source-adapted representations.

No encoder fitting, GPU extraction, or shutdown actions. Run with PYTHONPATH=src:scripts.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from benchmark_evaluation import grouped_splits, GRID
from fm_readout_v4.data import metadata, DATA
from fm_matrix.data import ROOT, sha
from paper_completion import states, feature_dir
from paper_robustness import assemble_labels
from readout_sensitivity import paired_rows
from sf100_benchmark import RUN as SF, matches

RUN = ROOT/'runs/intervention_prepost_v1'
COHORTS = ('act', 'cogte', 'breathe', 'psopt', 'hrv')
MODES = ('baseline', 'change', 'baseline_plus_change')
SPLIT_SEEDS = (0, 1, 2)
K = 5


def read(p):
    return json.loads(Path(p).read_text())


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.generic): return clean(o.item())
    if isinstance(o, float) and not np.isfinite(o): return None
    return o


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n'); tmp.replace(p)


def state_name(s):
    return '_'.join(map(str, s))


def feature_path(s, co):
    return (SF/'features'/co/(s[1]+'.npy') if s[0]=='comparator'
            else feature_dir(s)/(co+'.npy'))


def all_states():
    return list(states()) + [('comparator', n, 'frozen', 'none', 0) for n in ('sf100_fc', 'braingfm')]


def clinical_frame(m, co):
    nums = ['baseline_EF'] if co=='hrv' else ['baseline_MEM', 'baseline_EF', 'age']
    cats = ['age_category', 'sex', 'arm'] if co=='hrv' else ['sex', 'arm']
    return m[nums+cats].copy(), nums, cats


def build_inputs():
    files = {}; plans = {}; coverage = []
    registered = read(ROOT/'runs/paper_completion_v1/input_audit.json')['files_sha256']
    for co in COHORTS:
        subjects, (ca, ia), (cb, ib) = paired_rows(co)
        a, sources = assemble_labels(ca); b = metadata(cb)
        ma, mb = matches(ca), matches(cb)
        mask = np.array([ma[i]['available'] and mb[j]['available'] for i,j in zip(ia,ib)])
        mask &= a.iloc[ia].eligible.to_numpy(bool) & b.iloc[ib].eligible.to_numpy(bool)
        for cc in {ca, cb}:
            sources += [DATA/(cc+'_domains.csv' if cc.startswith('hrv') else cc+'.csv'), SF/'matches'/(cc+'.json')]
        for s in all_states():
            for cc, indices in [(ca, ia), (cb, ib)]:
                p = feature_path(s, cc)
                if str(p) not in files:
                    h = sha(p)
                    if s[0]!='comparator':
                        if str(p) not in registered or registered[str(p)] != h:
                            raise ValueError('Unregistered or changed encoder features: '+str(p))
                        stamp = feature_dir(s)/'complete.json'
                        sig = read(stamp)['signature']
                        if sig['model'] != s[1]: raise ValueError('Model mismatch')
                        if s[1]=='brainlm13' and sig.get('readout')!='CLS_final': raise ValueError('Wrong readout')
                        if s[1]=='brainlm13' and s[2]=='supervised' and sig.get('training_readout')!='CLS_final':
                            raise ValueError('Wrong supervised training readout')
                        files[str(stamp)] = sha(stamp)
                    else:
                        stamp=SF/'features'/cc/'complete.json'
                        if read(stamp)['files_sha256'][s[1]]!=h: raise ValueError('Comparator changed')
                        if read(stamp)['scan_ids']!=metadata(cc).scan.tolist(): raise ValueError('Comparator ordering')
                        files[str(stamp)] = sha(stamp)
                    files[str(p)] = h
                x = np.load(p, mmap_mode='r')
                if len(x)!=len(metadata(cc)): raise ValueError('Feature row count')
                finite = np.isfinite(x[indices]).all(1)
                if not finite[mask].all(): raise ValueError('Nonfinite features on shared paired scans')
        aa=a.iloc[ia].reset_index(drop=True); bb=b.iloc[ib].reset_index(drop=True)
        assert aa.subject.astype(str).tolist()==bb.subject.astype(str).tolist()
        assert not aa.subject.duplicated().any()
        # The same supplied cohort scale is used at both visits; no visit-wise restandardization.
        for e in ('MEM','EF'):
            np.testing.assert_allclose(aa[e], aa['baseline_'+e], equal_nan=True)
            np.testing.assert_allclose(bb[e], aa['follow_'+e], equal_nan=True)
        tasks={}
        for task in ('MEM_change', 'EF_change', 'arm'):
            if task=='arm':
                valid=mask & aa.arm.notna().to_numpy()
                classes=sorted(aa.loc[valid,'arm'].astype(str).unique())
                y=np.array([classes.index(str(v)) for v in aa.loc[valid,'arm']])
            else:
                e=task.split('_')[0]
                valid=mask & np.isfinite(aa[e].to_numpy(float)) & np.isfinite(bb[e].to_numpy(float))
                y=bb.loc[valid,e].to_numpy(float)-aa.loc[valid,e].to_numpy(float); classes=[]
            if not len(y):
                coverage.append(dict(cohort=co,task=task,n=0,status='unavailable',reason='No paired memory measurements'))
                continue
            ix=np.flatnonzero(valid); g=aa.iloc[ix].subject.astype(str).to_numpy()
            cf, nums, cats=clinical_frame(aa.iloc[ix],co)
            splits=[]
            for seed in SPLIT_SEEDS:
                for fold,(tr,te) in enumerate(grouped_splits(y,g,seed)):
                    if task=='arm':
                        ii,jj=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr,y[tr]))
                    else:
                        ii,jj=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr,groups=g[tr]))
                    it,iv=tr[ii],tr[jj]
                    assert not set(g[tr]) & set(g[te])
                    assert not set(it) & set(iv) and set(it)|set(iv)==set(tr)
                    if task=='arm':
                        for z in (tr,te,it,iv):
                            if set(y[z]) != set(range(len(classes))): raise ValueError('Missing class in split')
                    splits.append(dict(seed=seed,fold=fold,train=tr,test=te,inner_train=it,validation=iv))
            tasks[task]=dict(y=y,subjects=g,pre_indices=ia[ix],post_indices=ib[ix],
                pre_scans=aa.iloc[ix].scan.tolist(),post_scans=bb.iloc[ix].scan.tolist(),classes=classes,
                clinical=cf.to_dict(orient='list'),numeric_covariates=nums,categorical_covariates=cats,splits=splits)
            coverage.append(dict(cohort=co,task=task,n=len(y),status='ready',classes=classes))
        plans[co]=dict(pre_cohort=ca,post_cohort=cb,tasks=tasks)
        for p in sources: files[str(p)]=sha(p)
    protocol=dict(imaging_components=K,inputs=list(MODES),split_seeds=SPLIT_SEEDS,adaptation_seeds=[42,43,44],
        objectives=['frozen','ADNI/NACC MEM supervised','ADNI/NACC EF supervised','ADNI/NACC native SSL'],
        split='Five participant folds; cognitive-quintile or arm-stratified outer splits; inner 20% validation (arm-stratified for classification).',
        features='Float64 post-minus-pre subtraction before fold-fitted scaling/PCA; baseline+change concatenated before a single PCA with the same maximum of five components.',
        targets='MEM/EF follow-up minus baseline in supplied cohort score units; no visit-specific standardization; no responder thresholds; HRV MEM change unavailable.',
        regressors='Ridge; training-only scaling/PCA; tuning by validation Spearman. Clinical-only and clinical+imaging regressions use baseline cognitive scores/demographics and arm, with train-only imputation and categorical encoding.',
        classifiers='Logistic regression (multinomial for >2 arms); validation binary/macro OvR AUC; probabilities averaged across splits and adaptation seeds before metrics.',
        reference='Outer-training mean change or outer-training class probabilities; clinical regression reference. Same task participants/folds across all models/states/modes.',
        reporting='Scores of OOF predictions averaged over split seeds; adapted main scores additionally average adaptation seeds. Individual adaptation-seed scores retained. Point estimates only; no significance claims.',
        interpretation='Post hoc paired-visit association / arm discrimination; not prospective baseline prognosis or causal treatment-effect estimation.',
        missing='No cognitive-outcome imputation. Clinical covariate medians/categories fitted within each training partition. Common SF100+four-FM paired scan population.',
        preprocessing_caveat='SF100/BrainGFM use supplied whole-scan FC; FM configurations retain model-specific atlases/contexts. Source composite construction is inherited, not re-estimated.',
        automatic_shutdown=False)
    signature=hashlib.sha256(json.dumps(clean(dict(files=files,plans=plans,protocol=protocol)),sort_keys=True).encode()).hexdigest()
    result=dict(status='complete',signature=signature,files_sha256=files,plans=plans,coverage=coverage,protocol=protocol,states=all_states())
    old=RUN/'inputs.json'
    if old.exists() and read(old)['signature']!=signature: raise ValueError('Inputs/protocol changed; use a new run')
    save(old,result)
    pd.DataFrame(coverage).to_csv(RUN/'coverage.csv',index=False)
    return result


def project(x, tr, te):
    """Exact training PCA through the small sample Gram matrix; deterministic."""
    sc=StandardScaler().fit(x[tr]); a=sc.transform(x[tr]); b=sc.transform(x[te])
    k=min(K,len(tr)-1,x.shape[1])
    vals,u=eigh(a@a.T,subset_by_index=[len(tr)-k,len(tr)-1])
    order=np.argsort(vals)[::-1];vals=vals[order];u=u[:,order]
    good=vals>max(float(vals[0]),1)*1e-12
    v=(a.T@u[:,good])/np.sqrt(vals[good])[None,:]
    if not v.shape[1]: return np.zeros((len(tr),1)),np.zeros((len(te),1))
    return a@v,b@v


def clinical_transform(plan,tr,te):
    m=pd.DataFrame(plan['clinical']); nums=plan['numeric_covariates']; cats=plan['categorical_covariates']
    a=m.iloc[tr][nums].astype(float);b=m.iloc[te][nums].astype(float)
    med=a.median().fillna(0);a=a.fillna(med).to_numpy();b=b.fillna(med).to_numpy()
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    ac=m.iloc[tr][cats].fillna('missing').astype(str);bc=m.iloc[te][cats].fillna('missing').astype(str)
    a=np.c_[a,enc.fit_transform(ac)];b=np.c_[b,enc.transform(bc)]
    sc=StandardScaler().fit(a)
    return sc.transform(a),sc.transform(b)


def primary(y,p,classes):
    if classes:
        return roc_auc_score(y,p[:,1]) if len(classes)==2 else roc_auc_score(y,p,average='macro',multi_class='ovr')
    if np.std(p)==0 or np.std(y)==0: return float('nan')
    return float(spearmanr(y,p).statistic)


def fit_predict(a,y,b,c,classes):
    if classes:
        model=LogisticRegression(C=c,max_iter=2000,solver='lbfgs').fit(a,y)
        assert np.array_equal(model.classes_,np.arange(len(classes)))
        return model.predict_proba(b)
    return Ridge(alpha=1/c).fit(a,y).predict(b)


def tuned(y,tr,te,it,iv,outer,inner,classes):
    best=GRID[0];value=-np.inf
    for c in GRID:
        p=fit_predict(inner[0],y[it],inner[1],c,classes); v=primary(y[iv],p,classes)
        if np.isfinite(v) and v>value: best=c;value=v
    return fit_predict(outer[0],y[tr],outer[1],best,classes),float(best)


def metrics(y,p,classes):
    if classes:
        return dict(auc=primary(y,p,classes),balanced_accuracy=balanced_accuracy_score(y,p.argmax(1)),
                    accuracy=accuracy_score(y,p.argmax(1)),log_loss=log_loss(y,p,labels=np.arange(len(classes))))
    return dict(rho=primary(y,p,classes),mae=mean_absolute_error(y,p),rmse=np.sqrt(mean_squared_error(y,p)),r2=r2_score(y,p))


def run_job(arg):
    s,co=arg;data=read(RUN/'inputs.json');plan=data['plans'][co]
    dest=RUN/'jobs'/state_name(s)/(co+'.json')
    if dest.exists():
        old=read(dest)
        if old['input_signature']!=data['signature'] or old['script_sha256']!=sha(__file__): raise ValueError('Stale job')
        return str(dest)
    ca,cb=plan['pre_cohort'],plan['post_cohort']
    arrays={}
    for cc in {ca,cb}:
        p=feature_path(s,cc)
        if sha(p)!=data['files_sha256'][str(p)]: raise ValueError('Changed input')
        arrays[cc]=np.load(p,mmap_mode='r')
    rows=[]
    for task,t in plan['tasks'].items():
        y=np.asarray(t['y']);classes=t['classes']; n=len(y)
        pre=np.asarray(arrays[ca][t['pre_indices']],dtype=np.float64)
        post=np.asarray(arrays[cb][t['post_indices']],dtype=np.float64)
        delta=post-pre
        xs=dict(baseline=pre,change=delta,baseline_plus_change=np.c_[pre,delta])
        arms=list(MODES)+([] if classes else ['clinical']+[k+'+clinical' for k in MODES])+['null']
        shape=(3,n,len(classes)) if classes else (3,n)
        pred={k:np.full(shape,np.nan) for k in arms};selected={k:[] for k in arms if k!='null'}
        for split in t['splits']:
            seed=split['seed'];tr,te,it,iv=[np.asarray(split[k]) for k in ('train','test','inner_train','validation')]
            pred['null'][seed,te]=(np.bincount(y[tr].astype(int),minlength=len(classes))/len(tr) if classes else y[tr].mean())
            if not classes:
                oc=clinical_transform(t,tr,te);ic=clinical_transform(t,it,iv)
                pp,c=tuned(y,tr,te,it,iv,oc,ic,classes);pred['clinical'][seed,te]=pp;selected['clinical'].append(c)
            for name,x in xs.items():
                outer=project(x,tr,te);inner=project(x,it,iv)
                pp,c=tuned(y,tr,te,it,iv,outer,inner,classes);pred[name][seed,te]=pp;selected[name].append(c)
                if not classes:
                    op=tuple(np.c_[a,b] for a,b in zip(oc,outer));ip=tuple(np.c_[a,b] for a,b in zip(ic,inner))
                    pp,c=tuned(y,tr,te,it,iv,op,ip,classes);pred[name+'+clinical'][seed,te]=pp;selected[name+'+clinical'].append(c)
        for name,pp in pred.items():
            if not np.isfinite(pp).all(): raise ValueError('Incomplete OOF predictions')
            if classes: np.testing.assert_allclose(pp.sum(2),1,atol=1e-8)
            p=pp.mean(0)
            rows.append(dict(task=task,mode=name,n=n,subjects=t['subjects'],y=y,classes=classes,
                prediction=p,predictions_by_split_seed=pp,selected_C=selected.get(name,[]),metrics=metrics(y,p,classes)))
    save(dest,dict(status='complete',state=s,cohort=co,input_signature=data['signature'],script_sha256=sha(__file__),rows=rows))
    return str(dest)


def aggregate(data):
    individual=[]; groups={}; clinical={}
    for s in all_states():
        for co in COHORTS:
            d=read(RUN/'jobs'/state_name(s)/(co+'.json'))
            for r in d['rows']:
                if r['mode'] in ('clinical','null'):
                    key=(co,r['task'],r['mode'])
                    if key in clinical: np.testing.assert_allclose(r['prediction'],clinical[key]['prediction'],atol=0,rtol=0)
                    else: clinical[key]=r
                    continue
                source,model,obj,target,seed=s
                info=dict(cohort=co,task=r['task'],source=source,model=model,objective=obj,target=target,adaptation_seed=seed,mode=r['mode'],n=r['n'])
                individual.append(dict(**info,**r['metrics']))
                key=(co,r['task'],source,model,obj,target,r['mode'])
                groups.setdefault(key,[]).append(r)
    rows=[]; preds=[]
    for key,rr in groups.items():
        co,task,source,model,obj,target,mode=key
        assert len(rr)==(1 if obj=='frozen' else 3)
        r=rr[0]
        for other in rr[1:]:
            assert r['subjects']==other['subjects'] and r['y']==other['y'] and r['classes']==other['classes']
        p=np.mean([x['prediction'] for x in rr],axis=0); y=np.asarray(r['y'])
        info=dict(cohort=co,task=task,source=source,model=model,objective=obj,target=target,mode=mode,n=r['n'],adaptation_seeds=len(rr))
        rows.append(dict(**info,**metrics(y,p,r['classes'])))
        preds.append(dict(**info,subjects=r['subjects'],y=y,classes=r['classes'],prediction=p))
    for (co,task,mode),r in clinical.items():
        info=dict(cohort=co,task=task,source='none',model=mode,objective='reference',target='none',mode=mode,n=r['n'],adaptation_seeds=0)
        rows.append(dict(**info,**r['metrics']));preds.append(dict(**info,subjects=r['subjects'],y=r['y'],classes=r['classes'],prediction=r['prediction']))
    scores=pd.DataFrame(rows);seeds=pd.DataFrame(individual)
    scores.to_csv(RUN/'scores.csv',index=False);seeds.to_csv(RUN/'scores_by_adaptation_seed.csv',index=False)
    save(RUN/'predictions.json',preds)
    # Descriptive paired point differences, with no uncomputed inferential claims.
    contrasts=[]
    for r in rows:
        if r['objective']=='reference':continue
        metric='auc' if r['task']=='arm' else 'rho'
        candidates=[]
        if r['mode'] in ('change','baseline_plus_change','change+clinical','baseline_plus_change+clinical'):
            candidates.append(('input_vs_baseline',dict(r,mode='baseline'+('+clinical' if '+clinical' in r['mode'] else ''))))
        if r['objective']!='frozen':candidates.append(('adapted_vs_frozen',dict(r,source='none',objective='frozen',target='none')))
        for label,query in candidates:
            keys=('cohort','task','source','model','objective','target','mode')
            ref=[a for a in rows if all(a[k]==query[k] for k in keys)]
            assert len(ref)==1 and ref[0]['n']==r['n']
            contrasts.append(dict(**{k:r[k] for k in keys},contrast=label,metric=metric,reference_score=ref[0][metric],candidate_score=r[metric],difference=r[metric]-ref[0][metric]))
    diffs=pd.DataFrame(contrasts);diffs.to_csv(RUN/'paired_point_differences.csv',index=False)
    dest=ROOT/'results/generated/INTERVENTION_PREPOST_RESULTS.xlsx'
    with pd.ExcelWriter(dest,engine='xlsxwriter') as writer:
        scores.to_excel(writer,sheet_name='All scores',index=False)
        seeds.to_excel(writer,sheet_name='Adaptation seeds',index=False)
        diffs.to_excel(writer,sheet_name='Paired point differences',index=False)
        pd.DataFrame(data['coverage']).to_excel(writer,sheet_name='Coverage',index=False)
        pd.DataFrame([{'item':k,'value':json.dumps(v) if not isinstance(v,str) else v} for k,v in data['protocol'].items()]).to_excel(writer,sheet_name='Protocol',index=False)
    save(RUN/'validation.json',dict(status='complete',jobs=390,score_rows=len(scores),individual_seed_rows=len(seeds),
         comparisons=len(diffs),input_signature=data['signature'],script_sha256=sha(__file__),
         checks=['All 390 jobs complete','All source-feature checksums verified','Common participants and partitions across every state and method',
                 'Outer and inner subject partitions disjoint','All arm classes present in each partition','Complete finite OOF predictions',
                 'Clinical/null predictions exactly identical across representation states','Three adaptation seeds for every adapted summary'],
         inference='Point estimates only; differences have no CI or multiplicity-adjusted significance test.',
         outputs_sha256={str(p):sha(p) for p in [RUN/'scores.csv',RUN/'scores_by_adaptation_seed.csv',RUN/'predictions.json',RUN/'paired_point_differences.csv',dest]}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--audit-only',action='store_true');parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    data=build_inputs();print('Inputs verified:',len(data['files_sha256']),'files',flush=True)
    if args.audit_only:return
    jobs=[(s,co) for s in all_states() for co in COHORTS]; start=time.time();done=0
    if args.summarize_only:
        # Preserve the producer script: only export code was changed after all fits completed.
        original=RUN/'computation_script.py'
        assert original.read_text().split('def aggregate(data):')[0] == Path(__file__).read_text().split('def aggregate(data):')[0]
        for s,co in jobs:
            result=read(RUN/'jobs'/state_name(s)/(co+'.json'))
            assert result['script_sha256']==sha(original) and result['input_signature']==data['signature']
        aggregate(data)
        v=read(RUN/'validation.json');v['computation_script_sha256']=sha(original)
        v['export_only_fix']='Used installed xlsxwriter instead of unavailable openpyxl; computation functions verified byte-identical to archived producer.'
        save(RUN/'validation.json',v)
        save(RUN/'status.json',dict(status='complete',completed=len(jobs),total=len(jobs),automatic_shutdown=False))
        return
    save(RUN/'status.json',dict(status='running',completed=0,total=len(jobs),pid=os.getpid(),automatic_shutdown=False))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_job,j):j for j in jobs}
        for f in concurrent.futures.as_completed(pending):
            try:path=f.result()
            except Exception as e:
                save(RUN/'status.json',dict(status='failed',completed=done,total=len(jobs),failed_job=pending[f],error=str(e),automatic_shutdown=False))
                raise
            done+=1
            save(RUN/'status.json',dict(status='running',completed=done,total=len(jobs),elapsed_seconds=time.time()-start,last_completed=path,automatic_shutdown=False))
            print(done,'/',len(jobs),path,flush=True)
    aggregate(data)
    save(RUN/'status.json',dict(status='complete',completed=done,total=len(jobs),elapsed_seconds=time.time()-start,automatic_shutdown=False))
    print('Complete',flush=True)


if __name__=='__main__':main()
