"""Conditional participant bootstrap and paired swaps for the pre/post panel.

Extends the existing binary/rho implementation to macro OvR multiclass AUC,
swapping a participant's entire probability vector as one indivisible unit.
"""
import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
import numpy as np
import pandas as pd
from paper_paired_stats import score_batch, weighted_scores, holm
from intervention_prepost import ROOT, RUN as BASE, read, save, sha, metrics

RUN=BASE/'inference_v1'
KEYS=('cohort','task','source','model','objective','target','mode')
N_BOOT=5000
N_PERM=20000


def key(r):return tuple(r[k] for k in KEYS)
def uid(r):return '|'.join(map(str,key(r)))
def seed_of(s):return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def objective(r):return 'frozen' if r['objective']=='frozen' else 'SSL' if r['objective']=='native_ssl' else r['target']


def primary_batch(y,p,classes):
    p=np.asarray(p,dtype=float)
    if not classes:return score_batch(y,p,False)
    if p.ndim==2:p=p[None,:,:]
    if len(classes)==2:return score_batch(y,p[:,:,1],True)
    return np.mean([score_batch((y==c).astype(int),p[:,:,c],True) for c in range(len(classes))],axis=0)


def bootstrap_metrics(y,p,w,classes):
    y=np.asarray(y);p=np.asarray(p);total=w.sum(1)
    if not classes:
        err=p-y;mse=(w@(err**2))/total
        var=(w@(y*y))/total-((w@y)/total)**2
        return dict(rho=weighted_scores(y,p,w),mae=(w@abs(err))/total,rmse=np.sqrt(mse),
                    r2=1-np.divide(mse,var,out=np.full(len(w),np.nan),where=var>0))
    perclass=[];recall=[];pred=p.argmax(1)
    for c in range(len(classes)):
        yy=(y==c).astype(int)
        perclass.append(weighted_scores(yy,p[:,c],w,True))
        counts=w@yy
        recall.append(np.divide(w@((pred==c)&(y==c)),counts,out=np.full(len(w),np.nan),where=counts>0))
    auc=perclass[1] if len(classes)==2 else np.mean(perclass,axis=0)
    losses=-np.log(np.clip(p[np.arange(len(y)),y.astype(int)],np.finfo(float).eps,1))
    return dict(auc=auc,balanced_accuracy=np.mean(recall,axis=0),accuracy=(w@(pred==y))/total,log_loss=(w@losses)/total)


def swaps(y,a,b,classes,seed,n_perm=N_PERM):
    observed=float(primary_batch(y,b,classes)[0]-primary_batch(y,a,classes)[0])
    if not np.isfinite(observed):
        return dict(estimable=False,permutation_p=None,reason='Undefined primary score',permutations_valid=0)
    if np.array_equal(a,b):
        return dict(estimable=True,permutation_p=1.,permutations_valid=n_perm,permutation_exceedances=n_perm,exact_identity=True)
    rng=np.random.default_rng(seed);exceed=0
    for start in range(0,n_perm,256):
        mask=rng.random((min(256,n_perm-start),len(y)))<.5
        if classes:mask=mask[:,:,None]
        pa=np.where(mask,b,a);pb=np.where(mask,a,b)
        null=primary_batch(y,pb,classes)-primary_batch(y,pa,classes)
        if not np.isfinite(null).all():raise ValueError('Nonestimable permutation draw')
        exceed+=int(np.count_nonzero(np.abs(null)>=abs(observed)-1e-12))
    return dict(estimable=True,permutation_p=(1+exceed)/(n_perm+1),permutations_valid=n_perm,
                permutation_exceedances=exceed,exact_identity=False)


def define_plan():
    validation=read(BASE/'validation.json');assert validation['status']=='complete'
    for p in [BASE/'predictions.json',BASE/'scores.csv',BASE/'paired_point_differences.csv']:
        assert validation['outputs_sha256'][str(p)]==sha(p)
    predictions=read(BASE/'predictions.json');lookup={key(r):r for r in predictions}
    assert len(lookup)==len(predictions)
    contrasts=[]
    def add(r,ref,block):
        a=lookup[key(ref)]
        assert r['subjects']==a['subjects'] and r['y']==a['y'] and r['classes']==a['classes']
        assert len(set(r['subjects']))==len(r['subjects'])
        context='clinical_plus_imaging' if '+clinical' in r['mode'] else 'imaging_only'
        family='|'.join([block,r['task'],objective(r),context])
        contrasts.append(dict(**{k:r[k] for k in KEYS},reference_id=uid(a),candidate_id=uid(r),
            contrast=block,family=family,metric='auc' if r['task']=='arm' else 'rho',n=len(r['y'])))
    for r in predictions:
        if r['objective']=='reference':continue
        if r['mode'] in ('change','baseline_plus_change','change+clinical','baseline_plus_change+clinical'):
            add(r,dict(r,mode='baseline'+('+clinical' if '+clinical' in r['mode'] else '')),'input_vs_baseline')
        if r['objective']!='frozen':
            add(r,dict(r,source='none',objective='frozen',target='none'),'adapted_vs_frozen')
        if '+clinical' in r['mode']:
            add(r,dict(r,source='none',model='clinical',objective='reference',target='none',mode='clinical'),'imaging_added_to_clinical')
        if r['objective']=='frozen' and r['model']!='sf100_fc':
            add(r,dict(r,source='comparator',model='sf100_fc',target='none'),'frozen_vs_sf100')
    family_sizes=Counter(c['family'] for c in contrasts)
    assert max(family_sizes.values())/(N_PERM+1)<.05
    old=pd.read_csv(BASE/'paired_point_differences.csv',keep_default_na=False)
    core=[c for c in contrasts if c['contrast'] in ('input_vs_baseline','adapted_vs_frozen')]
    assert len(core)==len(old)==3036
    oldkeys={tuple(r[k] for k in KEYS)+(r['contrast'],) for r in old.to_dict('records')}
    assert oldkeys=={key(c)+(c['contrast'],) for c in core}
    plan=dict(status='planned_before_resampling',source_predictions_sha256=sha(BASE/'predictions.json'),
        source_scores_sha256=sha(BASE/'scores.csv'),script_sha256=sha(__file__),
        shared_stats_sha256=sha(ROOT/'src/paper_paired_stats.py'),n_boot=N_BOOT,n_perm=N_PERM,
        contrasts=contrasts,family_sizes=dict(family_sizes),
        methods=dict(bootstrap='5000 ordinary participant resamples with replacement. Each row is one paired participant. Recompute ranks with multiplicities. Shared draws across methods within each cohort/task.',
          score_intervals='Pointwise percentile 95% CIs for all reported primary and secondary metrics; not simultaneous intervals.',
          paired_intervals='Percentile 95% CIs for candidate minus reference primary scores from the same paired draws.',
          test='20000 two-sided participant-wise swaps of fixed OOF predictions; (1+b)/(B+1). Entire multiclass probability vectors are swapped together.',
          correction='Holm within contrast block x task endpoint x adaptation objective (frozen/MEM/EF/SSL) x imaging-only/clinical-plus-imaging context. Pool models, source cohorts, target cohorts, and all applicable input modes within a family. Nonestimable comparisons retained with P=1.',
          scope='Exploratory conditional inference on seed-averaged fitted OOF predictions. No model refitting, algorithm-level uncertainty, or single correction across the whole benchmark.',
          direction='Positive delta favors candidate for rho/AUC. Paired hypotheses only concern rho/AUC; other metric CIs are descriptive.',
          boundaries='Tests compare methods, not above-chance performance or causal intervention effects. Failure to reject does not establish equivalence. Bootstrap CIs are not multiplicity adjusted.'),
        automatic_shutdown=False)
    if (RUN/'plan.json').exists():assert read(RUN/'plan.json')==plan,'Plan changed; use a new inference run'
    else:save(RUN/'plan.json',plan)
    return plan,predictions


def run_task(task):
    co,endpoint=task;plan=read(RUN/'plan.json');dest=RUN/'tasks'/(co+'_'+endpoint+'.json')
    if dest.exists():
        result=read(dest);assert result['plan_sha256']==sha(RUN/'plan.json');return str(dest)
    assert sha(BASE/'predictions.json')==plan['source_predictions_sha256']
    predictions=[r for r in read(BASE/'predictions.json') if r['cohort']==co and r['task']==endpoint]
    lookup={uid(r):r for r in predictions};first=predictions[0];y=np.asarray(first['y']);classes=first['classes'];n=len(y)
    for r in predictions:
        assert r['subjects']==first['subjects'] and r['y']==first['y'] and r['classes']==classes
    rng=np.random.default_rng(seed_of(co+'|'+endpoint+'|bootstrap'))
    w=rng.multinomial(n,np.ones(n)/n,size=N_BOOT)
    boots={};scores=[]
    for r in predictions:
        p=np.asarray(r['prediction'],float);met=metrics(y,p,classes);b=bootstrap_metrics(y,p,w,classes)
        boots[uid(r)]=b['auc' if classes else 'rho']
        entry=dict(**{k:r[k] for k in KEYS},n=n,**met)
        for name,draws in b.items():
            valid=draws[np.isfinite(draws)]
            if np.isfinite(met[name]) and len(valid)<.95*N_BOOT:raise ValueError('Too few valid bootstrap draws')
            interval=np.quantile(valid,[.025,.975]) if len(valid) else [None,None]
            entry.update({name+'_ci_low':interval[0],name+'_ci_high':interval[1],name+'_bootstrap_valid':len(valid)})
        scores.append(entry)
    stats=[];contrasts=[c for c in plan['contrasts'] if c['cohort']==co and c['task']==endpoint]
    for i,c in enumerate(contrasts):
        a=np.asarray(lookup[c['reference_id']]['prediction']);b=np.asarray(lookup[c['candidate_id']]['prediction'])
        sa,sb=primary_batch(y,a,classes)[0],primary_batch(y,b,classes)[0]
        draws=boots[c['candidate_id']]-boots[c['reference_id']];valid=draws[np.isfinite(draws)]
        ci=np.quantile(valid,[.025,.975]) if len(valid) else [None,None]
        s=seed_of(c['reference_id']+'|'+c['candidate_id']+'|'+c['contrast'])
        result=swaps(y,a,b,classes,s)
        stats.append(dict(**c,reference_score=sa,candidate_score=sb,delta=sb-sa,ci_low=ci[0],ci_high=ci[1],
                          bootstrap_valid=len(valid),permutation_seed=s,**result))
        if (i+1)%25==0:print(co,endpoint,'contrasts',i+1,'/',len(contrasts),flush=True)
    save(dest,dict(status='complete',cohort=co,task=endpoint,scores=scores,contrasts=stats,plan_sha256=sha(RUN/'plan.json')))
    return str(dest)


def aggregate():
    plan=read(RUN/'plan.json');scores=[];stats=[]
    tasks=sorted({(c['cohort'],c['task']) for c in plan['contrasts']})
    for co,task in tasks:
        result=read(RUN/'tasks'/(co+'_'+task+'.json'));assert result['plan_sha256']==sha(RUN/'plan.json')
        scores.extend(result['scores']);stats.extend(result['contrasts'])
    for family,n in plan['family_sizes'].items():
        rows=[r for r in stats if r['family']==family];assert len(rows)==n
        for r,p in zip(rows,holm([r['permutation_p'] for r in rows])):
            r.update(holm_p=p,family_size=n,significant_holm_05=bool(r['estimable'] and p<.05))
    a=pd.DataFrame(scores);b=pd.DataFrame(stats)
    assert len(a)==2093 and len(b)==len(plan['contrasts'])
    assert not a.duplicated(list(KEYS)).any()
    original=pd.read_csv(BASE/'scores.csv',keep_default_na=False).set_index(list(KEYS))
    for r in scores:
        for m in ('rho','mae','rmse','r2','auc','balanced_accuracy','accuracy','log_loss'):
            if m in r:np.testing.assert_allclose(float(original.loc[key(r),m]),r[m],rtol=1e-12,atol=1e-12,equal_nan=True)
    a.to_csv(RUN/'scores_with_ci.csv',index=False);b.to_csv(RUN/'paired_inference.csv',index=False)
    families=[]
    for family,n in plan['family_sizes'].items():
        rows=b[b.family==family]
        families.append(dict(family=family,n=n,positive_holm_05=int(((rows.holm_p<.05)&(rows.delta>0)).sum()),negative_holm_05=int(((rows.holm_p<.05)&(rows.delta<0)).sum()),min_holm_p=float(rows.holm_p.min())))
    pd.DataFrame(families).to_csv(RUN/'families.csv',index=False)
    output=ROOT/'results/generated/INTERVENTION_PREPOST_INFERENCE.xlsx'
    with pd.ExcelWriter(output,engine='xlsxwriter') as writer:
        a.to_excel(writer,sheet_name='Scores and 95pct CI',index=False)
        b.to_excel(writer,sheet_name='All paired tests',index=False)
        pd.DataFrame(families).to_excel(writer,sheet_name='Holm families',index=False)
        pd.DataFrame([dict(item=k,description=v) for k,v in plan['methods'].items()]).to_excel(writer,sheet_name='Methods',index=False)
        for sh in writer.sheets.values():
            sh.freeze_panes(1,0);sh.autofilter(0,0,sh.dim_rowmax,sh.dim_colmax);sh.set_column(0,sh.dim_colmax,20)
    save(RUN/'validation.json',dict(status='complete',score_rows=len(a),paired_tests=len(b),n_boot=N_BOOT,n_perm=N_PERM,
        families=len(families),plan_sha256=sha(RUN/'plan.json'),script_sha256=sha(__file__),
        primary_estimates_unchanged=True,positive_holm_05=int(((b.holm_p<.05)&(b.delta>0)).sum()),negative_holm_05=int(((b.holm_p<.05)&(b.delta<0)).sum()),
        exact_identity_tests=int(b.exact_identity.fillna(False).sum()),
        outputs_sha256={str(p):sha(p) for p in [RUN/'scores_with_ci.csv',RUN/'paired_inference.csv',RUN/'families.csv',output]}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);plan,predictions=define_plan()
    print('Plan:',len(plan['contrasts']),'contrasts;',len(plan['family_sizes']),'families',flush=True)
    if args.plan_only:return
    tasks=sorted({(c['cohort'],c['task']) for c in plan['contrasts']});done=0;start=time.time()
    save(RUN/'status.json',dict(status='running',completed_tasks=0,total_tasks=len(tasks),automatic_shutdown=False))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_task,t):t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            try:path=f.result()
            except Exception as e:
                save(RUN/'status.json',dict(status='failed',error=str(e),failed_task=futures[f],completed_tasks=done,total_tasks=len(tasks),automatic_shutdown=False));raise
            done+=1;save(RUN/'status.json',dict(status='running',completed_tasks=done,total_tasks=len(tasks),elapsed_seconds=time.time()-start,last_completed=path,automatic_shutdown=False))
            print('Finished',done,'/',len(tasks),path,flush=True)
    aggregate();save(RUN/'status.json',dict(status='complete',completed_tasks=done,total_tasks=len(tasks),elapsed_seconds=time.time()-start,automatic_shutdown=False))


if __name__=='__main__':main()
