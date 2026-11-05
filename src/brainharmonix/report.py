"""Seed-averaged additive results and conditional inference; no parent overwrites."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from brainharmonix.data import ROOT,RUN,MODELS,COHORTS,save,sha,metadata
from brainharmonix.evaluate import rowdir,read,state_name
from paper_paired_stats import paired,holm
from readout_sensitivity import seed_of,identification_scores,paired_identification
from benchmark_evaluation import score

def states(phase='all'):
    for model in MODELS:
        yield ('none',model,'frozen','none',0)
        if phase=='frozen':continue
        for source in ('adni','nacc'):
            for obj,target in [('supervised','MEM'),('supervised','EF'),('native_ssl','none')]:
                for seed in (42,43,44):yield source,model,obj,target,seed

def average(rr,key='prediction'):
    first=rr[0]
    for r in rr[1:]:
        for field in ('subjects','y','scan_ids'):
            if field in first and first[field]!=r[field]:raise ValueError('Unpaired seed aggregation '+field)
    return np.mean([r[key] for r in rr],axis=0)

def adjusted(rows):
    families=defaultdict(list)
    for i,r in enumerate(rows):families[r['family']].append(i)
    for ids in families.values():
        for i,p in zip(ids,holm([rows[i].get('permutation_p') for i in ids])):
            rows[i]['holm_p']=p;rows[i]['family_size']=len(ids)
    return rows

def report(phase):
    out=RUN/'reports'/phase;out.mkdir(parents=True,exist_ok=True);summary=[];contrasts=[];predictions=[]
    grouped=defaultdict(list)
    for s in states(phase):
        for co in COHORTS[:-1]:
            for p in (rowdir(s)/co).glob('*.json'):
                r=read(p)
                if 'prediction' not in r:continue
                key=(*s[:4],co,r['task'],r.get('method','primary'))
                grouped[key].append(r)
    for key,rr in grouped.items():
        source,model,obj,target,co,task,method=key
        if len(rr)!=(1 if obj=='frozen' else 3):raise ValueError('Incomplete adaptation seeds '+str(key))
        r=rr[0];p=average(rr);metric=r['metric'];y=np.asarray(r['y'])
        if metric=='multiclass':
            from demographic_benchmark import metrics
            met=metrics('AGE_GROUP',y,p);value=met.get('auc_macro',met.get('balanced_accuracy'))
        else:value=score(y,p,metric=='auc');met=dict(score=value)
        info=dict(source=source,model=model,objective=obj,target=target,cohort=co,task=task,method=method,n=len(y),metric=metric,**met)
        summary.append(info);predictions.append(dict(info,subjects=r['subjects'],scan_ids=r['scan_ids'],y=y,prediction=p))
        if metric=='multiclass':continue
        if r.get('reference_file'):
            old=read(r['reference_file'])
            if old['scan_ids']!=r['scan_ids'] or old['subjects']!=r['subjects'] or old['y']!=r['y']:raise ValueError('Reference population changed')
            if obj=='frozen':
                contrasts.append({**info,'reference':'sf100_fc','family':'local_'+task,
                    **paired(y,old['predictions']['sf100_fc'],p,r['subjects'],metric=metric,seed=seed_of(str(key)))})
            if 'clinical' in old['predictions']:
                family=('incremental_'+task if obj=='frozen' else 'adapted_incremental_'+task+'_'+obj+'_'+target)
                contrasts.append({**info,'reference':'clinical','family':family,
                    **paired(y,old['predictions']['clinical'],p,r['subjects'],metric=metric,seed=seed_of(str(key)+'clinical'))})
            if obj!='frozen':
                frozen=grouped[('none',model,'frozen','none',co,task,method)][0]
                if frozen['scan_ids']!=r['scan_ids']:raise ValueError('Frozen/adapted participants differ')
                contrasts.append({**info,'reference':model+'_frozen','family':'adaptation_local_'+task+'_'+obj+'_'+target,
                    **paired(y,frozen['prediction'],p,r['subjects'],metric=metric,seed=seed_of(str(key)))})
    # Recompute Holm over the existing + added frozen comparison family.
    for p in (ROOT/'runs/sf100_benchmark_v1/evaluation').glob('*/*.json'):
        old=read(p)
        if old.get('status')!='complete' or 'predictions' not in old:continue
        for r in old.get('rows',[]):
            if r['family'].startswith(('local_','incremental_')):contrasts.append(dict(r,origin='existing_frozen_panel'))
        for model,v in old['metrics'].items():
            summary.append(dict(source='none',model=model,objective='frozen',target='none',cohort=old['cohort'],task=old['task'],
                method='primary',n=len(old['y']),origin='existing_frozen_panel',**v))
    save(out/'local_predictions.json',predictions);pd.DataFrame(summary).to_csv(out/'local_scores.csv',index=False)
    save(out/'local_comparisons.json',adjusted(contrasts));pd.DataFrame(contrasts).to_csv(out/'local_comparisons.csv',index=False)
    transfer_report(phase,out)
    identity_report(phase,out)
    paired_report(phase,out)
    sensitivity_report(out)
    harmonization_report(phase,out)
    if phase=='all':source_report(out)
    save(out/'complete.json',dict(status='complete',phase=phase,models=list(MODELS),
        input_sha256=sha(RUN/'intake.json'),local_scores=len(summary),local_contrasts=len(contrasts),
        aggregation='Split predictions averaged before scoring; adapted predictions average42/43/44; identification averages similarity matrices.',
        inference='5000 participant-cluster bootstrap draws and20000 conditional swaps; expanded frozen families include archived raw P values; old files unchanged.',
        physiology='Separate independent-window reencoding supplement; native temporal-token panel is not pooled with it.',
        automatic_shutdown=False))

def transfer_report(phase,out):
    for action in ('axes','loso'):
        groups=defaultdict(list)
        for s in states(phase):
            for r in read(rowdir(s)/(action+'.json'))['rows']:
                k=(*s[:4],r.get('source',s[0]),r['cohort'],r.get('task',r.get('domain')));groups[k].append(r)
        rows=[];stats=[]
        for k,rr in groups.items():
            r=rr[0];p=average(rr);info=dict(source=k[0],model=k[1],objective=k[2],target=k[3],axis_source=k[4],cohort=k[5],task=k[6])
            rows.append(dict(info,n=len(p),rho=score(r['y'],p,False),subjects=r['subjects'],y=r['y'],prediction=p))
            if action=='loso' and k[2]=='frozen':
                old=read(ROOT/'runs/sf100_loso_v1/sf100_fc/loso.json')['rows']
                ref=next(a for a in old if a['cohort']==k[5] and a['domain']==k[6])
                for field in ('subjects','scan_ids','y','training_subjects'):
                    if ref[field]!=r[field]:raise ValueError('LOSO reference pairing mismatch '+field)
                stats.append(dict(info,reference='sf100_fc',family='LOSO_'+k[6],
                    **paired(r['y'],ref['prediction'],p,r['subjects'],seed=seed_of(str(k)))))
            if k[2]!='frozen':
                fk=('none',k[1],'frozen','none',k[4] if action=='axes' else 'none',k[5],k[6])
                a=groups[fk][0]
                if a['subjects']!=r['subjects'] or a['y']!=r['y']:raise ValueError('Transfer pairing mismatch')
                stats.append(dict(info,family=action+'_'+str(k[6])+'_'+k[2]+'_'+k[3],
                    **paired(r['y'],a['prediction'],p,r['subjects'],seed=seed_of(str(k)))))
        if action=='loso':stats.extend(read(ROOT/'runs/sf100_loso_v1/report.json')['rows'])
        save(out/(action+'_predictions.json'),rows);pd.DataFrame([{k:v for k,v in r.items() if k not in ('subjects','y','prediction')} for r in rows]).to_csv(out/(action+'_scores.csv'),index=False)
        save(out/(action+'_comparisons.json'),adjusted(stats))

def identity_report(phase,out):
    groups=defaultdict(list)
    for s in states(phase):
        for r in read(rowdir(s)/'identification.json')['rows']:groups[(*s[:4],r['cohort'])].append(r)
    rows=[];stats=[]
    for k,rr in groups.items():
        for r in rr[1:]:
            if r['subjects']!=rr[0]['subjects']:raise ValueError('Identification population mismatch')
        s=identification_scores(np.mean([r['similarity'] for r in rr],axis=0));info=dict(source=k[0],model=k[1],objective=k[2],target=k[3],cohort=k[4])
        rows.append(dict(info,**{n:v for n,v in s.items() if not n.startswith('per_subject')}))
        if k[2]=='frozen':
            old=read(ROOT/'runs/sf100_benchmark_v1/identification'/(k[4]+'.json'))
            if old['subjects']!=rr[0]['subjects']:raise ValueError('Identification reference population mismatch')
            ref=old['scores']['sf100_fc']
        else:ref=identification_scores(np.asarray(groups[('none',k[1],'frozen','none',k[4])][0]['similarity']))
        for metric in ('top1','I_diff'):
            family='identification_'+metric if k[2]=='frozen' else 'identification_'+metric+'_'+k[2]+'_'+k[3]
            stats.append(dict(info,metric=metric,family=family,
                **paired_identification(ref['per_subject_'+metric],s['per_subject_'+metric],seed=seed_of(str(k)+metric))))
    stats.extend(read(ROOT/'runs/sf100_benchmark_v1/identification/complete.json')['rows'])
    pd.DataFrame(rows).to_csv(out/'identification_scores.csv',index=False);save(out/'identification_comparisons.json',adjusted(stats))

def paired_report(phase,out):
    import intervention_prepost as engine
    import intervention_prepost_inference as infer
    groups=defaultdict(list)
    for s in states(phase):
        for co in engine.COHORTS:
            p=RUN/'paired_visit'/state_name(s)/'jobs'/engine.state_name(s)/(co+'.json')
            for r in read(p)['rows']:
                if r['mode'] in ('clinical','null'):continue
                groups[(co,r['task'],*s[:4],r['mode'])].append(r)
    predictions=[r for r in read(ROOT/'runs/intervention_prepost_v1/predictions.json') if r['model'] in ('sf100_fc','clinical','null')]
    for k,rr in groups.items():
        co,task,source,model,obj,target,mode=k;r=rr[0];p=average(rr)
        if len(rr)!=(1 if obj=='frozen' else 3):raise ValueError('Incomplete paired-visit seeds')
        predictions.append(dict(cohort=co,task=task,source=source,model=model,objective=obj,target=target,mode=mode,n=len(p),
            subjects=r['subjects'],y=r['y'],classes=r['classes'],prediction=p,adaptation_seeds=len(rr)))
    # Include all archived states so the expanded families are corrected together.
    base=out/'paired_visit';base.mkdir(parents=True,exist_ok=True);save(base/'predictions.json',predictions)
    scores=[dict(**{k:r[k] for k in infer.KEYS},**engine.metrics(np.asarray(r['y']),np.asarray(r['prediction']),r['classes'])) for r in predictions]
    pd.DataFrame(scores).to_csv(base/'scores.csv',index=False)
    define_paired_plan(base,predictions)
    with patch.multiple(infer,BASE=base,RUN=base/'inference'):
        for co,task in sorted({(r['cohort'],r['task']) for r in predictions}):infer.run_task((co,task))
    stats={};score_rows={}
    for directory in (ROOT/'runs/intervention_prepost_v1/inference_v1/tasks',base/'inference/tasks'):
        for p in directory.glob('*.json'):
            d=read(p)
            for r in d['scores']:score_rows[infer.uid(r)]=r
            for r in d['contrasts']:stats[(r['reference_id'],r['candidate_id'],r['contrast'])]=r
    expanded=adjusted(list(stats.values()))
    pd.DataFrame(score_rows.values()).to_csv(base/'scores_with_ci.csv',index=False)
    pd.DataFrame(expanded).to_csv(base/'paired_inference.csv',index=False)
    save(base/'complete.json',dict(status='complete',score_rows=len(score_rows),paired_tests=len(expanded),
        reuse='Archived raw contrasts/scores reused; expanded Holm families recomputed. No original predictions or manuscript artifacts overwritten.'))

from unittest.mock import patch

def paired_contrasts(predictions):
    """Parent contrast definitions without its historical fixed row-count assertion."""
    import intervention_prepost_inference as infer
    lookup={infer.key(r):r for r in predictions}
    if len(lookup)!=len(predictions):raise ValueError('Duplicate paired-visit prediction')
    rows=[]
    def add(r,ref,block):
        a=lookup[infer.key(ref)]
        for k in ('subjects','y','classes'):
            if r[k]!=a[k]:raise ValueError('Unpaired intervention '+k)
        if len(set(r['subjects']))!=len(r['subjects']):raise ValueError('Repeated paired participant')
        context='clinical_plus_imaging' if '+clinical' in r['mode'] else 'imaging_only'
        rows.append(dict(**{k:r[k] for k in infer.KEYS},reference_id=infer.uid(a),candidate_id=infer.uid(r),
            contrast=block,family='|'.join([block,r['task'],infer.objective(r),context]),
            metric='auc' if r['task']=='arm' else 'rho',n=len(r['y'])))
    for r in predictions:
        if r['objective']=='reference':continue
        if r['mode'] in ('change','baseline_plus_change','change+clinical','baseline_plus_change+clinical'):
            add(r,dict(r,mode='baseline'+('+clinical' if '+clinical' in r['mode'] else '')),'input_vs_baseline')
        if r['objective']!='frozen':add(r,dict(r,source='none',objective='frozen',target='none'),'adapted_vs_frozen')
        if '+clinical' in r['mode']:
            add(r,dict(r,source='none',model='clinical',objective='reference',target='none',mode='clinical'),'imaging_added_to_clinical')
        if r['objective']=='frozen' and r['model']!='sf100_fc':
            add(r,dict(r,source='comparator',model='sf100_fc',target='none'),'frozen_vs_sf100')
    return rows

def define_paired_plan(base,predictions):
    from collections import Counter
    import intervention_prepost_inference as infer
    rows=paired_contrasts(predictions);sizes=dict(Counter(r['family'] for r in rows))
    parent=ROOT/'runs/intervention_prepost_v1/inference_v1/plan.json'
    plan=dict(status='planned_before_resampling',contrasts=rows,family_sizes=sizes,
        source_predictions_sha256=sha(base/'predictions.json'),source_scores_sha256=sha(base/'scores.csv'),
        parent_protocol_sha256=sha(parent),methods=read(parent)['methods'],n_boot=5000,n_perm=20000,
        automatic_shutdown=False)
    p=base/'inference/plan.json'
    if p.exists() and read(p)!=plan:raise ValueError('Paired inference plan changed')
    save(p,plan)

def harmonization_report(phase,out):
    import embedding_harmonization as engine
    root=RUN/'harmonization';keys=set()
    for state in states(phase):
        source,model,obj,target,_=state
        for origin in (('adni','nacc') if obj=='frozen' else (source,)):
            keys.add((model,origin,obj,target))
    files=[]
    with patch.object(engine,'RUN',root):
        for model,source,obj,target in sorted(keys):
            label=engine.infer(model,source,obj,target)
            files.append(root/'inference'/(label+'.json'))
    rows=[];scores=[]
    # Existing raw contrasts expand the original families, without rewriting them.
    files+=sorted((ROOT/'runs/embedding_harmonization_v2/inference').glob('*.json'))
    for p in files:
        d=read(p)
        for r in d['rows']:
            if phase=='frozen' and r['objective']!='frozen':continue
            rows.append(r)
        for r in d['scores']:
            if phase!='frozen' or r['objective']=='frozen':scores.append(r)
    save(out/'harmonization_comparisons.json',adjusted(rows))
    pd.DataFrame(rows).to_csv(out/'harmonization_comparisons.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'harmonization_scores.csv',index=False)

def sensitivity_report(out):
    rows=[]
    for model in MODELS:
        d=read(RUN/'readouts'/(model+'.json'))
        for r in d['rows']:
            for candidate,p in r['predictions'].items():
                if candidate=='native':continue
                rows.append(dict(model=model,cohort=r['cohort'],task=r['task'],candidate=candidate,family='readout_'+r['task'],
                    **paired(r['y'],r['predictions']['native'],p,r['subjects'],seed=seed_of(model+r['cohort']+r['task']+candidate))))
        for r in d['identification']:
            native=r['scores']['native']
            for candidate,scores in r['scores'].items():
                if candidate=='native':continue
                for metric in ('top1','I_diff'):
                    rows.append(dict(model=model,cohort=r['cohort'],task='identification',metric=metric,
                        candidate=candidate,family='readout_identification_'+metric,
                        **paired_identification(native['per_subject_'+metric],scores['per_subject_'+metric],
                            seed=seed_of(model+r['cohort']+candidate+metric))))
    save(out/'readout_comparisons.json',adjusted(rows))
    fusion=[]
    for p in (RUN/'fusion').glob('*/*.json'):
        d=read(p)
        if 'rows' not in d:continue
        rs={r['method']:r for r in d['rows']};r=d['rows'][0]
        pairs=[('FC','FM'),('FC','FC+FM')] if 'FC' in rs else [('clinical+FC','clinical+FC+FM'),('clinical','clinical+FC+FM'),('clinical+FC5','clinical+FC+FM')]
        for a,b in pairs:
            fusion.append(dict(model=p.parent.name,cohort=r['cohort'],task=r['task'],reference=a,candidate=b,family='fusion_'+r['task']+'_'+a+'_'+b,
                **paired(r['y'],rs[a]['prediction'],rs[b]['prediction'],r['subjects'],seed=seed_of(str(p)+a+b))))
    save(out/'fusion_comparisons.json',adjusted(fusion))
    robustness=[]
    for p in (RUN/'sensitivity/evaluation').glob('*/*/*.json'):robustness.extend(read(p).get('rows',[]))
    save(out/'robustness_comparisons.json',adjusted(robustness))

def source_report(out):
    from brainharmonix.train import location
    rows=[]
    for model in MODELS:
        for source in ('adni','nacc'):
            for target in ('MEM','EF'):
                seeds=[]
                for seed in (42,43,44):
                    d=read(location(source,model,'supervised',target,seed,'cv')/'complete.json');parts=d['folds']
                    ix=np.concatenate([r['test_indices'] for r in parts]);order=np.argsort(ix)
                    seeds.append(dict(indices=ix[order],y=np.concatenate([r['y'] for r in parts])[order],
                        a=np.concatenate([r['frozen_prediction'] for r in parts])[order],b=np.concatenate([r['adapted_prediction'] for r in parts])[order]))
                for z in seeds[1:]:
                    np.testing.assert_array_equal(z['indices'],seeds[0]['indices']);np.testing.assert_array_equal(z['y'],seeds[0]['y'])
                a=np.mean([z['a'] for z in seeds],0);b=np.mean([z['b'] for z in seeds],0);z=seeds[0];g=metadata(source).iloc[z['indices']].subject.to_numpy()
                rows.append(dict(model=model,source=source,target=target,family='source_cv_'+target,
                    **paired(z['y'],a,b,g,seed=seed_of(model+source+target))))
    save(out/'source_cv_comparisons.json',adjusted(rows))
