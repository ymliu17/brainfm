"""Additional shared-protocol comparisons; no absolute-null refitting."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from brainharmonix.data import ROOT,RUN,DATA,COHORTS,prepare,dataset,save,sha
from brainharmonix.extract import folder,load
MODEL='brainharmonix_f'

def read(p):return json.loads(Path(p).read_text())

def validate():
    from brainharmonix.models import Encoder
    from brainharmonix.data import transform
    torch.manual_seed(42)
    x=transform(torch.randn(450,20),2.)[0][None]
    net=Encoder(native_ssl=True)
    with torch.no_grad():
        a=net(x);b=net(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Changing unobserved frames must not affect observed-token readouts.
        changed=x.clone();changed[:,:400,36:]=100
        torch.testing.assert_close(a,net(changed),rtol=1e-5,atol=1e-6)
    loss=net.ssl_loss(x,123);other=net.ssl_loss(x,123)
    torch.testing.assert_close(loss,other,rtol=0,atol=0)
    loss.backward();grads=[p.grad for n,p in net.named_parameters() if 'lora_B' in n and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().max()>0 for g in grads)
    assert a.shape==(1,768) and torch.isfinite(a).all()
    save(RUN/'compatibility.json',dict(status='complete',checkpoint_sha256=sha(net.checkpoint_path),
        deterministic_inference=True,deterministic_validation_masks=True,finite_nonzero_adapter_gradients=True,
        encoder=net.provenance(),finite_loss=float(loss.detach()),test_input='Synthetic20-frame nativeTR2 test; full-data GPU timing is separate.'))

def input_fc(co):
    from transfer_baseline import fc_feature
    x,m=dataset(co,MODEL);values=[]
    for start in range(0,len(x),16):
        b=x[start:start+16];values.append(fc_feature(b[:,:400].numpy(),b[:,400,0].numpy().astype(int)))
    path=RUN/'supplements/input_fc'/MODEL/(co+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,np.concatenate(values));save(path.with_suffix('.json'),dict(status='complete',kind=MODEL,cohort=co,sha256=sha(path),
        scan_ids=m.scan.tolist(),protocol='Exact retained native samples and400ROIs; metadata rows and temporal padding excluded.'))

def shared_sensitivity(action,co,target):
    import reviewer_completion_tasks as r
    import benchmark_completion_tasks as core
    import unified_supplements as u
    root=RUN/'supplements'
    with patch.multiple(r,RUN=root,MODELS=(MODEL,),family=lambda _:MODEL),patch.object(u,'feature_folder',folder),patch.object(core,'feature',load):
        if action=='pca':r.pca(MODEL,co)
        elif action=='pca_inference':r.pca_inference(MODEL,co)
        elif action=='diagnostic':r.diagnostic(MODEL,target)
        elif action=='input_compare':r.input_comparison(MODEL,co)
        elif action=='identity_control':
            # Avoid old-four-specific NACC branch; this model has its own verified export.
            with patch.object(r,'MODELS',()):r.identity(MODEL,co)

def reference_compare(source,obj,target):
    from paper_paired_stats import paired
    from readout_sensitivity import seed_of
    seeds=(0,) if obj=='frozen' else (42,43,44);rows=[]
    for co in COHORTS[:-1]:
        if obj=='supervised' and source==co:continue
        for p in sorted((RUN/'references/sch400'/co).glob('*.json')):
            if p.name=='complete.json':continue
            r=read(p);task=r['task']
            if obj=='supervised' and task in ('AGE','SEX','AGE_GROUP'):continue
            qs=[read(RUN/'evaluation'/folder((source,MODEL,obj,target,s)).name/co/p.name) for s in seeds]
            for q in qs:
                assert q['subjects']==r['subjects'] and q['scan_ids']==r['scan_ids'] and q['y']==r['y']
            pred=np.mean([q['prediction'] for q in qs],axis=0)
            for method,base in r['predictions'].items():
                rows.append(dict(model=MODEL,source=source,objective=obj,target=target,cohort=co,task=task,reference='sch400_'+method,
                    scope='source_exposed_SSL' if obj=='native_ssl' and source==co else 'external_or_frozen',
                    **paired(r['y'],base,pred,r['subjects'],metric=r['metric'],seed=seed_of(MODEL+source+obj+target+co+task+method))))
    save(RUN/'reference_comparisons'/('_'.join((MODEL,source,obj,target))+'.json'),dict(status='complete',rows=rows))

def context(co):
    """Refit all nine configurations on the same complete-context intersection."""
    import benchmark_completion_tasks as core
    import unified_supplements as u
    import sf100_benchmark as sf
    root=RUN/'complete_context_nine';m=sf.labels(co);n=pd.read_csv(DATA/(co+'.csv'))
    old=read(ROOT/'runs/reviewer_completion_v1/context'/co/'mask.json')
    assert old['scan_ids']==m.scan.tolist()==n.scan.tolist()
    keep=np.asarray(old['eligible'],bool)&n.complete_context.to_numpy(bool)
    save(root/'context'/co/'mask.json',dict(status='complete',scan_ids=m.scan.tolist(),eligible=keep,
        definition='Original8 complete-context intersection AND18real BrainHarmonix-F patches; identical scans for all9.'))
    save(root/'audit/complete.json',dict(status='complete',intake_sha256=sha(RUN/'intake.json'),
        old_mask_sha256=sha(ROOT/'runs/reviewer_completion_v1/context'/co/'mask.json')))
    oldfolder=u.feature_folder;models=(*u.CONFIGS,MODEL)
    def folders(state):return folder(state) if state[1]==MODEL else oldfolder(state)
    with patch.multiple(u,CONFIGS=models,feature_folder=folders),patch.object(core,'RUN',root):
        core.references(co,context=True)
        for model in models:
            core.local(('none',model,'frozen','none',0),co,context=True)
            core.local_inference(model,co,context=True)
    save(root/(co+'.json'),dict(status='complete',models=list(models),complete_scans=int(keep.sum()),
        complete_subjects=int(m.loc[keep,'subject'].nunique()),empty_subsets='Reported as unavailable; no eligibility relaxation.'))

def joint():
    import joint_inference as p
    import reviewer_completion_tasks as r
    base=ROOT/'results/generated/reviewer_completion_v1/joint/Comparisons.csv'
    rows=pd.read_csv(base).replace({np.nan:None}).to_dict('records')
    directory=RUN/'reports/all';cache=RUN/'joint_cache';cache.mkdir(exist_ok=True)
    clinical=p.RUN/'original_adapted_clinical_inference.json'
    with patch.multiple(p,NEW=directory,RUN=cache,ADDED=(MODEL,),original_adapted_clinical=lambda:(clinical,read(clinical)['rows'])):
        candidates=p.build_rows()
    for row in candidates:
        if row['model']!=MODEL:continue
        if row['protocol']=='new_TR2_context':
            panel=row['panel'];v=dict(row,family=row['family'].removeprefix(panel+'::'))
            p.add(rows,v,panel,ROOT/row['input_file_provenance'],row['input_row'],protocol='brainharmonix_native_TR_context')
        else:rows.append(row)
    with patch.multiple(r,RUN=RUN/'supplements'):
        sensitivity,_=r.collect_sensitivity();rows.extend(sensitivity)
    for path in (RUN/'complete_context_nine/context_inference').glob('*/*.json'):
        for i,v in enumerate(read(path)['rows']):p.add(rows,v,'complete_context_nine',path,i,protocol='nine_configuration_complete_context')
    for path in (RUN/'reference_comparisons').glob('*.json'):
        for i,v in enumerate(read(path)['rows']):
            from joint_inference import atlas_family
            panel='atlas_frozen_reference' if v['objective']=='frozen' else 'atlas_adapted_reference'
            family=(atlas_family(v['task'],v['cohort']) if v['objective']=='frozen' else 'atlas_adapted_'+v['task']+'_'+v['objective']+'_'+v['target']+'_'+v['scope'])
            p.add(rows,v,panel,path,i,family=family,protocol='shared_exact_local_probe')
    # Keep independent-window analyses separate from contextual temporal-token analyses.
    path=ROOT/'runs/brainharmonix_f_window_v1/reports/all/paired_comparisons.json'
    for i,v in enumerate(read(path)):
        ref='frozen' if v['reference']=='same_model_frozen' else v['reference']
        kind='adaptation' if ref=='frozen' else 'FC' if ref.endswith('_FC') else 'nuisance'
        fam=('independent_window_scan_mean' if v['scope']=='scan_mean' else 'independent_window_dynamic_'+kind) if v['objective']=='frozen' else 'independent_window_'+v['scope']+'_'+v['objective']+'_'+v['target']+'_'+kind
        p.add(rows,dict(v,reference=ref,task='heart_or_pulse_rate',metric='rho',candidate='FM'),
            'physiology_common_window',path,i,family=fam,protocol='independent_windows_shared_refitted_probe')
    # Recall LOSO: average adaptation seeds before paired inference.
    from brainharmonix.report import states
    from collections import defaultdict
    from paper_paired_stats import paired
    from readout_sensitivity import seed_of
    groups=defaultdict(list)
    for state in states():groups[state[:4]].append(read(RUN/'supplements/recall'/folder(state).name/'result.json'))
    ref=read(ROOT/'runs/hrv_recall_external_v1/sf100_fc.json')
    for key,rr in groups.items():
        pred=np.mean([q['prediction'] for q in rr],0);first=rr[0]
        for q in rr:assert q['subjects']==first['subjects'] and q['y']==first['y']
        baselines={'sf100_fc':ref['prediction']}
        if key[2]!='frozen':baselines['frozen']=groups[('none',MODEL,'frozen','none')][0]['prediction']
        for refname,baseline in baselines.items():
            stat=paired(first['y'],baseline,pred,first['subjects'],seed=seed_of(str(key)+refname))
            fam='HRV_recall_external' if key[2]=='frozen' else 'HRV_recall_'+key[2]+'_'+key[3]+'_'+refname
            p.add(rows,dict(model=MODEL,source=key[0],objective=key[2],target=key[3],cohort='hrv_pre',task='MEM_delayed_recall',
                reference=refname,family=fam,**stat),'loso_frozen' if key[2]=='frozen' else 'loso_adaptation',
                RUN/'supplements/recall'/folder((*key,0 if key[2]=='frozen' else 42)).name/'result.json',refname)
    models=(*r.MODELS,MODEL)
    with patch.object(r,'MODELS',models):
        result=r.export_tables(rows,ROOT/'results/generated/brainharmonix_f_v1/joint','Nine_configuration_SI')
    save(RUN/'joint_complete.json',dict(result,configuration_count=9,model_family_count=5,
        parent_comparisons_sha256=sha(base),manuscript_editing=False,automatic_shutdown=False))

def execute(a,state):
    action=a.action
    if action=='prepare':prepare()
    elif action=='validate':validate()
    elif action=='recall':
        import unified_supplements as u
        with patch.multiple(u,RUN=RUN/'supplements',feature_folder=folder):u.recall(state)
    elif action=='reference':
        from brainharmonix.references import fc_reference
        fc_reference(a.cohort,'sch400')
    elif action=='reference_compare':reference_compare(a.source,a.objective,a.target)
    elif action=='input_fc':input_fc(a.cohort)
    elif action=='context':context(a.cohort)
    elif action in ('pca','pca_inference','diagnostic','input_compare','identity_control'):shared_sensitivity(action,a.cohort,a.target)
    elif action.startswith('phys_'):
        from brainharmonix import physiology as p
        if action=='phys_prepare':
            p.prepare();save(p.RUN/'prepared.json',dict(status='complete',protocol_sha256=sha(p.RUN/'protocol.json')))
        elif action=='phys_extract':p.extract(state,a.device)
        elif action=='phys_evaluate':p.evaluate(state,a.cohort)
        else:p.report(a.phase)
    elif action=='joint':joint()
    else:raise ValueError(action)
