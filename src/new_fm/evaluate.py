"""Apply existing task populations, folds, probes and inference to the new encoders."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from new_fm.data import ROOT,RUN,COHORTS,MODELS,save,sha,metadata
from new_fm.extract import folder,load
from fm_readout_v4.evaluate import tag
from benchmark_evaluation import predict_prepared,prepare_nested,score
from paper_completion import project_block
import sf100_benchmark as sf
import sample_alignment as sa

def read(p):return json.loads(Path(p).read_text())
def state_name(s):return tag(*s)
def rowdir(s):return RUN/'evaluation'/state_name(s)

def local(state,co):
    out=rowdir(state)/co;out.mkdir(parents=True,exist_ok=True)
    if state[2]=='supervised' and co==state[0]:
        save(out/'complete.json',dict(status='not_applicable',reason='External source encoder cannot be tested on its own source; separate fold-specific source-CV jobs.'));return
    m=sf.labels(co);x=load(state,co)
    for p in sorted((sf.RUN/'evaluation'/co).glob('*.json')):
        if p.name=='complete.json':continue
        old=read(p)
        if old.get('status')!='complete':continue
        task=old['task']
        if state[2]=='supervised' and task in ('AGE','SEX','AGE_GROUP'):continue
        dest=out/p.name
        if dest.exists():continue
        ix=m.set_index('scan').index.get_indexer(old['scan_ids'])
        if np.any(ix<0):raise ValueError('Missing common scan')
        y=np.asarray(old['y'],float);g=m.iloc[ix].subject.astype(str).to_numpy();cov=old['covariates']
        if g.tolist()!=old['subjects']:raise ValueError('Participant mismatch')
        prepared={s:[] for s in (0,1,2)}
        for split in old['split_audit']:
            tr,te,it,iv=[np.asarray(split[k]) for k in ('train','test','inner_train','validation')]
            if set(g[tr])&set(g[te]) or set(g[it])&set(g[iv]):raise ValueError('Split overlap')
            ca,cb=sf.covariate_pair(m,cov,ix[tr],ix[te]);ci,cj=sf.covariate_pair(m,cov,ix[it],ix[iv])
            a,b=project_block(x[ix[tr]],x[ix[te]],5 if cov else 100)
            ai,bi=project_block(x[ix[it]],x[ix[iv]],5 if cov else 100)
            prepared[split['seed']].append((tr,te,(np.c_[ca,a],np.c_[cb,b]),it,iv,(np.c_[ci,ai],np.c_[cj,bi])))
        binary=old['metrics']['sf100_fc']['metric']=='auc'
        pred=[predict_prepared(prepared[s],y,binary=binary) for s in (0,1,2)];pp=np.mean(pred,axis=0)
        save(dest,dict(status='complete',state=state,cohort=co,task=task,scan_ids=old['scan_ids'],subjects=g,y=y,
            prediction=pp,predictions_by_seed=pred,split_audit=old['split_audit'],covariates=cov,metric='auc' if binary else 'rho',
            score=score(y,pp,binary),mae=None if binary else float(np.mean(abs(y-pp))),rmse=None if binary else float(np.sqrt(np.mean((y-pp)**2))),
            reference_file=str(p),reference_sha256=sha(p),
            scope='source_exposed_SSL' if state[2]=='native_ssl' and co==state[0] else 'external_or_frozen'))
        print('local',state_name(state),co,task,score(y,pp,binary),flush=True)
    # Released HRV categorical age task is a separate existing demographic endpoint.
    if co.startswith('hrv') and state[2]!='supervised':
        import demographic_benchmark as demo
        dm=pd.read_csv(demo.RUN/'labels'/(co+'.csv'));ix=np.flatnonzero(sa.common_mask(m,co)&dm.age_group.notna().to_numpy())
        y=dm.iloc[ix].age_group.to_numpy(float);g=m.iloc[ix].subject.to_numpy();pred=[]
        for seed in (0,1,2):pred.append(demo.predict_multiclass(prepare_nested([x[ix]],[100],y,g,seed),y,len(demo.AGE_BINS)))
        save(out/'AGE_GROUP.json',dict(status='complete',state=state,cohort=co,task='AGE_GROUP',scan_ids=m.iloc[ix].scan.tolist(),subjects=g,
            y=y,prediction=np.mean(pred,axis=0),predictions_by_seed=pred,metrics=demo.metrics('AGE_GROUP',y,np.mean(pred,axis=0)),
            metric='multiclass',scope='Exploratory released age category; not age in years'))
    if co=='hrv_pre':
        from fm_readout_v4.evaluate import clinical_features
        from fm_matched_v3.probes import oof_predictions
        c=clinical_features(m,co);y0=m.MEM_delayed_recall.to_numpy(float)
        ix=np.flatnonzero(sa.common_mask(m,co)&np.isfinite(y0)&np.isfinite(c).all(1));y=y0[ix];g=m.iloc[ix].subject.to_numpy()
        for label,blocks,dims in [('clinical',[c[ix]],[c.shape[1]]),('clinical+FM',[c[ix],x[ix]],[c.shape[1],5])]:
            pp=[oof_predictions(blocks,dims,y,s,g) for s in (0,1,2)]
            save(out/('MEM_delayed_recall_'+label+'.json'),dict(status='complete',state=state,cohort=co,
                task='MEM_delayed_recall_baseline_prediction',method=label,scan_ids=m.iloc[ix].scan.tolist(),subjects=g,y=y,
                prediction=np.mean(pp,axis=0),predictions_by_seed=pp,metric='rho',covariates=list(range(c.shape[1])),
                score=score(y,np.mean(pp,axis=0),False),scope='Existing clinical recall probe protocol'))
    save(out/'complete.json',dict(status='complete',state=state,cohort=co))

def axes(state):
    import paper_completion as pc
    original_mask=pc.primary_mask
    sources=('adni','nacc') if state[2]=='frozen' else (state[0],)
    def mask(m,co):return original_mask(m,co) if co in sources else sa.common_mask(m,co)
    with patch.multiple(pc,feature_dir=folder,primary_mask=mask):rows=pc.axes(state)
    save(rowdir(state)/'axes.json',dict(status='complete',state=state,rows=rows,
        protocol='Existing source-only StandardScaler/randomized PCA<=100/ridge10; common eligible target scans; HRV MEM maps to delayed recall.'))

def loso(state):
    from run_sf100_loso import fit_held,COHORTS as trials
    tables={co:sf.labels(co) for co in trials};masks={co:sa.common_mask(tables[co],co) for co in trials};arrays={co:load(state,co) for co in trials}
    rows=[]
    for domain in ('MEM','EF'):
        for held in (trials if domain=='EF' else trials[:-1]):rows.append(fit_held(arrays,tables,masks,domain,held))
    save(rowdir(state)/'loso.json',dict(status='complete',state=state,rows=rows))

def identification(state):
    from readout_sensitivity import similarity,identification_scores
    rows=[]
    for co in ('adni','nacc_repeated','act','cogte','breathe','psopt','hrv'):
        subjects,(ca,ia),(cb,ib)=sa.paired_rows(co)
        if state[2]=='supervised' and (co==state[0] or (co=='nacc_repeated' and state[0]=='nacc')):continue
        sim=similarity(load(state,ca)[ia],load(state,cb)[ib])
        rows.append(dict(cohort=co,subjects=subjects,similarity=sim,scores=identification_scores(sim),
            scans_a=metadata(ca).iloc[ia].scan.tolist(),scans_b=metadata(cb).iloc[ib].scan.tolist()))
    save(rowdir(state)/'identification.json',dict(status='complete',state=state,rows=rows))

def paired_visit(state):
    import intervention_prepost as engine
    out=RUN/'paired_visit'/state_name(state);out.mkdir(parents=True,exist_ok=True)
    parent=read(ROOT/'runs/intervention_prepost_v1/inputs.json')
    d=copy.deepcopy(parent);d['states']=[state];d['files_sha256']={}
    for co in ('act','cogte','breathe','psopt','hrv_pre','hrv_post'):
        p=folder(state)/(co+'.npy');d['files_sha256'][str(p)]=sha(p)
    import hashlib
    d['signature']=hashlib.sha256(json.dumps(dict(parent=parent['signature'],state=state,files=d['files_sha256']),sort_keys=True).encode()).hexdigest()
    save(out/'inputs.json',d)
    with patch.multiple(engine,RUN=out,feature_path=lambda s,co:folder(s)/(co+'.npy')):
        for co in engine.COHORTS:engine.run_job((state,co))
    save(out/'complete.json',dict(status='complete',state=state,parent_signature=parent['signature'],
        protocol='Exact existing paired participants/folds; baseline, post-minus-pre, concatenated inputs;5 PCA components; clinical-only/plus-imaging; MEM/EF change and arms.'))

def readouts(model):
    state=('none',model,'frozen','none',0);rows=[]
    for co in ('adni','nacc'):
        m=sf.labels(co);xs={'native':load(state,co),**dict(np.load(folder(state)/(co+'_alternatives.npz')))}
        for endpoint in ('MEM','EF'):
            old=read(sf.RUN/'evaluation'/co/(endpoint+'.json'));ix=m.set_index('scan').index.get_indexer(old['scan_ids']);y=np.asarray(old['y']);g=np.asarray(old['subjects']);pred={}
            for name,x in xs.items():
                byseed={s:[] for s in (0,1,2)}
                for split in old['split_audit']:
                    tr,te,it,iv=[np.asarray(split[k]) for k in ('train','test','inner_train','validation')]
                    byseed[split['seed']].append((tr,te,project_block(x[ix[tr]],x[ix[te]],100),it,iv,project_block(x[ix[it]],x[ix[iv]],100)))
                pred[name]=np.mean([predict_prepared(byseed[s],y,binary=False) for s in (0,1,2)],axis=0)
            rows.append(dict(cohort=co,task=endpoint,subjects=g,scan_ids=old['scan_ids'],y=y,predictions=pred,
                scores={k:score(y,v,False) for k,v in pred.items()}))
    from readout_sensitivity import similarity,identification_scores
    identity=[]
    for co in ('adni','nacc_repeated','act','cogte','breathe','psopt','hrv'):
        subjects,(ca,ia),(cb,ib)=sa.paired_rows(co)
        aa={'native':load(state,ca),**dict(np.load(folder(state)/(ca+'_alternatives.npz')))}
        bb={'native':load(state,cb),**dict(np.load(folder(state)/(cb+'_alternatives.npz')))}
        identity.append(dict(cohort=co,subjects=subjects,scores={k:identification_scores(similarity(aa[k][ia],bb[k][ib])) for k in aa}))
    save(RUN/'readouts'/(model+'.json'),dict(status='complete',rows=rows,identification=identity))

def canonical(state):
    rows=[]
    for co in COHORTS[:-1]:
        for p in (rowdir(state)/co).glob('*.json'):
            r=read(p)
            if 'prediction' not in r or r.get('metric')!='rho':continue
            task=r['task']
            if task in ('MEM','EF'):task+=('_concurrent_post' if co=='hrv_post' else '_baseline')
            if task=='MEM_delayed_recall':task='MEM_delayed_recall_baseline_prediction' if co=='hrv_pre' else 'MEM_delayed_recall_concurrent_post'
            row=dict(r,task=task,method=r.get('method','clinical+FM' if r.get('covariates') else 'FM'));rows.append(row)
            # HRV recall clinical arm follows existing local clinical protocol separately.
    save(RUN/'canonical'/(state_name(state)+'.json'),dict(status='complete',state=state,rows=rows,axes=read(rowdir(state)/'axes.json')['rows']))

def harmonization(state):
    import embedding_harmonization as engine
    # Reuse axis and fold-local alignment implementation; keep artifacts in additive release.
    out=RUN/'harmonization';out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():raise ValueError('Register queue protocol before alignment')
    with patch.multiple(engine,RUN=out,CANON=RUN/'canonical',load_features=lambda s,co:np.asarray(load(s,co),np.float64)):
        for source in (('adni','nacc') if state[2]=='frozen' else (state[0],)):engine.execute(state,source)
    save(out/(state_name(state)+'_complete.json'),dict(status='complete'))
