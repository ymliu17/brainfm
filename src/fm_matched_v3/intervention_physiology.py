"""Existing BREATHE/pSOPT window-probe protocol applied to common-readout exports."""
import json
import numpy as np
import pandas as pd
from fm_matched_v3.data import ROOT,RUN,DATA,save
from fm_matched_v3.supplement import window_dir as feature_dir
from fm_matched_v3.evaluate import tag
from fm_matrix.evaluate import array,row,comparison
from benchmark_evaluation import score

def physiology(source,model,objective,target,seed):
    f=feature_dir(source,model,objective,target,seed);f0=feature_dir('none',model,'frozen','none',0);rows=[];comparisons=[]
    from rct_inscanner_patch_hr import subj_key,COHORTS as HR
    from hrv_ppg_benchmark import valid_window_rows
    for name in ['breathe','psopt']:
        if name=='hrv':w=pd.read_csv(ROOT/'data/HRV_biofeedback/label/HRV_biofeedback_PPG_HRV_windows48s_full.csv');w=w.loc[valid_window_rows(w)&w.WindowIdx.between(0,7)].copy();w['key']=w.Subject;w['visit']=w.Timepoint;targets={'pulse_rate':'HRV_MeanHR','log_RMSSD':'HRV_RMSSD'};ds=['hrv_pre','hrv_post']
        else:
            w=pd.read_csv(ROOT/HR[name]['hr']);w=w[w.Usable_HR&w.MeanHR.between(40,120)&w.WindowIdx.between(0,7)].copy()
            if 'Run' in w:w=w[w.Run.eq('rest-run01')]
            w['key']=w.Subject.map(subj_key);w['visit']=w.Timepoint;targets={'heart_rate':'MeanHR'};ds=[name]
        lookup={};lookup0={};groups={}
        for d in ds:
            m=pd.read_csv(DATA/f'{d}.csv');idx=np.load(f/(d+'_window_indices.npy'));z=array(f,d+'_windows');z0=array(f0,d+'_windows')
            for j,i in enumerate(idx):
                r=m.iloc[i];key=r.subject if name=='hrv' else subj_key(r.subject);visit=r.visit;lookup[key,visit]=z[j];lookup0[key,visit]=z0[j];groups[key,visit]=r.subject
        for task,col in targets.items():
            X=[];X0=[];Y=[];G=[];scan=[]
            for (key,visit),tab in w.groupby(['key','visit']):
                if (key,visit) not in lookup or sorted(tab.WindowIdx.tolist())!=list(range(8)):continue
                y=tab.sort_values('WindowIdx')[col].to_numpy(float)
                if task.startswith('log_'):
                    if (y<=0).any():continue
                    y=np.log(y)
                if not np.isfinite(y).all():continue
                X.append(lookup[key,visit]);X0.append(lookup0[key,visit]);Y.extend(y-y.mean());G.extend([groups[key,visit]]*8);scan.extend([str(key)+'_'+visit]*8)
            if len(set(G))<5:
                rows.append(dict(cohort=name,task=task,status='insufficient_subjects',subjects=len(set(G))));continue
            X=np.concatenate(X);X0=np.concatenate(X0);y=np.asarray(Y);g=np.asarray(G)
            from benchmark_evaluation import prepare_nested,predict_prepared
            prepared=[prepare_nested([X],[10],np.zeros(len(y)),g,s) for s in [0,1,2]]
            def physiology_predict(yy):return np.mean([predict_prepared(v,yy,binary=False) for v in prepared],axis=0)
            p=physiology_predict(y);rr=row(name,task,y,p,g,scan=scan)
            # Circular-shift null refits the predictor and preserves temporal ordering within each scan.
            rng=np.random.default_rng(0);null=[]
            for _ in range(200):
                yy=y.copy()
                for s in np.unique(scan):
                    ix=np.flatnonzero(np.asarray(scan)==s);yy[ix]=np.roll(y[ix],int(rng.integers(1,len(ix))))
                null.append(score(yy,physiology_predict(yy),False))
            rr['circular_shift_p']=(1+sum(abs(v)>=abs(rr['rho']) for v in null))/201;rows.append(rr)
            if objective!='frozen':
                p0=np.mean([predict_prepared(prepare_nested([X0],[10],np.zeros(len(y)),g,s),y,binary=False) for s in [0,1,2]],axis=0)
                comparisons.append(dict(cohort=name,task=task,**comparison(y,p0,p,g)))
    save(RUN/'supplement'/tag(source,model,objective,target,seed)/'intervention_physiology.json',dict(status='complete',rows=rows,comparisons=comparisons,scope='Within-scan contextual association, participant-grouped probes; HRV is PPG-derived PRV. Circular-shift null assumes within-scan stationarity; no causal interpretation.'))

