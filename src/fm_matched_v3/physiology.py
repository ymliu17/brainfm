"""Matched HRV-ER trait/state probes with FC, temporal and fusion controls.

An additive, versioned analysis of existing matrix inputs/features. No source
encoder is refitted here. Held-out windows are offline contextual observations.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from fm_matched_v3.data import ROOT,WORK,DATA,RUN as MATRIX_RUN,dataset,save,sha
from fm_matched_v3.models import MODELS,Encoder
from fm_matched_v3.train import amp
from fm_matched_v3.evaluate import tag
from fm_matched_v3.supplement import window_dir as feature_dir, checkpoint_path, window_features
from benchmark_evaluation import grouped_splits,score,paired_bootstrap
from transfer_baseline import fc_feature,amp_feature
from model_input_protocol import build_input

RUN=ROOT/'runs/fm_matched_v3/physiology'
WINDOW_FILE=ROOT/'data/HRV_biofeedback/label/HRV_biofeedback_PPG_HRV_windows48s_full.csv'
PREPARED_WINDOWS=WORK/'data/derived/hrv_ppg_v1/windows.csv'
SEEDS=(0,1,2)
N_NULL=200
N_BOOT=2000
TARGETS={'pulse_rate':'HRV_MeanHR','log_RMSSD':'HRV_RMSSD'}


def states():
    out=[]
    for model in MODELS:
        for source,objective,target,seed in [('none','frozen','none',0),*[(s,o,t,42) for s in ('adni','nacc') for o,t in [('supervised','MEM'),('supervised','EF'),('native_ssl','none')]]]:
            out.append(dict(name=tag(source,model,objective,target,seed),model=model,source=source,objective=objective,target=target,seed=seed))
    return out


def arguments(state):return [state[k] for k in ('source','model','objective','target','seed')]


def center_scans(x,scans):
    x=np.asarray(x).copy()
    for scan in np.unique(scans):
        ids=np.flatnonzero(scans==scan);x[ids]-=x[ids].mean(0)
    return x


def shuffle_windows(x,seed):
    """One common frame permutation across all ROIs within each 20-frame window."""
    z=x.clone();rng=np.random.default_rng(seed)
    for i in range(len(x)):
        for w in range(8):
            ix=w*20+rng.permutation(20)
            z[i,:,w*20:(w+1)*20]=x[i,:,ix]
    return z


def file_map():
    files=[Path(__file__),ROOT/'scripts/fm_matched_supplement.py',ROOT/'src/fm_matched_v3/supplement.py',ROOT/'src/benchmark_evaluation.py',ROOT/'src/fm_matched_v3/models.py',ROOT/'src/fm_matrix/models.py',ROOT/'src/model_input_protocol.py',ROOT/'scripts/transfer_baseline.py',WINDOW_FILE,PREPARED_WINDOWS,MATRIX_RUN/'intake.json']
    return {str(p):sha(p) for p in files}


def verify():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,expected in p['files'].items():
        if sha(path)!=expected:raise ValueError('Registered physiology source changed: '+path)
    for name,expected in p['artifacts'].items():
        if sha(RUN/name)!=expected:raise ValueError('Physiology artifact changed: '+name)
    return p


def prepare():
    if (RUN/'protocol.json').exists():verify();return
    from hrv_ppg_benchmark import valid_window_rows
    RUN.mkdir(parents=True,exist_ok=True)
    w=pd.read_csv(PREPARED_WINDOWS)
    if w.duplicated(['Subject','Timepoint','WindowIdx']).any() or not valid_window_rows(w).all():raise ValueError('Invalid window grid')
    if not (w.VolumeEndIdxExclusive<=w.native_frames).all():raise ValueError('Windows exceed real BOLD')
    # Independently match the prepared values to the current source CSV.
    raw=pd.read_csv(WINDOW_FILE).set_index(['Subject','Timepoint','WindowIdx'])
    if raw.index.duplicated().any():raise ValueError('Ambiguous physiological keys')
    current=raw.loc[pd.MultiIndex.from_frame(w[['Subject','Timepoint','WindowIdx']])]
    for col in TARGETS.values():
        if not np.allclose(w[col],current[col],equal_nan=True):raise ValueError('Prepared physiology differs from current CSV')
    good=[]
    for _,b in w.groupby(['Subject','Timepoint']):
        if sorted(b.WindowIdx.tolist())!=list(range(8)):raise ValueError('Incomplete physiological scan')
        if np.isfinite(b[list(TARGETS.values())].to_numpy()).all() and (b.HRV_RMSSD>0).all():good.extend(b.index)
    w=w.loc[good].sort_values(['Subject','Timepoint','WindowIdx']).reset_index(drop=True)
    w['scan']=w.Subject+'_'+w.Timepoint
    if w.Subject.nunique()<20:raise ValueError('Insufficient participant sample')
    scans=w.drop_duplicates('scan')[['Subject','Timepoint','scan']].reset_index(drop=True)
    lookup={}
    for tp in ('pre','post'):
        m=pd.read_csv(DATA/f'hrv_{tp}.csv')
        for i,r in m.iterrows():lookup[r.subject,tp]=(i,r)
    metadata=[]
    for r in scans.itertuples():
        i,m=lookup[r.Subject,r.Timepoint]
        if int(m.native_frames)<160:raise ValueError('Insufficient real BOLD')
        metadata.append(dict(matrix_index=i,age_category=m.age_category,sex=m.sex,arm=m.arm))
    scans=pd.concat([scans,pd.DataFrame(metadata)],axis=1)
    scans.to_csv(RUN/'scans.csv',index=False);w.to_csv(RUN/'windows.csv',index=False)
    artifacts=['scans.csv','windows.csv']
    # All controls cover exactly the same first 160 real frames as FM windows.
    for atlas,model in [('a424','brainlm13'),('sch450','brainjepa')]:
        values={}
        for tp in ('pre','post'):values[tp]=dataset('hrv_'+tp,model)[0]
        x=torch.stack([values[r.Timepoint][r.matrix_index,:,:160] for r in scans.itertuples()])
        torch.save(x,RUN/f'{atlas}_real160.pt');artifacts.append(f'{atlas}_real160.pt')
        windows=x.reshape(len(x),x.shape[1],8,20).permute(0,2,1,3).reshape(-1,x.shape[1],20).numpy()
        for name,v in dict(window_FC=fc_feature(windows,None),window_AMP=amp_feature(windows,None),window_RAW=windows.reshape(len(windows),-1),scan_FC=fc_feature(x.numpy(),None),scan_AMP=amp_feature(x.numpy(),None)).items():
            name=f'{atlas}_{name}.npy';np.save(RUN/name,np.asarray(v,dtype=np.float32));artifacts.append(name)
    # Outcome-independent partitions are identical for every model, state and target.
    for scope,groups in [('dynamic',w.Subject.to_numpy()),('scan_mean',scans.Subject.to_numpy())]:
        for seed in SEEDS:
            folds=[]
            for tr,te in grouped_splits(np.zeros(len(groups)),groups,seed):
                a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=groups[tr]))
                folds.append(dict(train=tr,test=te,inner_train=tr[a],inner_validation=tr[b]))
            name=f'{scope}_splits_s{seed}.json';save(RUN/name,folds);artifacts.append(name)
    save(RUN/'protocol.json',dict(status='registered',evidence_status='Common final-token readout correction; exploratory, previous v2 results inspected',files=file_map(),artifacts={p:sha(RUN/p) for p in artifacts},states=states(),subjects=w.Subject.nunique(),scans=len(scans),windows=len(w),window_seconds=48,n_windows_per_scan=8,
        target_definitions={'pulse_rate':'Mean pulse rate in 48-second PPG windows','log_RMSSD':'Natural log of positive window RMSSD; PPG-derived PRV, not ECG HRV'},
        scan_mean='Mean of the eight window targets; mean log-RMSSD is not whole-recording RMSSD. Grouped OOF, per-visit scores and descriptive paired pre/post target correlation.',
        dynamic='Both features and outcomes demeaned within each scan; predictions demeaned too. Window-position by visit nuisance terms in every arm; offline full-scan context.',
        controls='Same-atlas 20-frame FC, ROI amplitude and ordered raw-ROI PCA; scan-mean controls use all 160 real frames. Context FC uses relative lags -7..7 of training-fitted 20-PC window FC, followed by training-only PCA to 10 PCs.',
        fusion='FC5+FM5 versus FC10 (matched total imaging dimensions); FC10+FM10 versus FC10 (increment) and FC20 (dimension control).',
        validation='Three seeds, five participant-grouped outer folds, grouped 80:20 inner tuning; training-only scaler/randomized PCA. Ridge C grid .001,.01,.1,1,10. Same folds in every target/state/null.',
        null='200 independent scan-wise nonzero circular shifts, refitting all supervised ridge/tuning steps with fixed label-independent transforms; same three-seed statistic. Requires circular stationarity; descriptive sensitivity.',
        shuffle='Frozen models only; one prespecified realization seed 20261001, all ROIs synchronously shuffled within each 20-frame window. FC/amplitude preserved. Off-manifold perturbation, not a unique proof of temporal mechanism.',
        uncertainty='2000 paired participant bootstraps conditional on OOF predictions; pooled windows/scans are not independent participants.',
        multiplicity='Holm across 28 states x 2 targets x 2 primary dynamic comparisons (FC10 vs FM10; FC20 vs FC10+FM10). Additional comparisons, circular-null tests, scan means and shuffle tests are separate exploratory families.',
        limitations=['Scanner triggers/spatial alignment not independently audited','No new motion or respiration adjustment without corresponding aligned measurements','No raw PPG waveform or beat-to-beat reconstruction','Stable trait and causal interpretation not established'],automatic_shutdown=False))


def embedding(state):
    f=feature_dir(*arguments(state));p=f/'complete.json'
    d=json.loads(p.read_text())
    from fm_matched_v3.supplement import window_signature
    expected=window_signature(*arguments(state))
    if d.get('status')!='complete' or d['signature']!=expected:raise ValueError('Stale common-readout window cache')
    scans=pd.read_csv(RUN/'scans.csv');lookup={};files={str(p):sha(p)}
    for tp in ('pre','post'):
        a=f/f'hrv_{tp}_windows.npy';b=f/f'hrv_{tp}_window_indices.npy'
        z=np.load(a);ix=np.load(b)
        if len(z)!=len(ix) or len(set(ix))!=len(ix) or z.shape[1]!=8:raise ValueError('Window feature indexing error')
        files.update({str(a):sha(a),str(b):sha(b)})
        for i,v in zip(ix,z):lookup[tp,int(i)]=v
    x=np.stack([lookup[r.Timepoint,r.matrix_index] for r in scans.itertuples()])
    if not np.isfinite(x).all():raise ValueError('Nonfinite FM windows')
    return x,files


def shuffled_features(model,device='cuda'):
    verify();out=RUN/'shuffled'/model;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    atlas='sch450' if model=='brainjepa' else 'a424'
    x=torch.load(RUN/f'{atlas}_real160.pt',weights_only=True);shuffled=shuffle_windows(x,20261001)
    net=Encoder(model,device);vals=[]
    with torch.no_grad():
        for i in range(len(x)):
            a=build_input(shuffled[i],net.length)[None].to(device)
            with amp(device):v=net.windows(a)
            vals.append(v.float().cpu().numpy())
            if i%10==0:print(model,'shuffled windows',i,len(x),flush=True)
    z=np.concatenate(vals)
    if not np.isfinite(z).all():raise ValueError('Nonfinite shuffled representation')
    np.save(out/'windows.npy',z)
    save(out/'complete.json',dict(status='complete',model=model,protocol_sha256=sha(RUN/'protocol.json'),features_sha256=sha(out/'windows.npy'),base_checkpoint_sha256=sha(net.checkpoint_path),seed=20261001))


def reduce_fit(x,tr,te,k):
    sc=StandardScaler().fit(x[tr]);a=sc.transform(x[tr]);b=sc.transform(x[te])
    if k<x.shape[1]:
        pc=PCA(n_components=min(k,len(tr)-1,x.shape[1]),svd_solver='randomized',random_state=0).fit(a)
        a,b=pc.transform(a),pc.transform(b)
    return np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)


def context_lags(z,scans):
    """All eight windows as relative lags; no rows cross scan boundaries."""
    result=np.zeros((len(z),15*z.shape[1]),dtype=z.dtype)
    for scan in np.unique(scans):
        ids=np.flatnonzero(scans==scan)
        if len(ids)!=8:raise ValueError('Context requires a complete scan')
        for t,row in enumerate(ids):
            for other in range(8):result[row,(other-t+7)*z.shape[1]:(other-t+8)*z.shape[1]]=z[ids[other]]
    return result


def transformed(x,tr,te,k,scans=None,context=False):
    if not context:return reduce_fit(x,tr,te,k)
    a,b=reduce_fit(x,tr,te,20);full=np.zeros((len(x),a.shape[1]));full[tr]=a;full[te]=b
    # Inner unused participants are zeros, but group isolation means they cannot
    # affect context rows belonging to train or validation/test scans.
    c=context_lags(full,scans)
    return reduce_fit(c,tr,te,k)


def cached_transforms(x,scope,k,context=False):
    # Fit the same top-20 basis once; retain nested 5/10-PC prefixes. This avoids
    # repeated high-dimensional FC decompositions across arms and model states.
    if not context and k<min(20,x.shape[1]):
        base=cached_transforms(x,scope,20)
        return [[tuple(v[:,:k] for v in fold) for fold in seed] for seed in base]
    h=hashlib.sha256(np.ascontiguousarray(x).tobytes());h.update(f'{scope}/{k}/{context}'.encode());h.update(sha(RUN/'protocol.json').encode())
    root=RUN/'transform_cache'/h.hexdigest();root.mkdir(parents=True,exist_ok=True)
    scans=pd.read_csv(RUN/'windows.csv').scan.to_numpy() if scope=='dynamic' else None
    context_base=cached_transforms(x,scope,20) if context else None
    outputs=[]
    for seed in SEEDS:
        folds=json.loads((RUN/f'{scope}_splits_s{seed}.json').read_text());parts=[]
        for i,f in enumerate(folds):
            file=root/f's{seed}_f{i}.npz'
            if file.exists():
                with np.load(file) as z:parts.append(tuple(z[k] for k in ('a','b','ai','bi')))
                continue
            tr,te,itr,iva=[np.asarray(f[k],int) for k in ('train','test','inner_train','inner_validation')]
            if context:
                values=[]
                for u,v,aa,bb in [(tr,te,*context_base[seed][i][:2]),(itr,iva,*context_base[seed][i][2:])]:
                    full=np.zeros((len(x),aa.shape[1]));full[u]=aa;full[v]=bb
                    values.extend(reduce_fit(context_lags(full,scans),u,v,k))
                a,b,ai,bi=values
            else:
                a,b=transformed(x,tr,te,k,scans,False);ai,bi=transformed(x,itr,iva,k,scans,False)
            tmp=file.with_suffix('.tmp.npz');np.savez(tmp,a=a,b=b,ai=ai,bi=bi);tmp.replace(file);parts.append((a,b,ai,bi))
        outputs.append(parts)
    return outputs


def correlations(y,p):
    a=rankdata(y,axis=0);b=rankdata(p,axis=0);a-=a.mean(0);b-=b.mean(0)
    den=np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return np.divide((a*b).sum(0),den,out=np.full(y.shape[1],np.nan),where=den>0)


def ridge_many(a,y,b,c):
    mean=a.mean(0);z=a-mean
    coef=np.linalg.solve(z.T@z+np.eye(a.shape[1])/c,z.T@(y-y.mean(0)))
    return (b-mean)@coef+y.mean(0)


def predict_many(prepared,ys,scope,scans=None):
    """Vectorized independent ridge fits/tuning for each observed/null outcome."""
    ys=np.asarray(ys,dtype=float);ys=ys[:,None] if ys.ndim==1 else ys
    results=[];grid=(.001,.01,.1,1.,10.)
    for seed,parts in zip(SEEDS,prepared):
        pred=np.full(ys.shape,np.nan)
        for f,(a,b,ai,bi) in zip(json.loads((RUN/f'{scope}_splits_s{seed}.json').read_text()),parts):
            tr,te,itr,iva=[np.asarray(f[k],int) for k in ('train','test','inner_train','inner_validation')]
            scores=np.stack([correlations(ys[iva],ridge_many(ai,ys[itr],bi,c)) for c in grid])
            chosen=np.argmax(np.where(np.isfinite(scores),scores,-np.inf),axis=0)
            for idx,c in enumerate(grid):
                cols=np.flatnonzero(chosen==idx)
                if len(cols):pred[np.ix_(te,cols)]=ridge_many(a,ys[tr][:,cols],b,c)
        if not np.isfinite(pred).all():raise ValueError('Missing OOF predictions')
        results.append(center_scans(pred,scans) if scope=='dynamic' else pred)
    return np.mean(results,axis=0)


def summarize(y,p):
    return dict(rho=score(y,p,False),mae=float(np.mean(abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))))


def evaluate(state,shuffle=False):
    verify();name=state['name'];out=RUN/'results'/(name+('_shuffle' if shuffle else '')+'.json')
    if out.exists():return
    fm,files=embedding(state)
    if shuffle:
        p=RUN/'shuffled'/state['model']/'complete.json';d=json.loads(p.read_text());f=p.parent/'windows.npy'
        if d['protocol_sha256']!=sha(RUN/'protocol.json') or d['features_sha256']!=sha(f):raise ValueError('Stale shuffle cache')
        fm=np.load(f);files.update({str(p):sha(p),str(f):sha(f)})
    w=pd.read_csv(RUN/'windows.csv');scanmeta=pd.read_csv(RUN/'scans.csv');atlas='sch450' if state['model']=='brainjepa' else 'a424';rows=[]
    scopes=('dynamic',) if shuffle else ('dynamic','scan_mean')
    for scope in scopes:
        dynamic=scope=='dynamic';m=w if dynamic else scanmeta;g=m.Subject.to_numpy();scan=m.scan.to_numpy()
        prefix='window' if dynamic else 'scan'
        blocks={k:np.load(RUN/f'{atlas}_{prefix}_{k}.npy') for k in ('FC','AMP')}
        if dynamic:blocks['RAW']=np.load(RUN/f'{atlas}_window_RAW.npy')
        blocks['FM']=fm.reshape(-1,fm.shape[-1]) if dynamic else fm.mean(1)
        if dynamic:
            blocks={k:center_scans(v,scan) for k,v in blocks.items()}
            c=pd.get_dummies(w.Timepoint+'_'+w.WindowIdx.astype(str),dtype=float).to_numpy();c=center_scans(c,scan)
        else:c=pd.get_dummies(scanmeta[['age_category','sex','arm','Timepoint']],drop_first=True,dtype=float).to_numpy()
        specs={'nuisance':[], 'FC10':[('FC',10,False)],'AMP10':[('AMP',10,False)],'FM10':[('FM',10,False)],'FC5+FM5':[('FC',5,False),('FM',5,False)],'FC10+FM10':[('FC',10,False),('FM',10,False)],'FC20':[('FC',20,False)]}
        if dynamic:specs.update(RAW10=[('RAW',10,False)],FC_context10=[('FC',10,True)])
        if shuffle:specs={'FM10':[('FM',10,False)]}
        nuisance=cached_transforms(c,scope,c.shape[1]);prepared={}
        for arm,bs in specs.items():
            parts=[nuisance]+[cached_transforms(blocks[k],scope,n,context) for k,n,context in bs]
            prepared[arm]=[[tuple(np.hstack([p[s][i][t] for p in parts]) for t in range(4)) for i in range(5)] for s in range(3)]
            print(name,scope,'prepared',arm,flush=True)
        for target,col in TARGETS.items():
            raw=w[col].to_numpy(float);raw=np.log(raw) if target=='log_RMSSD' else raw
            y=center_scans(raw,w.scan.to_numpy()) if dynamic else raw.reshape(-1,8).mean(1)
            ys=y[:,None]
            if dynamic and not shuffle:
                rng=np.random.default_rng(814);ys=np.repeat(y[:,None],N_NULL+1,axis=1)
                for i in range(1,N_NULL+1):
                    for key in np.unique(scan):
                        ix=np.flatnonzero(scan==key);ys[ix,i]=np.roll(y[ix],int(rng.integers(1,8)))
            preds={arm:predict_many(p,ys,scope,scan) for arm,p in prepared.items()}
            result=dict(scope=scope,target=target,n_subjects=len(set(g)),n_scans=len(set(scan)),n_rows=len(y),subjects=g,scans=scan,y=y,predictions={k:p[:,0] for k,p in preds.items()},scores={k:summarize(y,p[:,0]) for k,p in preds.items()},comparisons=[])
            pairs=[('FC10','FM10'),('AMP10','FM10'),('FC10','FC5+FM5'),('FC10','FC10+FM10'),('FC20','FC10+FM10')]
            if dynamic:pairs += [('RAW10','FM10'),('FC_context10','FM10'),('nuisance','FM10')]
            if shuffle:
                base=json.loads((RUN/'results'/f'{name}.json').read_text())
                ref=next(r for r in base['rows'] if r['scope']==scope and r['target']==target)
                if ref['subjects']!=g.tolist() or not np.allclose(ref['y'],y):raise ValueError('Unmatched shuffle comparison')
                preds['original']=np.asarray(ref['predictions']['FM10'])[:,None];pairs=[('FM10','original')]
            for a,b in pairs:
                delta,p,ci,details=paired_bootstrap(y,preds[a][:,0],preds[b][:,0],n_boot=N_BOOT,groups=g,return_details=True)
                result['comparisons'].append(dict(reference=a,test=b,delta_rho=delta,ci95=ci,raw_one_sided_bootstrap_p=p,primary_family=dynamic and not shuffle and (a,b) in [('FC10','FM10'),('FC20','FC10+FM10')],**details))
            if state['objective']!='frozen':
                baseline=RUN/'results'/(state['model']+'_frozen.json');frozen=json.loads(baseline.read_text())
                ref=next(r for r in frozen['rows'] if r['scope']==scope and r['target']==target)
                if ref['subjects']!=g.tolist() or not np.allclose(ref['y'],y):raise ValueError('Unmatched adaptation comparison')
                delta,p,ci,details=paired_bootstrap(y,ref['predictions']['FM10'],preds['FM10'][:,0],n_boot=N_BOOT,groups=g,return_details=True)
                result['comparisons'].append(dict(reference='frozen_FM10',test='FM10',delta_rho=delta,ci95=ci,raw_one_sided_bootstrap_p=p,primary_family=False,**details))
                files[str(baseline)]=sha(baseline)
            if dynamic and not shuffle:
                scores={arm:correlations(ys,p) for arm,p in preds.items()}
                result['circular_shift']={arm:dict(observed=v[0],null=v[1:],n_valid=int(np.isfinite(v[1:]).sum()),p=float((1+np.sum(v[1:]>=v[0]))/(N_NULL+1)) if np.isfinite(v).all() else None) for arm,v in scores.items()}
            if not dynamic:
                result['by_visit']={tp:{k:summarize(y[m.Timepoint.eq(tp)],p[m.Timepoint.eq(tp),0]) for k,p in preds.items()} for tp in ('pre','post')}
                tab=pd.DataFrame(dict(subject=g,visit=m.Timepoint,y=y)).pivot(index='subject',columns='visit',values='y').dropna()
                result['paired_target_stability']=dict(n=len(tab),rho=score(tab['pre'],tab['post'],False),interpretation='Intervention visits, not intervention-free reliability')
            rows.append(result);print(name,scope,target,result['scores'],flush=True)
            save(out.with_suffix('.progress.json'),dict(status='running',state=state,rows=rows))
    save(out,dict(status='complete',state=state,shuffle=shuffle,protocol_sha256=sha(RUN/'protocol.json'),feature_files=files,rows=rows))


def report():
    from confirm_matched_axis import holm
    comparisons=[];nulls=[]
    for p in sorted((RUN/'results').glob('*.json')):
        if p.name.endswith('.progress.json'):continue
        d=json.loads(p.read_text())
        for row in d['rows']:
            for c in row['comparisons']:comparisons.append(dict(state=d['state']['name'],shuffle=d['shuffle'],scope=row['scope'],target=row['target'],**c))
            for arm,v in row.get('circular_shift',{}).items():nulls.append(dict(state=d['state']['name'],target=row['target'],arm=arm,p=v['p']))
    families={}
    for i,c in enumerate(comparisons):
        family='shuffle' if c['shuffle'] else 'adaptation' if c['reference']=='frozen_FM10' else 'primary_dynamic' if c['primary_family'] else 'scan_mean' if c['scope']=='scan_mean' else 'exploratory_dynamic'
        families.setdefault(family,[]).append(i)
    for family,ix in families.items():
        for i,p in zip(ix,holm([comparisons[i]['raw_one_sided_bootstrap_p'] for i in ix])):comparisons[i].update(holm_family=family,holm_p=p)
    valid=[r for r in nulls if r['p'] is not None]
    for r,p in zip(valid,holm([r['p'] for r in valid])):r['holm_p']=p
    save(RUN/'report.json',dict(status='complete',comparisons=comparisons,circular_nulls=nulls,protocol_sha256=sha(RUN/'protocol.json'),interpretation='Exploratory conditional inference; dynamic windows and repeat scans clustered by participant'))
    pd.DataFrame(comparisons).to_csv(RUN/'comparisons.csv',index=False)
