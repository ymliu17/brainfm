#!/usr/bin/env python3
"""Four-RCT baseline MEM -> HRV-ER Week-5 delayed recall, frozen encoders."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import paper_completion as pc
import sample_alignment as sa
import sf100_benchmark as sf
from fm_readout_v4.supplement import source_projection,shared_features
from fm_readout_v4.data import dataset
from fm_matrix.data import save,sha
from paper_paired_stats import paired,holm
from benchmark_evaluation import score

RUN=ROOT/'runs/hrv_recall_external_v1'
TRAIN=('act','cogte','breathe','psopt')
METHODS=('sf100_fc','braingfm',*pc.MODELS,'a424_FC','a424_FC-Tucker','sch450_FC','sch450_FC-Tucker')
FILES={}

def file(p):FILES[str(p)]=sha(p);return p

def features(method,co):
    if method in ('sf100_fc','braingfm'):
        file(sf.RUN/'features'/co/(method+'.npy'))
        return sf.features(co,include_fm=False)[method]
    if method in pc.MODELS:
        p=pc.feature_dir(('none',method,'frozen','none',0))/(co+'.npy');pc.verify_input_files([p]);return np.load(file(p),mmap_mode='r')
    atlas=method.split('_')[0]
    if method.endswith('FC-Tucker'):
        x,_=dataset(co,'brainjepa' if atlas=='sch450' else 'brainlm13')
        from fm_readout_v4.data import DATA
        file(DATA/(co+'_'+atlas+'.pt'));return x.numpy()
    p=ROOT/'runs/fm_posttraining_matrix_v2/baselines'/atlas/(co+'_FC.npy')
    pc.verify_input_files([p]);return np.load(file(p),mmap_mode='r')

def main():
    RUN.mkdir(parents=True,exist_ok=True);started=time.time();rows=[]
    tables={co:sf.labels(co) for co in (*TRAIN,'hrv_pre')};indices={};targets=[];subjects=[];train_audit=[]
    for co,m in tables.items():
        file(sf.label_file(co));file(sf.RUN/'matches'/(co+'.json'))
        col='MEM_delayed_recall' if co=='hrv_pre' else 'MEM'
        indices[co]=np.flatnonzero(sa.common_mask(m,co)&np.isfinite(m[col].to_numpy(float)))
        if co in TRAIN:
            y=m.iloc[indices[co]][col].to_numpy(float);targets.append((y-y.mean())/max(y.std(),1e-8))
            subjects.extend((co+':'+m.iloc[indices[co]].subject.astype(str)).tolist())
            train_audit.append(dict(cohort=co,n=len(y),scan_ids=m.iloc[indices[co]].scan.tolist(),mean=float(y.mean()),sd=float(y.std())))
    test=tables['hrv_pre'].iloc[indices['hrv_pre']];y=test.MEM_delayed_recall.to_numpy(float)
    assert len(test)==57 and len(subjects)==293
    save(RUN/'protocol.json',dict(status='complete',source_cohorts=TRAIN,held_out='hrv_pre',endpoint='Week-5 single delayed recall count; baseline MRI at Week 2',
        fit='Per-source-cohort standardized MEM outcomes, training-only StandardScaler/PCA<=100, fixed ridge alpha=10; FC-Tucker basis fitted to all four training cohorts only',
        scope='One-way external memory prediction; HRV recall is not added to the training pool for the previously reported four-cohort MEM LOSO rows.',
        correction='Five post hoc conditional paired contrasts against SF100 FC; separate HRV_recall_external Holm family; 5000 participant bootstrap/20000 prediction swaps',
        encoder_training=False,automatic_shutdown=False))
    for method in METHODS:
        save(RUN/'queue_status.json',dict(status='running',running=[method],complete=len(rows),total=len(METHODS),failed={},automatic_shutdown=False))
        p=RUN/(method+'.json')
        if p.exists():r=sa.read(p);assert r['status']=='complete';rows.append(r);continue
        a=np.concatenate([features(method,co)[indices[co]] for co in TRAIN])
        b=np.asarray(features(method,'hrv_pre')[indices['hrv_pre']])
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite selected features')
        if method.endswith('FC-Tucker'):a,b=shared_features(a,b,'FC-Tucker')
        prediction=source_projection(a,b,np.concatenate(targets))
        r=dict(status='complete',model=method,cohort='hrv_pre',domain='MEM / delayed recall',n=len(y),rho=score(y,prediction,False),
            scan_ids=test.scan.tolist(),subjects=test.subject.tolist(),y=y,prediction=prediction,training_subjects=subjects,training_audit=train_audit,
            endpoint_definition='Week-5 single delayed recall; Week-2 baseline MRI',evaluation='four-RCT MEM to HRV recall external prediction')
        save(p,r);rows.append(r);print(method,r['rho'],flush=True)
    by={r['model']:r for r in rows};base=by['sf100_fc'];contrasts=[]
    for method in ('braingfm',*pc.MODELS):
        q=by[method]
        assert base['subjects']==q['subjects'] and base['scan_ids']==q['scan_ids']
        seed=int(hashlib.sha256(('hrv_recall_external'+method).encode()).hexdigest()[:8],16)
        stat=paired(y,base['prediction'],q['prediction'],base['subjects'],seed=seed)
        stat['uncertainty']='Conditional on fixed external predictions; training cohorts/models not resampled'
        contrasts.append(dict(cohort='hrv_pre',domain='MEM / delayed recall',reference='sf100_fc',candidate=method,family='HRV_recall_external',**stat))
    for r,p in zip(contrasts,holm([r['permutation_p'] for r in contrasts])):r.update(holm_p=p,family_size=5)
    pd.DataFrame([{k:r[k] for k in ('model','cohort','domain','n','rho','endpoint_definition')} for r in rows]).to_csv(RUN/'scores.csv',index=False)
    pd.json_normalize(contrasts).to_csv(RUN/'paired_comparisons.csv',index=False)
    file(Path(__file__))
    save(RUN/'report.json',dict(status='complete',rows=rows,contrasts=contrasts,files_sha256=FILES,elapsed_seconds=time.time()-started))
    save(RUN/'queue_status.json',dict(status='finished',running=[],complete=10,total=10,failed={},automatic_shutdown=False))

if __name__=='__main__':main()
