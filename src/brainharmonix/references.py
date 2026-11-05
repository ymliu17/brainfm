"""Shared FC references and independent-window physiology for all configurations."""
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from fm_matrix.data import ROOT,save,sha
from fm_readout_v4.data import dataset,metadata,DATA
from fm_readout_v4.evaluate import tag
from benchmark_evaluation import predict_prepared,score
from fm_matched_v3.probes import TuckerBlock
from paper_task_expansion import exact_project
import paper_completion as pc
import sf100_benchmark as sf
import unified_supplements as u

RUN=ROOT/'runs/brainharmonix_f_v1'
PARENT=ROOT/'runs/physiology_frozen_crosscohort_v2_no_usable'
PHYS=('psopt','breathe','hrv')

def read(p):return json.loads(Path(p).read_text())

def fc_reference(co,atlas):
    model='brainjepa'
    xx,_=dataset(co,model);x=xx[:,:400].numpy();m=sf.labels(co)
    from transfer_baseline import fc_feature
    fc=fc_feature(x,None);tucker=TuckerBlock(x,64);rows=[]
    for p in sorted((sf.RUN/'evaluation'/co).glob('*.json')):
        if p.name=='complete.json':continue
        old=read(p)
        if old.get('status')!='complete' or 'split_audit' not in old:continue
        dest=RUN/'references'/atlas/co/p.name
        if dest.exists():continue
        ix=m.set_index('scan').index.get_indexer(old['scan_ids']);assert (ix>=0).all()
        y=np.asarray(old['y'],float);g=m.iloc[ix].subject.astype(str).tolist();assert g==old['subjects']
        cov=old['covariates'];binary=old['metrics']['sf100_fc']['metric']=='auc';predictions={}
        for name in ('FC','FC-Tucker'):
            prep={s:[] for s in range(3)}
            for split in old['split_audit']:
                tr,te,it,iv=[np.asarray(split[k],int) for k in ('train','test','inner_train','validation')]
                if set(np.asarray(g)[tr])&set(np.asarray(g)[te]):raise ValueError('Subject leakage')
                ca,cb=sf.covariate_pair(m,cov,ix[tr],ix[te]);ci,cj=sf.covariate_pair(m,cov,ix[it],ix[iv])
                if name=='FC':aa,bb=fc[ix[tr]],fc[ix[te]];ai,bi=fc[ix[it]],fc[ix[iv]]
                else:aa,bb=tucker.fit_pair(ix[tr],ix[te]);ai,bi=tucker.fit_pair(ix[it],ix[iv])
                aa,bb=exact_project(aa,bb,5 if cov else 100);ai,bi=exact_project(ai,bi,5 if cov else 100)
                prep[split['seed']].append((tr,te,(np.c_[ca,aa],np.c_[cb,bb]),it,iv,(np.c_[ci,ai],np.c_[cj,bi])))
            predictions[name]=[predict_prepared(prep[s],y,binary=binary) for s in range(3)]
        pp={k:np.mean(v,axis=0) for k,v in predictions.items()}
        save(dest,dict(status='complete',cohort=co,atlas=atlas,task=old['task'],scan_ids=old['scan_ids'],subjects=g,y=y,covariates=cov,
            predictions=pp,predictions_by_seed=predictions,metric='auc' if binary else 'rho',scores={k:score(y,v,binary) for k,v in pp.items()},
            split_audit=old['split_audit'],parent_sha256=sha(p),input_sha256=sha(DATA/f'{co}_sch450.pt'),
            protocol='Schaefer17n400 cortical subset of canonical160-native-frame Schaefer450 input, unchanged SF100 samples/labels/splits/covariates/ridge-logistic grid; exact training-only PCA; Tucker rank64 fit in each outer/inner training partition.',
            caveat='Atlas-matched reference; model-specific TR/context differences retained. No claim of time-context-matched architecture ablation.'))
        print('FC references',co,atlas,old['task'],flush=True)
    save(RUN/'references'/atlas/co/'complete.json',dict(status='complete',cohort=co,atlas=atlas))
