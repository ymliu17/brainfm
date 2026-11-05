#!/usr/bin/env python3
"""Versioned entry point for corrected common-readout comparisons."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def smoke(model,device):
    import torch
    import numpy as np
    from fm_matched_v3.models import Encoder
    from fm_matched_v3.data import dataset,save,RUN
    from fm_matched_v3.train import amp
    x,m=dataset('nacc',model);ix=np.flatnonzero(m.native_frames>=200)[:2]
    net=Encoder(model,device);values=[]
    for i in ix:
        with amp(device):z=net(x[i:i+1].to(device))
        if not torch.isfinite(z).all() or z.shape!=(1,net.feature_dim):
            raise ValueError('Invalid common-readout features')
        values.append(z.detach().float().cpu());z.square().mean().backward()
    gradient=sum(float(p.grad.float().norm()) for n,p in net.named_parameters() if 'lora_' in n and p.grad is not None)
    if gradient<=0 or torch.allclose(values[0],values[1]):raise ValueError('No adapter gradient or no subject variation')
    with torch.no_grad(),amp(device):a=net(x[ix[:1]].to(device));b=net(x[ix[:1]].to(device))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    save(RUN/'smoke'/f'{model}.json',dict(status='passed',encoder=net.provenance(),adapter_gradient=gradient,
         checks=['finite final-token features','subject variation','exact deterministic readout','LoRA gradients']))


def main():
    from fm_matched_v3.models import MODELS
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','smoke','features','local','identification','supervised','hrv_labels','diagnosis','analysis','diagnosis_prepare'])
    p.add_argument('--model',choices=MODELS,default='brainlm13');p.add_argument('--source',default='none')
    p.add_argument('--objective',choices=['frozen','supervised','native_ssl'],default='frozen')
    p.add_argument('--target',default='none');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--mode',choices=['cv','external'],default='external');p.add_argument('--cohort',default='adni')
    p.add_argument('--method',choices=['FM','FC','FC-Tucker','BrainGFM'],default='FM')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    from fm_matched_v3 import data,evaluate
    if a.action=='prepare':data.prepare();return
    if a.action=='hrv_labels':data.build_hrv_composites();return
    if a.action=='smoke':smoke(a.model,a.device);return
    if a.action in ('diagnosis','diagnosis_prepare'):
        from fm_matched_v3 import diagnosis
        if a.action=='diagnosis_prepare':diagnosis.prepare()
        else:diagnosis.train(a.source,a.model,a.target,a.seed,a.mode,a.device)
        return
    if a.action=='supervised':
        from fm_matched_v3.train import fit_supervised
        fit_supervised(a.source,a.model,a.target,a.seed,a.mode,a.device);return
    args=[a.source,a.model,a.objective,a.target,a.seed]
    if a.action=='features':evaluate.extraction(*args,a.device)
    elif a.action=='local':evaluate.local(*args,a.cohort,a.method)
    elif a.action=='identification':evaluate.identification(*args)
    elif a.action=='analysis':
        for cohort in evaluate.COHORTS:
            if cohort=='nacc_repeated' or (a.objective!='frozen' and cohort==a.source):continue
            evaluate.local(*args,cohort,'FM')
        evaluate.identification(*args)
        evaluate.axes(*args)
        from fm_matched_v3.data import save,RUN
        save(RUN/'evaluation'/evaluate.tag(*args)/'complete.json',dict(status='complete',
             coverage=['local grouped probes','baseline-only follow-up prognosis','cross-visit identification','source-fitted cognitive axes'],
             excluded=['LOSO','physiology','harmonization','temporal controls','significance tests']))


if __name__=='__main__':main()
