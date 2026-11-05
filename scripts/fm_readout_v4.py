#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    from fm_readout_v4.models import MODELS
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','features','analysis','loso','calibration','smoke'])
    p.add_argument('--model',choices=MODELS,default='brainlm13');p.add_argument('--source',default='none')
    p.add_argument('--objective',choices=['frozen','supervised','native_ssl'],default='frozen')
    p.add_argument('--target',default='none');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--method',choices=['FM','FC','FC-Tucker'],default='FM');p.add_argument('--device',default='cuda')
    a=p.parse_args();import torch;torch.set_num_threads(2)
    from fm_readout_v4 import data,evaluate,supplement
    data.prepare();args=(a.source,a.model,a.objective,a.target,a.seed)
    if a.action=='prepare':return
    if a.action=='smoke':
        from fm_readout_v4.models import Encoder
        from fm_matrix.train import amp
        net=Encoder('brainlm13',a.device);x,_=data.dataset('hrv_pre','brainlm13');x=x[:2].to(a.device)
        with torch.no_grad(),amp(a.device):
            actual=net(x)
            expected=net.net.vit(signal_vectors=x,xyz_vectors=net.xyz[None].expand(2,-1,-1),noise=None,return_dict=True).last_hidden_state[:,0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        if not torch.isfinite(actual).all() or torch.allclose(actual[0],actual[1]):raise ValueError('CLS not finite/distinct across scans')
        data.save(data.RUN/'smoke.json',dict(status='passed',shape=list(actual.shape),readout='last_hidden_state[:,0]',exact_encoder_output=True,post_training=False))
        return
    if a.action=='features':evaluate.extraction(*args,a.device)
    elif a.action=='analysis':
        # Only the added CLS arm needs the original six cohorts re-evaluated.
        cohorts=evaluate.COHORTS[:-1] if a.model=='brainlm13' else ['hrv_pre','hrv_post']
        for co in cohorts:
            if a.objective!='frozen' and co==a.source:continue
            evaluate.local(*args,co)
        if a.model=='brainlm13':evaluate.identification(*args)
        evaluate.axes(*args,cohorts=None if a.model=='brainlm13' else ['hrv_pre'])
        data.save(data.RUN/'evaluation'/evaluate.tag(*args)/'complete.json',dict(status='complete',
            cohorts=cohorts,post_training_performed=False,original_non_HRV_results='Preserved in fm_posttraining_matrix_v2',
            source_cv='Existing original-readout encoder CV retained; no claim of CLS-trained adapters or independent source CV for source-fitted external adapters'))
    elif a.action=='loso':supplement.loso(*args,a.method)
    elif a.action=='calibration':supplement.calibration(*args,a.method)


if __name__=='__main__':main()
