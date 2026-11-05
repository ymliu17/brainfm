#!/usr/bin/env python3
"""Additive commands; immutable matched-v3 core remains unchanged."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','windows','loso','calibration','physiology','shuffle','shuffle_eval','physiology_report','intervention_physiology'])
    p.add_argument('--model',default='brainlm13');p.add_argument('--source',default='none')
    p.add_argument('--objective',default='frozen');p.add_argument('--target',default='none')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--method',choices=['FM','FC','FC-Tucker'],default='FM')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    from fm_matched_v3 import supplement as s
    args=(a.source,a.model,a.objective,a.target,a.seed)
    if a.action=='windows':s.window_features(*args,a.device)
    elif a.action=='loso':s.loso(*args,a.method)
    elif a.action=='calibration':s.calibration(*args,a.method)
    elif a.action=='intervention_physiology':
        from fm_matched_v3.intervention_physiology import physiology
        physiology(*args)
    else:
        from fm_matched_v3 import physiology as ph
        if a.action=='prepare':ph.prepare()
        elif a.action=='shuffle':ph.shuffled_features(a.model,a.device)
        elif a.action=='physiology_report':ph.verify();ph.report()
        else:
            state=dict(zip(('source','model','objective','target','seed'),args));state['name']=s.tag(*args)
            ph.evaluate(state,shuffle=a.action=='shuffle_eval')


if __name__=='__main__':main()
