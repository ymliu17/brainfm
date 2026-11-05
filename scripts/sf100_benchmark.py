#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','extract','evaluate','identification','transfer','report']);p.add_argument('--cohort');p.add_argument('--source');p.add_argument('--device',default='cpu');a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    import sf100_benchmark as task
    if a.action=='extract':task.extract(a.device)
    elif a.action=='evaluate':task.evaluate(a.cohort,a.device)
    elif a.action=='transfer':task.transfer(a.source)
    else:getattr(task,a.action)()
