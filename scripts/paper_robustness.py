#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['audit','extract','projections','evaluate','weighting','report'])
    p.add_argument('--model');p.add_argument('--cohort');p.add_argument('--atlas');p.add_argument('--device',default='cpu')
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    import paper_robustness as task
    if a.action=='audit':task.audit()
    elif a.action=='extract':task.extract(a.model,a.cohort,a.device)
    elif a.action=='projections':task.prepare_fc(a.atlas,a.cohort,a.device)
    elif a.action=='evaluate':task.evaluate(a.model,a.cohort)
    elif a.action=='weighting':task.weighting(a.model)
    else:task.report()


if __name__=='__main__':main()
