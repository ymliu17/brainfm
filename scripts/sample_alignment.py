#!/usr/bin/env python3
"""One resumable common-sample refit task; no encoder training."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import sample_alignment as sa

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['init','canonical','baseline','demographics','inference','loso','identification','projections','fusion','robust_projections','robust_evaluate','report'])
    p.add_argument('--state',type=int)
    p.add_argument('--model')
    p.add_argument('--atlas')
    p.add_argument('--method',default='FM')
    p.add_argument('--kind')
    p.add_argument('--source')
    p.add_argument('--objective')
    p.add_argument('--target')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.action=='init':sa.initialize()
    elif a.action=='canonical':sa.canonical(list(sa.pc.states())[a.state])
    elif a.action=='demographics':sa.demographics(list(sa.demo.states())[a.state])
    elif a.action=='baseline':sa.baseline(a.atlas,a.method)
    elif a.action=='inference':sa.inference(a.model,a.kind,a.source,a.objective,a.target)
    elif a.action=='loso':sa.loso(a.model,a.method)
    elif a.action=='identification':sa.identification(a.model)
    elif a.action in ('projections','fusion'):
        sa.configure_paper()
        if a.action=='projections':sa.pc.projections(a.atlas,'cogte',a.device)
        else:sa.pc.fusion(a.model,'cogte')
    elif a.action.startswith('robust_'):sa.robustness(a.action.removeprefix('robust_'),a.model,a.atlas,a.device)
    elif a.action=='report':sa.report()
if __name__=='__main__':main()
