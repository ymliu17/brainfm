#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['audit','canonical','projections','fusion','frozen','source','adaptation','demographics','report'])
    parser.add_argument('--model',default='brainlm13')
    parser.add_argument('--source',default='none')
    parser.add_argument('--objective',default='frozen')
    parser.add_argument('--target',default='none')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--cohort')
    parser.add_argument('--atlas')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    import paper_completion as task
    # The src directory precedes scripts, so this imports src/paper_completion.py.
    if args.action=='audit':task.audit()
    elif args.action=='canonical':task.canonical((args.source,args.model,args.objective,args.target,args.seed))
    elif args.action=='projections':task.projections(args.atlas,args.cohort,args.device)
    elif args.action=='fusion':task.fusion(args.model,args.cohort)
    elif args.action=='frozen':task.frozen_inference(args.model)
    elif args.action=='source':task.source_inference(args.model)
    elif args.action=='adaptation':task.adaptation_inference(args.model,args.source,args.objective,args.target)
    elif args.action=='demographics':task.demographic_inference(args.model)
    else:task.report()


if __name__=='__main__':main()
