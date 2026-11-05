#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import torch
    torch.set_num_threads(2)
    import demographic_benchmark as benchmark
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','features','evaluate','report'])
    p.add_argument('--model',choices=benchmark.MODELS,default='brainlm13');p.add_argument('--source',choices=['none','adni','nacc'],default='none')
    p.add_argument('--objective',choices=['frozen','native_ssl'],default='frozen');p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu')
    a=p.parse_args();state=(a.source,a.model,a.objective,'none',a.seed)
    if a.action in ('features','evaluate') and state not in set(benchmark.states()):p.error('Unregistered model state')
    benchmark.prepare()
    if a.action=='features':benchmark.feature_registration(state,a.device)
    elif a.action=='evaluate':benchmark.evaluation(state)
    elif a.action=='report':benchmark.report()


if __name__=='__main__':main()
