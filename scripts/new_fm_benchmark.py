#!/usr/bin/env python3
"""One resumable task from the additive new-model benchmark."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['features','local','axes','loso','identification','paired_visit','readouts','canonical','harmonization','short_features','fusion','robustness','supervised','native_ssl','report'])
    p.add_argument('--model',choices=['semantoks','fmrilm'],default='semantoks');p.add_argument('--source',default='none',choices=['none','adni','nacc'])
    p.add_argument('--objective',default='frozen',choices=['frozen','supervised','native_ssl']);p.add_argument('--target',default='none',choices=['none','MEM','EF'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',default='external',choices=['external','cv']);p.add_argument('--cohort',default='adni')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--phase',default='frozen',choices=['frozen','all']);a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    state=(a.source,a.model,a.objective,a.target,a.seed)
    if a.action=='features':
        from new_fm.extract import extract
        extract(state,a.device)
    elif a.action=='supervised':
        from new_fm.train import supervised
        supervised(a.source,a.model,a.target,a.seed,a.mode,a.device)
    elif a.action=='native_ssl':
        from new_fm.train import ssl
        ssl(a.source,a.model,a.seed,a.device)
    elif a.action in ('short_features','fusion','robustness'):
        from new_fm import sensitivity
        getattr(sensitivity,a.action)(a.model,a.cohort,a.device)
    elif a.action=='report':
        from new_fm.report import report
        report(a.phase)
    else:
        from new_fm import evaluate
        if a.action=='local':
            if a.cohort=='all':
                from new_fm.data import COHORTS,save
                for co in COHORTS[:-1]:evaluate.local(state,co)
                save(evaluate.rowdir(state)/'local_complete.json',dict(status='complete'))
            else:evaluate.local(state,a.cohort)
        elif a.action=='readouts':evaluate.readouts(a.model)
        else:getattr(evaluate,a.action)(state)

if __name__=='__main__':main()
