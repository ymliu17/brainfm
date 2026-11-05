#!/usr/bin/env python3
"""One resumable task from the additive new-model benchmark."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['features','local','axes','loso','identification','paired_visit','readouts','canonical','harmonization','short_features','fusion','robustness','supervised','native_ssl','report','prepare','validate','recall','reference','reference_compare','input_fc','input_compare','identity_control','pca','pca_inference','diagnostic','phys_prepare','phys_extract','phys_evaluate','phys_report','context','joint'])
    p.add_argument('--model',choices=['brainharmonix_f'],default='brainharmonix_f');p.add_argument('--source',default='none',choices=['none','adni','nacc'])
    p.add_argument('--objective',default='frozen',choices=['frozen','supervised','native_ssl']);p.add_argument('--target',default='none',choices=['none','MEM','EF','DX','AD_DX'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',default='external',choices=['external','cv']);p.add_argument('--cohort',default='adni')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--phase',default='frozen',choices=['frozen','all']);a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    state=(a.source,a.model,a.objective,a.target,a.seed)
    if a.action in ('prepare','validate','recall','reference','reference_compare','input_fc','input_compare','identity_control','pca','pca_inference','diagnostic','phys_prepare','phys_extract','phys_evaluate','phys_report','context','joint'):
        from brainharmonix.tasks import execute
        execute(a,state)
    elif a.action=='features':
        from brainharmonix.extract import extract
        extract(state,a.device)
    elif a.action=='supervised':
        from brainharmonix.train import supervised
        supervised(a.source,a.model,a.target,a.seed,a.mode,a.device)
    elif a.action=='native_ssl':
        from brainharmonix.train import ssl
        ssl(a.source,a.model,a.seed,a.device)
    elif a.action in ('short_features','fusion','robustness'):
        from brainharmonix import sensitivity
        getattr(sensitivity,a.action)(a.model,a.cohort,a.device)
    elif a.action=='report':
        from brainharmonix.report import report
        report(a.phase)
    else:
        from brainharmonix import evaluate
        if a.action=='local':
            if a.cohort=='all':
                from brainharmonix.data import COHORTS,save
                for co in COHORTS[:-1]:evaluate.local(state,co)
                save(evaluate.rowdir(state)/'local_complete.json',dict(status='complete'))
            else:evaluate.local(state,a.cohort)
        elif a.action=='readouts':evaluate.readouts(a.model)
        else:getattr(evaluate,a.action)(state)

if __name__=='__main__':main()
