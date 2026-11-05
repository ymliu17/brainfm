#!/usr/bin/env python3
"""Reuse feature sweeps with strict training-only nested probes on NACC.

Legacy ADNI sweeps fit transforms before inner tuning. This adapter replaces the
probe for new runs without changing historical ADNI artifacts.
"""
import sys,importlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import probe_readouts as pr
from benchmark_evaluation import oof_predictions,score

def strict_probe(F,y,seeds=(0,1,2),n_pca=200,grid=(.001,.01,.1,1.,10.),groups=None,covariates=None,site_out=None,**kw):
 groups=np.arange(len(y)) if groups is None else np.asarray(groups)
 if site_out is not None:groups=np.asarray(site_out)
 if len(np.unique(groups))<5:raise ValueError('Fewer than five independent groups for the fixed protocol')
 vals=[score(y,oof_predictions([F],[n_pca],y,s,grid=grid,groups=groups)) for s in seeds]
 return float(np.mean(vals)),float(np.std(vals))

def main():
 module=sys.argv[1]
 if module not in ['probe_readouts','learning_curve']:raise ValueError('Unsupported audited adapter')
 pr.probe=strict_probe;engine=importlib.import_module(module);engine.probe=strict_probe;sys.argv=[module+'.py',*sys.argv[2:]];engine.main()
if __name__=='__main__':main()
