#!/usr/bin/env python3
"""Repair duplicate metric metadata in reference comparison serialization only."""
import sys,argparse
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import paper_paired_stats as stats
import benchmark_gap_tasks as gap
original=stats.paired

def paired_without_duplicate_metric(*args,**kwargs):
    result=original(*args,**kwargs)
    # Caller already writes the exact same requested metric; no statistical change.
    if result['metric']!=kwargs.get('metric','rho'):raise ValueError('Metric mismatch')
    return {k:v for k,v in result.items() if k!='metric'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--objective',required=True);p.add_argument('--target',required=True);a=p.parse_args()
    with patch.object(stats,'paired',paired_without_duplicate_metric):gap.reference_comparisons(a.model,a.source,a.objective,a.target)
    dest=gap.RUN/'reference_comparisons'/('_'.join((a.model,a.source,a.objective,a.target))+'.json')
    d=gap.read(dest);d['metadata_serialization_fix']=dict(script=str(Path(__file__).resolve()),sha256=gap.sha(__file__),scope='Remove duplicate metric keyword; all numerical calculations and resampling settings unchanged.')
    gap.save(dest,d)
if __name__=='__main__':main()
