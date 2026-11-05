#!/usr/bin/env python3
"""Publish requested historical controls without refitting or changing old families."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fm_matrix.data import save,sha
from benchmark_evaluation import score

OUT=ROOT/'runs/legacy_exploratory_v1'
CAL=ROOT/'runs/unified_input_v3/workspace/runs/transfer_calibration_v1'
CTRL=ROOT/'runs/unified_input_v3/workspace/runs/benchmark_extension_v1'
FILES={}
def read(p):FILES[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())

def main():
 OUT.mkdir(exist_ok=True)
 FILES[str((CAL/'comparisons.csv').relative_to(ROOT))]=sha(CAL/'comparisons.csv')
 comparisons=pd.read_csv(CAL/'comparisons.csv');comparisons=comparisons[comparisons.method.eq('input_gain')].copy()
 rows=[]
 for p in sorted((CAL/'input_results').glob('*/*.json')):
  a=read(p);b=read(CAL/'embedding_results'/p.parent.name/p.name)
  for r in a['rows']:
   old=next(v for v in b['rows'] if all(v[k]==r[k] for k in ['kind','target','family']) and v['method']=='none')
   assert old['subjects']==r['subjects'] and old['y']==r['y']
   for v in (old,r):
    rho=score(v['y'],v['prediction'],False);assert np.isclose(rho,v['rho'] if 'rho' in v else v['imaging_metrics']['rho'])
    rows.append(dict(source=a['source'],domain=a['domain'],state=a['state'],cohort=a['target_cohort'],kind=v['kind'],target=v['target'],family=v['family'],method=v['method'],n=v['n'],n_subjects=len(set(v['subjects'])),rho=rho))
 scores=pd.DataFrame(rows);assert len(scores)==360 and len(comparisons)==180
 for _,c in comparisons.iterrows():
  g=scores.copy()
  for k in ['source','domain','state','cohort','kind','target','family']:g=g[g[k]==c[k]]
  assert len(g)==2
  expected=g.set_index('method').rho['input_gain']-g.set_index('method').rho['none']
  assert np.isclose(expected,c['delta'],atol=1e-12)
 scores.to_csv(OUT/'roi_gain_scores.csv',index=False);comparisons.to_csv(OUT/'roi_gain_paired.csv',index=False)
 summary=read(CTRL/'training_controls_summary.json');controls=[];seeds=[]
 for r in summary['rows']:
  cells=[read(CTRL/'training_controls'/f"{r['cohort']}_{r['target']}_{r['model']}_s{seed}"/'result.json') for seed in (42,43,44)]
  first=cells[0]
  for seed,d in zip((42,43,44),cells):
   assert d['subjects']==first['subjects'] and d['y']==first['y']
   for f in d['folds']:
    assert not set(f['test_subjects'])&(set(f['train_subjects'])|set(f['validation_subjects']))
   seeds.append(dict(cohort=r['cohort'],task=r['target'],model=r['model'],training_seed=seed,n=len(d['y']),n_subjects=len(set(d['subjects'])),**d['metrics']))
  p=np.mean([d['predictions'] for d in cells],axis=0);binary='AUC' in r['metrics'];metric='AUC' if binary else 'rho'
  assert np.isclose(score(first['y'],p,binary),r['metrics'][metric])
  controls.append(dict(cohort=r['cohort'],task=r['target'],model=r['model'],n=len(first['y']),n_subjects=len(set(first['subjects'])),readout='BrainLM13 ROI norm L4; GRU is a separate temporal architecture',**r['metrics']))
 inference=read(CTRL/'multiplicity_summary.json')
 ctrl_stats=[r for r in inference['comparisons'] if r.get('file','').startswith('training_controls/')]
 assert len(controls)==12 and len(seeds)==36 and len(ctrl_stats)==8
 pd.DataFrame(controls).to_csv(OUT/'training_control_scores.csv',index=False)
 pd.DataFrame(seeds).to_csv(OUT/'training_control_seeds.csv',index=False)
 pd.DataFrame(ctrl_stats).to_csv(OUT/'training_control_paired.csv',index=False)
 save(OUT/'report.json',dict(status='complete',historical_exploratory=True,refitting=False,files_sha256=FILES,
  roi_gain=dict(scores=360,paired=180,significant=int(comparisons.holm_within_contrast_family.lt(.05).sum()),readout='BrainLM13 layer3 embedding connectivity; frozen and historical ADNI/NACC cognitive adapters',limitations='Older samples, model/readout, source PCA20 and HRV individual-test endpoints; not the current four-model matched panel. Source-record historical FM cache identity caveats retained. Original 900-test calibration comparison family structure retained; no recorrection on the selected ROI subset.'),
  training_controls=dict(scores=12,seed_scores=36,paired=8,significant=sum(r['holm_extension_and_previous_nacc_fusion']<.05 for r in ctrl_stats),tasks='ADNI MEM; NACC MEM,EF,AD_DX',readout='BrainLM13 ROI norm layer4 for pretrained/random controls; separate GRU',protocol=first['protocol'],limitations='Matched finite-budget historical control, not exhaustive scratch optimization, not a four-model pretraining ablation, and not the current CLS sample. Original extension-wide Holm values retained.',original_correction_scope=inference['scope']),
  interpretation='Historical supplementary evidence only; no averaging or pooled significance with unified primary experiments.',automatic_shutdown=False))
 print('Exported historical ROI gain: 360 scores/180 contrasts; training controls:12 scores/36 seed scores/8 contrasts.')

if __name__=='__main__':main()
