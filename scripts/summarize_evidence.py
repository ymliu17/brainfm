"""Reconstruct cohort evidence and FC-superiority counts from released aggregate tests.

Uses archived full-family adjusted P values; does not correct plotted subsets.
Above-chance evidence and superiority are separate, nonordinal criteria.
"""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
from plot_results_base import METHODS,LABEL
ROOT=Path(__file__).resolve().parents[1]
T=Path(os.environ.get('BRAINFM_RESULTS_DIR',str(ROOT/'results'))).expanduser().resolve()/'tables';OUT=ROOT/'outputs/evidence'
TASKS={'AGE':'Age','SEX':'Sex','DX':'Diagnosis','MEM':'Memory','EF':'EF',
       'MEM_followup':'Follow-up memory','EF_followup':'Follow-up EF',
       'MEM_change':'Memory change','EF_change':'EF change','arm':'Arm','Identity':'Re-ID'}
OBS=['adni','nacc'];RCT=['act','cogte','breathe','psopt','hrv']
def read(n,s):return pd.read_csv(T/f'Dataset_S{n}__{s}.csv',low_memory=False)
def main():
    if not (T/'Dataset_S12__S12_cohort_evidence_tests.csv').is_file():
        raise SystemExit('Numerical inputs are not bundled. Set BRAINFM_RESULTS_DIR to your local result export; see reproducibility/RESULTS.md.')
    OUT.mkdir(parents=True,exist_ok=True)
    tests=read(12,'S12_cohort_evidence_tests')
    assert not tests.duplicated(['model','cohort','task']).any()
    tests['symbol']=np.where(tests.p_adjusted<.05,'●','○')
    assert tests.symbol.equals(tests.main_Table3_symbol)
    rows=[];counts=[]
    local=read(5,'S5_local_frozen');ids=read(5,'S5_ID_primary')
    follow=read(12,'S12_followup_FC_comparisons');paired=read(8,'S8_paired_visit')
    for model in METHODS[2:]:
        row={'Configuration':LABEL[model]}
        for task,label in TASKS.items():
            cohorts=OBS if task in ['AGE','SEX','DX','MEM','EF'] else OBS+RCT if task=='Identity' else RCT
            a=tests[tests.model.eq(model)&tests.task.eq(task)].set_index('cohort')
            symbols=[a.loc[c,'symbol'] if c in a.index else '–' for c in cohorts]
            row[label]=' '.join(symbols[:2])+'\n'+' '.join(symbols[2:]) if task=='Identity' else ' '.join(symbols)
            if task in ['AGE','SEX','DX','MEM','EF']:
                q=local[local.model.eq(model)&local.task.eq(task)&local.cohort.isin(OBS)]
            elif task=='Identity':q=ids[ids.model.eq(model)&ids.task.eq('top1')]
            elif task.endswith('followup'):q=follow[follow.model.eq(model)&follow.task.eq(task)&follow.reference.eq('sf100_fc')]
            else:q=paired[paired.model.eq(model)&paired.task.eq(task)&paired.objective.eq('frozen')&paired['mode'].eq('change')&paired.contrast.eq('frozen_vs_sf100')]
            assert q.cohort.nunique()==len(q),(model,task,'multiple protocols cannot be pooled')
            wins=(q.p_adjusted<.05)&(q.delta>0)
            counts.append(dict(model=model,task=task,absolute_tested_cohorts=len(a),absolute_positive_cohorts=int((a.p_adjusted<.05).sum()),FC_tested_cohorts=len(q),FC_advantage_cohorts=int(wins.sum()),FC_positive_direction_cohorts=int((q.delta>0).sum()),consistent_FC_advantage=bool(wins.sum()>=2 and (q.delta>0).all())))
        rows.append(row)
    matrix=pd.DataFrame(rows);expected=read(12,'S12_cohort_evidence_matrix')
    pd.testing.assert_frame_equal(matrix,expected)
    matrix.to_csv(OUT/'TableS14_cohort_evidence.csv',index=False)
    tests.rename(columns={'main_Table3_symbol':'archived_symbol'}).to_csv(OUT/'cohort_absolute_tests.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'criteria_counts.csv',index=False)
    # Physiology is a separate supplementary protocol, not an extra cell in Table S14.
    phys=read(12,'S12_absolute_tests');phys=phys[phys.task.eq('physiology')]
    phys.to_csv(OUT/'physiology_absolute_tests.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps({'matrix_matches_TableS14':True,'tests':len(tests),'configurations':9,'consistent_primary_configurations':int(pd.DataFrame(counts).groupby('model').consistent_FC_advantage.any().sum()),'statistical_tests_recomputed':False},indent=2)+'\n')
    print(OUT)
if __name__=='__main__':main()
