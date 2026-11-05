#!/usr/bin/env python3
"""Finalize only the fully completed external-cognition families.

Identification is a different task/metric family and is not included here.
Do not replace the full-panel report or the primary nine-model comparison table.
"""
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import pandas as pd
import run_adapted_readouts_focused as e
from fm_matrix.data import save,sha
from paper_paired_stats import holm

def main():
    manifest=e.verify()
    jobs=[j for j in e.jobs(manifest) if 'endpoint' in j]
    if len(jobs)!=108: raise ValueError('Unexpected cognitive scope')
    missing=[j['id'] for j in jobs if not Path(j['output']).exists()]
    if missing: raise ValueError('Incomplete cognitive families: '+str(missing))
    rows=[];scores=[];coverage=[]
    for j in jobs:
        d=e.read(j['output']);s=j['state']
        if j['cohort']==s[0]: raise ValueError('Source-exposed cognitive test')
        if len(d['rows'])!=5: raise ValueError('Expected five paired contrasts per group/task')
        for r in d['rows']:
            if r['task']!=j['endpoint'] or r['scope']!='external': raise ValueError('Wrong family')
            rows.append(dict(r,job_id=j['id']))
        meta=dict(model=s[1],source=s[0],objective=s[3] if s[2]=='supervised' else 'SSL',cohort=j['cohort'],task=j['endpoint'])
        scores.extend([dict(meta,**r) for r in d['scores']])
        coverage.append(dict(job_id=j['id'],path=j['output'],sha256=sha(Path(j['output'])),status='complete'))
    df=pd.DataFrame(rows)
    df['contrast']=df.family.str.split('::').str[1]
    for family,ix in df.groupby('family').groups.items():
        expected=18 if '::readout_vs_native::' in family else 36
        if len(ix)!=expected: raise ValueError('Incomplete family '+family)
        df.loc[ix,'holm_p']=holm(df.loc[ix,'permutation_p'].fillna(1).tolist())
        df.loc[ix,'family_size']=len(ix)
    df['inference_status']='final_external_cognitive_family'
    df['scope_note']='Post hoc computationally reduced exploratory panel; uniform native/connectivity subset; no identification hypotheses in these families.'
    dest=e.OUT/'external_cognition';dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'Comparisons.csv',index=False)
    pd.DataFrame(scores).to_csv(dest/'Scores.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'Coverage.csv',index=False)
    with pd.ExcelWriter(dest/'External_cognition.xlsx',engine='xlsxwriter') as w:
        df.to_excel(w,sheet_name='Comparisons',index=False)
        pd.DataFrame(scores).to_excel(w,sheet_name='Scores',index=False)
        pd.DataFrame(coverage).to_excel(w,sheet_name='Coverage',index=False)
    summary=dict(status='complete',jobs=len(jobs),comparisons=len(df),families=df.family.nunique(),
        corrected_positive=int(((df.holm_p<.05)&(df.delta>0)).sum()),
        corrected_negative=int(((df.holm_p<.05)&(df.delta<0)).sum()),
        full_queue_complete=False,identification_included=False,
        scope='Independent-source encoders, locally fitted ADNI/NACC cognitive probes; not fixed-source cognitive axis transfer.',
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        script_sha256=sha(Path(__file__)),manifest_sha256=sha(e.RUN/'manifest.json'))
    save(dest/'complete.json',summary);print(summary)
    print(df[df.holm_p<.05][['model','source','objective','cohort','task','contrast','reference','candidate','reference_score','candidate_score','delta','ci95','holm_p']].to_string(index=False))

if __name__=='__main__':main()
