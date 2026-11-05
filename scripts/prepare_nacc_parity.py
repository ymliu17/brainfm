#!/usr/bin/env python3
"""Build NACC-labelled inputs for legacy ADNI CLI filenames, without relabelling cohort."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_external_preliminary import DATA,normalize,dump
OUT=ROOT/'data/derived/nacc_parity_v1';OUT.mkdir(parents=True,exist_ok=True)
def main():
 m=pd.read_csv(DATA/'nacc.csv');raw=pd.read_csv(ROOT/'data/NACC/fc_cognitive_composites_NACC.csv',dtype={'session':str})
 m['session']=m.session.astype(str);m=m.merge(raw,left_on=['subject','session'],right_on=['ID','session'],validate='one_to_one',sort=False)
 x=torch.load(DATA/'nacc.pt',weights_only=True);assert len(m)==len(x)
 # Verified NACCUDSD: 1 normal, 2 impaired-not-MCI, 3 MCI, 4 dementia.
 stages=m.NACCUDSD.map({1:'CN',2:'impaired_not_MCI',3:'MCI',4:'Dementia'})
 tasks={'MEM':m.memory,'EF':m.executive,'AGE':m.age,'DX':m.NACCUDSD.map({1:0.,4:1.}),'DX_ord':m.NACCUDSD.map({1:0.,3:1.,4:2.})};summary={}
 for task,y in tasks.items():
  keep=np.isfinite(y);a=m.loc[keep].copy();a['PTID']=a.subject;a['FILE_NAME']=a.subject+'_'+a.session;a['label']=y[keep];a['AGE']=a.age;a['SITE']=a.NACCADC;a['dx_stage']=stages[keep]
  torch.save(x[torch.as_tensor(keep.to_numpy())],OUT/f'ADNI_x_nacc_{task}.pt');torch.save(torch.tensor(y[keep].to_numpy(),dtype=torch.float32),OUT/f'ADNI_y_nacc_{task}.pt');a.to_csv(OUT/f'ADNI_manifest_nacc_{task}.csv',index=False)
  summary[task]={'n':len(a),'classes':a.label.value_counts().to_dict() if task.startswith('DX') else None}
 # Repeated imaging from eligible subjects, with no invented longitudinal cognition.
 allowed=set(m.subject);scans=[];rows=[]
 for p in sorted((ROOT/'data/NACC/A424_GSR/raw').glob('*.json')):
  meta=json.loads(p.read_text())
  if meta['subject'] not in allowed:continue
  if str(meta.get('source_qc_exclude_recommend'))=='1':continue
  f=p.with_suffix('.pt')
  try:t=normalize(torch.load(f,weights_only=True))
  except ValueError:continue
  scans.append(t);rows.append({'PTID':meta['subject'],'session':meta['session'].removeprefix('ses-'),'source':str(f),'native_frames':meta['shape'][1]})
 allm=pd.DataFrame(rows);allx=torch.stack(scans);torch.save(allx,OUT/'repeated_scans.pt');allm.to_csv(OUT/'repeated_scans.csv',index=False)
 pairs=[];px=[]
 for subject,g in allm.groupby('PTID'):
  ids=g.sort_values('session').index.to_list()
  if len(ids)<2:continue
  a,b=ids[:2];interval=(pd.to_datetime(allm.loc[b,'session'])-pd.to_datetime(allm.loc[a,'session'])).days/30.4375
  pairs.append({'PTID':subject,'interval_months':interval,'session1':allm.loc[a,'session'],'session2':allm.loc[b,'session']});px.append(torch.stack([allx[a],allx[b]],dim=-1))
 if px:torch.save(torch.stack(px),OUT/'ADNI_x_nacc_pairs.pt');pd.DataFrame(pairs).to_csv(OUT/'ADNI_manifest_nacc_pairs.csv',index=False)
 summary['repeat_scans']=len(allm);summary['paired_subjects']=len(px);summary['paired_le6months']=sum(r['interval_months']<=6 for r in pairs)
 summary['diagnosis_definition']='CN versus dementia (not AD etiology); ordinal CN/MCI/dementia, excluding impaired-not-MCI'
 summary['diagnosis_source']='https://files.alz.washington.edu/documentation/uds3-rdd.pdf#page=269'
 summary['unavailable_fields']=['sex','education','AD-specific etiology','repeated cognitive outcomes']
 dump(ROOT/'runs/nacc_parity_v1/intake.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
