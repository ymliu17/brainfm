#!/usr/bin/env python3
"""Supplement NACC labels without overwriting running experiment manifests."""
import sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_external_preliminary import dump,sha
D=ROOT/'data/derived/nacc_uds_v1';R=ROOT/'runs/nacc_uds_v1'

def enrich(m,c,u):
 m=m.copy();c=c.copy();u=u.copy()
 for frame,col in [(m,'subject'),(c,'ID'),(u,'NACCID')]:frame['_key']=frame[col].astype(str).str.strip().str.replace(r'^sub-','',regex=True)
 if u._key.duplicated().any() or c._key.duplicated().any():raise ValueError('Expected one uploaded clinical record per participant')
 z=m.merge(c[['_key','VISITYR','VISITMO','NACCUDSD']],on='_key',validate='one_to_one',how='left',sort=False).merge(u[['_key','VISITYR','VISITMO','NACCAGE','SEX','EDUC','DEMENTED','NORMCOG','NACCALZD']],on='_key',validate='one_to_one',how='left',sort=False,suffixes=('_cognitive','_uds'))
 assert z.subject.tolist()==m.subject.tolist()
 z['uds_month_matches_cognition']=(z.VISITYR_cognitive==z.VISITYR_uds)&(z.VISITMO_cognitive==z.VISITMO_uds)
 z['SEX_binary_male']=z.SEX.map({1:1.,2:0.})
 z['education_years']=pd.to_numeric(z.EDUC,errors='coerce').where(lambda v:v.between(0,36))
 cn=z.uds_month_matches_cognition&z.NACCUDSD.eq(1)&z.NORMCOG.eq(1)&z.DEMENTED.eq(0)
 ad=z.uds_month_matches_cognition&z.NACCUDSD.eq(4)&z.NORMCOG.eq(0)&z.DEMENTED.eq(1)&z.NACCALZD.eq(1)
 z['AD_DX']=np.nan;z.loc[cn,'AD_DX']=0.;z.loc[ad,'AD_DX']=1.
 return z

def main():
 D.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 old=ROOT/'data/derived/external_preliminary_v1';m=pd.read_csv(old/'nacc.csv');c=pd.read_csv(ROOT/'data/NACC/fc_cognitive_composites_NACC.csv');u=pd.read_csv(ROOT/'data/NACC/uds_SelectedVariables.csv');z=enrich(m,c,u)
 z.to_csv(D/'nacc_enriched.csv',index=False);x=torch.load(old/'nacc.pt',weights_only=True);summary={}
 for task,col in [('SEX','SEX_binary_male'),('EDUC','education_years'),('AD_DX','AD_DX')]:
  keep=z[col].notna();a=z.loc[keep].copy();a['PTID']=a.subject;a['label']=a[col];a['AGE']=a.age;a['PTGENDER']=a.SEX.map({1:'Male',2:'Female'});a['PTEDUCAT']=a.education_years
  torch.save(x[torch.as_tensor(keep.to_numpy())],D/f'ADNI_x_nacc_{task}.pt');torch.save(torch.as_tensor(a[col].to_numpy(),dtype=torch.float32),D/f'ADNI_y_nacc_{task}.pt');a.to_csv(D/f'ADNI_manifest_nacc_{task}.csv',index=False)
  summary[task]=dict(n=len(a),classes=a[col].value_counts().to_dict() if task!='EDUC' else None)
 cc=c.copy();cc['_key']=cc.ID.str.removeprefix('sub-');uu=u.copy();uu['_key']=uu.NACCID.str.removeprefix('sub-');v=cc.merge(uu,on='_key',suffixes=('_c','_u'),validate='one_to_one');same=(v.VISITYR_c==v.VISITYR_u)&(v.VISITMO_c==v.VISITMO_u)
 summary.update(status='preliminary',uds_rows=len(u),cognitive_rows_matched=len(v),same_clinical_month=int(same.sum()),different_clinical_month=int((~same).sum()),same_month_current500=int(z.uds_month_matches_cognition.sum()),clinical_fields_source_sha256=sha(ROOT/'data/NACC/uds_SelectedVariables.csv'),original_cognitive_sha256=sha(ROOT/'data/NACC/fc_cognitive_composites_NACC.csv'),policy='Sex and education are cross-sectional source fields. Original age, cognition, eligibility and diagnosis remain unchanged. AD-dementia task restricted to same clinical month and concordant cognitive status; month agreement is not exact visit-date verification.',education_unknown_code=99,definition_source='https://files.alz.washington.edu/documentation/uds3-rdd.pdf',still_missing=['APOE','repeated cognition','exact UDS visit day'])
 dump(R/'intake.json',summary);print(summary)
if __name__=='__main__':main()
