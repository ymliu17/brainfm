#!/usr/bin/env python3
"""PPG-derived PRV tests: recording-level, pre/post change, and synchronized windows.

Primary population is the existing older-adult imaging/QC cohort. Whole-recording
PPG is not claimed to be exactly coextensive with BOLD. Window analyses use the
provided volume coordinates, QC flags and all full 48-second windows inside BOLD.
"""
from pathlib import Path
import argparse,sys,json
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from benchmark_extension import save,load,metrics
from benchmark_evaluation import prepare_nested,predict_prepared,paired_bootstrap,score
from confirm_matched_axis import holm
from checkpoint_selection import resolve_selection,selection_note
from run_external_preliminary import sha,CHECKPOINTS
R=ROOT/'runs/hrv_ppg_v1';D=ROOT/'data/derived/hrv_ppg_v1'
STATES={'frozen':CHECKPOINTS['frozen'],'ADNI_MEM':CHECKPOINTS['MEM'],'ADNI_EF':CHECKPOINTS['EF'],**{f'NACC_{t}':ROOT/f'runs/nacc_parity_v1/posttraining/{t}_external_seed42/encoder.pt' for t in ['MEM','EF']}}
WINDOW_FILE='HRV_biofeedback_PPG_HRV_windows48s_full.csv'
TARGETS={'pulse_rate':'HRV_MeanHR','log_RMSSD':'HRV_RMSSD','log_SDNN':'HRV_SDNN'}

def target_values(table,target):
 v=table[TARGETS[target]].to_numpy(float)
 if target.startswith('log_'):v=np.where(v>0,np.log(np.maximum(v,1e-12)),np.nan)
 return v

def valid_window_rows(b):
 valid=b.Usable&b.ScanAligned&b.FullWindow&b.TimeReference.eq('scan')&b.Task.eq('rest')&b.TRSec.eq(2.4)
 valid &= b.TRsPerWindow.eq(20)&np.isclose(b.WindowLengthSec,48)&np.isclose(b.StepSec,48)
 valid &= b.VolumeStartIdx.eq(b.WindowIdx*20)&b.VolumeEndIdxExclusive.eq((b.WindowIdx+1)*20)&np.isclose(b.ScanStartSec,b.VolumeStartIdx*2.4)&np.isclose(b.ScanEndSec,b.VolumeEndIdxExclusive*2.4)
 return valid

def complete_window_grid(b,n_windows):
 """Keep a complete contiguous grid; QC failures must not collapse time gaps."""
 b=b.loc[valid_window_rows(b)&b.WindowIdx.between(0,n_windows-1)].copy()
 complete=[subject for subject,g in b.groupby('Subject') if sorted(g.WindowIdx.tolist())==list(range(n_windows))]
 return b[b.Subject.isin(complete)].copy()

def require_current_features(state):
 p=R/'features'/state/'complete.json'
 if not p.exists():raise ValueError('Missing PPG features: '+state)
 d=load(p)
 if d.get('source_intake_sha256')!=sha(R/'intake.json') or d.get('checkpoint_sha256')!=sha(STATES[state]):
  raise ValueError('Stale PPG feature cache; rebuild for the current intake/checkpoint')
 return d

def prepare():
 D.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 base=ROOT/'data/HRV_biofeedback/label';scan=pd.read_csv(base/'HRV_biofeedback_PPG_HRV.csv');win=pd.read_csv(base/WINDOW_FILE)
 if scan.duplicated(['Subject','Timepoint']).any() or win.duplicated(['Subject','Timepoint','WindowIdx']).any():raise ValueError('Duplicate physiological keys')
 old=ROOT/'data/derived/external_preliminary_v1'
 n_windows=min(int(pd.read_csv(old/f'hrv_{tp}.csv').native_frames.min())//20 for tp in ['pre','post'])
 if n_windows<2:raise ValueError('Insufficient real BOLD frames for within-scan analysis')
 stats={};aligned=[]
 for tp in ['pre','post']:
  old=ROOT/'data/derived/external_preliminary_v1';m=pd.read_csv(old/f'hrv_{tp}.csv');x=torch.load(old/f'hrv_{tp}.pt',weights_only=True)
  m['scan_index']=np.arange(len(m));a=m.merge(scan[scan.Timepoint==tp],left_on='subject',right_on='Subject',how='left',validate='one_to_one',sort=False)
  a.to_csv(D/f'{tp}_recordings.csv',index=False);torch.save(x,D/f'{tp}.pt')
  b=win[(win.Timepoint==tp)&win.Subject.isin(m.subject)].copy()
  valid=valid_window_rows(b)
  b=b.loc[valid].copy()
  b=b.merge(m[['subject','scan_index','source','native_frames']],left_on='Subject',right_on='subject',validate='many_to_one')
  # Do not infer a compressed/censored frame map. Metadata must retain the native grid.
  checked={}
  for _,row in m.iterrows():
   meta=load((ROOT/row.source).with_suffix('.json'));checked[row.subject]=(not meta.get('censoring',False) and int(meta['n_tr'])==int(meta['shape'][1])==int(row.native_frames))
  b=b[b.Subject.map(checked)&(b.VolumeEndIdxExclusive<=b.native_frames)]
  b=complete_window_grid(b,n_windows)
  b['scan_key']=b.Subject+'_'+tp;b.to_csv(D/f'{tp}_windows.csv',index=False);aligned.append(b)
  stats[tp]=dict(imaging=len(m),matched_recordings=int(a.Subject.notna().sum()),usable_recordings=int(a.Usable.fillna(False).sum()),complete_window_scans=b.Subject.nunique(),windows_per_scan=n_windows,windows=len(b))
 allw=pd.concat(aligned,ignore_index=True);allw.to_csv(D/'windows.csv',index=False)
 save(R/'intake.json',dict(status='complete',evidence_status='preliminary',population='Older adults, existing strict imaging QC sample',input_revision='full-grid-v2',window_file=WINDOW_FILE,input_window_rows=len(win),windows_per_scan=n_windows,window_context_frames=20*n_windows,window_context_seconds=48*n_windows,scan_end_verified_in_csv=int(win.ScanEndVerified.fillna(False).sum()),counts=stats,window_subjects=allw.Subject.nunique(),window_scans=allw.scan_key.nunique(),window_rows=len(allw),source_hashes={f:sha(base/f) for f in ['HRV_biofeedback_PPG_HRV.csv',WINDOW_FILE]},protocol=dict(signal='PPG',variability='PRV, not ECG-derived HRV',primary='pulse_rate',secondary=['log_RMSSD','log_SDNN'],window_secondary=['log_RMSSD'],window_null='Nonzero circular shift within each scan; refit supervised steps; stationarity is an assumption',window_selection='Complete contiguous QC-passing grid of full windows within native BOLD extent; recording-only, partial and post-BOLD windows excluded',window_frequency_metrics='Not run: FrequencyEligible=False and all frequency columns missing',whole_recording_scope='Visit-matched association, not exact BOLD-time synchronization',window_scope='Provided scan-relative volume coordinates; tensor n_tr/shape match and censoring=False; upstream spatial alignment remains unverified',scan_grid_caveat='CSV ScanEndVerified flags are false; window end is bounded independently by imaging native_frames/n_tr/shape. Metadata-only verification; original physiological traces and scanner triggers were not re-audited',no_fixed_reliability_ceiling=True)))
 print(stats,flush=True)

def features(state):
 from model_loader import load_brainlm_forpretraining
 from probe_readouts import extract
 from rct_inscanner_patch_hr import patch_readouts,patch_amp
 from transfer_baseline import fc_feature,amp_feature
 from datasets import load_xyz
 from paths import XYZ_PATH
 selection=resolve_selection(STATES[state],'nacc',state.removeprefix('NACC_')) if state.startswith('NACC_') else None
 out=R/'features'/state;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():
  require_current_features(state);return
 n_windows=int(load(R/'intake.json')['windows_per_scan'])
 gpu_acquire()
 net=load_brainlm_forpretraining(STATES[state]).cuda().eval();xyz=load_xyz(XYZ_PATH)
 gpu_release()
 for tp in ['pre','post']:
  x=torch.load(D/f'{tp}.pt',weights_only=True)
  F=gpu_call(extract,net,x,xyz,[3],['emb_fc','roi_norm'],batch=4)
  for (_,name),a in F.items():np.save(out/f'{tp}_{name}.npy',a)
  np.save(out/f'{tp}_FC.npy',fc_feature(x.numpy(),None));np.save(out/f'{tp}_amp.npy',amp_feature(x.numpy(),None))
  # Real complete windows only; attention uses the full retained scan context.
  if x.shape[2]<20*n_windows:raise ValueError('Window grid exceeds stored tensor')
  xx=x[:,:,:20*n_windows].contiguous();P=gpu_call(patch_readouts,net,xx,xyz,3,batch=4);P['amp']=patch_amp(xx)
  patch=xx.reshape(len(xx),xx.shape[1],n_windows,20).permute(0,2,1,3).reshape(-1,xx.shape[1],20)
  P['FC']=fc_feature(patch.numpy(),None).reshape(len(xx),n_windows,-1)
  for name,a in P.items():
   if not np.isfinite(a).all():raise ValueError('Nonfinite PPG features')
   np.save(out/f'{tp}_patch_{name}.npy',a)
 save(out/'complete.json',dict(status='complete',checkpoint_sha256=sha(STATES[state]),source_intake_sha256=sha(R/'intake.json'),windows_per_scan=n_windows,state=state,adaptation_selection=selection,selected_epoch_note=selection_note(selection)))

def fitted(blocks,ks,y,g,seeds=(0,1,2)):
 ps=[]
 for seed in seeds:
  prepared=prepare_nested(blocks,ks,np.zeros(len(y)),g,seed);ps.append(predict_prepared(prepared,y,binary=False))
 return np.mean(ps,axis=0)

def comparisons(y,preds,g,pairs):
 rows=[]
 for a,b in pairs:
  delta,p,ci,detail=paired_bootstrap(y,preds[a],preds[b],n_boot=2000,groups=g,return_details=True);rows.append(dict(reference=a,test=b,delta=delta,raw_bootstrap_p=p,ci95=ci,**detail))
 return rows

def recordings(state):
 require_current_features(state)
 out=R/'recording_results'/f'{state}.json';fp=R/'features'/state;visits={};rows=[]
 for tp in ['pre','post']:
  m=pd.read_csv(D/f'{tp}_recordings.csv');F={name:np.load(fp/f'{tp}_{name}.npy') for name in ['FC','amp','emb_fc','roi_norm']};visits[tp]=(m,F)
  C=pd.get_dummies(m[['age_category','sex','arm']],drop_first=True,dtype=float).to_numpy()
  for target in TARGETS:
   y=target_values(m,target);keep=m.Usable.fillna(False).to_numpy(bool)&np.isfinite(y)&np.isfinite(C).all(1);g=m.loc[keep,'subject'].to_numpy();yy=y[keep]
   if len(yy)<20:rows.append(dict(kind='recording',visit=tp,target=target,status='insufficient_sample',n=len(yy)));continue
   arms={'demographics':([C[keep]],[C.shape[1]])}
   for name,f in F.items():arms[name]=([f[keep]],[10]);arms['demographics+'+name]=([C[keep],f[keep]],[C.shape[1],10])
   preds={name:fitted(bl,ks,yy,g) for name,(bl,ks) in arms.items()}
   tests=comparisons(yy,preds,g,[('amp','emb_fc'),('FC','emb_fc'),('demographics','demographics+emb_fc'),('demographics+amp','demographics+emb_fc')])
   rows.append(dict(kind='recording',status='complete',visit=tp,target=target,n=len(yy),subjects=g.tolist(),y=yy.tolist(),scores={k:metrics(yy,v) for k,v in preds.items()},predictions={k:v.tolist() for k,v in preds.items()},comparisons=tests))
   save(out,dict(status='running',state=state,rows=rows))
 a,fa=visits['pre'];b,fb=visits['post'];common=sorted(set(a.subject)&set(b.subject));ia=a.reset_index().set_index('subject').loc[common,'index'].to_numpy();ib=b.reset_index().set_index('subject').loc[common,'index'].to_numpy()
 for target in TARGETS:
  ya=target_values(a,target)[ia];yb=target_values(b,target)[ib];ok=np.isfinite(ya)&np.isfinite(yb)&a.Usable.fillna(False).to_numpy(bool)[ia]&b.Usable.fillna(False).to_numpy(bool)[ib];g=np.array(common)[ok];y=(yb-ya)[ok]
  if len(y)<20:rows.append(dict(kind='paired_change',target=target,status='insufficient_sample',n=len(y)));continue
  pred={name:fitted([(fb[name][ib]-fa[name][ia])[ok]],[10],y,g) for name in fa};tests=comparisons(y,pred,g,[('amp','emb_fc'),('FC','emb_fc')])
  rows.append(dict(kind='paired_change',status='complete',target=target,n=len(y),subjects=g.tolist(),y=y.tolist(),scores={k:metrics(y,v) for k,v in pred.items()},predictions={k:v.tolist() for k,v in pred.items()},comparisons=tests,interpretation='Association between pre/post changes, not baseline prognosis or a causal intervention effect',target_pre_post_rho=score(ya[ok],yb[ok],False)))
 tests=[t for row in rows for t in row.get('comparisons',[])]
 for t,p in zip(tests,holm([t['raw_bootstrap_p'] for t in tests])):t['holm_within_state']=p
 save(out,dict(status='complete',state=state,source_intake_sha256=sha(R/'intake.json'),evidence_status='preliminary',rows=rows,notes=['No ceilings imported from BREATHE or pSOPT','Whole-recording PPG may extend beyond analyzed BOLD','PRV changes are not equivalent to ECG HRV changes']))

def circular_null(y,scans,rng):
 yp=y.copy()
 for s in np.unique(scans):
  ix=np.flatnonzero(scans==s);yp[ix]=np.roll(y[ix],rng.integers(1,len(ix)))
 return yp

def windows(state):
 require_current_features(state)
 fp=R/'features'/state;w=pd.read_csv(D/'windows.csv').sort_values(['Subject','Timepoint','WindowIdx']).reset_index(drop=True);arrays={}
 for fam in ['FMmean','FMroi','amp','FC']:
  F={tp:np.load(fp/f'{tp}_patch_{fam}.npy',mmap_mode='r') for tp in ['pre','post']}
  arrays[fam]=np.stack([F[r.Timepoint][int(r.scan_index),int(r.WindowIdx)] for _,r in w.iterrows()])
 for fam,x in arrays.items():
  for scan in w.scan_key.unique():ix=np.flatnonzero(w.scan_key.to_numpy()==scan);x[ix]-=x[ix].mean(0)
 rows=[]
 for target in ['pulse_rate','log_RMSSD']:
  yy=target_values(w,target);valid=np.isfinite(yy)
  # Keep complete scans for circular-shift nulls; never compress gaps.
  valid &= pd.Series(valid).groupby(w.scan_key).transform('all').to_numpy()
  v=w.loc[valid].reset_index(drop=True);y=yy[valid];g=v.Subject.to_numpy();scans=v.scan_key.to_numpy()
  for scan in np.unique(scans):ix=np.flatnonzero(scans==scan);y[ix]-=y[ix].mean()
  if len(np.unique(g))<20:rows.append(dict(target=target,status='insufficient_sample',n_subjects=len(np.unique(g))));continue
  prepared={name:prepare_nested([a[valid]],[10],np.zeros(len(y)),g,0) for name,a in arrays.items()}
  preds={name:np.mean([predict_prepared(prepared[name],y,binary=False),*[fitted([a[valid]],[10],y,g,seeds=[s]) for s in [1,2]]],axis=0) for name,a in arrays.items()}
  tests=comparisons(y,preds,g,[('amp','FMroi'),('FC','FMroi'),('amp','FMmean')]);rng=np.random.default_rng(814);null={name:[] for name in ['FMroi','amp']}
  # Null and observed use exactly the same seed-0 pipeline/statistic.
  observed={name:score(y,predict_prepared(prepared[name],y,binary=False),False) for name in null}
  for p in range(200):
   yp=circular_null(y,scans,rng)
   for name in null:null[name].append(score(yp,predict_prepared(prepared[name],yp,binary=False),False))
   if p%25==0:print('null',state,target,p,flush=True)
  pvals={name:(1+sum(s>=observed[name] for s in null[name]))/201 for name in null}
  rows.append(dict(target=target,status='complete',n_subjects=len(np.unique(g)),n_scans=len(np.unique(scans)),n_windows=len(y),subjects=g.tolist(),scans=scans.tolist(),window_indices=v.WindowIdx.tolist(),y=y.tolist(),scores={k:metrics(y,p) for k,p in preds.items()},predictions={k:p.tolist() for k,p in preds.items()},comparisons=tests,circular_shift_null=dict(permutations=200,observed_seed0=observed,null=null,p=pvals,assumption='Circular stationarity over the retained complete nonoverlapping window grid; descriptive sensitivity, not an exact universal null')))
  save(R/'window_results'/f'{state}.json',dict(status='running',state=state,rows=rows))
 save(R/'window_results'/f'{state}.json',dict(status='complete',evidence_status='preliminary',state=state,source_intake_sha256=sha(R/'intake.json'),rows=rows,scope='Offline within-scan deviations, subject-held-out; windows and repeated visits are not independent subjects'))

def report():
 rows=[];nulls=[];adaptation=[]
 for folder in ['recording_results','window_results']:
  frozen=load(R/folder/'frozen.json')
  for state in STATES:
   d=load(R/folder/f'{state}.json')
   if d['status']!='complete':raise ValueError('Incomplete PPG analysis')
   if d.get('source_intake_sha256')!=sha(R/'intake.json'):raise ValueError('Stale PPG result intake')
   for row in d['rows']:
    if state!='frozen' and row.get('status')=='complete':
     baseline=next(q for q in frozen['rows'] if q.get('kind')==row.get('kind') and q.get('visit')==row.get('visit') and q['target']==row['target'])
     if baseline.get('status')!='complete' or baseline['subjects']!=row['subjects'] or baseline['y']!=row['y']:raise ValueError('Unpaired PPG model comparison')
     for name in row['predictions']:
      if name not in ['emb_fc','roi_norm','demographics+emb_fc','demographics+roi_norm','FMmean','FMroi']:continue
      delta,pv,ci,detail=paired_bootstrap(row['y'],baseline['predictions'][name],row['predictions'][name],n_boot=2000,groups=row['subjects'],return_details=True)
      adaptation.append(dict(family=folder,state=state,kind=row.get('kind'),visit=row.get('visit'),target=row['target'],readout=name,delta=delta,raw_bootstrap_p=pv,ci95=ci,**detail))
    for t in row.get('comparisons',[]):rows.append(dict(family=folder,state=state,kind=row.get('kind'),visit=row.get('visit'),target=row['target'],**t))
    for model,pv in row.get('circular_shift_null',{}).get('p',{}).items():nulls.append(dict(state=state,target=row['target'],model=model,p=pv))
 for r,p in zip(rows,holm([r['raw_bootstrap_p'] for r in rows])):r['holm_ppg_all_comparisons']=p
 for r,p in zip(nulls,holm([r['p'] for r in nulls])):r['holm_ppg_all_circular_nulls']=p
 for r,p in zip(adaptation,holm([r['raw_bootstrap_p'] for r in adaptation])):r['holm_ppg_adaptation_contrasts']=p
 pd.DataFrame(adaptation).to_csv(R/'adapted_vs_frozen.csv',index=False)
 save(R/'summary.json',dict(status='complete',comparisons=rows,circular_nulls=nulls,adapted_vs_frozen=adaptation,intake=load(R/'intake.json')));pd.DataFrame(rows).to_csv(R/'comparisons.csv',index=False)

from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','features','recordings','windows','report']);ap.add_argument('--state',choices=list(STATES),default='frozen');a=ap.parse_args();torch.set_num_threads(4)
 if a.stage in ['prepare','report']:globals()[a.stage]()
 else:globals()[a.stage](a.state)
if __name__=='__main__':main()
