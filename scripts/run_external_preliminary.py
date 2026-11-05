#!/usr/bin/env python3
"""Reproducible preliminary NACC/HRV benchmark; keeps original cohorts untouched.

Source alignment and NACC composite definitions remain unverified. Never promote
these outputs to final validation solely because the workflow completed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from benchmark_evaluation import oof_predictions, paired_bootstrap, score
from paths import OFFICIAL_BRAINLM_13M, XYZ_PATH
RUN=ROOT/'runs/external_preliminary_v1'; DATA=ROOT/'data/derived/external_preliminary_v1'
SEEDS=[0,1,2]; NBOOT=2000
CHECKPOINTS={'frozen':Path(OFFICIAL_BRAINLM_13M),'MEM':ROOT/'checkpoints/finetuned/brainlm13m_adniMEM_level.pt','EF':ROOT/'checkpoints/finetuned/brainlm13m_adniEF_level.pt'}
FM=['frozen_emb_fc','MEM_emb_fc','EF_emb_fc','frozen_roi_norm']
FEATURES=['FC','amplitude']+FM

def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));tmp.replace(path)
def status(stage,state,**kw):
 dump(RUN/f'status_{stage}.json',dict(stage=stage,state=state,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**kw))
def loadx(ds):return torch.load(DATA/f'{ds}.pt',map_location='cpu',weights_only=True)
def manifest(ds):return pd.read_csv(DATA/f'{ds}.csv')
def feat(ds,name):return np.load(RUN/'features'/f'{ds}_{name}.npy',mmap_mode='r')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def normalize(x,length=200):
 """Full observed-run centering/gain, crop; short runs padded with zero only."""
 x=x.float();sd=x.std(dim=1,unbiased=False)
 if x.ndim!=2 or x.shape[0]!=424 or not torch.isfinite(x).all():raise ValueError('invalid_tensor')
 if (sd<=1e-8).any():raise ValueError('constant_roi_full')
 if (x[:,:length].std(dim=1,unbiased=False)<=1e-8).any():raise ValueError('constant_roi_window')
 x=(x-x.mean(dim=1,keepdim=True))/sd.median()
 if x.shape[1]<length:x=torch.nn.functional.pad(x,(0,length-x.shape[1]))
 return x[:,:length].contiguous()
def save_dataset(name,records,tensors):
 assert len(records)==len(tensors)>0
 pd.DataFrame(records).to_csv(DATA/f'{name}.csv',index=False)
 torch.save(torch.stack(tensors),DATA/f'{name}.pt')
def prepare():
 DATA.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
 cfg=dict(protocol='external-preliminary-v1-2026-09-30',status='preliminary',source_alignment='Same script/atlas reported by user; cross-template alignment not independently verified',atlas='A424',external_gsr=True,source_adni='existing adni424_t200 amplitude-preserving builds; no additional GSR',normalization='full observed-run ROI centering and scan-median SD division; crop 200 or zero-pad; no stretching',nacc_age_min=60,nacc_max_abs_cog_interval_months=6,nacc_outcomes=['mem_composite_nan','exec_composite_nan'],hrv_group='OA',hrv_outcomes=['flanker','LSWM','PCPS'],hrv_score='uncorrected standardScore',exclude='nonfinite; any ROI SD <=1e-8 in full run or first 200 frames; source QC flag=1',seeds=SEEDS,outer_folds=5,inner='subject-grouped 20% training-only holdout as existing benchmark_evaluation',local_dims=100,fusion_dims=[50,50],clinical_imaging_dims=5,bootstrap=NBOOT,checkpoint_sha256={k:sha(v) for k,v in CHECKPOINTS.items()},blocking_final_claims=['spatial alignment unverified','NACC composite construction and missing-value coding unverified','pretraining/participant overlap not audited'],deferred=['NACC diagnosis until code dictionary verified','HRV five-study cognitive LOSO until constructs harmonized','larger FMs and learning curves; GSR/age/interval sensitivities'])
 configpath=RUN/'protocol.json'
 if configpath.exists():
  if json.loads(configpath.read_text())!=cfg:raise RuntimeError('Protocol changed; use a new run directory')
 else:dump(configpath,cfg)
 if (RUN/'intake.json').exists():return
 audit=[];rows=[];xx=[]
 raw=ROOT/'data/NACC/A424_GSR/raw'
 metas={}
 for p in sorted(raw.glob('*.json')):
  m=json.loads(p.read_text());key=(m['subject'],str(m['session']).removeprefix('ses-'))
  if key in metas:raise ValueError('Duplicate NACC subject/session file')
  metas[key]=(p.with_suffix('.pt'),m)
 tab=pd.read_csv(ROOT/'data/NACC/fc_cognitive_composites_NACC.csv')
 if tab.ID.duplicated().any():raise ValueError('NACC table changed; revisit grouped task sampling')
 for _,r in tab.iterrows():
  rec={'cohort':'nacc','subject':r.ID,'session':str(r.session)};reason=[]
  pair=metas.get((r.ID,str(r.session)))
  if not pair:reason.append('no_scan')
  if not np.isfinite(r.NACCAGE) or r.NACCAGE<60:reason.append('age')
  if not np.isfinite(r.cog_interval_months) or abs(r.cog_interval_months)>6:reason.append('cognitive_interval')
  if pair:
   f,m=pair
   if str(m.get('source_qc_exclude_recommend'))=='1':reason.append('source_QC')
   try:x=normalize(torch.load(f,map_location='cpu',weights_only=True))
   except ValueError as e:reason.append(str(e))
  audit.append(dict(rec,exclusion=';'.join(reason)))
  if reason:continue
  rec.update(age=float(r.NACCAGE),memory=r.mem_composite_nan,executive=r.exec_composite_nan,cog_interval_months=int(r.cog_interval_months),source=str(f.relative_to(ROOT)),source_sha256=sha(f),native_frames=int(m['shape'][-1]),native_tr=float(m['tr']))
  rows.append(rec);xx.append(x)
 save_dataset('nacc',rows,xx)
 h=ROOT/'data/HRV_biofeedback'
 with zipfile.ZipFile(h/'label/phenotype.zip') as z:
  demo=pd.read_csv(z.open('Demographics.tsv'),sep='\t');cog=pd.read_csv(z.open('NIHcognition_summary.tsv'),sep='\t')
 tab=demo.merge(cog,on=['participant_id','age_group'],validate='one_to_one').merge(pd.read_csv(h/'label/participants.tsv',sep='\t'),on='participant_id',suffixes=('','_p'),validate='one_to_one')
 for visit,timepoint in [('T1','pre'),('T2','post')]:
  rows=[];xx=[]
  for _,r in tab[tab.age_group=='OA'].iterrows():
   rec={'cohort':'hrv','subject':r.participant_id,'session':timepoint};reason=[]
   f=h/f'timeseries/A424_GSR/raw/{visit}/{r.participant_id}_ses-{timepoint}_timeseries.pt'
   if not f.exists():reason.append('no_scan')
   qc=r[f'QC_ses-{timepoint}_task-rest_1excludeRecommend_0goodTouse']
   if pd.isna(qc) or float(qc)!=0:reason.append('source_QC_not_pass')
   if f.exists():
    try:x=normalize(torch.load(f,map_location='cpu',weights_only=True))
    except ValueError as e:reason.append(str(e))
   audit.append(dict(rec,exclusion=';'.join(reason)))
   if reason:continue
   rec.update(sex=r.gender,age_category=str(r.age_category),arm=r.condition,source=str(f.relative_to(ROOT)),source_sha256=sha(f),native_frames=175,native_tr=2.4)
   for target in ['flanker','LSWM','PCPS']:
    for tp in ['pre','post']:rec[tp+'_'+target]=r[tp+'_'+target+'_standardScore']
   rows.append(rec);xx.append(x)
  save_dataset('hrv_'+timepoint,rows,xx)
 pd.DataFrame(audit).to_csv(DATA/'intake_audit.csv',index=False)
 summary={}
 for ds in ['nacc','hrv_pre','hrv_post']:
  m=manifest(ds);summary[ds]={'n':len(m),'finite_outcomes':{c:int(pd.to_numeric(m[c],errors='coerce').notna().sum()) for c in m if c in ['memory','executive'] or c.startswith(('pre_','post_'))}}
 a=manifest('hrv_pre');b=manifest('hrv_post');summary['hrv_paired_qc']=len(set(a.subject)&set(b.subject))
 dump(RUN/'intake.json',summary);print('INTAKE',summary,flush=True)

def datasets():return ['nacc','hrv_pre','hrv_post','adni_MEM','adni_EF']
def source_tensor(ds):
 if ds.startswith('adni_'):
  label=ds.split('_')[1];base=ROOT/'data/derived/adni424_t200';tag=f'raw_ADNI_{label}_all_t200_off'
  return torch.load(base/f'ADNI_x_{tag}.pt',map_location='cpu',weights_only=True).float()
 return loadx(ds)
def features():
 from model_loader import load_brainlm_forpretraining
 from datasets import load_xyz
 from probe_readouts import extract
 from transfer_baseline import fc_feature,amp_feature
 fd=RUN/'features';fd.mkdir(exist_ok=True);torch.set_num_threads(4)
 for ds in datasets():
  if all((fd/f'{ds}_{n}.npy').exists() for n in ['FC','amplitude']):continue
  x=source_tensor(ds)
  for n,fn in [('FC',fc_feature),('amplitude',amp_feature)]:
   path=fd/f'{ds}_{n}.npy'
   if path.exists():continue
   a=fn(x.numpy(),np.full(len(x),200));assert np.isfinite(a).all()
   temp=path.with_suffix('.tmp.npy');np.save(temp,a.astype(np.float32));temp.replace(path)
 xyz=load_xyz(XYZ_PATH)
 if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; baseline output retained')
 for state,checkpoint in CHECKPOINTS.items():
  model=load_brainlm_forpretraining(checkpoint).cuda().eval()
  specs=[(3,'emb_fc')]+([(4,'roi_norm')] if state=='frozen' else [])
  for ds in datasets():
   missing=[(l,r) for l,r in specs if not (fd/f'{ds}_{state}_{r}.npy').exists()]
   if not missing:continue
   status('features','running',dataset=ds,checkpoint=state)
   x=source_tensor(ds)
   # Extract each required combination once; extra cross-product readouts are not retained.
   vals=extract(model,x,xyz,sorted({l for l,r in missing}),sorted({r for l,r in missing}),batch=4)
   for l,r in missing:
    a=vals[l,r];assert np.isfinite(a).all();path=fd/f'{ds}_{state}_{r}.npy';temp=path.with_suffix('.tmp.npy');np.save(temp,a);temp.replace(path)
   del vals,x
  del model;torch.cuda.empty_cache()
 status('features','complete')

def wait_baseline(ds):
 paths=[RUN/'features'/f'{ds}_{n}.npy' for n in ['FC','amplitude']]
 while not all(p.exists() for p in paths):
  st=RUN/'status_features.json'
  if st.exists() and json.loads(st.read_text())['state']=='failed':raise RuntimeError('Feature worker failed')
  time.sleep(2)
def metrics(y,p):return dict(rho=score(y,p,False),mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)))
def fit_task(name,ds,y,mask,clinical=None,baseline_only=False):
 m=manifest(ds);mask=np.asarray(mask)&np.isfinite(y)
 if clinical is not None:mask &= np.isfinite(clinical).all(axis=1)
 ix=np.flatnonzero(mask);y=np.asarray(y)[ix];groups=m.subject.to_numpy()[ix]
 if len(y)<20 or np.std(y)==0:
  dump(RUN/'results'/f'{name}.json',dict(status='insufficient_data',n=len(y)));return
 assert len(np.unique(groups))==len(groups),'Current external tasks require one visit per participant'
 names=['FC','amplitude'] if baseline_only else FEATURES
 f={n:np.asarray(feat(ds,n)[ix]) for n in names}
 arms={n:([a],[100]) for n,a in f.items()}
 if not baseline_only:arms['FC50+FM50']=([f['FC'],f['frozen_emb_fc']],[50,50])
 if clinical is not None:
  clin=clinical[ix];arms={'clinical':([clin],[clin.shape[1]])}
  for n,a in f.items():arms['clinical+'+n]=([clin,a],[clin.shape[1],5])
  if not baseline_only:arms['clinical+FC5+FM5']=([clin,f['FC'],f['frozen_emb_fc']],[clin.shape[1],5,5])
 out=RUN/'results'/f'{name}.json'
 result=json.loads(out.read_text()) if out.exists() else dict(status='preliminary',task=name,dataset=ds,n=len(y),subjects=groups.tolist(),y=y.tolist(),scores={},predictions={},comparisons=[])
 for a,(blocks,ks) in arms.items():
  if a in result['scores']:continue
  status('evaluation','running',task=name,arm=a,n=len(y))
  print('EVALUATE',name,a,'n',len(y),flush=True)
  oofs=[oof_predictions(blocks,ks,y,s,groups=groups) for s in SEEDS];pred=np.mean(oofs,axis=0)
  result['scores'][a]=dict(metrics(y,pred),seed_mean_rho=float(np.mean([score(y,o,False) for o in oofs])),seed_sd_rho=float(np.std([score(y,o,False) for o in oofs])))
  result['predictions'][a]=pred.tolist();dump(out,result)
 if baseline_only:return
 prefix='clinical+' if clinical is not None else ''
 pairs=[(prefix+'FC',prefix+'frozen_emb_fc'),(prefix+'frozen_emb_fc',prefix+'MEM_emb_fc'),(prefix+'frozen_emb_fc',prefix+'EF_emb_fc')]
 if clinical is None:pairs.append(('FC','FC50+FM50'))
 else:pairs += [('clinical',prefix+'FC'),('clinical',prefix+'frozen_emb_fc'),('clinical',prefix+'MEM_emb_fc'),('clinical',prefix+'EF_emb_fc'),(prefix+'FC','clinical+FC5+FM5')]
 result['comparisons']=[]
 for a,b in pairs:
  delta,pv,ci,details=paired_bootstrap(y,result['predictions'][a],result['predictions'][b],n_boot=NBOOT,groups=groups,return_details=True)
  result['comparisons'].append(dict(a=a,b=b,delta_rho=delta,raw_one_sided_bootstrap_p=pv,ci95=ci,**details))
 from confirm_matched_axis import holm
 for r,h in zip(result['comparisons'],holm([r['raw_one_sided_bootstrap_p'] for r in result['comparisons']])):r['holm_within_task']=h
 result['correction_scope']='within task displayed comparisons; no across-task confirmatory claim';dump(out,result)

def evaluate(baseline_only=False):
 if not baseline_only:fingerprint()
 for ds in ['nacc','hrv_pre']:
  if baseline_only:wait_baseline(ds)
  m=manifest(ds)
  if ds=='nacc':
   for target in ['memory','executive']:fit_task('nacc_local_'+target,ds,m[target].to_numpy(float),np.ones(len(m),bool),baseline_only=baseline_only)
  else:
   cat=pd.get_dummies(m[['age_category','sex','arm']],drop_first=True,dtype=float).to_numpy()
   for target in ['flanker','LSWM','PCPS']:
    y=m['pre_'+target].to_numpy(float)
    fit_task('hrv_local_'+target,ds,y,np.ones(len(m),bool),baseline_only=baseline_only)
    clin=np.column_stack([y,cat]);post=m['post_'+target].to_numpy(float)
    fit_task('hrv_prognosis_'+target,ds,post,np.isfinite(y),clinical=clin,baseline_only=baseline_only)
 if not baseline_only:transfer();fingerprint();status('evaluation','complete')

def axis_training_target(y, stages, arm):
 """Exclude missing stage from stage-dependent fits; never invent a category."""
 stages = pd.Series(stages, dtype="string")
 known = stages.isin(["CN", "MCI", "Dementia"]).to_numpy()
 target = np.asarray(y, dtype=float).copy()
 if arm == "MCI":
  selected = stages.eq("MCI").fillna(False).to_numpy(dtype=bool)
 elif arm == "all_residualized":
  selected = known
  for stage in ["CN", "MCI", "Dementia"]:
   ix = stages.eq(stage).fillna(False).to_numpy(dtype=bool)
   if ix.any(): target[ix] -= target[ix].mean()
 else:
  raise ValueError(f"Unknown axis arm: {arm}")
 return selected, target

def transfer():
 from mci_axis_transfer import Axis
 from confirm_matched_axis import holm
 for domain in ['MEM','EF']:
  base=ROOT/'data/derived/adni424_t200';tag=f'raw_ADNI_{domain}_all_t200_off'
  y=torch.load(base/f'ADNI_y_{tag}.pt',map_location='cpu',weights_only=True).numpy().astype(float)
  sm=pd.read_csv(base/f'ADNI_manifest_{tag}.csv');stage=sm.dx_stage.to_numpy()
  out=RUN/'results'/f'transfer_{domain}.json'
  result=json.loads(out.read_text()) if out.exists() else dict(status='preliminary',source='ADNI '+domain,rows=[],predictions={},interpretation='NACC domain composites and HRV tests are not assumed to have identical scales or constructs')
  for family in FEATURES:
   F=feat('adni_'+domain,family)
   for arm in ['MCI','all_residualized']:
    key=family+'__'+arm
    if key in result['predictions']:continue
    sel,target=axis_training_target(y,stage,arm)
    status('evaluation','running',task='transfer_'+domain,arm=key)
    model=Axis(n_pca=200).fit(np.asarray(F[sel]),target[sel]);result['predictions'][key]={}
    for ds in ['nacc','hrv_pre']:
     m=manifest(ds);p=model.score(feat(ds,family));result['predictions'][key][ds]={'subjects':m.subject.tolist(),'prediction':p.tolist()}
     targets=['memory' if domain=='MEM' else 'executive'] if ds=='nacc' else ['pre_flanker','pre_LSWM','pre_PCPS']
     for t in targets:
      obs=m[t].to_numpy(float);good=np.isfinite(obs);rho=score(obs[good],p[good],False)
      rng=np.random.default_rng(0);null=[score(rng.permutation(obs[good]),p[good],False) for _ in range(1000)]
      pv=float((1+np.sum(np.abs(null)>=abs(rho)))/1001)
      result['rows'].append(dict(dataset=ds,target=t,family=family,axis=arm,n=int(good.sum()),rho=rho,raw_two_sided_permutation_p=pv,source_n=int(sel.sum()),source_missing_stage_excluded=int(pd.isna(stage).sum())))
    dump(out,result)
  # Paired model comparisons use identical external participants and outcomes.
  result['comparisons']=[]
  for arm in ['MCI','all_residualized']:
   for ds in ['nacc','hrv_pre']:
    m=manifest(ds);targets=['memory' if domain=='MEM' else 'executive'] if ds=='nacc' else ['pre_flanker','pre_LSWM','pre_PCPS']
    for t in targets:
     obs=m[t].to_numpy(float);good=np.isfinite(obs)
     for a,b in [('FC','frozen_emb_fc'),('frozen_emb_fc','MEM_emb_fc'),('frozen_emb_fc','EF_emb_fc')]:
      pa=np.asarray(result['predictions'][a+'__'+arm][ds]['prediction']);pb=np.asarray(result['predictions'][b+'__'+arm][ds]['prediction'])
      delta,pv,ci,detail=paired_bootstrap(obs[good],pa[good],pb[good],n_boot=NBOOT,groups=m.subject.to_numpy()[good],return_details=True)
      result['comparisons'].append(dict(dataset=ds,target=t,axis=arm,a=a,b=b,delta_rho=delta,ci95=ci,raw_one_sided_bootstrap_p=pv,**detail))
  for r,h in zip(result['rows'],holm([r['raw_two_sided_permutation_p'] for r in result['rows']])):r['holm_within_source_domain']=h
  for r,h in zip(result['comparisons'],holm([r['raw_one_sided_bootstrap_p'] for r in result['comparisons']])):r['holm_within_source_domain']=h
  dump(out,result)

def fingerprint():
 a=manifest('hrv_pre');b=manifest('hrv_post');ids=sorted(set(a.subject)&set(b.subject));ia=[a.index[a.subject==s][0] for s in ids];ib=[b.index[b.subject==s][0] for s in ids]
 rows=[]
 for family in FEATURES:
  A=np.array(feat('hrv_pre',family)[ia]);B=np.array(feat('hrv_post',family)[ib]);A-=A.mean(axis=1,keepdims=True);B-=B.mean(axis=1,keepdims=True)
  normsA=np.linalg.norm(A,axis=1);normsB=np.linalg.norm(B,axis=1)
  if np.any(normsA==0) or np.any(normsB==0):raise ValueError('Constant fingerprint representation')
  sim=(A/normsA[:,None])@(B/normsB[:,None]).T
  rows.append(dict(family=family,n=len(ids),chance=1/len(ids),top1_pre_post=float(np.mean(sim.argmax(axis=1)==np.arange(len(ids)))),top1_post_pre=float(np.mean(sim.argmax(axis=0)==np.arange(len(ids))))))
 dump(RUN/'results/fingerprint_hrv.json',dict(status='preliminary',rows=rows))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','features','baseline','evaluate','transfer','all'],default='all');args=ap.parse_args();torch.set_num_threads(4)
 worker=None
 try:
  status(args.stage,'running');prepare()
  if args.stage=='features':features()
  elif args.stage=='baseline':evaluate(True)
  elif args.stage=='evaluate':evaluate(False)
  elif args.stage=='transfer':
   status('all','recovering_transfer');transfer();fingerprint();status('evaluation','complete');status('all','complete')
  elif args.stage=='all':
   logdir=ROOT/'logs/external_preliminary_v1';logdir.mkdir(parents=True,exist_ok=True)
   with (logdir/'features.log').open('a') as log:
    worker=subprocess.Popen([sys.executable,__file__,'--stage','features'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    dump(RUN/'workers.json',dict(queue_pid=os.getpid(),feature_pid=worker.pid))
    evaluate(True)
    while worker.poll() is None:status('all','waiting_for_features',feature_pid=worker.pid);time.sleep(10)
    if worker.returncode:raise RuntimeError('Feature worker failed; see features.log')
   evaluate(False)
  status(args.stage,'complete')
 except Exception as e:
  if worker is not None and worker.poll() is None:worker.terminate()
  status(args.stage,'failed',error=str(e));traceback.print_exc();raise
if __name__=='__main__':main()
