#!/usr/bin/env python3
"""External evaluation of NACC-adapted encoders; no within-NACC test claims."""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_external_preliminary import dump,normalize,CHECKPOINTS
from model_loader import load_brainlm_forpretraining
from probe_readouts import brainlm_readout_batched
from datasets import load_xyz
from paths import XYZ_PATH
from benchmark_evaluation import score,oof_predictions,paired_bootstrap
from mci_axis_transfer import Axis,apply_gsr,apply_gain_match
from cohorts import load_cohort,center_rois
from rct_probe import upsample_short
from rct_positive_control import CONTROLS
from transfer_baseline import fc_feature

from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['MEM','EF'],required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--out-dir',type=Path);ap.add_argument('--adaptation-name',default='supervised');a=ap.parse_args();torch.set_num_threads(4)
 d=ROOT/'data/derived/nacc_parity_v1';out=a.out_dir or ROOT/f'runs/nacc_parity_v1/external_transfer/{a.target}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 ck=a.checkpoint or ROOT/f'runs/nacc_parity_v1/posttraining/{a.target}_external_seed{a.seed}/encoder.pt'
 if not ck.exists():raise ValueError('Required external-only checkpoint is missing')
 from run_external_preliminary import sha
 provenance=dict(checkpoint=str(ck),sha256=sha(ck),adaptation=a.adaptation_name,source='NACC',evaluation='external subjects only',target=a.target)
 pf=out/'checkpoint_provenance.json'
 if pf.exists() and json.loads(pf.read_text())!=provenance:raise ValueError('Checkpoint changed; use a new output directory')
 dump(pf,provenance)
 source_x=torch.load(d/f'ADNI_x_nacc_{a.target}.pt',weights_only=True);source_y=torch.load(d/f'ADNI_y_nacc_{a.target}.pt',weights_only=True).numpy();source_m=pd.read_csv(d/f'ADNI_manifest_nacc_{a.target}.csv')
 panel={'NACC_source':(source_x,None,None,None)}
 for co_name in ['act','cogte','breathe','psopt']:
  co=apply_gain_match(apply_gsr(center_rois(load_cohort(co_name,release='amp',require_labels=False)),co_name),co_name)
  co,_=upsample_short(co,200,20,fill='pad');ctrl=CONTROLS[co_name](co.subjects)
  clin=np.column_stack([co.baseline_levels('mem'),co.baseline_levels('ef'),ctrl['age'][0],ctrl['sex'][0]])
  target=a.target.lower();valid=co.analysis_mask()
  panel[co_name]=(co.bold[...,0][:,:,:200].contiguous(),co.subjects,{target:(np.asarray(co.baseline_levels(target)),np.asarray(co.followup_levels(target)),clin,valid)},None)
 p=ROOT/'data/derived/external_preliminary_v1';hm=pd.read_csv(p/'hrv_pre.csv');cat=pd.get_dummies(hm[['age_category','sex','arm']],drop_first=True,dtype=float).to_numpy()
 ht={t:(hm['pre_'+t].to_numpy(),hm['post_'+t].to_numpy(),np.column_stack([hm['pre_'+t],cat]),np.ones(len(hm),bool)) for t in ['flanker','LSWM','PCPS']}
 panel['hrv']=(torch.load(p/'hrv_pre.pt',weights_only=True),hm.subject.to_numpy(),ht,None)
 ad=ROOT/'data/derived/adni424_t200';tag='raw_ADNI_'+a.target+'_all_t200_off';am=pd.read_csv(ad/f'ADNI_manifest_{tag}.csv');ay=torch.load(ad/f'ADNI_y_{tag}.pt',weights_only=True).numpy()
 panel['adni']=(torch.load(ad/f'ADNI_x_{tag}.pt',weights_only=True),am.PTID.to_numpy(),{a.target.lower():(ay,None,None,np.ones(len(ay),bool))},None)
 xyz=load_xyz(XYZ_PATH)
 states={'frozen':CHECKPOINTS['frozen'],'NACC_adapted':ck,'ADNI_adapted':CHECKPOINTS[a.target]}
 gpu_acquire()
 for state,checkpoint in states.items():
  model=load_brainlm_forpretraining(checkpoint).cuda().eval()
  for name,(x,_,_,_) in panel.items():
   # ADNI-adapted scores on its own training cohort are not independent validation.
   if state=='ADNI_adapted' and name=='adni':continue
   f=out/(name+'_'+state+'.npy')
   if not f.exists():np.save(f,brainlm_readout_batched(model,x,xyz,3,'emb_fc',batch=4))
  del model;torch.cuda.empty_cache()
 from benchmark_gpu_phase import release_gpu_for_cpu_tail
 release_gpu_for_cpu_tail()
 for name,(x,_,_,_) in panel.items():
  f=out/(name+'_FC.npy')
  if not f.exists():np.save(f,fc_feature(x.numpy(),None).astype(np.float32))
 transfer=[];prognosis=[]
 for state in ['FC',*states]:
  source=np.load(out/('NACC_source_'+state+'.npy'),mmap_mode='r')
  axes={}
  for axis in ['all_raw','all_residualized','MCI']:
   y=source_y.copy();sel=np.ones(len(y),bool)
   if axis=='all_residualized':
    for st in source_m.dx_stage.unique():selst=source_m.dx_stage.to_numpy()==st;y[selst]-=y[selst].mean()
   if axis=='MCI':sel=source_m.dx_stage.to_numpy()=='MCI'
   axes[axis]=Axis(n_pca=200).fit(np.asarray(source[sel]),y[sel])
  for name,(x,subjects,targets,_) in panel.items():
   if name=='NACC_source' or (state=='ADNI_adapted' and name=='adni'):continue
   f=np.load(out/(name+'_'+state+'.npy'),mmap_mode='r')
   for axis,model in axes.items():
    pred=model.score(f)
    for target,(level,follow,clin,valid) in targets.items():
     mask=np.asarray(valid)&np.isfinite(level);transfer.append(dict(cohort=name,target=target,state=state,axis=axis,n=int(mask.sum()),rho=score(level[mask],pred[mask],False),subjects=np.asarray(subjects)[mask].tolist(),y=np.asarray(level)[mask].tolist(),prediction=pred[mask].tolist()))
   for target,(level,follow,clin,valid) in targets.items():
    if follow is None:continue
    mask=np.asarray(valid)&np.isfinite(follow)&np.isfinite(clin).all(1);ix=np.flatnonzero(mask)
    if len(ix)<20:continue
    y=np.asarray(follow)[ix];g=np.asarray(subjects)[ix];C=clin[ix]
    cache=out/(name+'_'+target+'_clinical.json')
    if cache.exists():pc=np.array(json.loads(cache.read_text())['prediction'])
    else:
     pc=np.mean([oof_predictions([C],[C.shape[1]],y,s,groups=g) for s in [0,1,2]],axis=0);dump(cache,dict(prediction=pc.tolist()))
    pf=np.mean([oof_predictions([C,np.asarray(f[ix])],[C.shape[1],5],y,s,groups=g) for s in [0,1,2]],axis=0)
    delta,pval,ci,details=paired_bootstrap(y,pc,pf,n_boot=2000,groups=g,return_details=True)
    prognosis.append(dict(cohort=name,target=target,state=state,n=len(ix),clinical_rho=score(y,pc,False),imaging_rho=score(y,pf,False),delta=delta,raw_bootstrap_p=pval,ci95=ci,subjects=g.tolist(),y=y.tolist(),clinical_prediction=pc.tolist(),imaging_prediction=pf.tolist(),**details))
   dump(out/'results.json',dict(status='preliminary_in_progress',source='NACC '+a.target,seed=a.seed,transfer=transfer,prognosis=prognosis))
 # Match adapted-minus-frozen predictions within each cohort/endpoint.
 comparisons=[]
 for r in prognosis:
  if r['state'] not in ['NACC_adapted','ADNI_adapted']:continue
  ref=next(v for v in prognosis if v['state']=='frozen' and v['cohort']==r['cohort'] and v['target']==r['target']);assert ref['subjects']==r['subjects']
  delta,pv,ci,detail=paired_bootstrap(r['y'],ref['imaging_prediction'],r['imaging_prediction'],n_boot=2000,groups=r['subjects'],return_details=True)
  comparisons.append(dict(cohort=r['cohort'],target=r['target'],state=r['state'],delta=delta,raw_bootstrap_p=pv,ci95=ci,**detail))
 from confirm_matched_axis import holm
 for r,pv in zip(comparisons,holm([r['raw_bootstrap_p'] for r in comparisons])):r['holm_within_source_seed']=pv
 dump(out/'results.json',dict(status='preliminary_complete',source='NACC '+a.target,seed=a.seed,transfer=transfer,prognosis=prognosis,adaptation_comparisons=comparisons,notes=['ADNI-adapted encoder is not independently evaluated on ADNI','Different cognitive constructs remain separately named','Clinical CIs conditional on fitted OOF; no causal treatment claims']))
if __name__=='__main__':main()
