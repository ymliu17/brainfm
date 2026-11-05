#!/usr/bin/env python3
"""Focused benchmark extension. All new results remain preliminary.

A fixed feature budget, paired outer splits, training-only transformations and
inner readout selection are shared by every arm. Existing outputs are untouched.
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, mean_squared_error, mean_absolute_error, r2_score)
from scipy.stats import spearmanr, chi2_contingency
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from benchmark_evaluation import grouped_splits, transform_blocks, _predict, score, paired_bootstrap
from run_external_preliminary import dump, sha
from confirm_matched_axis import holm
R=ROOT/'runs/benchmark_extension_v1';D=ROOT/'data/derived/benchmark_extension_v1'
GRID=[.001,.01,.1,1.,10.]

def clean_json(v):
 if isinstance(v,dict):return {str(k):clean_json(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean_json(x) for x in v]
 if isinstance(v,np.ndarray):return clean_json(v.tolist())
 if isinstance(v,np.generic):v=v.item()
 if isinstance(v,float) and not np.isfinite(v):return None
 return v

def save(p,d):dump(p,clean_json(d))
def load(p):return json.loads(Path(p).read_text())
def valid_normalize(x,length):
 x=x.float()
 if x.ndim!=2 or not torch.isfinite(x).all():raise ValueError('nonfinite_or_shape')
 sd=x.std(1,unbiased=False)
 if (sd<=1e-8).any() or (x[:,:length].std(1,unbiased=False)<=1e-8).any():raise ValueError('constant_roi')
 x=(x-x.mean(1,keepdim=True))/sd.median()
 return torch.nn.functional.pad(x,(0,max(0,length-x.shape[1])))[:,:length].contiguous()

def prepare():
 D.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 m=pd.read_csv(ROOT/'data/derived/nacc_uds_v1/nacc_enriched.csv')
 c=pd.read_csv(ROOT/'data/NACC/fc_cognitive_composites_NACC.csv')
 m=m.merge(c[['ID','NACCADC']],left_on='subject',right_on='ID',validate='one_to_one',sort=False)
 m['SITE']=m.NACCADC;m['MEM']=m.memory;m['EF']=m.executive
 m['DX']=m.NACCUDSD.map({1:0.,4:1.});m['DX_ord']=m.NACCUDSD.map({1:0.,3:.5,4:1.})
 h=pd.read_csv(ROOT/'data/derived/external_preliminary_v1/hrv_pre.csv')
 variants={'nacc_default':(m,'A424_GSR',200,0),'nacc_noGSR':(m,'A424_noGSR',200,0),
   'nacc_t100':(m,'A424_GSR',100,200),'nacc_t200long':(m,'A424_GSR',200,200),
   'nacc_sch160':(m,'Schaefer400Tian_GSR',160,160),'nacc_a160':(m,'A424_GSR',160,160),
   'hrv_default':(h,'A424_GSR',200,0),'hrv_t160':(h,'A424_GSR',160,0),
   'hrv_noGSR':(h,'A424_noGSR',200,0)}
 counts={};exclusions=[]
 for name,(tab,atlas,length,minframes) in variants.items():
  rows=[];xx=[]
  for _,r in tab.iterrows():
   if r.native_frames<minframes:continue
   p=ROOT/str(r.source).replace('A424_GSR',atlas)
   try:
    raw=torch.load(p,weights_only=True,map_location='cpu');x=valid_normalize(raw,length)
   except (ValueError,FileNotFoundError) as e:
    exclusions.append(dict(dataset=name,subject=r.subject,reason=str(e)));continue
   q=r.to_dict();q.update(feature_source=str(p.relative_to(ROOT)),feature_sha256=sha(p),analyzed_frames=length)
   rows.append(q);xx.append(x)
  pd.DataFrame(rows).to_csv(D/(name+'.csv'),index=False);torch.save(torch.stack(xx),D/(name+'.pt'));counts[name]=len(rows)
 save(R/'intake.json',dict(status='preliminary',counts=counts,exclusions=exclusions,
   missing=['NACC repeated cognitive visits','scan-level mean FD','independently verified atlas/BOLD registration and ROI-order record','exact UDS visit day'],
   policy='No fabricated nuisance values or longitudinal labels. Existing >=60 age and <=6 month cognition interval selection retained.',
   input_hashes={str(p):sha(ROOT/p) for p in ['data/NACC/fc_cognitive_composites_NACC.csv','data/NACC/uds_SelectedVariables.csv']},
   analysis_status='Exploratory follow-up after inspecting earlier results; new random splits do not constitute independent replication'))
 print(counts,flush=True)

def feature_paths(ds,model):return R/'features'/f'{ds}_{model}'
def features(ds,model):
 from transfer_baseline import fc_feature, amp_feature
 from paths import OFFICIAL_BRAINLM_13M, XYZ_PATH, BRAINLM_111M
 from datasets import load_xyz
 from model_loader import load_brainlm_forpretraining
 from probe_readouts import extract,extract_vitmae
 p=feature_paths(ds,model);p.mkdir(parents=True,exist_ok=True)
 if (p/'complete.json').exists():return
 x=torch.load(D/(ds+'.pt'),weights_only=True);m=pd.read_csv(D/(ds+'.csv'))
 torch.manual_seed(42)
 if model=='shuffle':
  # Permute observed frames only; keep padding at the end. Same permutation for all ROIs.
  for i,n in enumerate(m.native_frames):
   k=min(int(n),x.shape[2]);ix=torch.randperm(k);x[i,:,:k]=x[i,:,:k][:,ix]
 out={'FC':fc_feature(x.numpy(),None).astype('float32'),'amplitude':amp_feature(x.numpy(),None).astype('float32')}
 ck=None
 gpu_acquire()
 if model in ['brainlm13','shuffle','random']:
  net=load_brainlm_forpretraining(OFFICIAL_BRAINLM_13M)
  if model=='random':
   from brainlm_mae.modeling_brainlm import BrainLMForPretraining
   net=BrainLMForPretraining(net.config)
  else:ck=OFFICIAL_BRAINLM_13M
  net=net.cuda().eval();xyz=load_xyz(XYZ_PATH)
  f=extract(net,x,xyz,[3,4],['emb_fc','roi_norm'],batch=4)
 elif model=='brainlm111':
  from vitmae_input import load_vitmae
  from brainlm_input import roi_order_by_y
  ck=BRAINLM_111M;net,cfg=load_vitmae(Path(ck).parent,device='cuda')
  f=extract_vitmae(net,cfg,x,roi_order_by_y(XYZ_PATH),[6,12],['emb_fc','roi_norm','mean_real'],batch=2)
 elif model=='jepa':
  from brainjepa import load_brainjepa,jepa_readout_multi,DEFAULT_CKPT
  ck=DEFAULT_CKPT;net=load_brainjepa(device='cuda')
  f=jepa_readout_multi(net,x,layers=[6,12],readouts=['mean','roi_norm'],batch=1)
 else:raise ValueError(model)
 del net
 gpu_release()
 for (layer,readout),a in f.items():out[f'L{layer}_{readout}']=a.astype('float32')
 for k,a in out.items():
  if not np.isfinite(a).all():raise ValueError('Nonfinite features '+k)
  np.save(p/(k+'.npy'),a)
 if model=='shuffle':
  original=torch.load(D/(ds+'.pt'),weights_only=True).numpy();error=float(np.max(np.abs(out['FC']-fc_feature(original,None))))
  if error>1e-5:raise ValueError('Time shuffle changed static FC')
 else:error=None
 ckfile=(Path(ck)/'pytorch_model.bin' if ck and Path(ck).is_dir() else ck)
 save(p/'complete.json',dict(dataset=ds,model=model,subjects=m.subject.tolist(),keys=list(out),checkpoint=str(ck),checkpoint_sha256=sha(ckfile) if ckfile else None,random_seed=42,fc_shuffle_max_error=error))

def get_features(ds,model,subjects=None):
 p=feature_paths(ds,model);meta=load(p/'complete.json');ix=np.arange(len(meta['subjects']))
 if subjects is not None:
  lookup={s:i for i,s in enumerate(meta['subjects'])};ix=np.array([lookup[s] for s in subjects])
 return {k:np.asarray(np.load(p/(k+'.npy'),mmap_mode='r')[ix]) for k in meta['keys']}

def effective_dimensions(ks,n_train):
 """Keep the total imaging rank matched even in small training/site folds."""
 ks=list(ks);imaging=[i for i,k in enumerate(ks) if k>=50]
 if not imaging:return ks
 total=sum(ks[i] for i in imaging);budget=min(total,n_train-1)
 if budget<len(imaging):raise ValueError('Too few training samples for matched feature blocks')
 remaining=budget
 for pos,i in enumerate(imaging):
  k=remaining if pos==len(imaging)-1 else max(1,int(budget*ks[i]/total))
  ks[i]=k;remaining-=k
 return ks

def nested_predict(candidates,y,groups,seed,sites=None,independent=False):
 """Choose readout AND regularization in the inner training split only."""
 groups=np.asarray(groups);binary=set(np.unique(y))=={0,1};splitgroups=groups if sites is None else np.asarray(sites)
 if independent:
  outer=list(GroupKFold(5).split(np.zeros(len(y)),groups=splitgroups))
 else:outer=list(grouped_splits(y,splitgroups,seed))
 pred=np.full(len(y),np.nan);audit=[]
 for fold,(tr,te) in enumerate(outer):
  # Multiple deterministic attempts avoid single-class validation in small diagnostic cells.
  for attempt in range(100):
   a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=seed*1000+fold*100+attempt).split(np.zeros(len(tr)),groups=splitgroups[tr]))
   it,iv=tr[a],tr[b]
   if not binary or (len(np.unique(y[it]))==2 and len(np.unique(y[iv]))==2):break
  else:raise ValueError('No valid two-class inner split')
  best=-np.inf;choice=None
  for name,(blocks,ks) in candidates.items():
   A,B=transform_blocks(blocks,effective_dimensions(ks,len(it)),it,iv)
   for c in GRID:
    s=score(y[iv],_predict(A,y[it],B,c,binary),binary)
    if np.isfinite(s) and s>best:best=s;choice=(name,c)
  if choice is None:raise ValueError('No finite inner validation scores')
  name,c=choice;blocks,ks=candidates[name];A,B=transform_blocks(blocks,effective_dimensions(ks,len(tr)),tr,te)
  pred[te]=_predict(A,y[tr],B,c,binary)
  assert not set(groups[tr])&set(groups[te])
  if sites is not None:assert not set(sites[tr])&set(sites[te])
  audit.append(dict(fold=fold,selected=name,C=c,train=groups[tr].tolist(),test=groups[te].tolist(),inner_train=groups[it].tolist(),inner_validation=groups[iv].tolist()))
 if not np.isfinite(pred).all():raise ValueError('Incomplete predictions')
 return pred,audit

def metrics(y,p):
 if set(np.unique(y))=={0,1}:
  return dict(AUC=roc_auc_score(y,p),PR_AUC=average_precision_score(y,p),balanced_accuracy_at_0_5=balanced_accuracy_score(y,p>=.5),brier=brier_score_loss(y,p),prevalence=float(np.mean(y)))
 return dict(rho=score(y,p,False),RMSE=mean_squared_error(y,p)**.5,MAE=mean_absolute_error(y,p),R2=r2_score(y,p))

def arms_from(F,C=None):
 prefix=[] if C is None else [C];kp=[] if C is None else [C.shape[1]]
 fms=[k for k in F if k.startswith('L')]
 arms={k:{k:(prefix+[F[k]],kp+[100])} for k in ['FC','amplitude']}
 if C is not None:arms['covariates']={'covariates':([C],[C.shape[1]])}
 arms['FM_selected']={k:(prefix+[F[k]],kp+[100]) for k in fms}
 arms['FC+FM_selected']={k:(prefix+[F['FC'],F[k]],kp+[50,50]) for k in fms}
 for k in fms:arms[k]={k:(prefix+[F[k]],kp+[100])}
 return arms

def evaluate(ds,target,model,mode,seeds,comparison_ds=None,comparison_model=None):
 stem=f'{ds}_{target}_{model}_{mode}'
 if comparison_ds:stem+='__'+comparison_ds+'_'+comparison_model
 out=R/'evaluation'/f'{stem}.json'
 if out.exists() and load(out).get('status')=='complete':return
 m=pd.read_csv(D/(ds+'.csv'));keep=np.isfinite(pd.to_numeric(m[target],errors='coerce'))
 covcols=['age','SEX_binary_male','education_years'] if mode in ['covariates','acquisition','sites'] else []
 if mode=='acquisition':covcols+=['native_tr','native_frames']
 if covcols:keep&=np.isfinite(m[covcols].to_numpy(float)).all(1)
 if comparison_ds:
  ids=load(feature_paths(comparison_ds,comparison_model)/'complete.json')['subjects'];keep&=m.subject.isin(ids)
 m=m.loc[keep].reset_index(drop=True);y=m[target].to_numpy(float);g=m.subject.to_numpy();C=m[covcols].to_numpy(float) if covcols else None
 F=get_features(ds,model,g);arms=arms_from(F,C)
 if comparison_ds:
  G=get_features(comparison_ds,comparison_model,g)
  # Compare both models/variants on exactly the same individuals and fixed splits.
  arms={**{'reference/'+k:v for k,v in arms.items() if k in ['FC','FM_selected','FC+FM_selected']},
        **{'comparison/'+k:v for k,v in arms_from(G,C).items() if k in ['FC','FM_selected','FC+FM_selected']}}
 result=dict(status='running',evidence_status='preliminary_exploratory',dataset=ds,target=target,model=model,mode=mode,n=len(y),subjects=g.tolist(),y=y.tolist(),seeds=list(range(seeds)),covariates=covcols,arms={},comparisons=[],split_policy='five site-grouped folds' if mode=='sites' else 'five subject-grouped folds',dimension_budget='min(100, training_n-1), summed over imaging blocks; covariates additional')
 for name,candidates in arms.items():
  ps=[];audits=[]
  for seed in range(seeds):
   p,a=nested_predict(candidates,y,g,seed,sites=m.SITE.to_numpy() if mode=='sites' else None);ps.append(p);audits.append(a)
   print(stem,name,seed,metrics(y,p),flush=True)
  p=np.mean(ps,axis=0);result['arms'][name]=dict(metrics=metrics(y,p),per_seed_metrics=[metrics(y,p) for p in ps],predictions=p.tolist(),per_seed_predictions=[p.tolist() for p in ps],split_audit=audits);save(out,result)
 pairs=[('FC','FM_selected'),('FC','FC+FM_selected')]
 if C is not None:pairs += [('covariates','FC'),('covariates','FM_selected')]
 if comparison_ds:pairs=[('reference/'+k,'comparison/'+k) for k in ['FC','FM_selected','FC+FM_selected']]
 for a,b in pairs:
  delta,p,ci,detail=paired_bootstrap(y,result['arms'][a]['predictions'],result['arms'][b]['predictions'],n_boot=5000,groups=g,return_details=True)
  result['comparisons'].append(dict(reference=a,test=b,delta=delta,raw_one_sided_bootstrap_p=p,ci95=ci,**detail))
 for row,p in zip(result['comparisons'],holm([r['raw_one_sided_bootstrap_p'] for r in result['comparisons']])):row['holm_within_analysis']=p
 result['status']='complete';save(out,result)

def diagnostics():
 m=pd.read_csv(D/'nacc_default.csv');out={}
 for t in ['AD_DX','DX','DX_ord','MEM','EF']:
  z=m[m[t].notna()];d=dict(n=len(z),site_counts=z.SITE.value_counts().to_dict())
  if t in ['AD_DX','DX','DX_ord']:
   table=pd.crosstab(z.SITE,z[t]);d['site_by_target']=table.to_dict();chi,p,_,_=chi2_contingency(table);d['site_association_descriptive_chi2_p']=p
  d['target_vs_acquisition_spearman']={k:spearmanr(z[t],z[k]).statistic for k in ['native_tr','native_frames','age']};out[t]=d
 from model_loader import load_brainlm_forpretraining
 from paths import OFFICIAL_BRAINLM_13M
 base=load_brainlm_forpretraining(OFFICIAL_BRAINLM_13M)
 b=base.state_dict();updates=[]
 checkpoints=[ROOT/'runs/nacc_parity_v1/reconstruction/nacc_w200.pth',*[ROOT/f'runs/nacc_parity_v1/contrastive/nacc_{x}.pth' for x in ['session','scale','both']]]
 for p in checkpoints:
  net=load_brainlm_forpretraining(p)
  if hasattr(net,'merge_and_unload'):net=net.merge_and_unload()
  s=net.state_dict();keys=[k for k in b if k.startswith('vit.') and k in s and b[k].is_floating_point()]
  num=sum(float(((s[k]-b[k]).double()**2).sum()) for k in keys);den=sum(float((b[k].double()**2).sum()) for k in keys)
  updates.append(dict(checkpoint=str(p.relative_to(ROOT)),sha256=sha(p),encoder_relative_l2=(num/den)**.5,matched_encoder_tensors=len(keys)))
 save(R/'diagnostics.json',dict(status='complete',data=out,checkpoint_updates=updates,notes=['No motion variables were fabricated','Training loss is not independent downstream utility']))

def clinical_hrv(ds='hrv_default'):
 m=pd.read_csv(D/(ds+'.csv'));F=get_features(ds,'brainlm13');result=dict(status='running',dataset=ds,results=[])
 for t in ['flanker','LSWM','PCPS']:
  C=np.column_stack([m['pre_'+t],pd.get_dummies(m[['sex','age_category','arm']],drop_first=True,dtype=float)])
  keep=np.isfinite(m['post_'+t])&np.isfinite(C).all(1);y=m.loc[keep,'post_'+t].to_numpy();g=m.loc[keep,'subject'].to_numpy();c=C[keep];f={k:v[keep] for k,v in F.items()};arms=arms_from(f,c)
  # Small RCTs use a five-dimensional imaging budget; fusion totals five (2+3).
  for choices in arms.values():
   for name,(blocks,ks) in choices.items():
    if len(ks)==2:ks[-1]=5
    if len(ks)==3:ks[-2:]=[2,3]
  preds={};entries=[]
  for name in ['covariates','FC','FM_selected','FC+FM_selected']:
   ps=[nested_predict(arms[name],y,g,s)[0] for s in range(10)];preds[name]=np.mean(ps,axis=0);entries.append(dict(arm=name,metrics=metrics(y,preds[name]),seed_metrics=[metrics(y,p) for p in ps],predictions=preds[name].tolist()))
  comparisons=[]
  for a,b in [('covariates','FC'),('covariates','FM_selected'),('FC','FM_selected'),('FC','FC+FM_selected')]:
   delta,p,ci,detail=paired_bootstrap(y,preds[a],preds[b],n_boot=5000,groups=g,return_details=True);comparisons.append(dict(reference=a,test=b,delta=delta,raw_one_sided_bootstrap_p=p,ci95=ci,**detail))
  result['results'].append(dict(target=t,n=len(y),subjects=g.tolist(),y=y.tolist(),arms=entries,comparisons=comparisons))
  save(R/'clinical'/f'{ds}.json',result)
 rows=[r for t in result['results'] for r in t['comparisons']]
 for row,p in zip(rows,holm([r['raw_one_sided_bootstrap_p'] for r in rows])):row['holm_across_hrv_endpoints']=p
 result['status']='complete';result['scope']='Baseline prognosis of followup cognition; not treatment-effect prediction';save(R/'clinical'/f'{ds}.json',result)

def report():
 # Multiplicity across all extension comparisons, plus the previously explored fusion results.
 rows=[]
 for p in (R/'evaluation').glob('*.json'):
  d=load(p)
  if d['status']!='complete':raise ValueError('Incomplete analysis '+str(p))
  rows += [dict(file=str(p.relative_to(ROOT)),**r) for r in d['comparisons']]
 for p in (ROOT/'runs/nacc_parity_v1').glob('fusion_*.json'):
  for r in load(p)['increments']:rows.append(dict(file=str(p.relative_to(ROOT)),reference=r['control'],test=r['arm'],delta=r['delta'],raw_one_sided_bootstrap_p=r['p_one_sided'],ci95=r['ci95']))
 for p in (ROOT/'runs/nacc_uds_v1').glob('fusion_*.json'):
  for r in load(p)['increments']:rows.append(dict(file=str(p.relative_to(ROOT)),reference=r['control'],test=r['arm'],delta=r['delta'],raw_one_sided_bootstrap_p=r['p_one_sided'],ci95=r['ci95']))
 # Aggregate matched-budget training controls across the same three seeds.
 training=[]
 for cohort,target in [('nacc','MEM'),('nacc','EF'),('nacc','AD_DX'),('adni','MEM')]:
  bymodel={}
  for model in ['pretrained','random','gru']:
   cells=[load(R/'training_controls'/f'{cohort}_{target}_{model}_s{seed}'/'result.json') for seed in [42,43,44]]
   ref=cells[0]
   if any(c['status']!='complete' or c['subjects']!=ref['subjects'] or c['y']!=ref['y'] for c in cells):raise ValueError('Training controls are not aligned')
   pred=np.mean([c['predictions'] for c in cells],axis=0);bymodel[model]=(ref,pred)
   training.append(dict(cohort=cohort,target=target,model=model,metrics=metrics(np.array(ref['y']),pred),selected_epochs=[[f['best_epoch'] for f in c['folds']] for c in cells]))
  for reference,test in [('random','pretrained'),('gru','pretrained')]:
   ref,pa=bymodel[reference];other,pb=bymodel[test]
   if ref['subjects']!=other['subjects']:raise ValueError('Model subjects differ')
   delta,pv,ci,detail=paired_bootstrap(ref['y'],pa,pb,n_boot=5000,groups=ref['subjects'],return_details=True)
   rows.append(dict(file='training_controls/'+cohort+'_'+target,reference=reference,test=test,delta=delta,raw_one_sided_bootstrap_p=pv,ci95=ci,**detail))
 save(R/'training_controls_summary.json',dict(status='complete',rows=training))
 # Include all HRV endpoint tests and all new adaptation comparisons in the family.
 clinical_files=list((R/'clinical').glob('*.json'))
 for p in clinical_files:
  for result in load(p)['results']:
   rows += [dict(file=str(p.relative_to(ROOT)),target=result['target'],**r) for r in result['comparisons']]
 external=[]
 for p in (R/'external_transfer').glob('*/results.json'):
  d=load(p)
  if d['status']!='preliminary_complete':raise ValueError('Incomplete external evaluation')
  for r in d['adaptation_comparisons']:
   row={k:v for k,v in r.items() if k!='raw_bootstrap_p'};row['raw_one_sided_bootstrap_p']=r['raw_bootstrap_p'];row['reference']='frozen';row['test']=r['state'];row['file']=str(p.relative_to(ROOT));rows.append(row)
  for r in d['prognosis']:
   external.append(dict(file=str(p.relative_to(ROOT)),cohort=r['cohort'],target=r['target'],state=r['state'],n=r['n'],clinical=metrics(np.array(r['y']),np.array(r['clinical_prediction'])),imaging=metrics(np.array(r['y']),np.array(r['imaging_prediction']))))
 save(R/'external_utility_summary.json',dict(status='complete',rows=external))
 # Paired HRV preprocessing sensitivity on identical subjects and endpoints.
 ref=load(R/'clinical/hrv_default.json')
 for variant in ['hrv_t160','hrv_noGSR']:
  comp=load(R/'clinical'/f'{variant}.json')
  for aa,bb in zip(ref['results'],comp['results']):
   if aa['target']!=bb['target'] or aa['subjects']!=bb['subjects']:raise ValueError('HRV sensitivity samples differ')
   for aarm,barm in zip(aa['arms'],bb['arms']):
    if aarm['arm']!=barm['arm']:raise ValueError('HRV arms differ')
    delta,pv,ci,detail=paired_bootstrap(aa['y'],aarm['predictions'],barm['predictions'],n_boot=5000,groups=aa['subjects'],return_details=True)
    rows.append(dict(file='clinical/sensitivity',target=aa['target'],reference='hrv_default/'+aarm['arm'],test=variant+'/'+barm['arm'],delta=delta,raw_one_sided_bootstrap_p=pv,ci95=ci,**detail))
 for row,p in zip(rows,holm([r['raw_one_sided_bootstrap_p'] for r in rows])):row['holm_extension_and_previous_nacc_fusion']=p
 save(R/'multiplicity_summary.json',dict(status='complete',comparisons=rows,scope='All extension paired model, training-control, HRV, and adaptation comparisons plus previously inspected NACC fusion comparisons; conditional bootstrap tails are not permutation p-values',interpretation='Exploratory, not a preregistered confirmatory family'))
 pd.DataFrame(rows).to_csv(R/'comparisons.csv',index=False)
 save(R/'completion_audit.json',dict(status='complete',missing_data=load(R/'intake.json')['missing'],notes=['No equivalence claim: minimum practically meaningful effect was not specified','Do not infer treatment effect from prognosis','Spatial alignment remains unverified']))

from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','features','evaluate','diagnostics','clinical','report']);ap.add_argument('--dataset',default='nacc_default');ap.add_argument('--model',default='brainlm13');ap.add_argument('--target',default='MEM');ap.add_argument('--mode',default='plain',choices=['plain','covariates','acquisition','sites']);ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--comparison-dataset');ap.add_argument('--comparison-model');a=ap.parse_args();torch.set_num_threads(4)
 if a.stage=='prepare':prepare()
 elif a.stage=='features':features(a.dataset,a.model)
 elif a.stage=='evaluate':evaluate(a.dataset,a.target,a.model,a.mode,a.seeds,a.comparison_dataset,a.comparison_model)
 elif a.stage=='diagnostics':diagnostics()
 elif a.stage=='clinical':clinical_hrv(a.dataset)
 else:report()
if __name__=='__main__':main()
