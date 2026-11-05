#!/usr/bin/env python3
"""Target-to-source calibration of fixed ADNI/NACC encoders and cognitive axes.

Source fits never see target outcomes. Target calibration uses only baseline scans
from label-independent subject folds. Clinical probes repeat calibration inside
inner folds. No transductive use of outer-test scans for calibration.
"""
from pathlib import Path
import argparse,sys,json,pickle
import numpy as np,pandas as pd,torch
from sklearn.model_selection import KFold,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from benchmark_extension import save,load,metrics
from benchmark_evaluation import paired_bootstrap,score
from confirm_matched_axis import holm
from run_external_preliminary import CHECKPOINTS,sha
R=ROOT/'runs/transfer_calibration_v1';D=ROOT/'data/derived/transfer_calibration_v1'
SEEDS=[0,1,2];METHODS=['none','moments','coral'];GRID=[.01,.1,1.,10.,100.]

def subject_folds(groups,seed):
 u=np.unique(groups)
 for tr,te in KFold(5,shuffle=True,random_state=seed).split(u):
  yield np.flatnonzero(np.isin(groups,u[tr])),np.flatnonzero(np.isin(groups,u[te]))

def subject_mean(a,g):
 return np.stack([a[np.asarray(g)==s].mean(0) for s in np.unique(g)])

def covariance_power(c,power):
 v,q=np.linalg.eigh(c);v=np.maximum(v,max(float(v.max())*1e-8,1e-10));return (q*v**power)@q.T

class Align:
 def fit(self,source,target,mode):
  self.mode=mode;self.ms=source.mean(0);self.mt=target.mean(0)
  if mode=='moments':self.gain=np.maximum(source.std(0),1e-6)/np.maximum(target.std(0),1e-6)
  if mode=='coral':
   cs=LedoitWolf().fit(source).covariance_;ct=LedoitWolf().fit(target).covariance_
   self.matrix=covariance_power(ct,-.5)@covariance_power(cs,.5)
  return self
 def transform(self,x):
  if self.mode=='none':return x.copy()
  z=x-self.mt
  return z*self.gain+self.ms if self.mode=='moments' else z@self.matrix+self.ms

def observed_sd(x):
 # Synthetic padding is exactly zero across every ROI; do not include it in amplitude statistics.
 mask=(x.abs().sum(1)>0).to(x.dtype);n=mask.sum(1).clamp_min(2)
 mean=(x*mask[:,None,:]).sum(2)/n[:,None]
 var=(((x-mean[:,:,None])**2)*mask[:,None,:]).sum(2)/n[:,None]
 return var.clamp_min(0).sqrt()

def roi_gains(source_x,target_calibration_x):
 # Per-ROI positive scaling preserves FC. Square-root shrinkage is fixed in advance.
 s=torch.median(observed_sd(source_x),dim=0).values
 t=torch.median(observed_sd(target_calibration_x),dim=0).values
 return torch.sqrt(s.clamp_min(1e-8)/t.clamp_min(1e-8))

def prepare():
 from cohorts import load_cohort,center_rois
 from mci_axis_transfer import apply_gsr,apply_gain_match
 from rct_probe import upsample_short
 from rct_positive_control import CONTROLS
 D.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 for domain in ['MEM','EF']:
  for source in ['adni','nacc']:
   if source=='adni':d=ROOT/'data/derived/adni424_t200';tag=f'raw_ADNI_{domain}_all_t200_off'
   else:d=ROOT/'data/derived/nacc_parity_v1';tag='nacc_'+domain
   x=torch.load(d/f'ADNI_x_{tag}.pt',weights_only=True);m=pd.read_csv(d/f'ADNI_manifest_{tag}.csv');y=torch.load(d/f'ADNI_y_{tag}.pt',weights_only=True).numpy()
   name=source+'_'+domain;torch.save(x,D/(name+'.pt'));save(D/(name+'.json'),dict(subjects=m.PTID.astype(str).tolist(),endpoints={domain.lower():dict(level=y.tolist(),follow=None,clinical=None)},source_labels=y.tolist(),input_manifest_sha256=sha(d/f'ADNI_manifest_{tag}.csv')))
 for co_name in ['act','cogte','breathe','psopt']:
  co=apply_gain_match(apply_gsr(center_rois(load_cohort(co_name,release='amp',require_labels=False)),co_name),co_name);co,_=upsample_short(co,200,20,fill='pad');ctrl=CONTROLS[co_name](co.subjects)
  core=np.column_stack([co.baseline_levels('mem'),co.baseline_levels('ef'),ctrl['age'][0],ctrl['sex'][0]])
  # Use observed baseline arm labels; missing labels exclude prognosis, not baseline transfer.
  t=co.table;cols=[c for c in ['group_assignment_T1','arm_name'] if c in t]
  if cols:
   raw=t[cols[0]];arm=pd.get_dummies(raw.astype('string'),drop_first=True,dtype=float).to_numpy();arm[raw.isna().to_numpy()]=np.nan
  elif all(c in t for c in ['GROUP_1.0','GROUP_2.0']):arm=t[['GROUP_1.0','GROUP_2.0']].to_numpy(float)
  else:raise ValueError('Missing intervention arm '+co_name)
  C=np.column_stack([core,arm]);eps={}
  for domain in ['mem','ef']:
   level=np.asarray(co.baseline_levels(domain),float);follow=np.asarray(co.followup_levels(domain),float);level[~co.analysis_mask()]=np.nan;follow[~co.analysis_mask()]=np.nan
   eps[domain]=dict(level=level.tolist(),follow=follow.tolist(),clinical=C.tolist())
  torch.save(co.bold[...,0][:,:,:200].contiguous(),D/(co_name+'.pt'));save(D/(co_name+'.json'),dict(subjects=co.subjects.tolist(),endpoints=eps,clinical_columns=['baseline_MEM','baseline_EF','age','sex','intervention_arm_dummies']))
 d=ROOT/'data/derived/external_preliminary_v1';m=pd.read_csv(d/'hrv_pre.csv');cat=pd.get_dummies(m[['age_category','sex','arm']],drop_first=True,dtype=float).to_numpy();eps={}
 for t in ['flanker','LSWM','PCPS']:eps[t]=dict(level=m['pre_'+t].tolist(),follow=m['post_'+t].tolist(),clinical=np.column_stack([m['pre_'+t],cat]).tolist())
 torch.save(torch.load(d/'hrv_pre.pt',weights_only=True),D/'hrv.pt');save(D/'hrv.json',dict(subjects=m.subject.tolist(),endpoints=eps))
 save(R/'protocol.json',dict(status='preliminary_exploratory',source_domains=['ADNI','NACC'],source_tasks=['MEM','EF'],source_reference='Existing source-task pool, including its observed disease mixture; no claim to isolate scanner from biological composition',methods=['none','ROI-amplitude input gain (square-root shrinkage)','embedding mean/SD matching','Ledoit-Wolf CORAL-style target-to-source covariance alignment'],source_projection_components=20,source_axis_alpha=10,target_calibration='Label-independent subject folds; baseline scans only; inner clinical validation recalibrated using inner-training scans only',target_labels='Used only for target clinical head fitting and final metrics, never domain alignment',source_model_states='frozen and source-adapted; NACC adaptation seeds 42,43,44 all retained',spatial_alignment='Unverified; preliminary',roi_gain_caveat='May remove biological cohort amplitude differences; interpreted as sensitivity, not ground-truth correction',references=['https://ojs.aaai.org/index.php/AAAI/article/view/10306']))

def checkpoint(source,domain,state):
 if state=='frozen':return CHECKPOINTS['frozen']
 if source=='adni':return CHECKPOINTS[domain]
 return ROOT/f'runs/nacc_parity_v1/posttraining/{domain}_external_seed{state.removeprefix("s")}/encoder.pt'

def feature_cache(source,domain,state,cohort,family):
 seed=42 if source=='adni' or state=='frozen' else int(state[1:]);base=ROOT/f'runs/nacc_parity_v1/external_transfer/{domain}_seed{seed}'
 dataset='NACC_source' if cohort=='nacc' else cohort
 st='FC' if family=='FC' else ('frozen' if state=='frozen' else ('ADNI_adapted' if source=='adni' else 'NACC_adapted'))
 if cohort=='adni' and st=='ADNI_adapted':return ROOT/f'runs/external_preliminary_v1/features/adni_{domain}_{domain}_emb_fc.npy'
 return base/f'{dataset}_{st}.npy'

def refdir(source,domain,state):return R/'references'/f'{source}_{domain}_{state}'
def target_names(source):return ['nacc' if source=='adni' else 'adni','act','cogte','breathe','psopt','hrv']
def dataset_name(cohort,domain):return cohort+'_'+domain if cohort in ['adni','nacc'] else cohort

def verify_fc_rows(x,cached):
 """Check row correspondence using 32 deterministic FC edges in every scan.

 Historical FM caches do not contain an ID manifest. This checks the companion
 FC cache and common producer order; it is not a historical FM-ID certificate.
 """
 ii,jj=np.triu_indices(x.shape[1],1);edges=np.linspace(0,len(ii)-1,min(32,len(ii)),dtype=int)
 a=x[:,ii[edges],:].numpy().astype(float);b=x[:,jj[edges],:].numpy().astype(float)
 a-=a.mean(2,keepdims=True);b-=b.mean(2,keepdims=True)
 corr=(a*b).sum(2)/np.maximum(np.sqrt((a*a).sum(2)*(b*b).sum(2)),1e-15)
 expected=np.arctanh(np.clip(corr,-.999999,.999999))
 if len(cached)!=len(x) or not np.allclose(expected,cached[:,edges],atol=2e-6,rtol=2e-5):
  raise ValueError('Companion FC cache does not match current tensor row order')
 return dict(rows=len(x),edges_per_row=len(edges),max_absolute_error=float(np.max(np.abs(expected-cached[:,edges]))))

def reference(source,domain,state):
 out=refdir(source,domain,state);out.mkdir(parents=True,exist_ok=True)
 m=load(D/f'{source}_{domain}.json');y=np.array(m['source_labels']);expected=len(y);refs={};audit={};hashes={}
 for co in [source,*target_names(source)]:
  x=torch.load(D/(dataset_name(co,domain)+'.pt'),weights_only=True)
  audit[co]=verify_fc_rows(x,np.load(feature_cache(source,domain,state,co,'FC'),mmap_mode='r'))
  hashes[co]={f:sha(feature_cache(source,domain,state,co,f)) for f in ['FC','FM']}
 del x
 for family in ['FC','FM']:
  file=feature_cache(source,domain,state,source,family);F=np.load(file)
  if len(F)!=expected or not np.isfinite(F).all():raise ValueError('Source cache misalignment')
  sc=StandardScaler().fit(F);z=sc.transform(F);pc=PCA(n_components=20,random_state=0).fit(z);z=pc.transform(z);axis=Ridge(alpha=10).fit(z,y)
  refs[family]=dict(sc=sc,pc=pc,axis=axis,source=z,source_groups=np.array(m['subjects']))
  np.save(out/(family+'_source.npy'),z)
  for co in target_names(source):
   f=feature_cache(source,domain,state,co,family);F=np.load(f,mmap_mode='r');tm=load(D/(dataset_name(co,domain)+'.json'))
   if len(F)!=len(tm['subjects']):raise ValueError('Target cache misalignment '+co)
   np.save(out/f'{co}_{family}.npy',pc.transform(sc.transform(F)))
 with (out/'reference.pkl').open('wb') as f:pickle.dump(refs,f)
 save(out/'complete.json',dict(status='complete',source=source,domain=domain,state=state,checkpoint=str(checkpoint(source,domain,state)),checkpoint_sha256=sha(checkpoint(source,domain,state)),source_manifest_sha256=sha(D/f'{source}_{domain}.json'),feature_cache_sha256=hashes,companion_fc_row_audit=audit,cache_identity_limit='Historical FM files lack ID manifests; companion FC rows and shared producer ordering checked'))

def clinical_predict(C,y,Z,source_Z,g,method,seed,input_inner=None,input_outer=None):
 """Every calibration, standardizer and ridge tuning step is training-only."""
 pred=np.full(len(y),np.nan);audit=[]
 for fold,(tr,te) in enumerate(subject_folds(g,seed)):
  it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed*10+fold).split(C[tr],groups=g[tr]));it,iv=tr[it],tr[iv]
  def design(cal,targets,which):
   if Z is None:return C[cal],C[targets]
   if input_inner is not None:
    zz=(input_inner if which=='inner' else input_outer)[fold]
   else:zz=Align().fit(source_Z,subject_mean(Z[cal],g[cal]),method).transform(Z)
   # Source projection is fixed. Target PCA reduces imaging to five dimensions in training only.
   sc=StandardScaler().fit(zz[cal]);pc=PCA(n_components=min(5,len(cal)-1),random_state=0).fit(sc.transform(zz[cal]));f=pc.transform(sc.transform(zz))
   return np.column_stack([C[cal],f[cal]]),np.column_stack([C[targets],f[targets]])
  A,B=design(it,iv,'inner');sc=StandardScaler().fit(A);A,B=sc.transform(A),sc.transform(B);best=-np.inf;alpha=GRID[0]
  for a in GRID:
   ss=score(y[iv],Ridge(alpha=a).fit(A,y[it]).predict(B),False)
   if np.isfinite(ss) and ss>best:alpha=a;best=ss
  A,B=design(tr,te,'outer');sc=StandardScaler().fit(A);pred[te]=Ridge(alpha=alpha).fit(sc.transform(A),y[tr]).predict(sc.transform(B))
  audit.append(dict(fold=fold,calibration=g[tr].tolist(),test=g[te].tolist(),inner_calibration=g[it].tolist(),inner_validation=g[iv].tolist()))
 return pred,audit

def evaluate(source,domain,state,input_mode=False):
 out=refdir(source,domain,state)
 with (out/'reference.pkl').open('rb') as f:refs=pickle.load(f)
 net=None
 if input_mode:
  from model_loader import load_brainlm_forpretraining
  from probe_readouts import brainlm_readout_batched
  from datasets import load_xyz
  from paths import XYZ_PATH
  gpu_acquire()
  net=load_brainlm_forpretraining(checkpoint(source,domain,state)).cuda().eval();xyz=load_xyz(XYZ_PATH)
  gpu_release()
  source_x=torch.load(D/f'{source}_{domain}.pt',weights_only=True)
 targetdir=R/('input_results' if input_mode else 'embedding_results')/f'{source}_{domain}_{state}';targetdir.mkdir(parents=True,exist_ok=True)
 for co in target_names(source):
  dest=targetdir/(co+'.json')
  if dest.exists() and load(dest).get('status')=='complete':continue
  meta=load(D/(dataset_name(co,domain)+'.json'));groups=np.array(meta['subjects']);endpoints=meta['endpoints'];rows=[];comparisons=[]
  if input_mode:tx=torch.load(D/(dataset_name(co,domain)+'.pt'),weights_only=True)
  for family in (['FM'] if input_mode else ['FC','FM']):
   ref=refs[family];zs=subject_mean(ref['source'],ref['source_groups']);zt=np.load(out/f'{co}_{family}.npy');source_axis=ref['axis']
   for method in (['input_gain'] if input_mode else METHODS):
    # Source-axis transfer: calibration folds are built without target labels, across all baseline scans.
    seed_axes=[];fold_audit=[]
    for seed in SEEDS:
     axis_pred=np.full(len(groups),np.nan)
     for k,(tr,te) in enumerate(subject_folds(groups,seed)):
      if input_mode:
       gain=roi_gains(source_x,tx[tr]);raw=gpu_call(brainlm_readout_batched,net,tx[te]*gain[None,:,None],xyz,3,'emb_fc',batch=4);z=ref['pc'].transform(ref['sc'].transform(raw));axis_pred[te]=source_axis.predict(z)
      else:axis_pred[te]=source_axis.predict(Align().fit(zs,subject_mean(zt[tr],groups[tr]),method).transform(zt[te]))
      fold_scores={}
      for endpoint,e in endpoints.items():
       yy=np.array(e['level'],float)[te];ok=np.isfinite(yy)
       fold_scores[endpoint]=score(yy[ok],axis_pred[te][ok],False) if ok.sum()>3 else None
      fold_audit.append(dict(seed=seed,fold=k,calibration=groups[tr].tolist(),test=groups[te].tolist(),heldout_fold_rho=fold_scores))
     seed_axes.append(axis_pred)
    axis_pred=np.mean(seed_axes,axis=0)
    selected=endpoints if co=='hrv' else {domain.lower():endpoints[domain.lower()]}
    for name,e in selected.items():
     y=np.array(e['level'],float);ok=np.isfinite(y)
     rows.append(dict(kind='fixed_source_axis',target=name,family=family,method=method,n=int(ok.sum()),subjects=groups[ok].tolist(),y=y[ok].tolist(),prediction=axis_pred[ok].tolist(),rho=score(y[ok],axis_pred[ok],False),calibration_audit=fold_audit))
     if e['follow'] is None:continue
     y=np.array(e['follow'],float);C=np.array(e['clinical'],float);keep=np.isfinite(y)&np.isfinite(C).all(1);g=groups[keep];y=y[keep];C=C[keep];Z=zt[keep]
     if len(y)<20:raise ValueError('Too few complete prognosis subjects '+co+' '+name)
     preds=[];baselines=[];audits=[]
     for seed in SEEDS:
      inner=None;outer=None
      if input_mode:
       xx=tx[torch.tensor(keep)];inner=[];outer=[]
       for k,(tr,te) in enumerate(subject_folds(g,seed)):
        ii,_=next(GroupShuffleSplit(1,test_size=.2,random_state=seed*10+k).split(C[tr],groups=g[tr]));it=tr[ii]
        for cal,container in [(it,inner),(tr,outer)]:
         gain=roi_gains(source_x,xx[cal]);raw=gpu_call(brainlm_readout_batched,net,xx*gain[None,:,None],xyz,3,'emb_fc',batch=4);container.append(ref['pc'].transform(ref['sc'].transform(raw)))
      pred,audit=clinical_predict(C,y,Z,zs,g,method,seed,inner,outer);preds.append(pred);audits.append(audit);baselines.append(clinical_predict(C,y,None,zs,g,'none',seed)[0])
     pred=np.mean(preds,axis=0);base=np.mean(baselines,axis=0);delta,p,ci,details=paired_bootstrap(y,base,pred,n_boot=2000,groups=g,return_details=True)
     rows.append(dict(kind='clinical_increment',target=name,family=family,method=method,n=len(y),subjects=g.tolist(),y=y.tolist(),prediction=pred.tolist(),clinical_prediction=base.tolist(),clinical_metrics=metrics(y,base),imaging_metrics=metrics(y,pred),delta=delta,raw_bootstrap_p=p,ci95=ci,calibration_audit=audits,**details))
    save(dest,dict(status='running',source=source,domain=domain,state=state,target_cohort=co,rows=rows))
  save(dest,dict(status='complete',evidence_status='preliminary_exploratory',source=source,domain=domain,state=state,target_cohort=co,rows=rows,notes=['Source cognitive axis and projection fixed before target evaluation','Calibration uses only target baseline training subjects','Clinical increment evaluates a separately trained target clinical probe','PPG and cognition tasks are not pooled']))
  print('COMPLETE',source,domain,state,co,'input' if input_mode else 'embedding',flush=True)

def report():
 rows=[];adaptation=[];interactions=[];panels={}
 # Require the complete registered panel, even if called directly outside the queue.
 for source in ['adni','nacc']:
  for domain in ['MEM','EF']:
   for state in (['frozen','adapted'] if source=='adni' else ['frozen','s42','s43','s44']):
    for co in target_names(source):
     tag=f'{source}_{domain}_{state}';d=load(R/'embedding_results'/tag/(co+'.json'));inp=load(R/'input_results'/tag/(co+'.json'))
     if d['status']!='complete' or inp['status']!='complete':raise ValueError('Incomplete calibration '+tag)
     panels[source,domain,state,co]=d['rows']+inp['rows']
 def key(r):return (r['kind'],r['target'],r['family'],r['method'])
 def contrast(a,b):
  if a['subjects']!=b['subjects'] or a['y']!=b['y']:raise ValueError('Nonpaired contrast')
  delta,pv,ci,details=paired_bootstrap(b['y'],a['prediction'],b['prediction'],n_boot=2000,groups=b['subjects'],return_details=True)
  return dict(delta=delta,raw_bootstrap_p=pv,ci95=ci,**details)
 for (source,domain,state,co),cells in panels.items():
  index={key(r):r for r in cells};frozen={key(r):r for r in panels[source,domain,'frozen',co]}
  for r in cells:
   ident=dict(source=source,domain=domain,state=state,cohort=co,kind=r['kind'],target=r['target'],family=r['family'],method=r['method'])
   k=key(r);base=index[k[:-1]+('none',)]
   if r['method']!='none':rows.append(dict(**ident,**contrast(base,r)))
   if state!='frozen' and r['family']=='FM':
    adaptation.append(dict(**ident,**contrast(frozen[k],r)))
    if r['method']!='none':
     fbase=frozen[k[:-1]+('none',)]
     def rho(q):return score(q['y'],q['prediction'],False)
     interactions.append(dict(**ident,delta_calibration_adapted=rho(r)-rho(base),delta_calibration_frozen=rho(frozen[k])-rho(fbase),difference_of_differences=(rho(r)-rho(base))-(rho(frozen[k])-rho(fbase)),interpretation='Descriptive interaction, no interaction p-value or CI'))
 for family in [rows,adaptation]:
  for r,pv in zip(family,holm([r['raw_bootstrap_p'] for r in family])):r['holm_within_contrast_family']=pv
 save(R/'summary.json',dict(status='complete',comparisons=rows,adapted_vs_frozen=adaptation,descriptive_interactions=interactions,interpretation='Exploratory target-to-source calibration; no causal effect or pure scanner-correction claim; bootstrap conditional on fitted OOF'))
 pd.DataFrame(rows).to_csv(R/'comparisons.csv',index=False);pd.DataFrame(adaptation).to_csv(R/'adapted_vs_frozen.csv',index=False);pd.DataFrame(interactions).to_csv(R/'descriptive_interactions.csv',index=False)

from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','reference','embedding','input','report']);ap.add_argument('--source',choices=['adni','nacc']);ap.add_argument('--domain',choices=['MEM','EF']);ap.add_argument('--state',default='frozen');a=ap.parse_args();torch.set_num_threads(4)
 if a.stage=='prepare':prepare()
 elif a.stage=='reference':reference(a.source,a.domain,a.state)
 elif a.stage in ['embedding','input']:evaluate(a.source,a.domain,a.state,a.stage=='input')
 else:report()
if __name__=='__main__':main()
