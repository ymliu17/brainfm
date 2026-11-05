"""Source-coordinate moment/CORAL sensitivity on immutable current FM embeddings."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold,GroupShuffleSplit
import paper_completion as pc
import sample_alignment as sa
import sf100_benchmark as sf
from fm_matrix.data import ROOT,save,sha
from fm_readout_v4.data import metadata
from benchmark_evaluation import grouped_splits,score
from paper_paired_stats import paired,holm

RUN=ROOT/'runs/embedding_harmonization_v2'
CANON=sa.PAPER/'canonical'
METHODS=('none','moments','coral')
GRID=(.01,.1,1.,10.,100.)

def read(p):return json.loads(Path(p).read_text())
def key(state,source):return pc.tag(*state)+'__'+source
def jobs():
 for state in pc.states():
  for source in (('adni','nacc') if state[2]=='frozen' else (state[0],)):
   yield tuple(state),source
def subject_mean(z,g):
 g=np.asarray(g).astype(str)
 return np.stack([z[g==s].mean(0) for s in np.unique(g)])
def power(c,p):
 e,v=np.linalg.eigh(c);e=np.maximum(e,max(float(e.max())*1e-8,1e-10))
 return (v*e**p)@v.T

class Alignment:
 def __init__(self,source):
  source=np.asarray(source,float)
  self.mean=source.mean(0);self.sd=np.maximum(source.std(0),1e-6)
  self.root=power(LedoitWolf().fit(source).covariance_,.5)
 def fit(self,target,method):
  self.method=method;target=np.asarray(target,float);self.mt=target.mean(0)
  if method=='moments':self.gain=self.sd/np.maximum(target.std(0),1e-6)
  elif method=='coral':self.matrix=power(LedoitWolf().fit(target).covariance_,-.5)@self.root
  elif method!='none':raise ValueError(method)
  return self
 def transform(self,z):
  if self.method=='none':return np.asarray(z,float).copy()
  z=np.asarray(z,float)-self.mt
  return (z*self.gain if self.method=='moments' else z@self.matrix)+self.mean

def audit():
 RUN.mkdir(parents=True,exist_ok=True)
 known=read(ROOT/'runs/paper_task_expansion_v1/input_audit.json')['files_sha256']
 files={}
 for state,source in jobs():
  folder=pc.feature_dir(state)
  for co in pc.COHORTS:
   if co=='hrv_post':continue
   p=folder/(co+'.npy');rel=str(p.relative_to(ROOT))
   if rel not in files:
    assert sha(p)==known[rel],rel
    files[rel]=known[rel]
   a=np.load(p,mmap_mode='r');assert len(a)==len(metadata(co)) and np.isfinite(a).all()
  p=CANON/(pc.tag(*state)+'.json');files[str(p.relative_to(ROOT))]=sha(p)
 for co in pc.COHORTS:
  for p in [sf.label_file(co),sf.RUN/'matches'/(co+'.json')]:files[str(p.relative_to(ROOT))]=sha(p)
 files[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 protocol=dict(status='frozen_before_new_evaluation',files_sha256=files,jobs=list(jobs()),encoder_training=False,gpu_required=False,automatic_shutdown=False,
  states='Four models, frozen plus ADNI/NACC MEM/EF/SSL adapters, seeds 42/43/44; current native scan-level readouts.',
  samples='Same endpoint-specific scans as current source-axis and clinical panels. No new QC. ADNI repeat scans remain participant-grouped.',
  projection='Source outcome-available sample fits float64 StandardScaler and randomized PCA<=100 (random_state=0); frozen source ridge alpha10. All alignment arms, including none, share this single fit. This is a refitted numerical baseline, not the archived float32 predictions; primary paper scores are not overwritten. Alignment is in fixed source-coordinate space, not raw ROI space.',
  alignment='Source and target calibration moments/covariances use participant-mean embeddings. Mean/SD matching uses population SD with floor1e-6. CORAL uses Ledoit-Wolf covariance and eigenvalue floor max(eigenvalue_max*1e-8,1e-10). No outcomes enter alignment parameters.',
  source_axis='Three label-independent target subject KFold splits (5 folds; seeds0/1/2). Fit target alignment using other folds from common eligible baseline scans, including unlabeled participants. Predict held-out scans, then score the exact published endpoint sample. Source labels fit the axis only; target labels are evaluation-only.',
  clinical='Five participant-grouped outer folds with 20% grouped inner validation, seeds0/1/2. Target alignment and target StandardScaler/PCA<=5 refitted inside inner/outer training partitions. Clinical+imaging Ridge alpha in [.01,.1,1,10,100] selected by inner Spearman. HRV MEM=Week5 recall from pre MRI, clinical baseline EF/category/sex/arm. These source-projected local probes are a separate sensitivity, not replacements for S2/S5 scores.',
  moment_invariance='Target-local feature standardization algebraically cancels a positive coordinate-wise moment transform. Clinical moments predictions reuse the identical none predictions after a numerical standardized-design invariance check. Their zero differences are a pipeline property, not an empirical absence-of-domain-shift result.',
  aggregation='Split-seed predictions averaged; adapter-seed predictions then averaged within objective/source/model. Never average feature coordinates across adapters.',
  inference='5000 paired participant-cluster bootstraps;20000 two-sided conditional prediction swaps; no model refit in inference. Holm per setting x endpoint x state-objective x alignment method across all models/sources/targets. This extension is exploratory, not prospectively registered. Exact identical predictions have analytical delta=0,CI=[0,0],P=1.',
  scope='External source-axis cognitive transfer and baseline-imaging follow-up/recall prediction only. No new diagnosis, demographic, identification or physiological harmonization endpoints.')
 path=RUN/'protocol.json'
 if path.exists():assert read(path)==json.loads(json.dumps(protocol)), 'Inputs/protocol changed; create a new version'
 else:save(path,protocol)
 return protocol

def load_features(state,co):return np.load(pc.feature_dir(state)/(co+'.npy')).astype(np.float64)

def design(c,z,train,test,reference,method):
 a=Alignment(reference).fit(subject_mean(z[train],design.groups[train]),method)
 tr,te=a.transform(z[train]),a.transform(z[test])
 sc=StandardScaler().fit(tr);tr,te=sc.transform(tr),sc.transform(te)
 p=PCA(n_components=min(5,len(tr)-1,tr.shape[1]),svd_solver='full').fit(tr)
 aa=np.column_stack([c[train],p.transform(tr)]);bb=np.column_stack([c[test],p.transform(te)])
 scaler=StandardScaler().fit(aa)
 return scaler.transform(aa),scaler.transform(bb)

def fit_ridge_tuned(y,it,iv,tr,te,inner,outer):
 aa,bb=inner;best=-np.inf;chosen=GRID[0]
 for alpha in GRID:
  v=score(y[iv],Ridge(alpha=alpha).fit(aa,y[it]).predict(bb),False)
  if np.isfinite(v) and v>best:chosen,best=alpha,v
 aa,bb=outer
 return Ridge(alpha=chosen).fit(aa,y[tr]).predict(bb),chosen

def clinical_predictions(c,y,z,g,reference):
 pred={m:[] for m in (*METHODS,'clinical')};audits=[]
 design.groups=np.asarray(g)
 for seed in (0,1,2):
  pp={m:np.full(len(y),np.nan) for m in (*METHODS,'clinical')}
  for fold,(tr,te) in enumerate(grouped_splits(y,g,seed)):
   a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=np.asarray(g)[tr]));it,iv=tr[a],tr[b]
   assert not set(g[tr])&set(g[te]) and not set(g[it])&set(g[iv])
   selected={}
   for method in ('none','coral'):
    inner=design(c,z,it,iv,reference,method);outer=design(c,z,tr,te,reference,method)
    pp[method][te],selected[method]=fit_ridge_tuned(y,it,iv,tr,te,inner,outer)
   # Check the invariance before applying the algebraic identity, not fitted outcomes.
   align=Alignment(reference).fit(subject_mean(z[tr],g[tr]),'moments')
   raw=StandardScaler().fit(z[tr]).transform(z[te])
   shifted=StandardScaler().fit(align.transform(z[tr])).transform(align.transform(z[te]))
   np.testing.assert_allclose(raw,shifted,rtol=1e-7,atol=1e-7)
   pp['moments'][te]=pp['none'][te]
   def clinical_only(a,b):
    sc=StandardScaler().fit(c[a]);return sc.transform(c[a]),sc.transform(c[b])
   pp['clinical'][te],selected['clinical']=fit_ridge_tuned(y,it,iv,tr,te,clinical_only(it,iv),clinical_only(tr,te))
   audits.append(dict(seed=seed,fold=fold,train=sorted(set(g[tr])),test=sorted(set(g[te])),inner_train=sorted(set(g[it])),inner_validation=sorted(set(g[iv])),selected_alpha=selected))
  for method in pred:
   assert np.isfinite(pp[method]).all();pred[method].append(pp[method])
 return {m:np.mean(v,axis=0) for m,v in pred.items()},audits

def execute(state,source):
 out=RUN/'evaluation'/(key(state,source)+'.json')
 if out.exists():return key(state,source)
 old=read(CANON/(pc.tag(*state)+'.json'));m=metadata(source);x=load_features(state,source);rows=[]
 for domain in ('MEM','EF'):
  keep=pc.primary_mask(m,source)&m[domain].notna().to_numpy();sc=StandardScaler().fit(x[keep]);a=sc.transform(x[keep])
  proj=PCA(n_components=min(100,len(a)-1,a.shape[1]),svd_solver='randomized',random_state=0).fit(a)
  zs=proj.transform(a);head=Ridge(alpha=10).fit(zs,m.loc[keep,domain]);ref=subject_mean(zs,m.loc[keep,'subject'].to_numpy())
  for co in pc.COHORTS:
   if co in (source,'hrv_post'):continue
   tm=metadata(co);f=load_features(state,co);zt=proj.transform(sc.transform(f)).astype(float)
   endpoint='MEM_delayed_recall' if co=='hrv_pre' and domain=='MEM' else domain
   base=sa.common_mask(tm,co);ix=np.flatnonzero(base&tm[endpoint].notna().to_numpy());pool=np.flatnonzero(base)
   g=tm.subject.astype(str).to_numpy();u=np.unique(g[pool]);pred={method:[] for method in METHODS};audits=[]
   published=next(r for r in old['axes'] if r['source']==source and r['cohort']==co and r['task']==domain+'_axis_to_'+endpoint)
   assert published['scan_ids']==tm.iloc[ix].scan.tolist() and published['subjects']==g[ix].tolist()
   np.testing.assert_array_equal(published['y'],tm.iloc[ix][endpoint].to_numpy(float))
   fresh=head.predict(zt[ix]);assert np.isfinite(fresh).all()
   for seed in (0,1,2):
    pp={method:np.full(len(tm),np.nan) for method in METHODS}
    for fold,(train,test) in enumerate(KFold(5,shuffle=True,random_state=seed).split(u)):
     tr=pool[np.isin(g[pool],u[train])];te=pool[np.isin(g[pool],u[test])]
     assert not set(g[tr])&set(g[te])
     for method in METHODS:
      alg=Alignment(ref).fit(subject_mean(zt[tr],g[tr]),method)
      pp[method][te]=head.predict(alg.transform(zt[te]))
     audits.append(dict(seed=seed,fold=fold,calibration=sorted(set(g[tr])),test=sorted(set(g[te]))))
    for method in METHODS:
     assert np.isfinite(pp[method][ix]).all();pred[method].append(pp[method][ix])
   predictions={method:np.mean(v,axis=0) for method,v in pred.items()}
   np.testing.assert_allclose(predictions['none'],fresh,rtol=1e-12,atol=1e-12)
   predictions['none']=fresh
   rows.append(dict(setting='source_axis',cohort=co,domain=domain,task=domain+'_axis_to_'+endpoint,n=len(ix),scan_ids=tm.iloc[ix].scan.tolist(),subjects=g[ix].tolist(),y=published['y'],predictions=predictions,fold_audit=audits,reference_prediction_max_abs_error=float(np.max(np.abs(fresh-np.array(published['prediction']))))))
   if co not in pc.TRIALS:continue
   clinical=pc.clinical_features(tm,co)
   col='MEM_delayed_recall' if co=='hrv_pre' and domain=='MEM' else 'follow_'+domain
   task='MEM_delayed_recall_baseline_prediction' if col=='MEM_delayed_recall' else domain+'_followup'
   y=tm[col].to_numpy(float);idx=np.flatnonzero(base&np.isfinite(y)&np.isfinite(clinical).all(1))
   published_local=next(r for r in old['rows'] if r['cohort']==co and r['task']==task and r['method']=='clinical+FM')
   assert published_local['scan_ids']==tm.iloc[idx].scan.tolist()
   np.testing.assert_array_equal(published_local['y'],y[idx])
   pred,audits=clinical_predictions(clinical[idx],y[idx],zt[idx],g[idx],ref)
   rows.append(dict(setting='local_followup',cohort=co,domain=domain,task=task,n=len(idx),scan_ids=tm.iloc[idx].scan.tolist(),subjects=g[idx].tolist(),y=y[idx],predictions=pred,fold_audit=audits))
 save(out,dict(status='complete',state=state,source=source,rows=rows,protocol_sha256=sha(RUN/'protocol.json')))
 return key(state,source)

def bundles():
 for model in pc.MODELS:
  for source in ('adni','nacc'):
   for obj,target in [('frozen','none'),('supervised','MEM'),('supervised','EF'),('native_ssl','none')]:yield model,source,obj,target

def infer(model,source,obj,target):
 label='_'.join((model,source,obj,target));out=RUN/'inference'/(label+'.json')
 if out.exists():return label
 states=[('none',model,'frozen','none',0)] if obj=='frozen' else [(source,model,obj,target,s) for s in pc.SEEDS]
 ds=[read(RUN/'evaluation'/(key(s,source)+'.json')) for s in states];rows=[];scores=[]
 for cells in zip(*(d['rows'] for d in ds)):
  r=cells[0]
  for b in cells[1:]:
   for k in ('setting','cohort','domain','task','subjects','scan_ids','y'):assert b[k]==r[k],k
  pp={method:np.mean([b['predictions'][method] for b in cells],axis=0) for method in r['predictions']}
  ident=dict(model=model,source=source,objective=obj,adaptation_target=target,**{k:r[k] for k in ('setting','cohort','domain','task','n')})
  for method,pred in pp.items():scores.append(dict(**ident,method=method,rho=score(r['y'],pred,False),n_subjects=len(set(r['subjects']))))
  for method in ('moments','coral'):
   family='_'.join([r['setting'],r['domain'],obj,target,method]);a,b=pp['none'],pp[method]
   seed=int(hashlib.sha256(json.dumps([ident,method],sort_keys=True).encode()).hexdigest()[:8],16)
   if np.array_equal(a,b):
    value=score(r['y'],a,False);stats=dict(reference_score=value,candidate_score=value,delta=0.,n=r['n'],n_subjects=len(set(r['subjects'])),metric='rho',estimable=True,ci95=[0.,0.],permutation_p=1.,bootstrap_se=0.,inference='Exact identical predictions; conditional resampling statistic is identically zero',bootstrap_requested=5000,permutations_requested=20000,analytic_identity=True)
   else:stats=paired(r['y'],a,b,r['subjects'],seed=seed)
   rows.append(dict(**ident,reference='none',candidate=method,family=family,**{k:v for k,v in stats.items() if k not in ident}))
 save(out,dict(status='complete',rows=rows,scores=scores));return label

def report():
 d=pd.DataFrame([r for p in sorted((RUN/'inference').glob('*.json')) for r in read(p)['rows']])
 assert len(d)==1408
 for fam,ix in d.groupby('family').groups.items():
  d.loc[ix,'holm_p']=holm(d.loc[ix,'permutation_p']);d.loc[ix,'family_size']=len(ix)
 assert not d.duplicated(['model','source','objective','adaptation_target','setting','cohort','domain','candidate']).any()
 d.to_csv(RUN/'paired_comparisons.csv',index=False)
 scores=pd.DataFrame([r for p in sorted((RUN/'inference').glob('*.json')) for r in read(p)['scores']]);scores.to_csv(RUN/'scores.csv',index=False)
 seeds=[]
 for p in sorted((RUN/'evaluation').glob('*.json')):
  ds=read(p);state=ds['state'];assert ds['protocol_sha256']==sha(RUN/'protocol.json')
  for r in ds['rows']:
   for fold in r['fold_audit']:
    assert not set(fold.get('train',fold.get('calibration')))&set(fold['test'])
    if 'inner_train' in fold:assert not set(fold['inner_train'])&set(fold['inner_validation'])
   for method,pp in r['predictions'].items():seeds.append(dict(model=state[1],source=ds['source'],objective=state[2],adaptation_target=state[3],adapter_seed=state[4],**{k:r[k] for k in ('setting','cohort','domain','task','n')},method=method,rho=score(r['y'],pp,False)))
 pd.DataFrame(seeds).to_csv(RUN/'scores_by_seed.csv',index=False)
 sig=d[d.holm_p<.05]
 save(RUN/'report.json',dict(status='complete',jobs=80,comparisons=len(d),scores=len(scores),seed_scores=len(seeds),positive=int((sig.delta>0).sum()),negative=int((sig.delta<0).sum()),families=d.groupby('family').size().to_dict(),checks='Published endpoint samples preserved; all arms share the same float64 source fit; calibration/train/test subject disjointness verified; clinical moment invariance checked; adapter seeds aligned.',automatic_shutdown=False))
