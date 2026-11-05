#!/usr/bin/env python3
"""NACC supervised post-training, with encoder fitting isolated from outer tests.

Full-source checkpoints are for external evaluation only. CV checkpoints are
fold-specific and are never evaluated on subjects used for early stopping.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge,LogisticRegression
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from finetune_adni_probe import ProbeHeadClassifier,frozen_features,predict
from benchmark_evaluation import grouped_splits,score,paired_bootstrap
from datasets import load_xyz
from paths import OFFICIAL_BRAINLM_13M,XYZ_PATH
from run_external_preliminary import dump,sha
DEVICE=torch.device('cuda')

def split_inner(groups,seed):
 tr,va=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(np.zeros(len(groups)),groups=groups))
 assert not set(groups[tr])&set(groups[va]);return tr,va

def fit_head(model,F,y,groups,seed,binary):
 tr,va=split_inner(groups,seed)
 def transform(A,B):
  sc=StandardScaler().fit(A);a=sc.transform(A);pc=PCA(n_components=min(200,len(A)-1,A.shape[1]),random_state=0).fit(a)
  return sc,pc,pc.transform(a),pc.transform(sc.transform(B))
 _,_,a,b=transform(F[tr],F[va]);best=-np.inf;chosen=.001
 for c in [.001,.01,.1,1.,10.]:
  if binary and len(np.unique(y[tr]))<2:continue
  m=(LogisticRegression(C=c,max_iter=5000) if binary else Ridge(alpha=1/c)).fit(a,y[tr]);p=m.predict_proba(b)[:,1] if binary else m.predict(b);s=score(y[va],p,binary)
  if np.isfinite(s) and s>best:best=s;chosen=c
 sc,pc,a,_=transform(F,F[:1]);m=(LogisticRegression(C=chosen,max_iter=5000) if binary else Ridge(alpha=1/chosen)).fit(a,y)
 for key,v in [('pca_mean',pc.mean_),('pca_comp',pc.components_),('scaler_mean',sc.mean_),('scaler_scale',sc.scale_)]:setattr(model,key,torch.as_tensor(v,dtype=torch.float32,device=DEVICE))
 model.head=nn.Linear(a.shape[1],1).to(DEVICE)
 with torch.no_grad():
  model.head.weight.copy_(torch.as_tensor(np.atleast_2d(m.coef_),dtype=torch.float32,device=DEVICE));model.head.bias.copy_(torch.as_tensor(np.atleast_1d(m.intercept_),dtype=torch.float32,device=DEVICE))
 return chosen

def train(x,y,groups,tr,va,te,args,folder):
 assert not set(groups[tr])&set(groups[va]);assert not (set(groups[tr])|set(groups[va]))&set(groups[te])
 torch.manual_seed(args.seed);np.random.seed(args.seed)
 model=ProbeHeadClassifier(pretrained_backbone_path=str(OFFICIAL_BRAINLM_13M),lora_r=8,readout='emb_fc',layer=3,n_pca=200).to(DEVICE).eval();xyz=load_xyz(XYZ_PATH).to(DEVICE)
 binary=args.target in ['DX','SEX','AD_DX'];f=frozen_features(model,x[tr],xyz,args.batch_size);c=fit_head(model,f,y[tr],groups[tr],args.seed,binary)
 p0=predict(model,x[te],xyz,args.batch_size,binary) if len(te) else None
 frozen_val=score(y[va],predict(model,x[va],xyz,args.batch_size,binary),binary)
 best=frozen_val;best_epoch=0;stale=0;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 opt=torch.optim.AdamW([{'params':[p for p in model.brainlm.parameters() if p.requires_grad],'lr':1e-4},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.01)
 loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y[tr]==0).sum()/max(1,(y[tr]==1).sum())],device=DEVICE)) if binary else nn.MSELoss()
 loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[tr],torch.tensor(y[tr],dtype=torch.float32)),batch_size=args.batch_size,shuffle=True)
 history=[];folder.mkdir(parents=True,exist_ok=True)
 dump(folder/'split.json',dict(train=groups[tr].tolist(),validation=groups[va].tolist(),test=groups[te].tolist(),C=c,target=args.target,seed=args.seed,source_checkpoint_sha256=sha(OFFICIAL_BRAINLM_13M),data_dir=str(args.data_dir),label_sha256=sha(args.data_dir/f'ADNI_y_nacc_{args.target}.pt'),protocol='NACC-posttraining-v1',status='preliminary',epoch0_eligible_for_selection=True,score_scope='external_only' if not len(te) else 'outer_held_out'))
 for epoch in range(1,args.epochs+1):
  model.train();total=0
  for xb,yb in loader:
   xb,yb=xb.to(DEVICE),yb.to(DEVICE);opt.zero_grad();v=loss(model(xb,xyz.unsqueeze(0).expand(len(xb),-1,-1)),yb);v.backward();opt.step();total+=v.item()*len(xb)
  sv=score(y[va],predict(model,x[va],xyz,args.batch_size,binary),binary)
  history.append(dict(epoch=epoch,loss=total/len(tr),validation_score=sv));dump(folder/'history.json',history)
  print(args.target,args.mode,args.seed,'epoch',epoch,'val',sv,flush=True)
  if np.isfinite(sv) and (not np.isfinite(best) or sv>best):best=sv;best_epoch=epoch;stale=0;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:stale+=1
  if stale>=args.patience:break
 model.load_state_dict(beststate)
 pf=predict(model,x[te],xyz,args.batch_size,binary) if len(te) else None
 # Preserve an optional head for audit; the exported merged encoder is standalone.
 torch.save(beststate,folder/'model_with_head.pt');enc=model.brainlm.merge_and_unload();torch.save({k:v.cpu() for k,v in enc.state_dict().items()},folder/'encoder.pt')
 meta=dict(target=args.target,seed=args.seed,mode=args.mode,n_train=len(tr),n_validation=len(va),n_test=len(te),best_epoch=best_epoch,epochs_run=len(history),inner_val_frozen=frozen_val,inner_val_selected=best,independent_test=bool(len(te)),status='preliminary')
 if len(te):meta.update(subjects=groups[te].tolist(),y=y[te].tolist(),frozen_prediction=p0.tolist(),adapted_prediction=pf.tolist())
 dump(folder/'result.json',meta);del model;torch.cuda.empty_cache();return meta

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['MEM','EF','DX','DX_ord','AGE','SEX','EDUC','AD_DX'],required=True);ap.add_argument('--mode',choices=['external','cv'],required=True);ap.add_argument('--data-dir',type=Path,default=ROOT/'data/derived/nacc_parity_v1');ap.add_argument('--out-root',type=Path,default=ROOT/'runs/nacc_parity_v1/posttraining');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--patience',type=int,default=5);ap.add_argument('--batch_size',type=int,default=8);a=ap.parse_args();torch.set_num_threads(4)
 # The active common-atlas diagnosis extension owns DX/AD_DX fitting.
 canonical=next((p for p in [ROOT,*ROOT.parents] if (p/'runs/diagnosis_posttraining_v1/protocol.json').exists()),None)
 if canonical is not None and a.target in ('DX','AD_DX') and a.out_root.resolve().is_relative_to(canonical/'runs/unified_input_v3/workspace/runs'):
  sys.path[:0]=[str(canonical/'src'),str(canonical/'scripts')]
  from diagnosis_posttraining import delegate_legacy
  if delegate_legacy(a):return
 d=a.data_dir;x=torch.load(d/f'ADNI_x_nacc_{a.target}.pt',weights_only=True);y=torch.load(d/f'ADNI_y_nacc_{a.target}.pt',weights_only=True).numpy();m=pd.read_csv(d/f'ADNI_manifest_nacc_{a.target}.csv');groups=m.PTID.to_numpy()
 out=a.out_root/f'{a.target}_{a.mode}_seed{a.seed}'
 if a.mode=='external':
  if (out/'result.json').exists():return
  tr,va=split_inner(groups,a.seed);train(x,y,groups,tr,va,np.array([],dtype=int),a,out)
 else:
  results=[]
  for k,(pool,te) in enumerate(grouped_splits(y,groups,a.seed,5)):
   f=out/f'fold{k}';tr,va=split_inner(groups[pool],a.seed*100+k)
   r=json.loads((f/'result.json').read_text()) if (f/'result.json').exists() else train(x,y,groups,pool[tr],pool[va],te,a,f);results.append(r)
  ys=np.concatenate([r['y'] for r in results]);p0=np.concatenate([r['frozen_prediction'] for r in results]);pf=np.concatenate([r['adapted_prediction'] for r in results]);subs=np.concatenate([r['subjects'] for r in results]);delta,p,ci,details=paired_bootstrap(ys,p0,pf,n_boot=10000,groups=subs,return_details=True)
  dump(out/'result.json',dict(status='preliminary',folds=results,frozen_score=score(ys,p0),adapted_score=score(ys,pf),delta=delta,raw_one_sided_bootstrap_p=p,ci95=ci,**details))
if __name__=='__main__':main()
