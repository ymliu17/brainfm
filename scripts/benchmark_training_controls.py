#!/usr/bin/env python3
"""Matched-budget pretrained/random BrainLM and a compact temporal GRU control.

Train and early-stop inside each subject-grouped outer fold. The pretrained and
random encoders have identical architecture, readout, optimizer and training budget.
This is a finite-budget comparison, not an upper bound on training from scratch.
"""
import argparse,sys,copy
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from benchmark_extension import save,load,R,D,metrics
from benchmark_evaluation import grouped_splits,score,paired_bootstrap
from sklearn.model_selection import GroupShuffleSplit
from datasets import load_xyz
from paths import OFFICIAL_BRAINLM_13M,XYZ_PATH
from model_loader import load_brainlm_forpretraining

class BrainControl(nn.Module):
 def __init__(self,random=False):
  super().__init__();head_seed=torch.initial_seed();base=load_brainlm_forpretraining(OFFICIAL_BRAINLM_13M)
  if random:
   from brainlm_mae.modeling_brainlm import BrainLMForPretraining
   base=BrainLMForPretraining(base.config)
  self.vit=base.vit
  for p in self.vit.parameters():p.requires_grad_(True)
  torch.manual_seed(head_seed+100000) # Same head initialization across pretrained/random encoders.
  self.head=nn.Sequential(nn.LayerNorm(424),nn.Linear(424,1))
 def forward(self,x,xyz):
  self.vit.eval() # Disable pretraining masking; gradients remain enabled.
  h=self.vit(signal_vectors=x,xyz_vectors=xyz[None].expand(len(x),-1,-1),return_dict=True,output_hidden_states=True).hidden_states[4][:,1:]
  f=h.reshape(len(x),424,x.shape[-1]//20,-1).mean(2).norm(dim=-1)
  return self.head(f).flatten()

class TemporalControl(nn.Module):
 def __init__(self):
  super().__init__();self.input=nn.Sequential(nn.Linear(424,64),nn.LayerNorm(64),nn.GELU());self.gru=nn.GRU(64,32,batch_first=True);self.head=nn.Linear(32,1)
 def forward(self,x,xyz):
  # Native trailing padding is detected from the centered tensor; do not pool pads.
  lengths=(x.abs().sum(1)>0).sum(1).clamp(min=1).cpu()
  a=self.input(x.transpose(1,2));packed=nn.utils.rnn.pack_padded_sequence(a,lengths,batch_first=True,enforce_sorted=False)
  _,h=self.gru(packed);return self.head(h[-1]).flatten()

@torch.no_grad()
def predict(net,x,xyz,binary,mean,sd):
 net.eval();p=torch.cat([net(x[i:i+4].cuda(),xyz).cpu() for i in range(0,len(x),4)])
 return (p.sigmoid() if binary else p*sd+mean).numpy()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['nacc','adni'],required=True);ap.add_argument('--target',required=True);ap.add_argument('--model',choices=['pretrained','random','gru'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();torch.set_num_threads(4)
 out=R/'training_controls'/f'{a.cohort}_{a.target}_{a.model}_s{a.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists() and not a.smoke:return
 if a.cohort=='nacc':
  m=pd.read_csv(D/'nacc_default.csv');keep=m[a.target].notna();x=torch.load(D/'nacc_default.pt',weights_only=True)[torch.tensor(keep.to_numpy())];y=m.loc[keep,a.target].to_numpy(float);groups=m.loc[keep,'subject'].to_numpy()
 else:
  d=ROOT/'data/derived/adni424_t200';tag='raw_'+('ADNI_'+a.target if a.target in ['MEM','EF'] else a.target)+'_all_t200_off';x=torch.load(d/f'ADNI_x_{tag}.pt',weights_only=True);y=torch.load(d/f'ADNI_y_{tag}.pt',weights_only=True).numpy();groups=pd.read_csv(d/f'ADNI_manifest_{tag}.csv').PTID.to_numpy()
 binary=set(np.unique(y))=={0,1};xyz=load_xyz(XYZ_PATH).cuda();pred=np.full(len(y),np.nan);initial=pred.copy();folds=[]
 for k,(pool,te) in enumerate(grouped_splits(y,groups,a.seed)):
  folder=out/f'fold{k}';folder.mkdir(exist_ok=True)
  if (folder/'result.json').exists() and not a.smoke:
   r=load(folder/'result.json');pred[te]=r['prediction'];initial[te]=r['initial_prediction'];folds.append(r);continue
  for attempt in range(100):
   it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=a.seed*1000+k*100+attempt).split(np.zeros(len(pool)),groups=groups[pool]));tr,va=pool[it],pool[iv]
   if not binary or (len(np.unique(y[tr]))==2 and len(np.unique(y[va]))==2):break
  else:raise ValueError('Unable to split classes')
  torch.manual_seed(a.seed*10+k);np.random.seed(a.seed*10+k)
  net=(TemporalControl() if a.model=='gru' else BrainControl(a.model=='random')).cuda()
  if a.smoke:
   z=net(x[tr[:2]].cuda(),xyz);z.sum().backward();print('SMOKE',a.model,z.shape,flush=True);return
  mean,sd=(0.,1.) if binary else (float(y[tr].mean()),max(float(y[tr].std()),1e-8))
  initial[te]=predict(net,x[te],xyz,binary,mean,sd);baseline=score(y[va],predict(net,x[va],xyz,binary,mean,sd))
  best=baseline;epochbest=0;state=copy.deepcopy({k:v.cpu() for k,v in net.state_dict().items()});opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
  loss=nn.BCEWithLogitsLoss() if binary else nn.MSELoss();hist=[];stale=0
  for epoch in range(1,a.epochs+1):
   net.train();order=np.random.permutation(tr);total=0.
   for i in range(0,len(order),4):
    ix=order[i:i+4];opt.zero_grad();p=net(x[ix].cuda(),xyz);v=loss(p,torch.tensor((y[ix]-mean)/sd,dtype=torch.float32,device='cuda'));v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();total+=float(v)*len(ix)
   sv=score(y[va],predict(net,x[va],xyz,binary,mean,sd));hist.append(dict(epoch=epoch,loss=total/len(tr),validation_score=sv));print(a.cohort,a.target,a.model,a.seed,k,epoch,sv,flush=True)
   if np.isfinite(sv) and (not np.isfinite(best) or sv>best):best=sv;epochbest=epoch;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};stale=0
   else:stale+=1
   if stale>=5:break
  net.load_state_dict(state);pred[te]=predict(net,x[te],xyz,binary,mean,sd);torch.save(state,folder/'model.pt')
  r=dict(fold=k,train_subjects=groups[tr].tolist(),validation_subjects=groups[va].tolist(),test_subjects=groups[te].tolist(),test_indices=te.tolist(),prediction=pred[te].tolist(),initial_prediction=initial[te].tolist(),history=hist,best_epoch=epochbest,validation_score=best)
  assert not set(groups[tr])&set(groups[te]) and not set(groups[va])&set(groups[te]);save(folder/'result.json',r);folds.append(r);del net;torch.cuda.empty_cache()
 save(out/'result.json',dict(status='complete',evidence_status='preliminary',cohort=a.cohort,target=a.target,model=a.model,seed=a.seed,subjects=groups.tolist(),y=y.tolist(),predictions=pred.tolist(),metrics=metrics(y,pred),folds=folds,protocol=dict(lr=.0001,max_epochs=a.epochs,patience=5,batch=4,weight_decay=.01,brainlm_readout='roi_norm L4',selection='inner validation only including epoch zero',scope='Matched finite-budget control; no claim that scratch tuning is exhaustive')))
if __name__=='__main__':main()
