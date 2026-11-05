#!/usr/bin/env python3
"""Independent HRV fingerprinting of NACC-trained encoders, with matched subjects."""
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from benchmark_extension import R,save
from run_external_preliminary import sha
from model_loader import load_brainlm_forpretraining
from probe_readouts import extract
from datasets import load_xyz
from paths import XYZ_PATH,OFFICIAL_BRAINLM_13M
from transfer_baseline import fc_feature

from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--name',required=True);a=ap.parse_args();torch.set_num_threads(4)
 d=ROOT/'data/derived/external_preliminary_v1';m=pd.read_csv(d/'hrv_pre.csv');n=pd.read_csv(d/'hrv_post.csv');ids=sorted(set(m.subject)&set(n.subject));xs=[]
 for tag,tab in [('pre',m),('post',n)]:
  ix=tab.reset_index().set_index('subject').loc[ids,'index'].to_numpy();xs.append(torch.load(d/f'hrv_{tag}.pt',weights_only=True)[ix])
 xyz=load_xyz(XYZ_PATH);rows=[]
 for state,ck in [('frozen',OFFICIAL_BRAINLM_13M),(a.name,a.checkpoint)]:
  gpu_acquire()
  net=load_brainlm_forpretraining(ck).cuda().eval();feats=[extract(net,x,xyz,[3,4],['emb_fc','roi_norm'],batch=4) for x in xs];del net;torch.cuda.empty_cache();gpu_release()
  for key in feats[0]:
   aa,bb=[f[key] for f in feats];aa=aa-aa.mean(1,keepdims=True);bb=bb-bb.mean(1,keepdims=True);aa/=np.maximum(np.linalg.norm(aa,axis=1,keepdims=True),1e-8);bb/=np.maximum(np.linalg.norm(bb,axis=1,keepdims=True),1e-8);sim=aa@bb.T
   rows.append(dict(state=state,layer=key[0],readout=key[1],n=len(ids),chance=1/len(ids),pre_post=float(np.mean(sim.argmax(1)==np.arange(len(ids)))),post_pre=float(np.mean(sim.argmax(0)==np.arange(len(ids))))))
 save(R/'adapted_fingerprint'/f'{a.name}.json',dict(status='complete',evidence_status='preliminary',subjects=ids,checkpoint_sha256=sha(a.checkpoint),rows=rows,scope='HRV participants independent of NACC adaptation. Pre/post-intervention identifiability, not intervention-free test-retest reliability.'))
if __name__=='__main__':main()
