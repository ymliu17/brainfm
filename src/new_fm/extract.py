"""Deterministic scan features and fixed alternative readouts."""
import json
import numpy as np
import torch
from new_fm.data import RUN,COHORTS,dataset,save,sha
from new_fm.models import Encoder
from new_fm.train import location
from fm_matrix.train import amp
from fm_readout_v4.evaluate import tag

def folder(state):return RUN/'features'/tag(*state)

def extract(state,device='cuda'):
    source,model,obj,target,seed=state;out=folder(state);out.mkdir(parents=True,exist_ok=True)
    ck=None if obj=='frozen' else location(source,model,obj,target,seed,'external')/('fold0/adapter.pt' if obj=='supervised' else 'adapter.pt')
    signature=dict(model=model,objective=obj,readout={'semantoks':'teacher_CLS_final','fmrilm':'stage2_brain_only_Qwen_mean_final'}[model],
        intake_sha256=sha(RUN/'intake.json'),checkpoint_sha256=sha(ck) if ck else None)
    request=out/'request.json'
    if request.exists() and json.loads(request.read_text())!=signature:raise ValueError('Stale partial export')
    save(request,signature)
    if (out/'complete.json').exists():return
    net=Encoder(model,device)
    if ck:
        d=torch.load(ck,map_location='cpu',weights_only=True)
        if d['model']!=model or d['source_checkpoint_sha256']!=sha(net.checkpoint_path) or d['intake_sha256']!=sha(RUN/'intake.json'):raise ValueError('Adapter provenance mismatch')
        net.load_adapter(d['adapter'])
    checks={}
    for co in COHORTS:
        path=out/(co+'.npy');alt=out/(co+'_alternatives.npz')
        if not path.exists() or (obj=='frozen' and not alt.exists()):
            x,m=dataset(co,model);features=[];alternatives={}
            with torch.no_grad():
                for start in range(0,len(x),4):
                    with amp(device):
                        tokens=net.tokens(x[start:start+4].to(device))
                        primary=tokens[:,0] if model=='semantoks' else tokens.mean(1)
                    if not torch.isfinite(primary).all():raise ValueError('Nonfinite features')
                    features.append(primary.float().cpu().numpy())
                    if obj=='frozen':
                        if model=='semantoks':
                            observed=tokens[:,1:].reshape(len(tokens),9,5,-1)[:,:8]
                            vals=dict(content_mean=observed.mean((1,2)),cls_avg=torch.cat((observed.mean((1,2)),tokens[:,0]),dim=1),network_flat=observed.mean(2).flatten(1))
                        else:
                            with amp(device):image,_=net.net.tokenizer(x[start:start+4,:,:160].to(device))
                            vals=dict(image_encoder_mean=image.mean(2),regional_norm=tokens.norm(dim=2))
                        for key,val in vals.items():alternatives.setdefault(key,[]).append(val.float().cpu().numpy())
            tmp=path.with_suffix('.tmp.npy');np.save(tmp,np.concatenate(features));tmp.replace(path)
            if obj=='frozen':
                tmp=alt.with_suffix('.tmp.npz');np.savez(tmp,**{k:np.concatenate(v) for k,v in alternatives.items()});tmp.replace(alt)
            print('features',tag(*state),co,len(x),flush=True)
        checks[co]=sha(path)
    save(out/'complete.json',dict(status='complete',signature=signature,encoder=net.provenance(),feature_sha256=checks,
        source_checkpoint_sha256=sha(net.checkpoint_path),readout_alternatives='Fixed a priori, not chosen using downstream test outcomes',
        input_caveat='fMRI-LM per-scan robust-scaling benchmark variant; official pretraining cohort statistics unavailable.'))

def load(state,co):
    f=folder(state);d=json.loads((f/'complete.json').read_text());p=f/(co+'.npy')
    if sha(p)!=d['feature_sha256'][co]:raise ValueError('Changed features')
    return np.load(p,mmap_mode='r')
