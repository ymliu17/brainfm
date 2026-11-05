"""Frozen and adapted encoders share observed-token pooling and input construction."""
import json
import numpy as np
import torch
from brainharmonix.data import RUN,COHORTS,dataset,save,sha
from brainharmonix.models import Encoder
from brainharmonix.train import location
from fm_matrix.train import amp
from fm_readout_v4.evaluate import tag

def folder(state):return RUN/'features'/tag(*state)

def extract(state,device='cuda'):
    source,model,obj,target,seed=state;out=folder(state);out.mkdir(parents=True,exist_ok=True)
    ck=None if obj=='frozen' else location(source,model,obj,target,seed,'external')/('fold0/adapter.pt' if obj=='supervised' else 'adapter.pt')
    signature=dict(model=model,objective=obj,readout='EMA_final_valid_token_mean',intake_sha256=sha(RUN/'intake.json'),checkpoint_sha256=sha(ck) if ck else None)
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
                for i in range(len(x)):
                    with amp(device):z,mask=net.token_outputs(x[i:i+1].to(device))
                    z=z.float();primary=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
                    if not torch.isfinite(primary).all():raise ValueError('Nonfinite features')
                    features.append(primary.cpu().numpy())
                    if obj=='frozen':
                        reg=z.reshape(1,400,18,768).sum(2)/mask.reshape(1,400,18).sum(2)[:,:,None]
                        centered=torch.nn.functional.normalize(reg-reg.mean(-1,keepdim=True),dim=-1)
                        a=centered@centered.transpose(1,2);ii,jj=torch.triu_indices(400,400,offset=1,device=z.device)
                        vals=dict(regional_norm=reg.norm(dim=-1),regional_connectivity=a[:,ii,jj],
                            token_std=torch.sqrt(((z-primary[:,None]).square()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]))
                        for key,val in vals.items():alternatives.setdefault(key,[]).append(val.cpu().numpy())
                    if i%100==0:print('extract',tag(*state),co,i,len(x),flush=True)
            tmp=path.with_suffix('.tmp.npy');np.save(tmp,np.concatenate(features));tmp.replace(path)
            if obj=='frozen':
                tmp=alt.with_suffix('.tmp.npz');np.savez(tmp,**{k:np.concatenate(v) for k,v in alternatives.items()});tmp.replace(alt)
        checks[co]=sha(path)
    save(out/'complete.json',dict(status='complete',signature=signature,encoder=net.provenance(),feature_sha256=checks,
        source_checkpoint_sha256=sha(net.checkpoint_path),readout_alternatives='Fixed regional norm/connectivity and token SD; no outcome-based readout selection.',
        input_caveat='Native-TR18patch context; scan-local robust-normalization variant. Not duration-matched to other models.'))

def load(state,co):
    f=folder(state);d=json.loads((f/'complete.json').read_text());p=f/(co+'.npy')
    if sha(p)!=d['feature_sha256'][co]:raise ValueError('Changed features')
    return np.load(p,mmap_mode='r')
