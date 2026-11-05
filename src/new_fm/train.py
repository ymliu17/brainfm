"""Reuse published supervised fitting; native image-only SSL with source-only splits."""
import json
from unittest.mock import patch
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from new_fm.data import RUN,DATA,ROOT,dataset,save,sha
from new_fm.models import Encoder
from fm_matrix.train import amp,features

def location(source,model,objective,target,seed,mode):
    return RUN/'training'/source/model/f'{objective}_{target}_s{seed}_{mode}'

class InputPaths:
    def __init__(self,model):self.model=model
    def __truediv__(self,name):return DATA/name.replace('_a424.pt','_'+self.model+'.pt').replace('_sch450.pt','_'+self.model+'.pt')

def supervised(source,model,target,seed,mode,device):
    from fm_matrix import train as engine
    root=location(source,model,'supervised',target,seed,mode)
    signature=dict(intake_sha256=sha(RUN/'intake.json'),engine_sha256=sha(Path(engine.__file__)),
        model=model,source=source,target=target,seed=seed,mode=mode)
    p=root/'request.json'
    if p.exists() and json.loads(p.read_text())!=signature:raise ValueError('Changed supervised request')
    save(p,signature)
    with patch.multiple(engine,Encoder=Encoder,RUN=RUN,DATA=InputPaths(model),dataset=dataset):
        engine.fit_supervised(source,model,target,seed,mode,device)

from pathlib import Path

def ssl(source,model,seed,device,epochs=30):
    root=location(source,model,'native_ssl','none',seed,'external')
    if (root/'complete.json').exists():return
    ds='nacc_repeated' if source=='nacc' else 'adni';x,m=dataset(ds,model)
    # Preserve existing four-model source population and exact participant split.
    ix=np.flatnonzero(m.native_frames.to_numpy()>=200);groups=m.iloc[ix].subject.to_numpy()
    tr,va=next(GroupShuffleSplit(1,test_size=.15,random_state=seed).split(ix,groups=groups))
    assert not set(groups[tr])&set(groups[va])
    torch.manual_seed(seed);rng=np.random.default_rng(seed);net=Encoder(model,device,native_ssl=True)
    params=[p for p in net.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.05)
    def loss_epoch(ids,training,epoch):
        order=rng.permutation(ids) if training else ids;values=[]
        for start in range(0,len(order),8):
            batch=order[start:start+8]
            # Self-distillation covariance regularization requires >=2 samples.
            if model=='semantoks' and len(batch)==1:
                batch=np.r_[batch,order[0]]
            if training:opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training),amp(device):
                loss=net.ssl_loss(x[ix[batch]].to(device),seed+int(start)+(epoch*10000 if training else 999999))
            if not torch.isfinite(loss):raise ValueError('Nonfinite native SSL loss')
            if training:
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();net.update_teacher()
            values.append((float(loss.detach()),min(8,len(order)-start)))
        return sum(a*b for a,b in values)/sum(b for a,b in values)
    base=loss_epoch(va,False,0);best=base;best_epoch=0;state=net.adapter_state();history=[];stale=0
    for ep in range(1,epochs+1):
        tl=loss_epoch(tr,True,ep);vl=loss_epoch(va,False,ep)
        history.append(dict(epoch=ep,train=tl,validation=vl));save(root/'history.json',history)
        print(model,source,'native_ssl',seed,ep,tl,vl,flush=True)
        if vl<best:best=vl;best_epoch=ep;state=net.adapter_state();stale=0
        else:stale+=1
        if stale>=8:break
    root.mkdir(parents=True,exist_ok=True)
    tmp=root/'adapter.tmp.pt';torch.save(dict(adapter=state,model=model,provenance=net.provenance(),
        intake_sha256=sha(RUN/'intake.json'),source_checkpoint_sha256=sha(net.checkpoint_path),best_epoch=best_epoch),tmp);tmp.replace(root/'adapter.pt')
    save(root/'complete.json',dict(status='complete',source=source,model=model,objective='native_ssl',seed=seed,
        train_subjects=groups[tr],validation_subjects=groups[va],source_indices=ix,
        best_epoch=best_epoch,base_validation=base,best_validation=best,history=history,protocol=net.provenance(),
        effective_batch=8,lr=1e-5,weight_decay=.05,max_epochs=epochs,patience=8,epoch_zero_eligible=True,
        sampling='Same native>=200 frame source pool and85:15 participant split as four-model SSL. Semantoks singleton tail duplicated only for covariance loss.',
        scope='Source-exposed SSL is not independent within-source encoder validation.'))
