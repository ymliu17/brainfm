"""Participant-isolated training with identical splits, budgets and selection rules."""
from pathlib import Path
import contextlib,copy,json,time
import numpy as np,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from fm_matrix.models import Encoder
from fm_matrix.data import DATA,RUN,dataset,splits,save,sha
from benchmark_evaluation import score


def amp(device):return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()

@torch.no_grad()
def features(net,x,device):
    vals=[]
    for i in range(len(x)):
        with amp(device):v=net(x[i:i+1].to(device))
        if not torch.isfinite(v).all():raise ValueError('Nonfinite encoder representation')
        vals.append(v.float().cpu())
    return torch.cat(vals).numpy()


class Projection(nn.Module):
    def __init__(self,F,dim=100):
        super().__init__();sc=StandardScaler().fit(F);pc=PCA(n_components=min(dim,len(F)-1,F.shape[1]),svd_solver='randomized',random_state=0).fit(sc.transform(F))
        for key,value in [('mean',sc.mean_),('scale',sc.scale_),('pmean',pc.mean_),('components',pc.components_)]:self.register_buffer(key,torch.tensor(value,dtype=torch.float32))
        self.dim=len(pc.components_)
    def forward(self,z):return ((z.float()-self.mean)/self.scale-self.pmean)@self.components.T


def location(source,model,objective,target,seed,mode):return RUN/'training'/source/model/f'{objective}_{target}_s{seed}_{mode}'


def fit_supervised(source,model,target,seed,mode,device='cuda',epochs=15):
    root=location(source,model,'supervised',target,seed,mode)
    if (root/'complete.json').exists():return
    x,m=dataset(source,model);y=m[target].to_numpy(float);g=m.subject.astype(str).to_numpy();rows=[]
    for fold,(tr,va,te) in enumerate(splits(source,target,seed,mode)):
        out=root/f'fold{fold}'
        if (out/'result.json').exists():rows.append(json.loads((out/'result.json').read_text()));continue
        torch.manual_seed(seed*100+fold);rng=np.random.default_rng(seed*100+fold)
        net=Encoder(model,device);ft=features(net,x[tr],device);project=Projection(ft).to(device)
        mean=float(y[tr].mean());sd=max(float(y[tr].std()),1e-8)
        z=project(torch.as_tensor(ft,device=device)).detach().cpu().numpy();ridge=Ridge(alpha=10).fit(z,(y[tr]-mean)/sd)
        head=nn.Linear(project.dim,1).to(device)
        with torch.no_grad():head.weight.copy_(torch.as_tensor(ridge.coef_[None],device=device));head.bias.fill_(float(ridge.intercept_))
        @torch.no_grad()
        def predict(ix):
            f=features(net,x[ix],device);return (head(project(torch.as_tensor(f,device=device))).squeeze(1).cpu().numpy()*sd+mean)
        p0=predict(te) if len(te) else [];v0=score(y[va],predict(va),False);best=v0;epochbest=0;stale=0
        adapter=net.adapter_state();headstate=copy.deepcopy(head.state_dict());history=[]
        opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.requires_grad],'lr':1e-4},{'params':head.parameters(),'lr':1e-4}],weight_decay=.01)
        for epoch in range(1,epochs+1):
            losses=[];order=rng.permutation(tr)
            for start in range(0,len(order),8):
                ids=order[start:start+8];opt.zero_grad()
                for idx in ids:
                    with amp(device):pred=head(project(net(x[idx:idx+1].to(device)))).squeeze();loss=(pred-(y[idx]-mean)/sd)**2
                    if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss')
                    (loss/len(ids)).backward();losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad]+list(head.parameters()),1.);opt.step()
            sv=score(y[va],predict(va),False);history.append(dict(epoch=epoch,loss=np.mean(losses),validation=sv));save(out/'history.json',history)
            print(source,model,target,seed,mode,fold,epoch,sv,flush=True)
            if np.isfinite(sv) and (not np.isfinite(best) or sv>best):best=sv;epochbest=epoch;stale=0;adapter=net.adapter_state();headstate=copy.deepcopy(head.state_dict())
            else:stale+=1
            if stale>=5:break
        net.load_adapter(adapter);head.load_state_dict(headstate);pa=predict(te) if len(te) else []
        out.mkdir(parents=True,exist_ok=True)
        torch.save(dict(adapter=adapter,model=model,provenance=net.provenance(),intake_sha256=sha(RUN/'intake.json'),source_checkpoint_sha256=sha(net.checkpoint_path),best_epoch=epochbest),out/'adapter.pt')
        torch.save(dict(projection=project.cpu().state_dict(),head={k:v.cpu() for k,v in head.state_dict().items()},target_mean=mean,target_sd=sd),out/'head.pt')
        result=dict(status='complete',source=source,model=model,target=target,seed=seed,mode=mode,fold=fold,train_subjects=g[tr],validation_subjects=g[va],test_subjects=g[te],test_indices=te,y=y[te],frozen_prediction=p0,adapted_prediction=pa,best_epoch=epochbest,validation_frozen=v0,validation_selected=best,history=history,protocol=net.provenance(),input_sha256=sha(DATA/f'{source}_{"sch450" if model=="brainjepa" else "a424"}.pt'))
        save(out/'result.json',result);rows.append(result);del net,project,head,opt;torch.cuda.empty_cache()
    save(root/'complete.json',dict(status='complete',folds=rows,scope='External export only' if mode=='external' else 'Five outer participant folds, each with independent encoder training and inner validation',effective_batch=8,microbatch=1,max_epochs=epochs,patience=5,lr=1e-4,rank=8,alpha=16,projection_components=100,seed=seed,epoch_zero_eligible=True))


def contrast_loss(a,b,temperature=.1):
    a=torch.nn.functional.normalize(a,dim=-1);b=torch.nn.functional.normalize(b,dim=-1);z=torch.cat([a,b]);n=len(a);sim=z@z.T/temperature
    sim.fill_diagonal_(-torch.inf);target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return torch.nn.functional.cross_entropy(sim,target)


def ssl_samples(source,objective,model):
    ds='nacc_repeated' if source=='nacc' else 'adni';x,m=dataset(ds,model)
    pairs=[]
    if objective=='native_ssl':
        # Identical eligibility across models, even though JEPA consumes 160 real frames.
        for i,r in m.iterrows():
            if r.native_frames>=200:pairs.append((i,i))
    else:
        for _,g in m.groupby('subject',sort=True):
            ix=g.sort_values('visit').index.to_list()
            if len(ix)>=2:pairs.append((ix[0],ix[1]))
    if len(pairs)<10:raise ValueError('Too few SSL source subjects')
    groups=np.asarray([m.iloc[a].subject for a,b in pairs]);return x,m,np.asarray(pairs),groups


def fit_ssl(source,model,objective,seed,device='cuda',epochs=None):
    root=location(source,model,objective,'none',seed,'external')
    if (root/'complete.json').exists():return
    x,m,pairs,g=ssl_samples(source,objective,model);tr,va=next(GroupShuffleSplit(1,test_size=.15,random_state=seed).split(pairs,groups=g));assert not set(g[tr])&set(g[va])
    torch.manual_seed(seed);rng=np.random.default_rng(seed);native=objective=='native_ssl';net=Encoder(model,device,native_ssl=native)
    epochs=epochs or (30 if native else 20);batch=8 if native else 16;project=None;head=None
    if not native:
        f=features(net,x[pairs[tr,0]],device);project=Projection(f).to(device);head=nn.Sequential(nn.Linear(project.dim,128),nn.GELU(),nn.Linear(128,128)).to(device)
    params=[p for p in net.parameters() if p.requires_grad]+([] if head is None else list(head.parameters()))
    opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.05 if native else .01)
    def epoch_loss(ids,train,epoch):
        losses=[];order=rng.permutation(ids) if train else ids
        # Preserve reproducible validation augmentations/masks across every epoch.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []):
            torch.manual_seed(seed+epoch if train else seed+999999)
            for start in range(0,len(order),batch):
                inds=order[start:start+batch]
                if not native and len(inds)<2:continue
                if train:opt.zero_grad()
                with torch.set_grad_enabled(train):
                    if native:
                        total=0.
                        for idx in inds:
                            xb=x[pairs[idx,0]:pairs[idx,0]+1].to(device)
                            with amp(device):loss=net.masked_loss(xb,seed+int(idx)+epoch*10000 if train else seed+int(idx))
                            if not torch.isfinite(loss):raise ValueError('Nonfinite masked objective')
                            if train:(loss/len(inds)).backward()
                            total+=float(loss.detach())
                        value=total/len(inds)
                    else:
                        views=[[],[]]
                        for idx in inds:
                            a,b=pairs[idx];b=a if objective=='contrast_scale' else b
                            for side,k in enumerate([a,b]):
                                xb=x[k:k+1].to(device)
                                if objective in ['contrast_scale','contrast_both']:xb=xb*torch.exp(torch.empty((),device=device).uniform_(np.log(.5),np.log(2.)))
                                with amp(device):views[side].append(head(project(net(xb))))
                        loss=contrast_loss(torch.cat(views[0]).float(),torch.cat(views[1]).float())
                        if not torch.isfinite(loss):raise ValueError('Nonfinite contrastive loss')
                        if train:loss.backward()
                        value=float(loss.detach())
                    if train:torch.nn.utils.clip_grad_norm_(params,1.);opt.step();net.update_teacher()
                    losses.append((value,len(inds)))
        if not losses:raise ValueError('No eligible SSL batches')
        return sum(a*b for a,b in losses)/sum(b for a,b in losses)
    base=epoch_loss(va,False,0);best=base;best_epoch=0;state=net.adapter_state();history=[];stale=0
    for ep in range(1,epochs+1):
        tl=epoch_loss(tr,True,ep);vl=epoch_loss(va,False,ep);history.append(dict(epoch=ep,train=tl,validation=vl));save(root/'history.json',history);print(source,model,objective,seed,ep,tl,vl,flush=True)
        if vl<best:best=vl;best_epoch=ep;state=net.adapter_state();stale=0
        else:stale+=1
        if native and stale>=8:break
    root.mkdir(parents=True,exist_ok=True);torch.save(dict(adapter=state,model=model,provenance=net.provenance(),intake_sha256=sha(RUN/'intake.json'),source_checkpoint_sha256=sha(net.checkpoint_path),best_epoch=best_epoch),root/'adapter.pt')
    # Epoch zero always yields an explicit unchanged encoder, never a missing checkpoint.
    save(root/'complete.json',dict(status='complete',source=source,model=model,objective=objective,seed=seed,train_subjects=g[tr],validation_subjects=g[va],best_epoch=best_epoch,base_validation=base,best_validation=best,history=history,protocol=net.provenance(),effective_batch=batch,microbatch=1,lr=1e-5,epoch_zero_eligible=True,scope='Source-only SSL; external downstream tests only; no independent within-source SSL claim'))
