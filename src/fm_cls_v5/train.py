"""Participant-isolated training with identical splits, budgets and selection rules."""
from pathlib import Path
import contextlib,copy,json,time
import numpy as np,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from fm_cls_v5.models import Encoder
from fm_matrix.data import DATA,dataset,splits,save,sha
from fm_cls_v5.data import RUN
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
    if model!='brainlm13' or source not in ('adni','nacc') or target not in ('MEM','EF') or mode not in ('cv','external'):
        raise ValueError('Unsupported CLS supervised configuration')
    root=location(source,model,'supervised',target,seed,mode)
    signature=dict(source=source,model=model,target=target,seed=seed,mode=mode,max_epochs=epochs,
                   readout='CLS_final',intake_sha256=sha(RUN/'intake.json'))
    request=root/'request.json'
    if request.exists() and json.loads(request.read_text())!=signature:
        raise ValueError('Changed training configuration cannot reuse partial results')
    if not request.exists():save(request,signature)
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
        torch.save(dict(adapter=adapter,model=model,provenance=net.provenance(),intake_sha256=sha(RUN/'intake.json'),source_checkpoint_sha256=sha(net.checkpoint_path),best_epoch=epochbest),out/'adapter.tmp.pt')
        (out/'adapter.tmp.pt').replace(out/'adapter.pt')
        torch.save(dict(projection=project.cpu().state_dict(),head={k:v.cpu() for k,v in head.state_dict().items()},target_mean=mean,target_sd=sd),out/'head.tmp.pt')
        (out/'head.tmp.pt').replace(out/'head.pt')
        result=dict(status='complete',source=source,model=model,target=target,seed=seed,mode=mode,fold=fold,train_subjects=g[tr],validation_subjects=g[va],test_subjects=g[te],test_indices=te,y=y[te],frozen_prediction=p0,adapted_prediction=pa,best_epoch=epochbest,validation_frozen=v0,validation_selected=best,history=history,protocol=net.provenance(),input_sha256=sha(DATA/f'{source}_{"sch450" if model=="brainjepa" else "a424"}.pt'))
        save(out/'result.json',result);rows.append(result);del net,project,head,opt;torch.cuda.empty_cache()
    save(root/'complete.json',dict(status='complete',signature=signature,folds=rows,training_readout='CLS_final',initialization='Original pretrained BrainLM13 checkpoint; no previous supervised adapter loaded',scope='External export only' if mode=='external' else 'Five outer participant folds, each with independent encoder training and inner validation',effective_batch=8,microbatch=1,max_epochs=epochs,patience=5,lr=1e-4,rank=8,alpha=16,projection_components=100,seed=seed,epoch_zero_eligible=True))
