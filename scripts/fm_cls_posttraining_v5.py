#!/usr/bin/env python3
"""Train and evaluate BrainLM13 with final CLS throughout supervised adaptation."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def smoke(device):
    import torch
    from fm_cls_v5.models import Encoder
    from fm_cls_v5.data import dataset,RUN,save
    from fm_cls_v5.train import amp
    torch.manual_seed(42);net=Encoder('brainlm13',device);x,_=dataset('nacc','brainlm13');x=x[:2].to(device)
    before=net.adapter_state();head=torch.nn.Linear(512,1).to(device)
    with torch.no_grad(),amp(device):
        actual=net(x)
        expected=net.net.vit(signal_vectors=x,xyz_vectors=net.xyz[None].expand(2,-1,-1),noise=None,return_dict=True).last_hidden_state[:,0]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    with amp(device):loss=(head(net(x)).flatten()-torch.tensor([-1.,1.],device=device)).float().square().mean()
    loss.backward()
    gradients=[p.grad for n,p in net.named_parameters() if 'lora_' in n and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Missing/nonfinite CLS adapter gradient')
    grad=sum(float(g.float().norm()) for g in gradients)
    if grad<=0:raise ValueError('CLS loss does not update adapters')
    if any(p.grad is not None for p in net.parameters() if not p.requires_grad):raise ValueError('Frozen base received a gradient')
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-4);opt.step()
    changed=any(not torch.equal(before[k],v) for k,v in net.adapter_state().items())
    if not changed:raise ValueError('CLS optimizer step did not update an adapter')
    net.load_adapter(before)
    with torch.no_grad(),amp(device):restored=net(x)
    torch.testing.assert_close(actual,restored,rtol=0,atol=0)
    save(RUN/'smoke.json',dict(status='passed',shape=list(actual.shape),readout='CLS_final',
         exact_final_CLS=True,supervised_adapter_gradient=grad,optimizer_updates_adapters=changed,
         base_frozen=True,adapter_roundtrip=True,loss=float(loss.detach()),protocol=net.provenance()))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','supervised','features','analysis','loso','calibration','report'])
    p.add_argument('--source',choices=['none','adni','nacc'],default='none');p.add_argument('--model',choices=['brainlm13'],default='brainlm13')
    p.add_argument('--target',choices=['none','MEM','EF'],default='none');p.add_argument('--objective',choices=['frozen','supervised'],default='frozen')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--mode',choices=['cv','external'],default='external')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    from fm_cls_v5 import data,evaluate,supplement,train
    data.prepare();args=(a.source,a.model,a.objective,a.target,a.seed)
    if a.action=='prepare':return
    if a.action=='smoke':smoke(a.device)
    elif a.action=='supervised':train.fit_supervised(a.source,a.model,a.target,a.seed,a.mode,a.device)
    elif a.action=='features':evaluate.extraction(*args,a.device)
    elif a.action=='analysis':
        cohorts=[co for co in evaluate.COHORTS[:-1] if a.objective=='frozen' or co!=a.source]
        for co in cohorts:evaluate.local(*args,co)
        evaluate.identification(*args);evaluate.axes(*args)
        data.save(data.RUN/'evaluation'/evaluate.tag(*args)/'complete.json',dict(status='complete',cohorts=cohorts,
            readout='CLS_final',training_readout='CLS_final' if a.objective=='supervised' else None,
            source_cv='Only fold-specific training outputs support within-source evaluation; external source-fitted adapters excluded.'))
    elif a.action=='loso':supplement.loso(*args)
    elif a.action=='calibration':supplement.calibration(*args)
    elif a.action=='report':
        from fm_cls_v5.report import report
        report()


if __name__=='__main__':main()
