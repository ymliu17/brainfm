#!/usr/bin/env python3
"""Entry point for the matched four-model post-training matrix."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def smoke(model,device):
    import torch,numpy as np
    from fm_matrix.models import Encoder
    from fm_matrix.data import dataset,RUN,save
    from fm_matrix.train import amp,contrast_loss
    torch.manual_seed(42);net=Encoder(model,device,native_ssl=True);x,m=dataset('nacc',model);i=int(np.flatnonzero(m.native_frames>=200)[0]);x=x[i:i+1].to(device)
    before=net.adapter_state()
    with amp(device):z=net(x);w=net.windows(x)
    assert torch.isfinite(z).all() and w.shape[1]==8
    z.float().square().mean().backward()
    grad=sum(float(p.grad.float().norm()) for n,p in net.named_parameters() if 'lora_' in n and p.grad is not None)
    assert grad>0,'No encoder adapter gradient from supervised readout'
    net.zero_grad(set_to_none=True)
    with amp(device):loss=net.masked_loss(x,42)
    assert torch.isfinite(loss);loss.backward()
    native_grad=sum(float(p.grad.float().norm()) for n,p in net.named_parameters() if n.startswith('net.') and 'lora_' in n and p.grad is not None)
    assert native_grad>0,'Masked objective does not update encoder adapters'
    assert all(p.grad is None for n,p in net.net.named_parameters() if '.base.' in n),'Frozen base was updated'
    net.zero_grad(set_to_none=True)
    with torch.no_grad():a=net(x).float().cpu();b=net(x).float().cpu()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    # A real optimizer step followed by adapter restoration must recover the representation.
    with amp(device):v=net(x).float().square().mean()
    v.backward();opt=torch.optim.SGD([p for p in net.net.parameters() if p.requires_grad],lr=1e-3);opt.step();net.load_adapter(before)
    with torch.no_grad():c=net(x).float().cpu()
    torch.testing.assert_close(a,c,rtol=0,atol=0)
    out=RUN/'smoke'/f'{model}_{device}.json';save(out,dict(status='passed',model=model,device=device,supervised_adapter_gradient=grad,native_ssl_loss=float(loss.detach()),native_ssl_adapter_gradient=native_grad,feature_shape=list(z.shape),window_shape=list(w.shape),checks=['finite forward','encoder gradients for supervised and native SSL','base parameters frozen','deterministic unmasked readout','adapter round trip'],protocol=net.provenance()));print(out,flush=True)


def report():
    import numpy as np
    from fm_matrix.data import RUN,save
    from benchmark_evaluation import paired_bootstrap,score
    from confirm_matched_axis import holm
    rows=[];scope=[]
    for p in sorted((RUN/'evaluation').glob('*/*.json')):
        d=json.loads(p.read_text());scope.append(dict(file=str(p.relative_to(RUN)),status=d['status']))
        for r in d.get('comparisons',[]):rows.append(dict(file=str(p.relative_to(RUN)),**r))
    cv=[]
    for p in sorted((RUN/'training').glob('*/*/supervised_*_cv/complete.json')):
        d=json.loads(p.read_text());folds=d['folds'];y=np.concatenate([f['y'] for f in folds]);a=np.concatenate([f['frozen_prediction'] for f in folds]);b=np.concatenate([f['adapted_prediction'] for f in folds]);g=np.concatenate([f['test_subjects'] for f in folds]);delta,pv,ci,details=paired_bootstrap(y,a,b,n_boot=2000,groups=g,return_details=True)
        cv.append(dict(file=str(p.relative_to(RUN)),frozen_rho=score(y,a,False),adapted_rho=score(y,b,False),delta_rho=delta,raw_one_sided_bootstrap_p=pv,ci95=ci,**details))
    # Compare models only when their actual prediction rows and labels match exactly.
    from fm_matrix.models import MODELS
    between=[]
    for ref in sorted((RUN/'evaluation').glob('brainlm13_*/cognitive.json')):
        state=ref.parent.name.removeprefix('brainlm13_');a=json.loads(ref.read_text())
        for model in MODELS[1:]:
            other=RUN/'evaluation'/(model+'_'+state)/'cognitive.json'
            if not other.exists():raise ValueError('Missing matched model result '+str(other))
            b=json.loads(other.read_text());lookup={(r['cohort'],r['task'],r['method']):r for r in b['rows']}
            for r in a['rows']:
                if r['method'] not in ['FM','clinical+FM','source_axis']:continue
                q=lookup[r['cohort'],r['task'],r['method']]
                if r['subjects']!=q['subjects'] or not np.allclose(r['y'],q['y']):raise ValueError('Unmatched cross-model prediction rows')
                delta,pv,ci,detail=paired_bootstrap(r['y'],r['prediction'],q['prediction'],n_boot=2000,groups=r['subjects'],return_details=True)
                between.append(dict(state=state,reference='brainlm13',test=model,cohort=r['cohort'],task=r['task'],method=r['method'],delta_rho=delta,raw_one_sided_bootstrap_p=pv,ci95=ci,**detail))
    for collection in [rows,cv,between]:
        ps=[r.get('raw_one_sided_bootstrap_p') for r in collection];ix=[i for i,p in enumerate(ps) if p is not None and np.isfinite(p)]
        for i,p in zip(ix,holm([ps[i] for i in ix])):collection[i]['holm_within_matrix_block']=p
    save(RUN/'report.json',dict(status='complete',scope=scope,external_comparisons=rows,within_source_supervised=cv,between_models=between,multiplicity='Exploratory Holm families: external paired comparisons, within-source supervised comparisons, and between-model comparisons, separately. Bootstrap intervals condition on fitted predictions.'))


def main():
    from fm_matrix.models import MODELS
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','smoke','supervised','ssl','features','cognitive','cognitive_strict','identification','loso','physiology','report']);ap.add_argument('--model',choices=MODELS,default='brainlm13');ap.add_argument('--source',choices=['adni','nacc','none'],default='adni');ap.add_argument('--target',choices=['MEM','EF','none'],default='MEM');ap.add_argument('--objective',choices=['supervised','native_ssl','contrast_session','contrast_scale','contrast_both','frozen'],default='supervised');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--mode',choices=['external','cv'],default='external');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=ap.parse_args()
    import torch
    torch.set_num_threads(4)
    if a.action=='prepare':
        from fm_matrix.data import prepare
        prepare();return
    if a.action=='smoke':smoke(a.model,a.device);return
    if a.action=='report':report();return
    if a.action=='supervised':
        from fm_matrix.train import fit_supervised
        fit_supervised(a.source,a.model,a.target,a.seed,a.mode,a.device);return
    if a.action=='ssl':
        from fm_matrix.train import fit_ssl
        fit_ssl(a.source,a.model,a.objective,a.seed,a.device);return
    from fm_matrix import evaluate
    fn=evaluate.extraction if a.action=='features' else getattr(evaluate,a.action)
    args=[a.source,a.model,a.objective,a.target,a.seed]
    if a.action=='features':args.append(a.device)
    fn(*args)

if __name__=='__main__':main()
