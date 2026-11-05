#!/usr/bin/env python3
"""Export a canonical matrix encoder for supplementary analyses, without training.

Only encoder LoRA deltas are exported. This is not a native-SSL decoder checkpoint
and no old-protocol predictions or validation scores are synthesized.
"""
from pathlib import Path
import argparse,json,sys,os
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fm_matrix.models import Encoder,LoRALinear
from fm_matrix.data import RUN,dataset,sha,save
from fm_matrix.evaluate import checkpoint_path
from fm_matrix.train import location


def merged_state(net):
    state={k.replace('.base.','.'):v.detach().cpu().clone() for k,v in net.state_dict().items() if 'lora_' not in k}
    for name,module in net.named_modules():
        if isinstance(module,LoRALinear):
            state[name+'.weight']=(module.base.weight+module.scale*(module.lora_B@module.lora_A)).detach().cpu()
    return state


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['adni','nacc'],required=True);ap.add_argument('--objective',choices=['supervised','native_ssl','contrast_session','contrast_scale','contrast_both'],required=True);ap.add_argument('--target',choices=['MEM','EF','none'],default='none');ap.add_argument('--seed',type=int,required=True);ap.add_argument('--destination',type=Path,required=True);ap.add_argument('--alpha-dir',type=Path);a=ap.parse_args();torch.set_num_threads(2)
    checkpoint=checkpoint_path(a.source,'brainlm13',a.objective,a.target,a.seed)
    result=location(a.source,'brainlm13',a.objective,a.target,a.seed,'external')/'complete.json'
    if not result.exists() or json.loads(result.read_text())['status']!='complete':raise ValueError('Canonical source training is not complete')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['model']!='brainlm13' or ck['intake_sha256']!=sha(RUN/'intake.json'):raise ValueError('Canonical adapter/input provenance mismatch')
    net=Encoder('brainlm13','cpu')
    if ck['source_checkpoint_sha256']!=sha(net.checkpoint_path):raise ValueError('Pretrained checkpoint changed')
    net.load_adapter(ck['adapter']);state=merged_state(net.net)
    from model_loader import load_brainlm_forpretraining
    dst=a.destination;dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.tmp');torch.save(state,tmp)
    plain=load_brainlm_forpretraining(tmp).eval();x,_=dataset(a.source,'brainlm13');x=x[:1]
    with torch.no_grad():
        before=net(x)
        h=plain.vit(signal_vectors=x,xyz_vectors=net.xyz[None],noise=None,output_hidden_states=True,return_dict=True).hidden_states[3][:,1:].reshape(1,424,10,-1).mean(2)
        h=torch.nn.functional.normalize(h-h.mean(-1,keepdim=True),dim=-1,eps=1e-8);v=h@h.transpose(1,2);i,j=torch.triu_indices(424,424,1);after=v[:,i,j]
    torch.testing.assert_close(before,after,atol=2e-5,rtol=2e-4);error=float((before-after).abs().max())
    # Preserve any earlier standalone export with a content-addressed archived name.
    if dst.exists() and sha(dst)!=sha(tmp):
        archived=dst.with_name(dst.name+'.superseded-'+sha(dst)[:12]);dst.replace(archived)
        meta=dst.with_suffix('.meta.json')
        if meta.exists():meta.replace(archived.with_name(archived.name+'.meta.json'))
    tmp.replace(dst)
    if a.alpha_dir:
        a.alpha_dir.mkdir(parents=True,exist_ok=True);base=torch.load(net.checkpoint_path,map_location='cpu',weights_only=True)
        if base.keys()!=state.keys():raise ValueError('Merged encoder keys changed')
        for alpha in [0,1,2]:
            scaled={k:base[k]+alpha*(state[k]-base[k]) if base[k].is_floating_point() else base[k].clone() for k in base}
            p=a.alpha_dir/f'brainlm13m_adni{a.target}_level_a{alpha}.pt';q=p.with_suffix('.tmp');torch.save(scaled,q);q.replace(p)
    save(dst.with_suffix('.meta.json'),dict(status='complete',operation='CPU merge of canonical matrix encoder; no additional training',canonical_checkpoint=str(checkpoint),canonical_checkpoint_sha256=sha(checkpoint),canonical_result=str(result),canonical_intake_sha256=sha(RUN/'intake.json'),source=a.source,objective=a.objective,target=a.target,seed=a.seed,sha256=sha(dst),merge_forward_max_abs_error=error,scope='Encoder only; native SSL decoder is not exported. Auxiliary analyses may use separately documented readouts and evaluation procedures.'))
    print('Materialized',dst,'max merge error',error,flush=True)

if __name__=='__main__':main()
