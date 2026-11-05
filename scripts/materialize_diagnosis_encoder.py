#!/usr/bin/env python3
"""Materialize the one canonical diagnosis encoder for existing consumers."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from diagnosis_posttraining import RUN,verify,location
from fm_matrix.data import dataset,sha,save
from fm_matrix.models import Encoder
from materialize_matrix_encoder import merged_state


def materialize(target,seed,destination):
    verify(); root=location('nacc','brainlm13',target,seed,'external')
    if not (root/'complete.json').exists(): raise ValueError('Canonical diagnosis training incomplete')
    checkpoint=root/'fold0/adapter.pt'; ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['diagnosis_protocol_sha256']!=sha(RUN/'protocol.json'): raise ValueError('Diagnosis provenance mismatch')
    net=Encoder('brainlm13','cpu')
    if sha(net.checkpoint_path)!=ck['source_checkpoint_sha256']: raise ValueError('Base checkpoint changed')
    net.load_adapter(ck['adapter']); state=merged_state(net.net)
    dst=Path(destination); dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_suffix('.tmp'); torch.save(state,tmp)
    from model_loader import load_brainlm_forpretraining
    plain=load_brainlm_forpretraining(tmp).eval(); x,_=dataset('nacc','brainlm13'); x=x[:1]
    with torch.no_grad():
        before=net(x)
        h=plain.vit(signal_vectors=x,xyz_vectors=net.xyz[None],noise=None,output_hidden_states=True,return_dict=True).hidden_states[3][:,1:].reshape(1,424,10,-1).mean(2)
        h=torch.nn.functional.normalize(h-h.mean(-1,keepdim=True),dim=-1,eps=1e-8); v=h@h.transpose(1,2); i,j=torch.triu_indices(424,424,1); after=v[:,i,j]
    torch.testing.assert_close(before,after,atol=2e-5,rtol=2e-4)
    if dst.exists() and sha(dst)!=sha(tmp): dst.replace(dst.with_name(dst.name+'.superseded-'+sha(dst)[:12]))
    tmp.replace(dst)
    save(dst.with_suffix('.meta.json'),dict(status='complete',canonical_checkpoint=str(checkpoint),canonical_checkpoint_sha256=sha(checkpoint),diagnosis_protocol_sha256=sha(RUN/'protocol.json'),sha256=sha(dst),operation='Encoder-only LoRA merge; no extra training',max_abs_forward_error=float((before-after).abs().max())))
