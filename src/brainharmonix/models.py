"""Official F encoder and predictor; SDPA adapter preserves attention-mask semantics."""
import copy
import sys
import types
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from fm_matrix.models import LoRALinear, checkpoint_blocks
from brainharmonix.data import ROOT, PATCHES

REPO=ROOT/'third_party/brain_harmony'
CKPT=ROOT/'checkpoints/pretrained/brainharmonix_f/model.pth'

def sdpa(self,x,attention_mask=None,output_attentions=False):
    if output_attentions:raise ValueError('Attention maps not used')
    b,n,c=x.shape
    q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
    mask=None if attention_mask is None else attention_mask[:,None,None,:].bool()
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=self.scale)
    return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,c))),None

def native_modules():
    # Run in isolated workers so upstream generic `libs` cannot collide with other models.
    sys.path.insert(0,str(REPO))
    from libs import flex_transformer as ft
    from libs.position_embedding import BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed as Pos
    ft.Attention.forward=sdpa
    c=types.SimpleNamespace(grid_size=(400,18),embed_dim=768,predictor_embed_dim=384,
        cls_token=False,grad_dim=30,geoh_dim=200,use_pos_embed_decoder=True,
        gradient=str(REPO/'brainharmony_pos_embed/gradient_mapping_400.csv'),
        geo_harm=str(REPO/'brainharmony_pos_embed/schaefer400_roi_eigenmodes.csv'))
    pos=Pos('cpu',c)
    # Upstream stores these as plain tensors; buffers ensure .to(device) moves them.
    for key in ('geo_harm','gradient'):
        v=getattr(pos,key);delattr(pos,key);pos.register_buffer(key,v,persistent=False)
    net=ft.FlexVisionTransformer(pos_embed=pos,cls_token=None,img_size=(400,48*PATCHES),patch_size=48,
        embed_dim=768,depth=12,num_heads=12,mlp_ratio=4,qkv_bias=True,
        norm_layer=partial(nn.LayerNorm,eps=1e-6),attn_mode='eager')
    return net,ft

def load_strict(net,sd):
    sd=dict(sd);regenerated=[]
    for key in list(sd):
        if key.endswith(('emb_h_encoder','emb_h_decoder')):
            # Released weights have11 temporal patches; official downstream uses18.
            # Only deterministic sine/cosine positions may be regenerated.
            if sd[key].shape!=net.state_dict()[key].shape:
                sd[key]=net.state_dict()[key];regenerated.append(key)
    net.load_state_dict(sd,strict=True)
    return regenerated

class Encoder(nn.Module):
    def __init__(self,name='brainharmonix_f',device='cpu',native_ssl=False):
        super().__init__()
        if name!='brainharmonix_f':raise ValueError(name)
        self.name=name;self.readout='EMA_final_valid_token_mean';self.feature_dim=768
        self.rois=400;self.length=None;self.checkpoint_path=CKPT;self.teacher=None;self.predictor=None
        state=torch.load(CKPT,map_location='cpu',weights_only=True,mmap=True)
        self.net,ft=native_modules()
        regenerated=load_strict(self.net,{k.removeprefix('encoder_ema.'):v for k,v in state.items() if k.startswith('encoder_ema.')})
        self.net.requires_grad_(False);self.adapter_modules=[]
        for name,layer in list(self.net.named_modules()):
            if name.startswith('blocks.') and isinstance(layer,nn.Linear) and name.split('.')[-1] in ('qkv','proj','fc1','fc2'):
                parent,_,leaf=name.rpartition('.');setattr(self.net.get_submodule(parent),leaf,LoRALinear(layer));self.adapter_modules.append(name)
        if native_ssl:
            self.predictor=ft.VisionTransformerPredictor(pos_embed=copy.deepcopy(self.net.pos_embed),cls_token=None,
                embed_dim=768,predictor_embed_dim=384,depth=6,num_heads=12,attn_mode='eager')
            load_strict(self.predictor,{k.removeprefix('decoder.'):v for k,v in state.items() if k.startswith('decoder.')})
            self.predictor.predictor_pos_embed.requires_grad_(False)
            self.teacher=copy.deepcopy(self.net).requires_grad_(False)
            checkpoint_blocks(self.predictor.predictor_blocks)
        checkpoint_blocks(self.net.blocks)
        self.load_audit=dict(strict_learned_weights=True,regenerated_fixed_positions=regenerated,
            attention='SDPA with explicit valid-key masks; no causal mask',native_predictor=native_ssl)
        self.to(device);self.eval()

    def train(self,mode=True):
        super().train(False);return self

    @staticmethod
    def unpack(row):
        n=int(row[400,0]);patch=int(row[401,0]);total=PATCHES*patch
        if not 0<n<=total or total>row.shape[-1]:raise ValueError('Invalid packed native input')
        valid=torch.arange(PATCHES,device=row.device)<((n+patch-1)//patch)
        return row[None,None,:400,:total],patch,valid.repeat(400)[None]

    def token_outputs(self,x):
        out=[];masks=[]
        for row in x:
            signal,ps,mask=self.unpack(row)
            ix=mask[0].nonzero().flatten()
            z=self.net(signal,ps,masks=ix[None])
            # Dropping fully masked tokens is equivalent for observed queries.
            full=z.new_zeros(1,400*PATCHES,768).index_copy(1,ix,z)
            out.append(full);masks.append(mask)
        return torch.cat(out),torch.cat(masks)

    def tokens(self,x):
        z,mask=self.token_outputs(x)
        # Rescale so tokens.mean(1) equals valid-token mean; exported raw summaries
        # below use token_outputs and never interpret these padded values as signal.
        return z*mask[:,:,None]*(z.shape[1]/mask.sum(1))[:,None,None]

    def forward(self,x):
        z,m=self.token_outputs(x);return (z*m[:,:,None]).sum(1)/m.sum(1)[:,None]

    def ssl_loss(self,x,seed):
        gen=torch.Generator(device=x.device).manual_seed(seed);losses=[]
        for row in x:
            signal,ps,mask=self.unpack(row)
            valid=mask[0].nonzero().flatten();perm=valid[torch.randperm(len(valid),device=x.device,generator=gen)]
            k=max(1,round(len(valid)*.25));context=perm[:k][None];target=perm[k:][None]
            z=self.net(signal,ps,masks=context,attention_mask=torch.ones_like(context,dtype=torch.bool))
            pred=self.predictor(z,context,target,attention_masks=torch.ones(1,len(valid),dtype=torch.bool,device=x.device))
            with torch.no_grad():
                h=self.teacher(signal,ps,masks=valid[None])
                h=F.layer_norm(h,(h.shape[-1],))[:,torch.searchsorted(valid,target[0])]
            losses.append(F.smooth_l1_loss(pred.float(),h.float()))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update_teacher(self,momentum=.996):
        if self.teacher is not None:
            for a,b in zip(self.teacher.parameters(),self.net.parameters()):a.mul_(momentum).add_(b,alpha=1-momentum)

    def adapter_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.net.state_dict().items() if 'lora_' in k}

    def load_adapter(self,state):
        if set(state)!={k for k in self.net.state_dict() if 'lora_' in k}:raise ValueError('Adapter key mismatch')
        self.net.load_state_dict(state,strict=False)

    def provenance(self):
        return dict(model=self.name,readout=self.readout,features=768,rank=8,alpha=16,
            adapter_modules=self.adapter_modules,trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            rois=400,temporal_patches=18,patch_seconds=35.28,native_TR=True,load_audit=self.load_audit,
            native_ssl='Masked latent prediction using released predictor;75% uniform observed-token masking, EMA0.996, smoothL1. Benchmark masking adaptation; padded tokens excluded.',
            normalization='Per-scan median/IQR benchmark variant; original corpus scaling statistics unavailable')
