"""Differentiable native model paths and matched rank-8 encoder adapters."""
import copy,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODELS=('brainlm13','brainlm111','brainlm650','brainjepa')

class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16):
        super().__init__();self.base=base;self.base.requires_grad_(False);self.scale=alpha/rank
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features));self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=5**.5)
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.lora_A),self.lora_B)*self.scale


def add_lora(net,jepa=False):
    names=[];suffixes={'qkv','proj','fc1','fc2'} if jepa else {'query','key','value','dense'}
    for name,m in list(net.named_modules()):
        if isinstance(m,nn.Linear) and name.split('.')[-1] in suffixes:
            parent,_,leaf=name.rpartition('.');setattr(net.get_submodule(parent) if parent else net,leaf,LoRALinear(m));names.append(name)
    if not names:raise ValueError('No LoRA modules matched')
    return names


def sdpa_forward(self,x,return_attn=None):
    if return_attn:raise ValueError('Attention maps are not used in the matched training protocol')
    b,n,c=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
    y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop.p if self.training else 0.,scale=self.scale)
    return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,c))),None


def efficient_jepa(net):
    from brainjepa_official.models.vision_transformer import Attention
    for m in net.modules():
        if isinstance(m,Attention):m.forward=types.MethodType(sdpa_forward,m)


def checkpoint_blocks(blocks):
    for block in blocks:
        original=block.forward
        def forward(*args,_original=original,**kw):
            if torch.is_grad_enabled():return checkpoint(_original,*args,use_reentrant=False,**kw)
            return _original(*args,**kw)
        block.forward=forward


def overlap_weights(starts,ends,n_windows=8,width=20):
    starts=np.asarray(starts);ends=np.asarray(ends);weights=[]
    for i in range(n_windows):
        a=np.maximum(0,np.minimum(ends,(i+1)*width)-np.maximum(starts,i*width))
        if a.sum()==0:raise ValueError('No real tokens overlap physiological window')
        weights.append(a/a.sum())
    return np.asarray(weights,dtype=np.float32)


class Encoder(nn.Module):
    def __init__(self,name,device='cpu',native_ssl=False):
        super().__init__();self.name=name;self.length=160 if name=='brainjepa' else 200;self.rois=450 if name=='brainjepa' else 424
        self.native_ssl=native_ssl;self.predictor=None;self.teacher=None
        if name=='brainlm13':
            import config as C
            C.NUM_TIMEPOINTS_PER_VOXEL=200
            from model_loader import load_brainlm_forpretraining
            from paths import OFFICIAL_BRAINLM_13M,XYZ_PATH
            from datasets import load_xyz
            self.checkpoint_path=Path(OFFICIAL_BRAINLM_13M);self.net=load_brainlm_forpretraining(self.checkpoint_path)
            self.register_buffer('xyz',load_xyz(XYZ_PATH));self.feature_dim=424*423//2;self.readout='emb_fc_L3'
        elif name in ['brainlm111','brainlm650']:
            from vitmae_input import load_vitmae
            from brainlm_input import roi_order_by_y
            from paths import BRAINLM_111M,BRAINLM_650M,XYZ_PATH
            self.checkpoint_path=Path(BRAINLM_111M if name=='brainlm111' else BRAINLM_650M);self.net,self.cfg=load_vitmae(self.checkpoint_path.parent)
            self.order=roi_order_by_y(XYZ_PATH);self.feature_dim=self.cfg.hidden_size;self.readout='CLS_final'
        elif name=='brainjepa':
            from brainjepa import load_brainjepa,DEFAULT_CKPT
            self.checkpoint_path=Path(DEFAULT_CKPT);self.net=load_brainjepa(verbose=False);efficient_jepa(self.net)
            self.feature_dim=self.net.embed_dim;self.readout='mean_final'
        else:raise ValueError(name)
        self.net.requires_grad_(False)
        self.adapter_modules=add_lora(self.net if name=='brainjepa' else self.net.vit,name=='brainjepa')
        if native_ssl and name=='brainjepa':
            from brainjepa_official.models.vision_transformer import vit_predictor
            ck=torch.load(self.checkpoint_path,map_location='cpu',weights_only=False);sd={k.removeprefix('module.'):v for k,v in ck['predictor'].items()}
            depth=1+max(int(k.split('.')[1]) for k in sd if k.startswith('predictor_blocks.'))
            self.predictor=vit_predictor(num_patches=4500,num_patches_2d=(450,10),embed_dim=768,predictor_embed_dim=sd['predictor_embed.weight'].shape[0],depth=depth,num_heads=self.net.num_heads,gradient_pos_embed=self.net.gradient_pos_embed.clone(),attn_mode='normal',add_w='mapping')
            self.predictor.load_state_dict(sd,strict=True)
            g=self.predictor.gradient_pos_embed;del self.predictor.gradient_pos_embed;self.predictor.register_buffer('gradient_pos_embed',g,persistent=False)
            efficient_jepa(self.predictor)
            # Clone before attaching checkpoint closures so teacher never calls student blocks.
            self.teacher=copy.deepcopy(self.net).requires_grad_(False).eval()
            checkpoint_blocks(self.predictor.predictor_blocks)
        blocks=self.net.blocks if name=='brainjepa' else self.net.vit.encoder.layer
        checkpoint_blocks(blocks)
        if native_ssl and name!='brainjepa':
            # Native reconstruction decoder is optimized with the encoder adapters.
            self.net.decoder.requires_grad_(True)
            checkpoint_blocks(self.net.decoder.decoder_layers)
        self.to(device);self.eval()

    def train(self,mode=True):
        super().train(False)  # Deterministic readouts; gradients remain enabled.
        return self

    def tokens(self,x):
        if self.name=='brainlm13':
            o=self.net.vit(signal_vectors=x,xyz_vectors=self.xyz[None].expand(len(x),-1,-1),noise=None,output_hidden_states=True,return_dict=True)
            return o.hidden_states[3][:,1:].reshape(len(x),424,x.shape[2]//20,-1)
        if self.name=='brainjepa':
            from brainjepa import normalise
            return self.net(normalise(x).unsqueeze(1)).reshape(len(x),450,10,-1)
        from vitmae_input import prepare_batch
        pixels=prepare_batch(x,self.order,self.cfg.image_size);n=self.net.vit.embeddings.patch_embeddings.num_patches
        return self.net.vit(pixel_values=pixels,noise=torch.zeros(len(x),n,device=x.device)).last_hidden_state

    def forward(self,x):
        z=self.tokens(x)
        if self.name=='brainlm13':
            z=z.mean(2);z=F.normalize(z-z.mean(-1,keepdim=True),dim=-1,eps=1e-8);a=z@z.transpose(1,2);i,j=torch.triu_indices(424,424,offset=1,device=x.device);return a[:,i,j]
        if self.name=='brainjepa':return z.mean((1,2))
        return z[:,0]

    def windows(self,x,n_windows=8):
        z=self.tokens(x)
        if self.name in ['brainlm13','brainjepa']:
            width=20 if self.name=='brainlm13' else 16;t=z.mean(1);starts=np.arange(t.shape[1])*width;ends=starts+width
        else:
            patch=self.cfg.patch_size;patch=patch[0] if isinstance(patch,(tuple,list)) else patch
            size=self.cfg.image_size;size=size[0] if isinstance(size,(tuple,list)) else size
            grid=size//patch;z=z[:,1:].reshape(len(x),grid,grid,-1)
            left=(size-x.shape[2])//2;top=(size-x.shape[1])//2
            spatial=np.maximum(0,np.minimum((np.arange(grid)+1)*patch,top+x.shape[1])-np.maximum(np.arange(grid)*patch,top))
            spatial=torch.as_tensor(spatial/spatial.sum(),device=z.device,dtype=z.dtype)
            t=(z*spatial[None,:,None,None]).sum(1);starts=np.arange(grid)*patch-left;ends=starts+patch
        w=torch.as_tensor(overlap_weights(starts,ends,n_windows),device=t.device,dtype=t.dtype)
        return torch.einsum('wt,btd->bwd',w,t)

    def masked_loss(self,x,seed):
        # Called only with full real windows. Masks held fixed for validation.
        gen=torch.Generator(device=x.device).manual_seed(seed)
        if self.name=='brainjepa':
            from brainjepa import normalise
            xx=normalise(x).unsqueeze(1);order=torch.rand(len(x),4500,device=x.device,generator=gen).argsort(1);context=order[:,:1125];target=order[:,1125:]
            with torch.no_grad():
                y=F.layer_norm(self.teacher(xx),(self.feature_dim,));y=torch.gather(y,1,target[:,:,None].expand(-1,-1,y.shape[-1]))
            pred=self.predictor(self.net(xx,masks=context),context,target)
            return F.smooth_l1_loss(pred.float(),y.float())
        if self.name=='brainlm13':
            emb=self.net.vit.embeddings;emb.training=True;emb.mask_ratio=.75;self.net.config.mask_ratio=.75
            noise=torch.rand(len(x),424*(x.shape[2]//20),generator=gen,device=x.device)
            try:return self.net(signal_vectors=x,xyz_vectors=self.xyz[None].expand(len(x),-1,-1),noise=noise).loss
            finally:emb.training=False
        # Use native decoder; explicitly exclude spatial/image padding from loss.
        from vitmae_input import prepare_batch
        pixels=prepare_batch(x,self.order,self.cfg.image_size);n=self.net.vit.embeddings.patch_embeddings.num_patches
        old=self.net.config.mask_ratio;mode=self.net.config.train_mode;self.net.config.mask_ratio=.75;self.net.config.train_mode='inference'
        order=torch.rand(len(x),n,device=x.device,generator=gen).argsort(1)
        noise=torch.ones(len(x),n,device=x.device);noise.scatter_(1,order[:,:int(n*.25)],0.)
        try:
            o=self.net.vit(pixel_values=pixels,noise=noise);pred=self.net.decoder(o.last_hidden_state,o.ids_restore).logits
            image=self.net.unpatchify(pred);mask=self.net.unpatchify(o.mask[:,:,None].expand_as(pred))
            h,w=pixels.shape[-2:];top=(h-424)//2;left=(w-x.shape[2])//2
            image=image[:,:,top:top+424,left:left+x.shape[2]];mask=mask[:,:,top:top+424,left:left+x.shape[2]]
            target=pixels[:,:,top:top+424,left:left+x.shape[2]]
            return (((image-target).float()**2)*mask).sum()/mask.sum().clamp_min(1)
        finally:self.net.config.mask_ratio=old;self.net.config.train_mode=mode

    @torch.no_grad()
    def update_teacher(self,momentum=.996):
        if self.teacher is not None:
            for a,b in zip(self.teacher.parameters(),self.net.parameters()):a.mul_(momentum).add_(b,alpha=1-momentum)

    def adapter_state(self):return {k:v.detach().cpu().clone() for k,v in self.net.state_dict().items() if 'lora_' in k}
    def load_adapter(self,state):
        expected=set(self.adapter_state())
        if set(state)!=expected:raise ValueError('Adapter keys mismatch')
        self.net.load_state_dict(state,strict=False)
    def provenance(self):
        return dict(model=self.name,readout=self.readout,rank=8,alpha=16,adapter_modules=self.adapter_modules,trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),length=self.length,rois=self.rois,native_ssl='masked_latent_prediction_EMA' if self.name=='brainjepa' else 'masked_BOLD_reconstruction')
