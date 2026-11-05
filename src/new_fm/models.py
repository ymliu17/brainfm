"""Pinned official encoders, strict loading, frozen readouts and rank8 adapters."""
import copy
import json
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
from new_fm.data import ROOT
from fm_matrix.models import LoRALinear, checkpoint_blocks

ASSETS=ROOT/'checkpoints/pretrained/new_fm_v1'
READOUTS={'semantoks':'teacher_CLS_final','fmrilm':'stage2_brain_only_Qwen_mean_final'}

def semantoks_backbone():
    sys.path.insert(0,str(ROOT/'third_party/brain_semantoks'))
    from model.semantoks import CNN_TF
    import yaml
    c=yaml.safe_load((ASSETS/'semantoks/config_used.yaml').read_text());m=c['model'];d=c['data']
    net=CNN_TF(target_time_length=d['target_signal_length'],patch_size=d['patch_size'],
        network_data_path=str(ASSETS/'semantoks/network_mapping.npz'),
        atlas_names=['schaefer400','tian3','buckner7'],atlas_network_counts=[7,1,1],
        embedding_dim=m['embedding_dim'],depth=m['depth'],heads=m['embedding_dim']//64,
        mlp_dim=m['embedding_dim']*4,drop_path_rate=m['drop_path_rate'],
        layer_scale_init_value=m['layer_scale_init_value'],emb_dropout=m['emb_dropout'],
        do_masking=True,tokenizer_config=m['tokenizer']['config'],
        tokenizer_final_norm=m['tokenizer']['final_norm'],flash_attention='normal')
    return net,c

def adapters(net,kind):
    names=[]
    for name,layer in list(net.named_modules()):
        if not isinstance(layer,nn.Linear):continue
        leaf=name.split('.')[-1]
        use=(('transformer_blocks.' in name and ('.attn.' in name or '.mlp.' in name)) if kind=='semantoks'
             else (('.blocks.' in name and leaf in ('qkv','proj','fc1','fc2')) or
                   (name.startswith('qwen.layers.') and leaf in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'))))
        if use:
            parent,_,key=name.rpartition('.');setattr(net.get_submodule(parent),key,LoRALinear(layer));names.append(name)
    if not names:raise ValueError('No adapter sites')
    return names

class FMRILMBranch(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0,str(ROOT/'third_party/fmrilm_2026'))
        from quantizers.tokenizer import Encoder as TokenEncoder
        from omegaconf import OmegaConf
        from transformers import Qwen3Config,Qwen3Model
        p=ASSETS/'fmrilm/vq-contrastive'
        first=torch.load(p/'stage1-tokenizer.pt',map_location='cpu',weights_only=False,mmap=True)
        cfg=OmegaConf.create(OmegaConf.to_container(first['conf'],resolve=True));cfg.model.attn_mode='normal'
        self.tokenizer=TokenEncoder(cfg);self.tokenizer_config=cfg
        stage=torch.load(p/'stage2-pretrain-Qwen3-0.6B.pt',map_location='cpu',weights_only=False,mmap=True)
        sd=stage['model'];enc={k.removeprefix('tokenizer.encoder.'):v for k,v in sd.items() if k.startswith('tokenizer.encoder.')}
        self.tokenizer.load_state_dict(enc,strict=True)
        for key in ('weight','bias'):
            torch.testing.assert_close(sd['tokenizer.patch_embed.proj.'+key],enc['patch_embed.proj.'+key],rtol=0,atol=0)
        qc=Qwen3Config.from_dict(json.loads((ASSETS/'fmrilm/qwen_config.json').read_text()));qc._attn_implementation='sdpa'
        self.qwen=Qwen3Model(qc)
        prefix='llm.base_model.base_model.model.model.'
        q={k.removeprefix(prefix):v.float() for k,v in sd.items() if k.startswith(prefix)}
        # Merge the released rank1 language-pretraining adapters into immutable base weights.
        merged={}
        for k,v in q.items():
            if '.lora_' in k:continue
            key=k.replace('.base_layer.','.')
            if '.base_layer.weight' in k:
                base=k.removesuffix('.base_layer.weight')
                v=v+(q[base+'.lora_B.default.weight']@q[base+'.lora_A.default.weight'])*(stage['model_args']['lora_alpha']/stage['model_args']['lora_r'])
            merged[key]=v
        self.qwen.load_state_dict(merged,strict=True)
        self.projection=nn.Sequential(nn.Linear(768,1024),nn.GELU())
        self.projection[0].load_state_dict({k:sd['encode_transform_layer.0.'+k] for k in ('weight','bias')},strict=True)
        self.position=nn.Embedding(450,1024);self.position.load_state_dict({'weight':sd['pos_embed.weight']},strict=True)
        self.load_audit=dict(tokenizer_tensors=len(enc),qwen_tensors=len(merged),rank1_pretraining_lora_merged=True,
            attention='Fully connected brain block, constant temporal position0, matching official fMRI-only path; no text/labels/prompts.',
            readout='Mean final Qwen hidden state across450 brain tokens; benchmark-specified scan readout.')

    def tokens(self,x):
        features,_=self.tokenizer(x[:,:,:160]);h=self.projection(features.transpose(1,2))+self.position.weight[None]
        b,n,_=h.shape
        # Explicit all-zero additive 4D mask prevents implicit causal masking.
        return self.qwen(inputs_embeds=h,attention_mask=torch.zeros(b,1,n,n,device=h.device,dtype=h.dtype),
            position_ids=torch.zeros(b,n,device=h.device,dtype=torch.long),use_cache=False,return_dict=True).last_hidden_state
    def forward(self,x):return self.tokens(x).mean(1)

class Encoder(nn.Module):
    def __init__(self,name,device='cpu',native_ssl=False):
        super().__init__();self.name=name;self.native_ssl=native_ssl;self.teacher=None;self.aux=None
        self.readout=READOUTS[name];self.length=100 if name=='semantoks' else 160
        if name=='semantoks':
            self.rois=457;self.feature_dim=768
            self.checkpoint_path=ASSETS/'semantoks/brainsemantoks_ckpt_epoch_100.pth'
            blob=torch.load(self.checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
            self.net,self.config=semantoks_backbone()
            state=blob['model_state_dict'];s={k.removeprefix('teacher_encoder.'):v for k,v in state.items() if k.startswith('teacher_encoder.')}
            self.net.load_state_dict(s,strict=True)
            self.load_audit=dict(strict_teacher_tensors=len(s),missing_cerebellum='Pretrained mask embedding replaces all Buckner7 tokens post-tokenizer')
        elif name=='fmrilm':
            self.rois=450;self.feature_dim=1024;self.checkpoint_path=ASSETS/'fmrilm/vq-contrastive/stage2-pretrain-Qwen3-0.6B.pt'
            self.net=FMRILMBranch();self.load_audit=self.net.load_audit
        else:raise ValueError(name)
        self.net.requires_grad_(False);self.adapter_modules=adapters(self.net,name)
        if native_ssl:
            if name=='semantoks':
                from model.dino_head import DINOHead
                m=self.config['model']
                self.aux=DINOHead(768,m['projection_output_dim'],hidden_dim=m['projection_hidden_dim'],
                    bottleneck_dim=m['projection_bottleneck_dim'],nlayers=m['projection_nlayers'],remove_last_layer=True,normalize=True)
                self.aux.load_state_dict({k.removeprefix('teacher_predictor.'):v for k,v in state.items() if k.startswith('teacher_predictor.')},strict=True)
                self.teacher=copy.deepcopy(self.net).requires_grad_(False)
                self.teacher_head=copy.deepcopy(self.aux).requires_grad_(False)
            else:
                from quantizers.tokenizer import Tokenizer
                self.aux=Tokenizer(self.net.tokenizer_config)
                first=torch.load(ASSETS/'fmrilm/vq-contrastive/stage1-tokenizer.pt',map_location='cpu',weights_only=False,mmap=True)
                self.aux.load_state_dict({k.removeprefix('quantizer.'):v for k,v in first['model'].items() if k.startswith('quantizer.')},strict=True)
                self.aux.encoder=self.net.tokenizer
                self.aux.requires_grad_(False)
                for n,p in self.net.tokenizer.named_parameters():
                    if 'lora_' in n:p.requires_grad_(True)
                self.aux.decoder.requires_grad_(True)
                self.aux.quantize.requires_grad_(True)
                self.net.tokenizer.conv_out.requires_grad_(True)
        self.to(device);self.eval()

    def train(self,mode=True):
        super().train(False)
        return self

    @staticmethod
    def sem_tokens(net,x,mask=None):
        mask=torch.zeros(len(x),9,5,dtype=torch.bool,device=x.device) if mask is None else mask.clone()
        mask[:,8]=True
        return net(x[:,:,:100],mask=mask)['tokens']

    def tokens(self,x):
        return self.sem_tokens(self.net,x) if self.name=='semantoks' else self.net.tokens(x)

    def forward(self,x):
        z=self.tokens(x);return z[:,0] if self.name=='semantoks' else z.mean(1)

    def ssl_loss(self,x,seed):
        # Coding-rate log determinants are unstable with bf16 covariance products.
        if torch.is_autocast_enabled(x.device.type):
            with torch.autocast(x.device.type,enabled=False):return self.ssl_loss(x.float(),seed)
        if self.name=='fmrilm':
            xx=x[:,:,:160];_,_,z,q=self.aux.encode(xx);rec=self.aux.decode(z)
            # Architectural right-padding is not an observed BOLD target.
            valid=xx.abs().sum(1,keepdim=True)>0
            error=(rec.float().reshape_as(xx)-xx.float()).square()
            reconstruction=(error*valid).sum()/(valid.sum()*xx.shape[1]).clamp_min(1)
            return reconstruction+self.aux.quantizer_weight*q['quantizer_loss']
        from model.simdino_loss import MCRLoss,CosinePatchLoss
        gen=torch.Generator(device=x.device).manual_seed(seed);views=[];masks=[]
        for v in range(2):
            starts=[]
            for row in x:
                valid=(row[:450].abs().sum(0)>0).nonzero().flatten()
                last=int(valid[-1])+1 if len(valid) else 100
                starts.append(int(torch.randint(max(1,min(last,180)-100+1),(1,),device=x.device,generator=gen)))
            views.append(torch.stack([row[:,s:s+100] for row,s in zip(x,starts)]))
            mask=torch.rand(len(x),9,5,device=x.device,generator=gen)<.75;mask[:,8]=False;masks.append(mask)
        with torch.no_grad():tz=[self.sem_tokens(self.teacher,v) for v in views]
        sz=[self.sem_tokens(self.net,v,m) for v,m in zip(views,masks)]
        lossfn=MCRLoss(out_dim=256,expa_type=1,reduce_cov=0,eps=.05,coeff=1,center=False)
        sc=[self.aux(z[:,0]) for z in sz]
        with torch.no_grad():tc=[self.teacher_head(z[:,0]) for z in tz]
        loss=lossfn(sc,tc,no_diag=True,normalized=True)[0]
        sn=[self.aux(z[:,1:].reshape(len(x),9,5,768)[:,:8].mean(2)).reshape(-1,128) for z in sz]
        with torch.no_grad():tn=[self.teacher_head(z[:,1:].reshape(len(x),9,5,768)[:,:8].mean(2)).reshape(-1,128) for z in tz]
        loss=loss+.5*lossfn(sn,tn,no_diag=True,normalized=True)[0]
        flat=torch.cat([m.flatten(1) for m in masks]);sp=torch.cat([self.aux(z[:,1:][m.flatten(1)]) for z,m in zip(sz,masks)])
        with torch.no_grad():tp=torch.cat([self.teacher_head(z[:,1:][m.flatten(1)]) for z,m in zip(tz,masks)])
        weights=(1/flat.sum(-1).clamp_min(1))[:,None].expand_as(flat)[flat]
        patch=CosinePatchLoss(patch_out_dim=256,center=False).forward_masked(sp,tp,flat,len(sp),weights)[0]
        return (loss+.5*patch).mean()

    @torch.no_grad()
    def update_teacher(self,momentum=.996):
        if self.teacher is not None:
            for a,b in zip(self.teacher.parameters(),self.net.parameters()):a.mul_(momentum).add_(b,alpha=1-momentum)
            for a,b in zip(self.teacher_head.parameters(),self.aux.parameters()):a.mul_(momentum).add_(b,alpha=1-momentum)

    def adapter_state(self):
        # Inference remains the pretrained-teacher branch for Semantoks SSL.
        net=self.teacher if self.teacher is not None else self.net
        return {k:v.detach().cpu().clone() for k,v in net.state_dict().items() if 'lora_' in k}

    def load_adapter(self,state):
        expected={k for k in self.net.state_dict() if 'lora_' in k}
        if set(state)!=expected:raise ValueError('Adapter keys mismatch')
        self.net.load_state_dict(state,strict=False)

    def provenance(self):
        return dict(model=self.name,readout=self.readout,features=self.feature_dim,rank=8,alpha=16,
            adapter_modules=self.adapter_modules,trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            length=self.length,rois=self.rois,load_audit=self.load_audit,
            native_ssl='Self-distillation with MCR and masked-token cosine loss; observed networks only' if self.name=='semantoks' else
                'Image-only VQ reconstruction + quantization objective; stage1 decoder and stage2 image encoder, Qwen retained; paired-text contrastive loss unavailable without text')
