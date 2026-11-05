"""Strict stage-2 loaders and variant-native image objectives for fMRI-LM."""
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from new_fm.models import Encoder as ParentEncoder, FMRILMBranch, adapters
from fm_matrix.models import checkpoint_blocks

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'checkpoints/pretrained/new_fm_v1/fmrilm'


class Branch(FMRILMBranch):
    def __init__(self, variant):
        nn.Module.__init__(self)
        sys.path.insert(0, str(ROOT / 'third_party/fmrilm_2026'))
        from quantizers.tokenizer import Encoder as TokenEncoder
        from transformers import Qwen3Config, Qwen3Model
        first = torch.load(ASSETS / variant / 'stage1-tokenizer.pt', map_location='cpu', weights_only=False, mmap=True)
        if variant == 'mae':
            cfg = OmegaConf.create({'model': first['model_config']['tokenizer_model_config']})
        else:
            cfg = OmegaConf.create(OmegaConf.to_container(first['conf'], resolve=True))
        cfg.model.attn_mode = 'normal'
        self.tokenizer = TokenEncoder(cfg)
        self.tokenizer_config = cfg
        stage = torch.load(ASSETS / variant / 'stage2-pretrain-Qwen3-0.6B.pt', map_location='cpu', weights_only=False, mmap=True)
        sd = stage['model']
        enc = {k.removeprefix('tokenizer.encoder.'): v for k, v in sd.items() if k.startswith('tokenizer.encoder.')}
        self.tokenizer.load_state_dict(enc, strict=True)
        for key in ('weight', 'bias'):
            torch.testing.assert_close(sd['tokenizer.patch_embed.proj.' + key], enc['patch_embed.proj.' + key], rtol=0, atol=0)
        config = Qwen3Config.from_dict(json.loads((ASSETS / 'qwen_config.json').read_text()))
        config._attn_implementation = 'sdpa'
        self.qwen = Qwen3Model(config)
        prefix = 'llm.base_model.base_model.model.model.'
        q = {k.removeprefix(prefix): v.float() for k, v in sd.items() if k.startswith(prefix)}
        merged = {}
        for key, value in q.items():
            if '.lora_' in key:
                continue
            target = key.replace('.base_layer.', '.')
            if '.base_layer.weight' in key:
                base = key.removesuffix('.base_layer.weight')
                value = value + (q[base + '.lora_B.default.weight'] @ q[base + '.lora_A.default.weight']) * (stage['model_args']['lora_alpha'] / stage['model_args']['lora_r'])
            merged[target] = value
        self.qwen.load_state_dict(merged, strict=True)
        self.projection = nn.Sequential(nn.Linear(768, 1024), nn.GELU())
        self.projection[0].load_state_dict({k: sd['encode_transform_layer.0.' + k] for k in ('weight', 'bias')}, strict=True)
        self.position = nn.Embedding(450, 1024)
        self.position.load_state_dict({'weight': sd['pos_embed.weight']}, strict=True)
        self.load_audit = dict(variant=variant, tokenizer_tensors=len(enc), qwen_tensors=len(merged),
                               rank1_pretraining_lora_merged=True, strict=True,
                               attention='Full brain-block attention; no text; temporal position zero',
                               readout='Final brain-token Qwen mean, 1024 dimensions')


class VariantEncoder(ParentEncoder):
    variant = None

    def __init__(self, name='fmrilm', device='cpu', native_ssl=False):
        nn.Module.__init__(self)
        if name != 'fmrilm' or self.variant not in ('mae', 'vq-domain'):
            raise ValueError('Explicit fMRI-LM variant required')
        self.name = name
        self.native_ssl = native_ssl
        self.teacher = None
        self.aux = None
        self.length, self.rois, self.feature_dim = 160, 450, 1024
        self.readout = 'stage2_brain_only_Qwen_mean_final'
        self.checkpoint_path = ASSETS / self.variant / 'stage2-pretrain-Qwen3-0.6B.pt'
        self.net = Branch(self.variant)
        self.load_audit = self.net.load_audit
        self.net.requires_grad_(False)
        self.adapter_modules = adapters(self.net, 'fmrilm')
        checkpoint_blocks(self.net.tokenizer.blocks)
        checkpoint_blocks(self.net.qwen.layers)
        if native_ssl:
            first = torch.load(ASSETS / self.variant / 'stage1-tokenizer.pt', map_location='cpu', weights_only=False, mmap=True)
            if self.variant == 'mae':
                from brain_encoder.vit_mae import MaskedAutoencoderViT
                config = first['model_config']
                self.aux = MaskedAutoencoderViT(img_size=(450, 160), patch_size=160,
                    in_chans=config['in_chans'], embed_dim=768, depth=12, num_heads=12,
                    decoder_embed_dim=config['decoder_embed_dim'], decoder_depth=config['decoder_depth'],
                    decoder_num_heads=config['decoder_num_heads'], mlp_ratio=config['mlp_ratio'],
                    qkv_bias=config['qkv_bias'], qk_scale=config['qk_scale'],
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0., attn_mode='normal',
                    sep_pos_embed=config['sep_pos_embed'], cls_embed=config['cls_embed'],
                    norm_pix_loss=config['norm_pix_loss'])
                self.aux.load_state_dict(first['model'], strict=True)
                self.aux.requires_grad_(False)
                # The pretrained decoder receives the *stage-2* encoder whose
                # weights and LoRA adapters are shared with the downstream path.
                self.aux.patch_embed = self.net.tokenizer.patch_embed
                self.aux.blocks = self.net.tokenizer.blocks
                self.aux.norm = self.net.tokenizer.ln_post
                self.aux.pos_embed = self.net.tokenizer.pos_embed
                for param_name, param in self.aux.named_parameters():
                    if param_name.startswith(('decoder_', 'mask_token')):
                        param.requires_grad_(True)
                checkpoint_blocks(self.aux.decoder_blocks)
            else:
                from quantizers.tokenizer import Tokenizer
                self.aux = Tokenizer(self.net.tokenizer_config)
                self.aux.load_state_dict({k.removeprefix('quantizer.'): v for k, v in first['model'].items() if k.startswith('quantizer.')}, strict=True)
                self.aux.requires_grad_(False)
                self.aux.encoder = self.net.tokenizer
                self.aux.decoder.requires_grad_(True)
                self.aux.quantize.requires_grad_(True)
                self.net.tokenizer.conv_out.requires_grad_(True)
        self.to(device)
        self.eval()

    def ssl_loss(self, x, seed):
        if self.variant != 'mae':
            return super().ssl_loss(x, seed)
        xx = x[:, :, :160]
        devices = [xx.device.index] if xx.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            latent, mask, restore = self.aux.forward_encoder(xx, mask_ratio=.75)
            reconstruction = self.aux.forward_decoder(latent, restore).reshape_as(xx)
        valid = (xx.abs().sum(1, keepdim=True) > 0).expand_as(xx)
        observed_mask = valid * mask[:, :, None]
        return ((reconstruction.float() - xx.float()).square() * observed_mask).sum() / observed_mask.sum().clamp_min(1)

    def provenance(self):
        data = super().provenance()
        data.update(checkpoint_variant=self.variant,
                    native_ssl=('masked_BOLD_reconstruction_75pct' if self.variant == 'mae'
                                else 'image_VQ_reconstruction_and_quantization'),
                    pretraining_domain_or_text_alignment='Not repeated without paired text; image-only source adaptation')
        return data
